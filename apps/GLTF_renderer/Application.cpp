/*
 * Copyright (c) 2013-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cuda/config.h>

#include "Application.h"
#include "CheckMacros.h"


#ifdef _WIN32
 // The cfgmgr32 header is necessary for interrogating driver information in the registry.
#include <cfgmgr32.h>
// For convenience the library is also linked in automatically using the #pragma command.
#pragma comment(lib, "Cfgmgr32.lib")
#else
#include <dlfcn.h>
#endif

// STB
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <map>

#include "cuda/hit_group_data.h"
#include "cuda/light_definition.h"
#include "cuda/vector_math.h"

#include "Record.h"
#include "Mesh.h"

#include <glm/gtc/matrix_access.hpp>

// CUDA Driver API version of the OpenGL interop header. 
#include <cudaGL.h>

#include <MyAssert.h>

#ifdef _WIN32
// Code based on helper function in optix_stubs.h
static void* optixLoadWindowsDll(void)
{
  const char* optixDllName = "nvoptix.dll";
  void* handle = NULL;

  // Get the size of the path first, then allocate
  unsigned int size = GetSystemDirectoryA(NULL, 0);
  if (size == 0)
  {
    // Couldn't get the system path size, so bail
    return NULL;
  }

  size_t pathSize = size + 1 + strlen(optixDllName);
  char* systemPath = (char*) malloc(pathSize);

  if (GetSystemDirectoryA(systemPath, size) != size - 1)
  {
    // Something went wrong
    free(systemPath);
    return NULL;
  }

  strcat(systemPath, "\\");
  strcat(systemPath, optixDllName);

  handle = LoadLibraryA(systemPath);

  free(systemPath);

  if (handle)
  {
    return handle;
  }

  // If we didn't find it, go looking in the register store.  Since nvoptix.dll doesn't
  // have its own registry entry, we are going to look for the OpenGL driver which lives
  // next to nvoptix.dll. 0 (null) will be returned if any errors occured.

  static const char* deviceInstanceIdentifiersGUID = "{4d36e968-e325-11ce-bfc1-08002be10318}";
  const ULONG        flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;
  ULONG              deviceListSize = 0;

  if (CM_Get_Device_ID_List_SizeA(&deviceListSize, deviceInstanceIdentifiersGUID, flags) != CR_SUCCESS)
  {
    return NULL;
  }

  char* deviceNames = (char*) malloc(deviceListSize);

  if (CM_Get_Device_ID_ListA(deviceInstanceIdentifiersGUID, deviceNames, deviceListSize, flags))
  {
    free(deviceNames);
    return NULL;
  }

  DEVINST devID = 0;

  // Continue to the next device if errors are encountered.
  for (char* deviceName = deviceNames; *deviceName; deviceName += strlen(deviceName) + 1)
  {
    if (CM_Locate_DevNodeA(&devID, deviceName, CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS)
    {
      continue;
    }

    HKEY regKey = 0;
    if (CM_Open_DevNode_Key(devID, KEY_QUERY_VALUE, 0, RegDisposition_OpenExisting, &regKey, CM_REGISTRY_SOFTWARE) != CR_SUCCESS)
    {
      continue;
    }

    const char* valueName = "OpenGLDriverName";
    DWORD       valueSize = 0;

    LSTATUS     ret = RegQueryValueExA(regKey, valueName, NULL, NULL, NULL, &valueSize);
    if (ret != ERROR_SUCCESS)
    {
      RegCloseKey(regKey);
      continue;
    }

    char* regValue = (char*) malloc(valueSize);
    ret = RegQueryValueExA(regKey, valueName, NULL, NULL, (LPBYTE) regValue, &valueSize);
    if (ret != ERROR_SUCCESS)
    {
      free(regValue);
      RegCloseKey(regKey);
      continue;
    }

    // Strip the OpenGL driver dll name from the string then create a new string with
    // the path and the nvoptix.dll name
    for (int i = valueSize - 1; i >= 0 && regValue[i] != '\\'; --i)
    {
      regValue[i] = '\0';
    }

    size_t newPathSize = strlen(regValue) + strlen(optixDllName) + 1;
    char* dllPath = (char*) malloc(newPathSize);
    strcpy(dllPath, regValue);
    strcat(dllPath, optixDllName);

    free(regValue);
    RegCloseKey(regKey);

    handle = LoadLibraryA((LPCSTR) dllPath);
    free(dllPath);

    if (handle)
    {
      break;
    }
  }

  free(deviceNames);

  return handle;
}
#endif


static void debugDumpTexture(const std::string& name, const MaterialData::Texture& t)
{
  std::cout << name 
            << ": ( index = " << t.index << ", radians = " << t.angle << ", object = " << t.object 
            // KHR_texture_transform
            << "), scale = (" << t.scale.x << ", " << t.scale.y
            << "), rotation = (" << t.rotation.x << ", " << t.rotation.y 
            << "), translation = (" << t.translation.x << ", " << t.translation.y 
            << ")\n";
}


static void debugDumpMaterial(const MaterialData& m)
{
  // PBR Metallic Roughness parameters:
  std::cout << "baseColorFactor = (" << m.baseColorFactor.x << ", " << m.baseColorFactor.y << ", " << m.baseColorFactor.z << ", " << m.baseColorFactor.w << ")\n";
  std::cout << "metallicFactor  = " << m.metallicFactor << "\n";
  std::cout << "roughnessFactor = " << m.roughnessFactor << "\n";;
  debugDumpTexture("baseColorTexture", m.baseColorTexture);
  debugDumpTexture("metallicRoughnessTexture", m.metallicRoughnessTexture);

  // Standard Material parameters:
  std::cout << "doubleSided = " << ((m.doubleSided) ? "true" : "false") << "\n";

  switch (m.alphaMode)
  {
  case MaterialData::ALPHA_MODE_OPAQUE:
    std::cout << "alpha_mode = ALPHA_MODE_OPAQUE\n";
    break;
  case MaterialData::ALPHA_MODE_MASK:
    std::cout << "alpha_mode = ALPHA_MODE_MASK\n";
    break;
  case MaterialData::ALPHA_MODE_BLEND:
    std::cout << "alpha_mode = ALPHA_MODE_BLEND\n";
    break;
  }
  std::cout << "alphaCutoff = " << m.alphaCutoff << "\n";;
  
  std::cout << "normalTextureScale = " << m.normalTextureScale << "\n";;
  debugDumpTexture("normalTexture", m.normalTexture);

  std::cout << "occlusionTextureStrength = " << m.occlusionTextureStrength << "\n";;
  debugDumpTexture("occlusionTexture", m.occlusionTexture);

  std::cout << "emissiveStrength = " << m.emissiveStrength << "\n";
  std::cout << "emissiveFactor = (" << m.emissiveFactor.x << ", " << m.emissiveFactor.y << ", " << m.emissiveFactor.z << ")\n";
  debugDumpTexture("emissiveTexture", m.emissiveTexture);
  
  std::cout << "flags = 0 ";
  //if (m.flags & FLAG_KHR_MATERIALS_IOR)
  //{
  //  std::cout << " | FLAG_KHR_MATERIALS_IOR";
  //}
  if (m.flags & FLAG_KHR_MATERIALS_SPECULAR)
  {
    std::cout << " | FLAG_KHR_MATERIALS_SPECULAR";
  }
  if (m.flags & FLAG_KHR_MATERIALS_TRANSMISSION)
  {
    std::cout << " | FLAG_KHR_MATERIALS_TRANSMISSION";
  }
  if (m.flags & FLAG_KHR_MATERIALS_VOLUME)
  {
    std::cout << " | FLAG_KHR_MATERIALS_VOLUME";
  }
  if (m.flags & FLAG_KHR_MATERIALS_CLEARCOAT)
  {
    std::cout << " | FLAG_KHR_MATERIALS_CLEARCOAT";
  }
  if (m.flags & FLAG_KHR_MATERIALS_ANISOTROPY)
  {
    std::cout << " | FLAG_KHR_MATERIALS_ANISOTROPY";
  }
  if (m.flags & FLAG_KHR_MATERIALS_SHEEN)
  {
    std::cout << " | FLAG_KHR_MATERIALS_SHEEN";
  }
  if (m.flags & FLAG_KHR_MATERIALS_IRIDESCENCE)
  {
    std::cout << " | FLAG_KHR_MATERIALS_IRIDESCENCE";
  }
  std::cout << "\n";

  // KHR_materials_ior
  std::cout << "ior = " << m.ior << "\n";;

  // KHR_materials_specular
  std::cout << "specularFactor = " << m.specularFactor << "\n";;
  debugDumpTexture("specularTexture", m.specularTexture);
  std::cout << "specularColorFactor = (" << m.specularColorFactor.x << ", " << m.specularColorFactor.y << ", " << m.specularColorFactor.z << ")\n";
  debugDumpTexture("specularColorTexture", m.specularColorTexture);

  // KHR_materials_transmission
  std::cout << "transmissionFactor = " << m.transmissionFactor << "\n";;
  debugDumpTexture("transmissionTexture", m.transmissionTexture);

  //  // KHR_materials_volume
  std::cout << "thicknessFactor = " << m.thicknessFactor << "\n";;
  //debugDumpTexture("thicknessTexture", m.thicknessTexture);
  std::cout << "attenuationDistance = " << m.attenuationDistance << "\n";;
  std::cout << "attenuationColor = (" << m.attenuationColor.x << ", " << m.attenuationColor.y << ", " << m.attenuationColor.z << ")\n";

  // KHR_materials_clearcoat
  std::cout << "clearcoatFactor = " << m.clearcoatFactor << "\n";;
  debugDumpTexture("clearcoatTexture", m.clearcoatTexture);
  std::cout << "clearcoatRoughnessFactor = " << m.clearcoatRoughnessFactor << "\n";;
  debugDumpTexture("clearcoatRoughnessTexture", m.clearcoatRoughnessTexture);
  debugDumpTexture("clearcoatNormalTexture", m.clearcoatNormalTexture);

  // KHR_materials_sheen
  std::cout << "sheenColorFactor = (" << m.sheenColorFactor.x << ", " << m.sheenColorFactor.y << ", " << m.sheenColorFactor.z << ")\n";
  debugDumpTexture("sheenColorTexture", m.sheenColorTexture);
  std::cout << "sheenRoughnessFactor = " << m.sheenRoughnessFactor << "\n";;
  debugDumpTexture("sheenRoughnessTexture", m.sheenRoughnessTexture);

  // KHR_materials_anisotropy
  std::cout << "anisotropyStrength = " << m.anisotropyStrength << "\n";;
  std::cout << "anisotropyRotation = " << m.anisotropyRotation << "\n";;
  debugDumpTexture("anisotropyTexture", m.anisotropyTexture);

  // KHR_materials_iridescence
  std::cout << "iridescenceFactor = " << m.iridescenceFactor << "\n";;
  debugDumpTexture("iridescenceTexture", m.iridescenceTexture);
  std::cout << "iridescenceIor = " << m.iridescenceIor << "\n";;
  std::cout << "iridescenceThicknessMinimum = " << m.iridescenceThicknessMinimum << "\n";;
  std::cout << "iridescenceThicknessMaximum = " << m.iridescenceThicknessMaximum << "\n";;
  debugDumpTexture("iridescenceThicknessTexture", m.iridescenceThicknessTexture);

  // KHR_materials_unlit
  std::cout << "unlit = " << ((m.unlit) ? "true" : "false") << "\n";
}


//void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
//{
//    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
//              << message << "\n";
//}


// Calculate the values which handle the access calculations.
// This is used by all three conversion routines.
static void determineAccess(const fastgltf::Accessor& accessor,
                            const fastgltf::BufferView& bufferView,
                            int16_t& bytesPerComponent,
                            size_t& strideInBytes)
{
  bytesPerComponent = fastgltf::getComponentBitSize(accessor.componentType) >> 3; // Returned.
  MY_ASSERT(0 < bytesPerComponent);

  if (bufferView.byteStride.has_value())
  {
    // This assumes that the bufferView.byteStride adheres to the glTF data alignment requirements!
    strideInBytes = bufferView.byteStride.value(); // Returned.
  }
  else
  {
    // BufferView has no byteStride value, means the data is tightly packed 
    // according to the glTF alignment rules (vector types are 4 bytes aligned).
    const uint8_t numComponents = fastgltf::getNumComponents(accessor.type);
    MY_ASSERT(0 < numComponents);

    // This is the number of bytes per element inside the source buffer without padding!
    size_t bytesPerElement = size_t(numComponents) * size_t(bytesPerComponent);

    // Now it gets awkward: 
    // The glTF specs "Data Alignment" chapter requires that start addresses of vectors must align to 4-byte.
    // That also affects the individual column vectors of matrices!
    // That means padding to 4-byte addresses of vectors is required in the following four cases:
    if (accessor.type == fastgltf::AccessorType::Vec3 && bytesPerComponent == 1)
    {
      bytesPerElement = 4;
    }
    else if (accessor.type == fastgltf::AccessorType::Mat2 && bytesPerComponent == 1)
    {
      bytesPerElement = 8;
    }
    else if (accessor.type == fastgltf::AccessorType::Mat3 && bytesPerComponent <= 2)
    {
      bytesPerElement = 12 * size_t(bytesPerComponent); // Can be 12 or 24 bytes stride.
    }
    
    // The bytesPerElement value is only used when the bufferView doesn't specify a byteStride.
    strideInBytes = bytesPerElement; // Returned.
  }
}


static unsigned short readComponentAsUshort(const unsigned char* src, 
                                            const fastgltf::ComponentType typeComponent)
{
  // This is only ever called for JOINTS_n which can be uchar or ushort.
  switch (typeComponent)
  {
    case fastgltf::ComponentType::UnsignedByte:
      return (unsigned short)(*reinterpret_cast<const unsigned char*>(src));

    case fastgltf::ComponentType::UnsignedShort:
      return *reinterpret_cast<const unsigned short*>(src);

    default:
      MY_ASSERT(!"readComponentAsUshort(): Illegal component type");
      return 0;
  }
}


static unsigned int readComponentAsUint(const unsigned char* src,
                                        const fastgltf::ComponentType typeComponent)
{
  switch (typeComponent)
  {
    case fastgltf::ComponentType::UnsignedByte:
      return (unsigned int)(*reinterpret_cast<const unsigned char*>(src));

    case fastgltf::ComponentType::UnsignedShort:
      return (unsigned int)(*reinterpret_cast<const unsigned short*>(src));

    case fastgltf::ComponentType::UnsignedInt:
      return *reinterpret_cast<const unsigned int*>(src);

    default:
      // This is only ever used for indices and they should only be uchar, ushort, uint.
      MY_ASSERT(!"readComponentAsUint(): Illegal component type"); // Normalized values are only allowed for 8 and 16 bit integers.
      return 0u;
  }
}


static float readComponentAsFloat(const unsigned char* src,
                                  const fastgltf::Accessor& accessor)
{
  float f;

  switch (accessor.componentType)
  {
    case fastgltf::ComponentType::Byte:
      f = float(*reinterpret_cast<const int8_t*>(src));
      return (accessor.normalized) ? std::max(-1.0f, f / 127.0f) : f;

    case fastgltf::ComponentType::UnsignedByte:
      f = float(*reinterpret_cast<const uint8_t*>(src));
      return (accessor.normalized) ? f / 255.0f : f;

    case fastgltf::ComponentType::Short:
      f = float(*reinterpret_cast<const int16_t*>(src));
      return (accessor.normalized) ? std::max(-1.0f, f / 32767.0f) : f;

    case fastgltf::ComponentType::UnsignedShort:
      f = float(*reinterpret_cast<const uint16_t*>(src));
      return (accessor.normalized) ? f / 65535.0f : f;

    case fastgltf::ComponentType::Float:
      return *reinterpret_cast<const float*>(src);

    default:
      // None of the vertex attributes supports normalized int32_t or uint32_t or double.
      MY_ASSERT(!"readComponentAsFloat() Illegal component type"); // Normalized values are only allowed for 8 and 16 bit integers.
      return 0.0f;
  }
}


static void convertToUshort(const fastgltf::Accessor& accessor,
                            const fastgltf::BufferView& bufferView,
                            const fastgltf::Buffer& buffer,
                            unsigned char* dest)
{
  int16_t bytesPerComponent;
  size_t  strideInBytes;

  determineAccess(accessor, bufferView, bytesPerComponent, strideInBytes);

  std::visit(fastgltf::visitor {
      [](auto& arg) {
        // Covers FilePathWithOffset, BufferView, ... which are all not possible
      },

      [&](const fastgltf::sources::Array& vector) {
        const unsigned char* ptrBase = vector.bytes.data() + bufferView.byteOffset + accessor.byteOffset;
        unsigned short* ptr = reinterpret_cast<unsigned short*>(dest);

        // Check if the data can simply be memcpy'ed.
        if (accessor.type          == fastgltf::AccessorType::Vec4 && 
            accessor.componentType == fastgltf::ComponentType::UnsignedShort &&
            strideInBytes          == 4 * sizeof(uint16_t))
        {
          memcpy(ptr, ptrBase, accessor.count * sizeof(uint16_t));
        }
        else
        {
          switch (accessor.type)
          {
            // This function will only ever be called for JOINTS_n which are uchar or ushort VEC4.
            case fastgltf::AccessorType::Vec4:
              for (size_t i = 0; i < accessor.count; ++i)
              {
                const unsigned char* ptrElement = ptrBase + i * strideInBytes; 

                ptr[0] = readComponentAsUshort(ptrElement,                         accessor.componentType);
                ptr[1] = readComponentAsUshort(ptrElement + bytesPerComponent,     accessor.componentType);
                ptr[2] = readComponentAsUshort(ptrElement + bytesPerComponent * 2, accessor.componentType);
                ptr[3] = readComponentAsUshort(ptrElement + bytesPerComponent * 3, accessor.componentType);
                ptr += 4;
              }
              break;

            default:
              MY_ASSERT(!"convertToUshort() Unexpected accessor type.")
              break;
          }
        }
      },

      [&](const fastgltf::sources::Vector& vector) {
        const unsigned char* ptrBase = vector.bytes.data() + bufferView.byteOffset + accessor.byteOffset;

        unsigned short* ptr = reinterpret_cast<unsigned short*>(dest);

        // Check if the data can simply be memcpy'ed.
        if (accessor.type          == fastgltf::AccessorType::Vec4 && 
            accessor.componentType == fastgltf::ComponentType::UnsignedShort &&
            strideInBytes          == 4 * sizeof(uint16_t))
        {
          memcpy(ptr, ptrBase, accessor.count * sizeof(uint16_t));
        }
        else
        {
          switch (accessor.type)
          {
            // This function will only ever be called for JOINTS_n which are uchar or ushort VEC4.
            case fastgltf::AccessorType::Vec4:
              for (size_t i = 0; i < accessor.count; ++i)
              {
                const unsigned char* ptrElement = ptrBase + i * strideInBytes; 

                ptr[0] = readComponentAsUshort(ptrElement,                         accessor.componentType);
                ptr[1] = readComponentAsUshort(ptrElement + bytesPerComponent,     accessor.componentType);
                ptr[2] = readComponentAsUshort(ptrElement + bytesPerComponent * 2, accessor.componentType);
                ptr[3] = readComponentAsUshort(ptrElement + bytesPerComponent * 3, accessor.componentType);
                ptr += 4;
              }
              break;

            default:
              MY_ASSERT(!"convertToUshort() Unexpected accessor type.")
              break;
          }
        }
      }
  }, buffer.data);
}


static void convertToUint(const fastgltf::Accessor& accessor,
                          const fastgltf::BufferView& bufferView,
                          const fastgltf::Buffer& buffer,
                          unsigned char* dest)
{
  int16_t bytesPerComponent;
  size_t  strideInBytes;

  determineAccess(accessor, bufferView, bytesPerComponent, strideInBytes);

  std::visit(fastgltf::visitor {
      [](auto& arg) {
        // Covers FilePathWithOffset, BufferView, ... which are all not possible
      },
      
      [&](const fastgltf::sources::Array& vector) {
        const unsigned char* ptrBase = vector.bytes.data() + bufferView.byteOffset + accessor.byteOffset;
        unsigned int *ptr = reinterpret_cast<unsigned int*>(dest);

        // Check if the data can simply be memcpy'ed.
        if (accessor.type          == fastgltf::AccessorType::Scalar && 
            accessor.componentType == fastgltf::ComponentType::UnsignedInt &&
            strideInBytes          == sizeof(uint32_t))
        {
          memcpy(ptr, ptrBase, accessor.count * sizeof(uint32_t));
        }
        else
        {
          switch (accessor.type)
          {
            // This function will only ever be called for vertex indices which are uchar, ushort or uint scalars.
            case fastgltf::AccessorType::Scalar:
              for (size_t i = 0; i < accessor.count; ++i)
              {
                const unsigned char* ptrElement = ptrBase + i * strideInBytes; 

                *ptr++ = readComponentAsUint(ptrElement, accessor.componentType);
              }
              break;

            default:
              MY_ASSERT(!"convertToUint() Unexpected accessor type.")
              break;
          }
        }
      },

      [&](const fastgltf::sources::Vector& vector) {
        const unsigned char* ptrBase = vector.bytes.data() + bufferView.byteOffset + accessor.byteOffset;
        unsigned int *ptr = reinterpret_cast<unsigned int*>(dest);
        
        // Check if the data can simply be memcpy'ed.
        if (accessor.type          == fastgltf::AccessorType::Scalar && 
            accessor.componentType == fastgltf::ComponentType::UnsignedInt &&
            strideInBytes          == sizeof(uint32_t))
        {
          memcpy(ptr, ptrBase, accessor.count * sizeof(uint32_t));
        }
        else
        {
          switch (accessor.type)
          {
            // This function will only ever be called for vertex indices which are uchar, ushort or uint scalars.
            case fastgltf::AccessorType::Scalar:
              for (size_t i = 0; i < accessor.count; ++i)
              {
                const unsigned char* ptrElement = ptrBase + i * strideInBytes; 

                *ptr++ = readComponentAsUint(ptrElement, accessor.componentType);
              }
              break;

            default:
              MY_ASSERT(!"convertToUint() Unexpected accessor type.")
              break;
          }
        }
      }
   }, buffer.data);
}


static void convertToFloat(const fastgltf::Accessor& accessor,
                           const fastgltf::BufferView& bufferView,
                           const fastgltf::Buffer& buffer,
                           unsigned char* dest,
                           const fastgltf::AccessorType typeTarget)
{
  const uint8_t numTargetComponents = fastgltf::getNumComponents(typeTarget);

  int16_t bytesPerComponent;
  size_t  strideInBytes;

  determineAccess(accessor, bufferView, bytesPerComponent, strideInBytes);

  std::visit(fastgltf::visitor {
      [](auto& arg) {
        // Covers FilePathWithOffset, BufferView, ... which are all not possible
      },

      [&](const fastgltf::sources::Array& vector) {
        const unsigned char* ptrBase = vector.bytes.data() + bufferView.byteOffset + accessor.byteOffset;
        float* ptr = reinterpret_cast<float*>(dest);

        // Check if the data can simply be memcpy'ed.
        if (accessor.type          == typeTarget && 
            accessor.componentType == fastgltf::ComponentType::Float &&
            strideInBytes          == numTargetComponents * sizeof(float))
        {
          memcpy(ptr, ptrBase, accessor.count * strideInBytes);
        }
        else
        {
          for (size_t i = 0; i < accessor.count; ++i)
          {
            const unsigned char* ptrElement = ptrBase + i * strideInBytes; 

            switch (accessor.type)
            {
              case fastgltf::AccessorType::Scalar:
                *ptr++ = readComponentAsFloat(ptrElement, accessor);
                break;

              case fastgltf::AccessorType::Vec2:
                ptr[0] = readComponentAsFloat(ptrElement,                     accessor);
                ptr[1] = readComponentAsFloat(ptrElement + bytesPerComponent, accessor);
                ptr += 2;
                break;

              case fastgltf::AccessorType::Vec3:
                ptr[0] = readComponentAsFloat(ptrElement,                         accessor);
                ptr[1] = readComponentAsFloat(ptrElement + bytesPerComponent,     accessor);
                ptr[2] = readComponentAsFloat(ptrElement + bytesPerComponent * 2, accessor);
                ptr += 3;
                // Special case for vec3f to vec4f color conversion.
                if (typeTarget == fastgltf::AccessorType::Vec4)
                {
                  *ptr++ = 1.0f; // Append an alpha = 1.0f value to the destination.
                }
                break;

              case fastgltf::AccessorType::Vec4:
                ptr[0] = readComponentAsFloat(ptrElement,                         accessor);
                ptr[1] = readComponentAsFloat(ptrElement + bytesPerComponent,     accessor);
                ptr[2] = readComponentAsFloat(ptrElement + bytesPerComponent * 2, accessor);
                ptr[3] = readComponentAsFloat(ptrElement + bytesPerComponent * 3, accessor);
                ptr += 4;
                break;

              case fastgltf::AccessorType::Mat2: // DEBUG Are these actually used as source data in glTF anywhere?
                if (1 < bytesPerComponent) // Standard case, no padding to 4-byte vectors needed.
                {
                  // glTF/OpenGL matrices are defined column-major!
                  ptr[0] = readComponentAsFloat(ptrElement,                         accessor); // m00
                  ptr[1] = readComponentAsFloat(ptrElement + bytesPerComponent,     accessor); // m10
                  ptr[2] = readComponentAsFloat(ptrElement + bytesPerComponent * 2, accessor); // m01
                  ptr[3] = readComponentAsFloat(ptrElement + bytesPerComponent * 3, accessor); // m11
                }
                else // mat2 with 1-byte components requires 2 bytes source data padding between the two vectors..
                {
                  MY_ASSERT(bytesPerComponent == 1);
                  ptr[0] = readComponentAsFloat(ptrElement + 0, accessor); // m00
                  ptr[1] = readComponentAsFloat(ptrElement + 1, accessor); // m10
                  // 2 bytes padding
                  ptr[2] = readComponentAsFloat(ptrElement + 4, accessor); // m01
                  ptr[3] = readComponentAsFloat(ptrElement + 5, accessor); // m11
                }
                ptr += 4;
                break;

              case fastgltf::AccessorType::Mat3: // DEBUG Are these actually used as source data in glTF anywhere?
                if (2 < bytesPerComponent) // Standard case, no padding to 4-byte vectors needed.
                {
                  // glTF/OpenGL matrices are defined column-major!
                  for (int element = 0; element < 9; ++element)
                  {
                    ptr[element] = readComponentAsFloat(ptrElement + bytesPerComponent * element, accessor);
                  }
                }
                else if (bytesPerComponent == 1) // mat3 with 1-byte components requires 2 bytes source data padding between the two vectors..
                {
                  ptr[0] = readComponentAsFloat(ptrElement +  0, accessor); // m00
                  ptr[1] = readComponentAsFloat(ptrElement +  1, accessor); // m10
                  ptr[2] = readComponentAsFloat(ptrElement +  2, accessor); // m20
                  // 1 byte padding
                  ptr[3] = readComponentAsFloat(ptrElement +  4, accessor); // m01
                  ptr[4] = readComponentAsFloat(ptrElement +  5, accessor); // m11
                  ptr[5] = readComponentAsFloat(ptrElement +  6, accessor); // m21
                  // 1 byte padding
                  ptr[6] = readComponentAsFloat(ptrElement +  8, accessor); // m02
                  ptr[7] = readComponentAsFloat(ptrElement +  9, accessor); // m12
                  ptr[8] = readComponentAsFloat(ptrElement + 10, accessor); // m22
                }
                else if (bytesPerComponent == 2) // mat3 with 2-byte components requires 2 bytes source data padding between the two vectors..
                {
                  ptr[0] = readComponentAsFloat(ptrElement +  0, accessor); // m00
                  ptr[1] = readComponentAsFloat(ptrElement +  2, accessor); // m10
                  ptr[2] = readComponentAsFloat(ptrElement +  4, accessor); // m20
                  // 2 bytes padding
                  ptr[3] = readComponentAsFloat(ptrElement +  8, accessor); // m01
                  ptr[4] = readComponentAsFloat(ptrElement + 10, accessor); // m11
                  ptr[5] = readComponentAsFloat(ptrElement + 12, accessor); // m21
                  // 2 bytes padding
                  ptr[6] = readComponentAsFloat(ptrElement + 16, accessor); // m02
                  ptr[7] = readComponentAsFloat(ptrElement + 18, accessor); // m12
                  ptr[8] = readComponentAsFloat(ptrElement + 20, accessor); // m22
                }
                ptr += 9;
                break;

              case fastgltf::AccessorType::Mat4:
                // glTF/OpenGL matrices are defined column-major!
                for (int element = 0; element < 16; ++element)
                {
                  ptr[element] = readComponentAsFloat(ptrElement + bytesPerComponent * element, accessor);
                }
                ptr += 16;
                break;

              default:
                MY_ASSERT(!"convertToFloat() Unexpected accessor type.")
                break;
            }
          }
        }

      },

      [&](const fastgltf::sources::Vector& vector) {
        const unsigned char* ptrBase = vector.bytes.data() + bufferView.byteOffset + accessor.byteOffset;
        float* ptr = reinterpret_cast<float*>(dest);

        // Check if the data can simply be memcpy'ed.
        if (accessor.type          == typeTarget && 
            accessor.componentType == fastgltf::ComponentType::Float &&
            strideInBytes          == numTargetComponents * sizeof(float))
        {
          memcpy(ptr, ptrBase, accessor.count * strideInBytes);
        }
        else
        {
          for (size_t i = 0; i < accessor.count; ++i)
          {
            const unsigned char* ptrElement = ptrBase + i * strideInBytes; 

            switch (accessor.type)
            {
              case fastgltf::AccessorType::Scalar:
                *ptr++ = readComponentAsFloat(ptrElement, accessor);
                break;

              case fastgltf::AccessorType::Vec2:
                ptr[0] = readComponentAsFloat(ptrElement,                     accessor);
                ptr[1] = readComponentAsFloat(ptrElement + bytesPerComponent, accessor);
                ptr += 2;
                break;

              case fastgltf::AccessorType::Vec3:
                ptr[0] = readComponentAsFloat(ptrElement,                         accessor);
                ptr[1] = readComponentAsFloat(ptrElement + bytesPerComponent,     accessor);
                ptr[2] = readComponentAsFloat(ptrElement + bytesPerComponent * 2, accessor);
                ptr += 3;
                // Special case for vec3f to vec4f color conversion.
                if (typeTarget == fastgltf::AccessorType::Vec4)
                {
                  *ptr++ = 1.0f; // Append an alpha = 1.0f value to the destination.
                }
                break;

              case fastgltf::AccessorType::Vec4:
                ptr[0] = readComponentAsFloat(ptrElement,                         accessor);
                ptr[1] = readComponentAsFloat(ptrElement + bytesPerComponent,     accessor);
                ptr[2] = readComponentAsFloat(ptrElement + bytesPerComponent * 2, accessor);
                ptr[3] = readComponentAsFloat(ptrElement + bytesPerComponent * 3, accessor);
                ptr += 4;
                break;

              case fastgltf::AccessorType::Mat2: // DEBUG Are these actually used as source data in glTF anywhere?
                if (1 < bytesPerComponent) // Standard case, no padding to 4-byte vectors needed.
                {
                  // glTF/OpenGL matrices are defined column-major!
                  ptr[0] = readComponentAsFloat(ptrElement,                         accessor); // m00
                  ptr[1] = readComponentAsFloat(ptrElement + bytesPerComponent,     accessor); // m10
                  ptr[2] = readComponentAsFloat(ptrElement + bytesPerComponent * 2, accessor); // m01
                  ptr[3] = readComponentAsFloat(ptrElement + bytesPerComponent * 3, accessor); // m11
                }
                else // mat2 with 1-byte components requires 2 bytes source data padding between the two vectors..
                {
                  MY_ASSERT(bytesPerComponent == 1);
                  ptr[0] = readComponentAsFloat(ptrElement + 0, accessor); // m00
                  ptr[1] = readComponentAsFloat(ptrElement + 1, accessor); // m10
                  // 2 bytes padding
                  ptr[2] = readComponentAsFloat(ptrElement + 4, accessor); // m01
                  ptr[3] = readComponentAsFloat(ptrElement + 5, accessor); // m11
                }
                ptr += 4;
                break;

              case fastgltf::AccessorType::Mat3: // DEBUG Are these actually used as source data in glTF anywhere?
                if (2 < bytesPerComponent) // Standard case, no padding to 4-byte vectors needed.
                {
                  // glTF/OpenGL matrices are defined column-major!
                  for (int element = 0; element < 9; ++element)
                  {
                    ptr[element] = readComponentAsFloat(ptrElement + bytesPerComponent * element, accessor);
                  }
                }
                else if (bytesPerComponent == 1) // mat3 with 1-byte components requires 2 bytes source data padding between the two vectors..
                {
                  ptr[0] = readComponentAsFloat(ptrElement +  0, accessor); // m00
                  ptr[1] = readComponentAsFloat(ptrElement +  1, accessor); // m10
                  ptr[2] = readComponentAsFloat(ptrElement +  2, accessor); // m20
                  // 1 byte padding
                  ptr[3] = readComponentAsFloat(ptrElement +  4, accessor); // m01
                  ptr[4] = readComponentAsFloat(ptrElement +  5, accessor); // m11
                  ptr[5] = readComponentAsFloat(ptrElement +  6, accessor); // m21
                  // 1 byte padding
                  ptr[6] = readComponentAsFloat(ptrElement +  8, accessor); // m02
                  ptr[7] = readComponentAsFloat(ptrElement +  9, accessor); // m12
                  ptr[8] = readComponentAsFloat(ptrElement + 10, accessor); // m22
                }
                else if (bytesPerComponent == 2) // mat3 with 2-byte components requires 2 bytes source data padding between the two vectors..
                {
                  ptr[0] = readComponentAsFloat(ptrElement +  0, accessor); // m00
                  ptr[1] = readComponentAsFloat(ptrElement +  2, accessor); // m10
                  ptr[2] = readComponentAsFloat(ptrElement +  4, accessor); // m20
                  // 2 bytes padding
                  ptr[3] = readComponentAsFloat(ptrElement +  8, accessor); // m01
                  ptr[4] = readComponentAsFloat(ptrElement + 10, accessor); // m11
                  ptr[5] = readComponentAsFloat(ptrElement + 12, accessor); // m21
                  // 2 bytes padding
                  ptr[6] = readComponentAsFloat(ptrElement + 16, accessor); // m02
                  ptr[7] = readComponentAsFloat(ptrElement + 18, accessor); // m12
                  ptr[8] = readComponentAsFloat(ptrElement + 20, accessor); // m22
                }
                ptr += 9;
                break;

              case fastgltf::AccessorType::Mat4:
                // glTF/OpenGL matrices are defined column-major!
                for (int element = 0; element < 16; ++element)
                {
                  ptr[element] = readComponentAsFloat(ptrElement + bytesPerComponent * element, accessor);
                }
                ptr += 16;
                break;

              default:
                MY_ASSERT(!"convertToFloat() Unexpected accessor type.")
                break;
            }
          }
        }
      }
   }, buffer.data);
}


static DeviceBuffer createDeviceBuffer(
  fastgltf::Asset&        asset,               // The asset contains all source data (Accessor, BufferView, Buffer)
  const int               indexAccessor,       // The accessor index defines the source data. -1 means no data.
  fastgltf::AccessorType  typeTarget,          // One of Scalar, Vec2, etc.)
  fastgltf::ComponentType typeTargetComponent) // One of UnsignedInt primitive indices, UnsignedShort JOINTS_n, everything else Float)
{
  DeviceBuffer deviceBuffer; // Default empty DeviceBuffer, all values zero.
 
   // Negative accessor index means the data is optional and an empty DeviceBuffer is returned initialize all dev::Primitive fields.
  if (indexAccessor < 0)
  {
    return deviceBuffer; // Empty!
  }

  // Accessor, BufferView, and Buffer together specify the source data.
  MY_ASSERT(indexAccessor < static_cast<int>(asset.accessors.size()));
  const fastgltf::Accessor& accessor = asset.accessors[indexAccessor];

  if (accessor.count == 0) // DEBUG Can there be accessors with count == 0?
  {
    std::cerr << "WARNING: createDeviceBuffer() Accessor.count == 0\n";
    return deviceBuffer;
  }
  
  // FIXME Could be a using sparse accessor, which this example is not supporting, yet.
  if (!accessor.bufferViewIndex.has_value())
  {
    std::cerr << "WARNING: createDeviceBuffer() Accessor.bufferViewIndex has no value\n";
    return deviceBuffer;
  }

  const size_t bufferViewIndex = accessor.bufferViewIndex.value();
  MY_ASSERT(bufferViewIndex < asset.bufferViews.size());
  
  const fastgltf::BufferView& bufferView = asset.bufferViews[bufferViewIndex];

  const fastgltf::Buffer& buffer = asset.buffers[bufferView.bufferIndex];

  // First calculate the size of the DeviceBuffer.
  const uint8_t numTargetComponents = fastgltf::getNumComponents(typeTarget);
  MY_ASSERT(0 < numTargetComponents);

  const int16_t sizeTargetComponentInBytes = fastgltf::getComponentBitSize(typeTargetComponent) >> 3;
  MY_ASSERT(0 < sizeTargetComponentInBytes);

  // Number of elements inside the source accessor times the target element size in bytes.
  const size_t sizeTargetBufferInBytes = accessor.count * size_t(numTargetComponents) * size_t(sizeTargetComponentInBytes);

  // Host target buffer allocation.
  deviceBuffer.h_ptr = new unsigned char[sizeTargetBufferInBytes]; // Allocate the host buffer.
  deviceBuffer.size  = sizeTargetBufferInBytes; // Size of the host and device buffers in bytes.
  deviceBuffer.count = accessor.count;

  // Convert all elements inside the source data to the expected target data format individually.
  switch (typeTargetComponent)
  {
    case fastgltf::ComponentType::UnsignedShort: // JOINTS_n are converted to ushort.
      convertToUshort(accessor, bufferView, buffer, deviceBuffer.h_ptr);
      break;

    case fastgltf::ComponentType::UnsignedInt: // Primitive indices are converted to uint.
      convertToUint(accessor, bufferView, buffer, deviceBuffer.h_ptr);
      break;

    case fastgltf::ComponentType::Float: // Everything else is float.
      // Only this conversion needs to know the target type to be able to expand COLOR_n from vec3f to vec4f.
      convertToFloat(accessor, bufferView, buffer, deviceBuffer.h_ptr, typeTarget);
      break;
  
    default:
      MY_ASSERT(!"createDeviceBuffer() Unexpected target component type.")
      break;
  }

  // If everything has been copied/converted to the host buffer, allocate the device buffer and copy the data there.
  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&deviceBuffer.d_ptr), deviceBuffer.size) );
  CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(deviceBuffer.d_ptr), deviceBuffer.h_ptr, deviceBuffer.size, cudaMemcpyHostToDevice) );

  return deviceBuffer;
}


void Application::initSheenLUT()
{
  // Create the sheen lookup table which is required to weight the sheen sampling.
  m_picSheenLUT = new Picture();

  if (!m_picSheenLUT->load("sheen_lut.hdr", IMAGE_FLAG_2D)) // This frees all images inside an existing Picture.
  {
    delete m_picSheenLUT;
    m_picSheenLUT = nullptr;

    throw std::exception("ERROR: initSheenLUT() Picture::load() failed.");
  }

  // Create a new texture to keep the old texture intact in case anything goes wrong.
  m_texSheenLUT = new Texture(m_allocator);

  m_texSheenLUT->setAddressMode(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP);

  if (!m_texSheenLUT->create(m_picSheenLUT, IMAGE_FLAG_2D | IMAGE_FLAG_SHEEN))
  {
    delete m_texSheenLUT;
    m_texSheenLUT = nullptr;

    throw std::exception("ERROR: initSheenLUT Texture::create() failed.");
  }
}


Application::Application(GLFWwindow* window,
                         Options const& options)
  : m_window(window)
  , m_logger(std::cerr)
{
  m_pathAsset = options.getFilename();
  m_width     = std::max(1, options.getClientWidth());
  m_height    = std::max(1, options.getClientHeight());
  m_launches  = options.getLaunches();
  m_interop   = options.getInterop();
  m_punctual  = options.getPunctual();
  m_missID    = options.getMiss();
  m_pathEnv   = options.getEnvironment();

  m_iterations.resize(m_launches); // The size of this vector must always match m_launches;

  m_colorEnv[0] = 1.0f;
  m_colorEnv[1] = 1.0f;
  m_colorEnv[2] = 1.0f;
  m_intensityEnv = 1.0f;
  m_rotationEnv[0] = 0.0f;
  m_rotationEnv[1] = 0.0f;
  m_rotationEnv[2] = 0.0f;

  m_pbo = 0;
  m_hdrTexture = 0;

  m_bufferHost = nullptr; // Allocated inside updateBuffers() when needed.

  m_glslVS = 0;
  m_glslFS = 0;
  m_glslProgram = 0;
  
#if 1 // Tonemapper defaults
    m_gamma          = 2.2f;
    m_colorBalance   = make_float3(1.0f, 1.0f, 1.0f);
    m_whitePoint     = 1.0f;
    m_burnHighlights = 0.8f;
    m_crushBlacks    = 0.2f;
    m_saturation     = 1.2f;
    m_brightness     = 1.0f;
#else // Neutral tonemapper settings.
    m_gamma          = 1.0f;
    m_colorBalance   = make_float3(1.0f, 1.0f, 1.0f);
    m_whitePoint     = 1.0f;
    m_burnHighlights = 1.0f;
    m_crushBlacks    = 0.0f;
    m_saturation     = 1.0f;
    m_brightness     = 1.0f;
#endif

  m_guiState = GUI_STATE_NONE;

  m_isVisibleGUI = true;

  m_mouseSpeedRatio = 100.0f;
  m_trackball.setSpeedRatio(m_mouseSpeedRatio);

  m_vboAttributes = 0;
  m_vboIndices = 0;
      
  m_positionLocation = -1;
  m_texCoordLocation = -1;
    
  m_cudaGraphicsResource = nullptr;
  
  // Setup ImGui binding.
  ImGui::CreateContext();

  ImGuiIO& io = ImGui::GetIO(); 
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Use Tab and arrow keys to navigate through widgets.
  //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
#ifdef _WIN32
  // HACK Only enable Multi-Viewport under Windows because of
  // https://github.com/ocornut/imgui/wiki/Multi-Viewports#issues
  // "The feature tends to be broken on Linux/X11 with many window managers.
  //  The feature doesn't work in Wayland."
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport/Platform Windows"
#endif
  io.ConfigWindowsResizeFromEdges      = true; // More consistent window resize behavior, esp. when using multi-viewports.
  io.ConfigWindowsMoveFromTitleBarOnly = true; // Prevent moving the GUI window when inadvertently clicking on an empty space.

  //ImGui::StyleColorsDark(); // default
  //ImGui::StyleColorsLight();
  //ImGui::StyleColorsClassic();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init();

  // This initializes ImGui resources like the font texture.
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  // Do nothing.
  ImGui::EndFrame();

  // This must always be called after each ImGui::EndFrame() when ImGuiConfigFlags_ViewportsEnable is set.
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
  {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    // Platform windows can change the OpenGL context.
    glfwMakeContextCurrent(m_window);
  }

  initCUDA();   // CUDA must be initialized before OpenGL to have the CUDA device UUID/LUID for the OpenGL-CUDA interop check.
  initOpenGL(); // OpenGL must be initialized before OptiX because that determines the OpenGL-CUDA interop mode and generates resources.
  initOptiX();
  initSheenLUT();

  // This uses fastgltf to load the glTF into Application::m_asset.
  loadGLTF(m_pathAsset);

  // Print which extensions the asset uses.
  // This is helpful when adding support for new extensions in loadGLTF().
  // Which material extensions are used is determined inside initMaterials() per individual material.
  std::cout << "extensionsUsed = {"<< '\n';
  for (const auto& extension : m_asset.extensionsUsed)
  {
    std::cout << "  " << extension << '\n';
  }
  std::cout << "}\n";

  // If this would list any extensions which aren't supported, the loadGLTF() above already threw an error.
  std::cout << "extensionsRequired = {"<< '\n';
  for (const auto& extension : m_asset.extensionsRequired)
  {
    std::cout << "  " << extension << '\n';
  }
  std::cout << "}\n";

  // Initialize the GLTF host and device resource vectors (sizes) upfront 
  // to match the GLTF indices used in various asset objects.
  initImages();
  initTextures();
  initMaterials();
  initMeshes();
  initLights(); // This just copies the data from the m_asset.lights. This is not the device side representation. 
  initCameras(); // This also creates a default camera when there isn't one inside the asset.
  // First time scene initialization, creating or using the default scene.
  initScene(-1);

  // Initialize all acceleration structures, pipeline, shader binding table.
  initRenderer();

  initTrackball(); // In case there was no camera inside the asset, this places and centers the added default camera according to the selected scene.
  
  initLaunchParameters();
}


Application::~Application()
{
  try
  {
    cleanup();

    delete m_allocator; // This frees all CUDA allocations done with the arena allocator!

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
  }
}


// Arena version of cudaMalloc(), but asynchronous!
CUdeviceptr Application::memAlloc(const size_t size, const size_t alignment, const cuda::Usage usage)
{
  return m_allocator->alloc(size, alignment, usage);
}

// Arena version of cudaFree(), but asynchronous!
void Application::memFree(const CUdeviceptr ptr)
{
  m_allocator->free(ptr);
}



void Application::reshape(int width, int height)
{
  // Zero sized interop buffers are not allowed in OptiX.
  if ((width != 0 && height != 0) && (m_width != width || m_height != height))
  {
    m_width  = width;
    m_height = height;

    glViewport(0, 0, m_width, m_height);

    m_isDirtyResize = true; // Trigger output buffer resize on next render call.
  }
}

// Convert between slashes and backslashes in paths depending on the operating system.
static void convertPath(std::string& path)
{
#if defined(_WIN32)
  std::string::size_type pos = path.find("/", 0);
  while (pos != std::string::npos)
  {
    path[pos] = '\\';
    pos = path.find("/", pos);
  }
#elif defined(__linux__)
  std::string::size_type pos = path.find("\\", 0);
  while (pos != std::string::npos)
  {
    path[pos] = '/';
    pos = path.find("\\", pos);
  }
#endif
}


void Application::drop(const int countPaths, const char* paths[])
{
  // DEBUG
  //std::cout << "drop(): count = " << countPaths << '\n'; 
  //for (int i = 0; i < countPaths; ++i)
  //{
  //  std::cout << paths[i] << '\n';
  //}
  //std::cout << std::endl;

  // Exchanging the light type in m_lightDefinitions[0] only works when it's the same type,
  // because the pipeline miss program is setup for that case.
  if (m_missID != 2)
  {
    std::cerr << "WARNING: Environment texture exchange requires spherical environment light. Use command line without --miss (-m) option or set it to 2.\n";
    return;
  }

  // Check if there is any *.hdr file inside the dropped paths.
  for (int i = 0; i < countPaths; ++i)
  {
    std::string strPath(paths[i]);
    convertPath(strPath);

    std::filesystem::path path(strPath);
    std::filesystem::path ext = path.extension();

    if (ext.string() == std::string(".hdr"))
    {
      // The first found *.hdr file is the current environment light. Nothing changes.
      if (m_pathEnv == strPath)
      {
        std::cerr << "WARNING: Environment light " << strPath << " already used.\n";
        return;
      }

      std::cout << "drop() Replacing environment light with image "<< strPath << '\n';

      CUDA_CHECK( cudaDeviceSynchronize() ); // Wait until all rendering is finished before deleting and existing environment light.
      
      // Create the new environment light.
      m_pathEnv = strPath;

      if (m_picEnv == nullptr)
      {
        m_picEnv = new Picture();
      }

      if (!m_picEnv->load(m_pathEnv, IMAGE_FLAG_2D)) // This frees all images inside an existing Picture.
      {
        return;
      }

      // Create a new texture to keep the old texture intact in case anything goes wrong.
      Texture *texture = new Texture(m_allocator);

      if (!texture->create(m_picEnv, IMAGE_FLAG_2D | IMAGE_FLAG_ENV))
      {
        delete texture;
        return;
      }

      if (m_texEnv != nullptr)
      {
        delete m_texEnv;
        //m_texEnv = nullptr;
      }

      m_texEnv = texture;

      // Replace the spherical environment light in entry 0.
      LightDefinition& light = m_lightDefinitions[0];
      MY_ASSERT(light.typeLight == TYPE_LIGHT_ENV_SPHERE);

      // Only change the values affected by the new texture.
      // Not reseting the orientation matrices means the 
      // m_environmentRotation values from the GUI stay intact.
      light.cdfU            = m_texEnv->getCDF_U(); 
      light.cdfV            = m_texEnv->getCDF_V();
      light.textureEmission = m_texEnv->getTextureObject();
      light.emission        = make_float3(1.0f);
      light.invIntegral     = 1.0f / m_texEnv->getIntegral();
      light.width           = m_texEnv->getWidth(); 
      light.height          = m_texEnv->getHeight();

      // Update the light definitions inside the launch parameters.
      CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_launchParameters.lightDefinitions), m_lightDefinitions.data(), m_lightDefinitions.size() * sizeof(LightDefinition), cudaMemcpyHostToDevice) );
      
      m_launchParameters.iteration = 0; // Restart accumulation when any launch parameter changes.

      return;
    }
  }
}


void Application::guiNewFrame()
{
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
}


void Application::guiReferenceManual()
{
  ImGui::ShowDemoWindow();
}


void Application::guiRender()
{
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  ImGuiIO& io = ImGui::GetIO();
  // This must always be called after each ImGui::EndFrame() when ImGuiConfigFlags_ViewportsEnable is set.
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
  {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    // Platform windows can change the OpenGL context.
    glfwMakeContextCurrent(m_window);
  }
}


void Application::getSystemInformation()
{
  int versionDriver = 0;
  CUDA_CHECK( cudaDriverGetVersion(&versionDriver) ); 
  
  // The version is returned as (1000 * major + 10 * minor).
  int major =  versionDriver / 1000;
  int minor = (versionDriver - major * 1000) / 10;
  std::cout << "Driver Version  = " << major << "." << minor << '\n';
  
  int versionRuntime = 0;
  CUDA_CHECK( cudaRuntimeGetVersion(&versionRuntime) );
  
  // The version is returned as (1000 * major + 10 * minor). 
  major =  versionRuntime / 1000;
  minor = (versionRuntime - major * 1000) / 10;
  std::cout << "Runtime Version = " << major << "." << minor << '\n';
  
  int countDevices = 0;
  CUDA_CHECK( cudaGetDeviceCount(&countDevices) );
  std::cout << "Device Count    = " << countDevices << '\n';

  for (int i = 0; i < countDevices; ++i)
  {
    cudaDeviceProp properties;

    CUDA_CHECK( cudaGetDeviceProperties(&properties, i) );

    //m_deviceProperties.push_back(properties);
    
    std::cout << "Device " << i << ": " << properties.name << '\n';
#if 1 // Condensed information    
    std::cout << "  SM " << properties.major << "." << properties.minor << '\n';
    std::cout << "  Total Mem = " << properties.totalGlobalMem << '\n';
    std::cout << "  ClockRate [kHz] = " << properties.clockRate << '\n';
    std::cout << "  MaxThreadsPerBlock = " << properties.maxThreadsPerBlock << '\n';
    std::cout << "  SM Count = " << properties.multiProcessorCount << '\n';
    std::cout << "  Timeout Enabled = " << properties.kernelExecTimeoutEnabled << '\n';
    std::cout << "  TCC Driver = " << properties.tccDriver << '\n';
#else // Dump every property.
    //std::cout << "name[256] = " << properties.name << '\n';
    std::cout << "uuid = " << properties.uuid.bytes << '\n';
    std::cout << "totalGlobalMem = " << properties.totalGlobalMem << '\n';
    std::cout << "sharedMemPerBlock = " << properties.sharedMemPerBlock << '\n';
    std::cout << "regsPerBlock = " << properties.regsPerBlock << '\n';
    std::cout << "warpSize = " << properties.warpSize << '\n';
    std::cout << "memPitch = " << properties.memPitch << '\n';
    std::cout << "maxThreadsPerBlock = " << properties.maxThreadsPerBlock << '\n';
    std::cout << "maxThreadsDim[3] = " << properties.maxThreadsDim[0] << ", " << properties.maxThreadsDim[1] << ", " << properties.maxThreadsDim[0] << '\n';
    std::cout << "maxGridSize[3] = " << properties.maxGridSize[0] << ", " << properties.maxGridSize[1] << ", " << properties.maxGridSize[2] << '\n';
    std::cout << "clockRate = " << properties.clockRate << '\n';
    std::cout << "totalConstMem = " << properties.totalConstMem << '\n';
    std::cout << "major = " << properties.major << '\n';
    std::cout << "minor = " << properties.minor << '\n';
    std::cout << "textureAlignment = " << properties.textureAlignment << '\n';
    std::cout << "texturePitchAlignment = " << properties.texturePitchAlignment << '\n';
    std::cout << "deviceOverlap = " << properties.deviceOverlap << '\n';
    std::cout << "multiProcessorCount = " << properties.multiProcessorCount << '\n';
    std::cout << "kernelExecTimeoutEnabled = " << properties.kernelExecTimeoutEnabled << '\n';
    std::cout << "integrated = " << properties.integrated << '\n';
    std::cout << "canMapHostMemory = " << properties.canMapHostMemory << '\n';
    std::cout << "computeMode = " << properties.computeMode << '\n';
    std::cout << "maxTexture1D = " << properties.maxTexture1D << '\n';
    std::cout << "maxTexture1DMipmap = " << properties.maxTexture1DMipmap << '\n';
    std::cout << "maxTexture1DLinear = " << properties.maxTexture1DLinear << '\n';
    std::cout << "maxTexture2D[2] = " << properties.maxTexture2D[0] << ", " << properties.maxTexture2D[1] << '\n';
    std::cout << "maxTexture2DMipmap[2] = " << properties.maxTexture2DMipmap[0] << ", " << properties.maxTexture2DMipmap[1] << '\n';
    std::cout << "maxTexture2DLinear[3] = " << properties.maxTexture2DLinear[0] << ", " << properties.maxTexture2DLinear[1] << ", " << properties.maxTexture2DLinear[2] << '\n';
    std::cout << "maxTexture2DGather[2] = " << properties.maxTexture2DGather[0] << ", " << properties.maxTexture2DGather[1] << '\n';
    std::cout << "maxTexture3D[3] = " << properties.maxTexture3D[0] << ", " << properties.maxTexture3D[1] << ", " << properties.maxTexture3D[2] << '\n';
    std::cout << "maxTexture3DAlt[3] = " << properties.maxTexture3DAlt[0] << ", " << properties.maxTexture3DAlt[1] << ", " << properties.maxTexture3DAlt[2] << '\n';
    std::cout << "maxTextureCubemap = " << properties.maxTextureCubemap << '\n';
    std::cout << "maxTexture1DLayered[2] = " << properties.maxTexture1DLayered[0] << ", " << properties.maxTexture1DLayered[1] << '\n';
    std::cout << "maxTexture2DLayered[3] = " << properties.maxTexture2DLayered[0] << ", " << properties.maxTexture2DLayered[1] << ", " << properties.maxTexture2DLayered[2] << '\n';
    std::cout << "maxTextureCubemapLayered[2] = " << properties.maxTextureCubemapLayered[0] << ", " << properties.maxTextureCubemapLayered[1] << '\n';
    std::cout << "maxSurface1D = " << properties.maxSurface1D << '\n';
    std::cout << "maxSurface2D[2] = " << properties.maxSurface2D[0] << ", " << properties.maxSurface2D[1] << '\n';
    std::cout << "maxSurface3D[3] = " << properties.maxSurface3D[0] << ", " << properties.maxSurface3D[1] << ", " << properties.maxSurface3D[2] << '\n';
    std::cout << "maxSurface1DLayered[2] = " << properties.maxSurface1DLayered[0] << ", " << properties.maxSurface1DLayered[1] << '\n';
    std::cout << "maxSurface2DLayered[3] = " << properties.maxSurface2DLayered[0] << ", " << properties.maxSurface2DLayered[1] << ", " << properties.maxSurface2DLayered[2] << '\n';
    std::cout << "maxSurfaceCubemap = " << properties.maxSurfaceCubemap << '\n';
    std::cout << "maxSurfaceCubemapLayered[2] = " << properties.maxSurfaceCubemapLayered[0] << ", " << properties.maxSurfaceCubemapLayered[1] << '\n';
    std::cout << "surfaceAlignment = " << properties.surfaceAlignment << '\n';
    std::cout << "concurrentKernels = " << properties.concurrentKernels << '\n';
    std::cout << "ECCEnabled = " << properties.ECCEnabled << '\n';
    std::cout << "pciBusID = " << properties.pciBusID << '\n';
    std::cout << "pciDeviceID = " << properties.pciDeviceID << '\n';
    std::cout << "pciDomainID = " << properties.pciDomainID << '\n';
    std::cout << "tccDriver = " << properties.tccDriver << '\n';
    std::cout << "asyncEngineCount = " << properties.asyncEngineCount << '\n';
    std::cout << "unifiedAddressing = " << properties.unifiedAddressing << '\n';
    std::cout << "memoryClockRate = " << properties.memoryClockRate << '\n';
    std::cout << "memoryBusWidth = " << properties.memoryBusWidth << '\n';
    std::cout << "l2CacheSize = " << properties.l2CacheSize << '\n';
    std::cout << "maxThreadsPerMultiProcessor = " << properties.maxThreadsPerMultiProcessor << '\n';
    std::cout << "streamPrioritiesSupported = " << properties.streamPrioritiesSupported << '\n';
    std::cout << "globalL1CacheSupported = " << properties.globalL1CacheSupported << '\n';
    std::cout << "localL1CacheSupported = " << properties.localL1CacheSupported << '\n';
    std::cout << "sharedMemPerMultiprocessor = " << properties.sharedMemPerMultiprocessor << '\n';
    std::cout << "regsPerMultiprocessor = " << properties.regsPerMultiprocessor << '\n';
    std::cout << "managedMemory = " << properties.managedMemory << '\n';
    std::cout << "isMultiGpuBoard = " << properties.isMultiGpuBoard << '\n';
    std::cout << "multiGpuBoardGroupID = " << properties.multiGpuBoardGroupID << '\n';
    std::cout << "singleToDoublePrecisionPerfRatio = " << properties.singleToDoublePrecisionPerfRatio << '\n';
    std::cout << "pageableMemoryAccess = " << properties.pageableMemoryAccess << '\n';
    std::cout << "concurrentManagedAccess = " << properties.concurrentManagedAccess << '\n';
    std::cout << "computePreemptionSupported = " << properties.computePreemptionSupported << '\n';
    std::cout << "canUseHostPointerForRegisteredMem = " << properties.canUseHostPointerForRegisteredMem << '\n';
    std::cout << "cooperativeLaunch = " << properties.cooperativeLaunch << '\n';
    std::cout << "cooperativeMultiDeviceLaunch = " << properties.cooperativeMultiDeviceLaunch << '\n';
    std::cout << "pageableMemoryAccessUsesHostPageTables = " << properties.pageableMemoryAccessUsesHostPageTables << '\n';
    std::cout << "directManagedMemAccessFromHost = " << properties.directManagedMemAccessFromHost << '\n';
#endif
  }
}



static bool matchLUID(const char* cudaLUID, const unsigned int cudaNodeMask,
                      const char* glLUID,   const unsigned int glNodeMask)
{
  if ((cudaNodeMask & glNodeMask) == 0)
  {
    return false;
  }
  for (int i = 0; i < GL_LUID_SIZE_EXT; ++i)
  {
    if (cudaLUID[i] != glLUID[i])
    {
      return false;
    }
  }
  return true;
}


static bool matchUUID(const CUuuid& cudaUUID, const char* glUUID)
{
  for (size_t i = 0; i < 16; ++i)
  {
    if (cudaUUID.bytes[i] != glUUID[i])
    {
      return false;
    }
  }
  return true;
}


void Application::initOpenGL()
{
  // Find out which device is running the OpenGL implementation to be able to allocate the PBO peer-to-peer staging buffer on the same device.
  // Needs these OpenGL extensions: 
  // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_external_objects.txt
  // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_external_objects_win32.txt
  // and on CUDA side the CUDA 10.0 Driver API function cuDeviceGetLuid().
  // While the extensions are named EXT_external_objects, the enums and functions are found under name string EXT_memory_object!
  if (GLEW_EXT_memory_object)
  {
    // LUID 
    // "The devices in use by the current context may also be identified by an (LUID, node) pair.
    //  To determine the LUID of the current context, call GetUnsignedBytev with <pname> set to DEVICE_LUID_EXT and <data> set to point to an array of LUID_SIZE_EXT unsigned bytes.
    //  Following the call, <data> can be cast to a pointer to an LUID object that will be equal to the locally unique identifier 
    //  of an IDXGIAdapter1 object corresponding to the adapter used by the current context.
    //  To identify which individual devices within an adapter are used by the current context, call GetIntegerv with <pname> set to DEVICE_NODE_MASK_EXT.
    //  A bitfield is returned with one bit set for each device node used by the current context.
    //  The bits set will be subset of those available on a Direct3D 12 device created on an adapter with the same LUID as the current context."
    if (GLEW_EXT_memory_object_win32) // LUID
    {
      // LUID only works under Windows and only in WDDM mode, not in TCC mode!
      // Get the LUID and node mask from the CUDA device.
      char cudaDeviceLUID[8];
      unsigned int cudaNodeMask = 0;
      
      memset(cudaDeviceLUID, 0, 8);
      CU_CHECK( cuDeviceGetLuid(cudaDeviceLUID, &cudaNodeMask, m_cudaDevice) ); // This means initCUDA() must run before initOpenGL().

      // Now compare that with the OpenGL device.
      GLubyte glDeviceLUID[GL_LUID_SIZE_EXT]; // 8 bytes identifier.
      GLint   glNodeMask = 0;                 // Node mask used together with the LUID to identify OpenGL device uniquely.

      // It is not expected that a single context will be associated with multiple DXGI adapters, so only one LUID is returned.
      memset(glDeviceLUID, 0, GL_LUID_SIZE_EXT);
      glGetUnsignedBytevEXT(GL_DEVICE_LUID_EXT, glDeviceLUID);
      glGetIntegerv(GL_DEVICE_NODE_MASK_EXT, &glNodeMask);

      if (!matchLUID(cudaDeviceLUID, cudaNodeMask, reinterpret_cast<const char*>(glDeviceLUID), glNodeMask))
      {
        // The CUDA and OpenGL devices do not match, there is no interop possible!
        std::cerr << "WARNING: OpenGL-CUDA interop disabled, LUID mismatch.\n";
        m_interop = INTEROP_OFF;
      }
    }
    else // UUID
    {
      // UUID works under Windows and Linux.
      CUuuid cudaDeviceUUID;

      memset(&cudaDeviceUUID, 0, 16);
      CU_CHECK( cuDeviceGetUuid(&cudaDeviceUUID, m_cudaDevice) ); // This means initCUDA() must run before initOpenGL().

      GLint numDevices = 0; // Number of OpenGL devices. Normally 1, unless multicast is enabled.

      // To determine which devices are used by the current context, first call GetIntegerv with <pname> set to NUM_DEVICE_UUIDS_EXT, 
      // then call GetUnsignedBytei_vEXT with <target> set to DEVICE_UUID_EXT, <index> set to a value in the range [0, <number of device UUIDs>),
      // and <data> set to point to an array of UUID_SIZE_EXT unsigned bytes. 
      glGetIntegerv(GL_NUM_DEVICE_UUIDS_EXT, &numDevices);
    
      int deviceMatch = -1;
      for (GLint i = 0; i < numDevices; ++i)
      {
        GLubyte glDeviceUUID[GL_UUID_SIZE_EXT];  // 16 bytes identifier. This example only supports one device but check up to 8 device in a machine.

        memset(glDeviceUUID, 0, GL_UUID_SIZE_EXT);
        glGetUnsignedBytei_vEXT(GL_DEVICE_UUID_EXT, i, glDeviceUUID);

        if (matchUUID(cudaDeviceUUID, reinterpret_cast<const char*>(glDeviceUUID)))
        {
          deviceMatch = i;
          break;
        }
      }
      if (deviceMatch == -1)
      {
        // The CUDA and OpenGL devices do not match, there is no interop possible!
        std::cerr << "WARNING: OpenGL-CUDA interop disabled, UUID mismatch.\n";
        m_interop = INTEROP_OFF;
      }
    }
  }

  // Report which OpenGL-CUDA interop mode is used.
  switch (m_interop) 
  {
    case INTEROP_OFF:
    default:
      std::cout << "OpenGL-CUDA interop OFF\n";
      break;
    case INTEROP_PBO:
      std::cout << "OpenGL-CUDA interop PBO\n";
      break;
    case INTEROP_TEX:
      std::cout << "OpenGL-CUDA interop TEX\n";
      break;
    case INTEROP_IMG:
      std::cout << "OpenGL-CUDA interop IMG\n";
      break;
  }

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

  glViewport(0, 0, m_width, m_height);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // glPixelStorei(GL_UNPACK_ALIGNMENT, 4); // default, works for BGRA8, RGBA16F, and RGBA32F.

  glDisable(GL_CULL_FACE);  // default
  glDisable(GL_DEPTH_TEST); // default

  glGenTextures(1, &m_hdrTexture);
  MY_ASSERT(m_hdrTexture != 0);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glBindTexture(GL_TEXTURE_2D, 0);

  // For all interop modes, updateBuffers() resizes m_hdrTexture before the first render() call and registers the resource as needed.
  switch (m_interop) 
  {
    // The "enum InteropMode" declaration documents what these OpenGL-CUDA interop modes do.
    case INTEROP_OFF:
    case INTEROP_TEX:
    case INTEROP_IMG:
    default:
      // Nothing else to initialize on OpenGL side when interop is OFF, TEX, or IMG.
      break;

    case INTEROP_PBO:
      glGenBuffers(1, &m_pbo); // PBO for OpenGL-CUDA interop.
      MY_ASSERT(m_pbo != 0); 
      // First time initialization of the PBO size happens in updateBuffers().
      break;
  }

  // GLSL shaders objects and program. 
  m_glslVS      = 0;
  m_glslFS      = 0;
  m_glslProgram = 0;

  m_positionLocation = -1;
  m_texCoordLocation = -1;

  initGLSL();

  // Two hardcoded triangles in the identity matrix pojection coordinate system with 2D texture coordinates.
  const float attributes[16] = 
  {
    // vertex2f,   texcoord2f
    -1.0f, -1.0f,  0.0f, 0.0f,
     1.0f, -1.0f,  1.0f, 0.0f,
     1.0f,  1.0f,  1.0f, 1.0f,
    -1.0f,  1.0f,  0.0f, 1.0f
  };

  unsigned int indices[6] = 
  {
    0, 1, 2, 
    2, 3, 0
  };

  glGenBuffers(1, &m_vboAttributes);
  MY_ASSERT(m_vboAttributes != 0);

  glGenBuffers(1, &m_vboIndices);
  MY_ASSERT(m_vboIndices != 0);

  // Setup the vertex arrays from the interleaved vertex attributes.
  glBindBuffer(GL_ARRAY_BUFFER, m_vboAttributes);
  glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) sizeof(float) * 16, (GLvoid const*) attributes, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vboIndices);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr) sizeof(unsigned int) * 6, (const GLvoid*) indices, GL_STATIC_DRAW);

  glVertexAttribPointer(m_positionLocation, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (GLvoid*) 0);
  //glEnableVertexAttribArray(m_positionLocation);

  glVertexAttribPointer(m_texCoordLocation, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (GLvoid*) (sizeof(float) * 2));
  //glEnableVertexAttribArray(m_texCoordLocation);
}


OptixResult Application::initOptiXFunctionTable()
{
#ifdef _WIN32
  void* handle = optixLoadWindowsDll();
  if (!handle)
  {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }

  void* symbol = reinterpret_cast<void*>(GetProcAddress((HMODULE) handle, "optixQueryFunctionTable"));
  if (!symbol)
  {
    return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
  }
#else
  void* handle = dlopen("libnvoptix.so.1", RTLD_NOW);
  if (!handle)
  {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }

  void* symbol = dlsym(handle, "optixQueryFunctionTable");
  if (!symbol)
  {
    return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
  }
#endif

  OptixQueryFunctionTable_t* optixQueryFunctionTable = reinterpret_cast<OptixQueryFunctionTable_t*>(symbol);

  return optixQueryFunctionTable(OPTIX_ABI_VERSION, 0, 0, 0, &m_api, sizeof(OptixFunctionTable));
}


void Application::initCUDA()
{
  getSystemInformation(); // This optionally dumps system information.

  cudaError_t cudaErr = cudaFree(0); // Creates a CUDA context.
  if (cudaErr != cudaSuccess)
  {
    std::cerr << "ERROR: initCUDA() cudaFree(0) failed: " << cudaErr << '\n';
    throw std::exception("initCUDA() cudaFree(0) failed");
  }

  // Get the CUdevice handle from the CUDA device ordinal.
  // This single-GPU example uses the first visible CUDA device ordinal.
  // Use the environment variable CUDA_VISIBLE_DEVICES to control which installed device is the first visible one.
  // Note that OpenGL interop is only possible of that CUDA device also runs the NVIDIA OpenGL implementation.
  // That is checked in initOpenGL() with this m_cudaDevice when m_interop != INTEROP_OFF.
  CU_CHECK( cuDeviceGet(&m_cudaDevice, 0) );

  CUresult cuRes = cuCtxGetCurrent(&m_cudaContext);
  if (cuRes != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initCUDA() cuCtxGetCurrent() failed: " << cuRes << '\n';
    throw std::exception("initCUDA() cuCtxGetCurrent() failed");
  }

  cudaErr = cudaStreamCreate(&m_cudaStream);
  if (cudaErr != cudaSuccess)
  {
    std::cerr << "ERROR: initCUDA() cudaStreamCreate() failed: " << cudaErr << '\n';
    throw std::exception("initCUDA() cudaStreamCreate() failed");
  }

  // The ArenaAllocator gets the default Arena size in bytes.
  m_allocator = new cuda::ArenaAllocator(m_sizeArena * 1024 * 1024);
}


void Application::initOptiX()
{
  OptixResult res = initOptiXFunctionTable();
  if (res != OPTIX_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() initOptiXFunctionTable() failed: " << res << '\n';
    throw std::exception("initOptiX() initOptiXFunctionTable() failed");
  }

  OptixDeviceContextOptions options = {};

  options.logCallbackFunction = &Logger::callback;
  options.logCallbackData     = &m_logger;
  options.logCallbackLevel    = 3; // Keep at warning level to suppress the disk cache messages.
#ifndef NDEBUG
  // PERF This incurs significant performance cost and should only be done during development!
  //options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

  res = m_api.optixDeviceContextCreate(m_cudaContext, &options, &m_optixContext);
  if (res != OPTIX_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() optixDeviceContextCreate() failed: " << res << '\n';
    throw std::exception("initOptiX() optixDeviceContextCreate() failed");
  }
}


void Application::updateBuffers()
{
  // Set the render resolution.
  m_launchParameters.resolution = make_int2(m_width, m_height);

  // Always resize the host output buffer.
  delete[] m_bufferHost;
  m_bufferHost = new float4[m_width * m_height];

  switch (m_interop)
  {
  case INTEROP_OFF:
  default:
    // Resize the native device buffer.
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferAccum)) );
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.bufferAccum), sizeof(float4) * m_width * m_height) );

    // Update the display texture size.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_width, (GLsizei) m_height, 0, GL_RGBA, GL_FLOAT, (GLvoid*) m_bufferHost); // RGBA32F
    break;

  case INTEROP_PBO:
    // Resize the OpenGL PBO.
    if (m_cudaGraphicsResource != nullptr)
    {
      CU_CHECK( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
    }
    // Buffer size must be > 0 or OptiX can't create a buffer from it.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * sizeof(float) * 4, (void*) 0, GL_DYNAMIC_DRAW); // RGBA32F from byte offset 0 in the pixel unpack buffer.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glFinish(); // Synchronize with following CUDA operations.
    // Keep the PBO buffer registered to only call the faster Map/Unmap around the launches.
    CU_CHECK( cuGraphicsGLRegisterBuffer(&m_cudaGraphicsResource, m_pbo, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) ); 

    // Update the display texture size.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_width, (GLsizei) m_height, 0, GL_RGBA, GL_FLOAT, (GLvoid*) m_bufferHost); // RGBA32F
    break;

  case INTEROP_TEX:
    // Resize the native device buffer.
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferAccum)) );
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.bufferAccum), sizeof(float4) * m_width * m_height) );

    if (m_cudaGraphicsResource != nullptr)
    {
      CU_CHECK( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
    }
    // Update the display texture size.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_width, (GLsizei) m_height, 0, GL_RGBA, GL_FLOAT, (GLvoid*) m_bufferHost); // RGBA32F
    glFinish(); // Synchronize with following CUDA operations.
    // Keep the texture image registered to only call the faster Map/Unmap around the launches.
    CU_CHECK( cuGraphicsGLRegisterImage(&m_cudaGraphicsResource, m_hdrTexture, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) );
    break;

  case INTEROP_IMG:
    if (m_cudaGraphicsResource != nullptr)
    {
      CU_CHECK( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
    }
    // Update the display texture size.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_width, (GLsizei) m_height, 0, GL_RGBA, GL_FLOAT, (GLvoid*) m_bufferHost); // RGBA32F
    glFinish(); // Synchronize with following CUDA operations.
    // Keep the texture image registered.
    CU_CHECK( cuGraphicsGLRegisterImage(&m_cudaGraphicsResource, m_hdrTexture, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST) ); // surface object read/write.
    break;
  }
}


bool Application::render()
{
  bool repaint = false;

  // The scene has been changed inside the GUI. 
  // Rebuild the IAS and the SBT and update the launch parameters.
  if (m_isDirtyScene)
  {
    updateScene();
    updateRenderer();
    updateLaunchParameters();
    m_isDirtyScene = false;
  }

  if (m_isDirtyLights)
  {
    updateLights();
    m_isDirtyLights = false;
  }

  if (m_isDirtyCamera || m_isDirtyResize)
  {
    updateCamera();
    m_isDirtyCamera = false;
  }

  if (m_isDirtyResize)
  {
    updateBuffers();
    m_isDirtyResize = false;
  }

  switch (m_interop)
  {
    case INTEROP_PBO:
    {
    // INTEROP_PBO renders directly into the linear OpenGL PBO buffer. Map/UnmapResource around optixLaunch calls.
      size_t size = 0;

      CU_CHECK( cuGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
      CU_CHECK( cuGraphicsResourceGetMappedPointer(reinterpret_cast<CUdeviceptr*>(&m_launchParameters.bufferAccum), &size, m_cudaGraphicsResource) ); // The pointer can change on every map!
      MY_ASSERT(m_launchParameters.resolution.x * m_launchParameters.resolution.y * sizeof(float4) <= size);
    }
    break;
    
    case INTEROP_IMG:
    {
      CUarray dstArray = nullptr;

      // Map the texture image surface directly.
      CU_CHECK( cuGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream )); // This is an implicit cuSynchronizeStream().
      CU_CHECK( cuGraphicsSubResourceGetMappedArray(&dstArray, m_cudaGraphicsResource, 0, 0) ); // arrayIndex = 0, mipLevel = 0

      CUDA_RESOURCE_DESC surfDesc{};

      surfDesc.resType = CU_RESOURCE_TYPE_ARRAY;
      surfDesc.res.array.hArray = dstArray;

      CU_CHECK( cuSurfObjectCreate(&m_launchParameters.surface, &surfDesc) );
      break;
    }
  }

  // Update all launch parameters on the device.
  CUDA_CHECK( cudaMemcpyAsync(reinterpret_cast<void*>(m_d_launchParameters), &m_launchParameters, sizeof(LaunchParameters), cudaMemcpyHostToDevice, m_cudaStream) );

  // If this render call should also shoot a single material picking ray at the screen coordinate (origin like launch index at lower-left).
  if (0.0f <= m_launchParameters.picking.x)
  {
    OPTIX_CHECK( m_api.optixLaunch(m_pipeline, m_cudaStream, reinterpret_cast<CUdeviceptr>(m_d_launchParameters), sizeof(LaunchParameters), &m_sbt, 1, 1, 1) );
    
    m_launchParameters.picking.x = -1.0f; // Disable picking on the host again. On the device this will automatically be disable on the next render() call.

    int32_t indexMaterial = -1;
    CUDA_CHECK( cudaMemcpy((void*) &indexMaterial, (const void*) m_launchParameters.bufferPicking, sizeof(int32_t), cudaMemcpyDeviceToHost) );
    if (0 <= indexMaterial) // Negative means missed all geometry.
    {
      m_indexMaterial = size_t(indexMaterial);
    }
    // repaint stays false here! No need to update the rendered image when only picking.
  }
  else // render
  {
    unsigned int iteration = m_launchParameters.iteration;

    if (m_benchmark)
    {
      CUDA_CHECK( cudaDeviceSynchronize() );
      std::chrono::steady_clock::time_point time0 = std::chrono::steady_clock::now(); // Start time.

      for (int i = 0; i < m_launches; ++i)
      {
        // Fill the vector with the iteration indices for the next m_launches.
        m_iterations[i] = iteration++;
        // Only update the iteration from the fixed vector every sub-frame.
        // This makes sure that the asynchronous copy finds the right data on the host when it's executed.
        CUDA_CHECK( cudaMemcpyAsync(reinterpret_cast<void*>(&m_d_launchParameters->iteration), &m_iterations[i], sizeof(unsigned int), cudaMemcpyHostToDevice, m_cudaStream) );
        OPTIX_CHECK( m_api.optixLaunch(m_pipeline, m_cudaStream, reinterpret_cast<CUdeviceptr>(m_d_launchParameters), sizeof(LaunchParameters), &m_sbt, m_width, m_height, 1) );
      }

      CUDA_CHECK( cudaDeviceSynchronize() ); // Wait until all kernels finished.
      std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now(); // End time.

      std::chrono::duration<double> timeRender = time1 - time0;
      const double milliseconds = std::chrono::duration<double, std::milli>(timeRender).count();
      const double sps = m_launches * 1000.0 / milliseconds;

      std::cout << sps << " samples per second (" << m_launches << " launches in " << milliseconds << " ms)\n";
    }
    else
    {
      for (int i = 0; i < m_launches; ++i)
      {
        m_iterations[i] = iteration++; // See comments above.
        CUDA_CHECK( cudaMemcpyAsync(reinterpret_cast<void*>(&m_d_launchParameters->iteration), &m_iterations[i], sizeof(unsigned int), cudaMemcpyHostToDevice, m_cudaStream) );
        OPTIX_CHECK( m_api.optixLaunch(m_pipeline, m_cudaStream, reinterpret_cast<CUdeviceptr>(m_d_launchParameters), sizeof(LaunchParameters), &m_sbt, m_width, m_height, 1) );
      }
      CUDA_CHECK( cudaDeviceSynchronize() ); // Wait for all kernels to have finished.
    }

    m_launchParameters.iteration += m_launches; // Skip the number of rendered sub frames inside the host launch parameters.
  
    repaint = true; // Indicate that there is a new image.
  }

  switch (m_interop)
  {
    case INTEROP_PBO:
      CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
      break;
    
    case INTEROP_IMG:
      CU_CHECK( cuSurfObjectDestroy(m_launchParameters.surface) );
      CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
      break;
  }

  return repaint;
}


void Application::display()
{
  glBindBuffer(GL_ARRAY_BUFFER, m_vboAttributes);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vboIndices);

  glEnableVertexAttribArray(m_positionLocation);
  glEnableVertexAttribArray(m_texCoordLocation);

  glUseProgram(m_glslProgram);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_hdrTexture);

  glDrawElements(GL_TRIANGLES, (GLsizei) 6, GL_UNSIGNED_INT, (const GLvoid*) 0);

  glUseProgram(0);

  glDisableVertexAttribArray(m_positionLocation);
  glDisableVertexAttribArray(m_texCoordLocation);
}


void Application::checkInfoLog(const char *msg, GLuint object)
{
  GLint  maxLength;
  GLint  length;
  GLchar *infoLog;

  if (glIsProgram(object))
  {
    glGetProgramiv(object, GL_INFO_LOG_LENGTH, &maxLength);
  }
  else
  {
    glGetShaderiv(object, GL_INFO_LOG_LENGTH, &maxLength);
  }
  if (maxLength > 1) 
  {
    infoLog = (GLchar *) malloc(maxLength);
    if (infoLog != NULL)
    {
      if (glIsShader(object))
      {
        glGetShaderInfoLog(object, maxLength, &length, infoLog);
      }
      else
      {
        glGetProgramInfoLog(object, maxLength, &length, infoLog);
      }
      //fprintf(fileLog, "-- tried to compile (len=%d): %s\n", (unsigned int)strlen(msg), msg);
      //fprintf(fileLog, "--- info log contents (len=%d) ---\n", (int) maxLength);
      //fprintf(fileLog, "%s", infoLog);
      //fprintf(fileLog, "--- end ---\n");
      std::cout << infoLog << '\n';
      // Look at the info log string here...
      free(infoLog);
    }
  }
}


void Application::initGLSL()
{
  static const std::string vsSource =
    "#version 330\n"
    "layout(location = 0) in vec2 attrPosition;\n"
    "layout(location = 1) in vec2 attrTexCoord;\n"
    "out vec2 varTexCoord;\n"
    "void main()\n"
    "{\n"
    "  gl_Position = vec4(attrPosition, 0.0, 1.0);\n"
    "  varTexCoord = attrTexCoord;\n"
    "}\n";

  static const std::string fsSource =
    "#version 330\n"
    "uniform sampler2D samplerHDR;\n"
    "uniform vec3  colorBalance;\n"
    "uniform float invWhitePoint;\n"
    "uniform float burnHighlights;\n"
    "uniform float saturation;\n"
    "uniform float crushBlacks;\n"
    "uniform float invGamma;\n"
    "in vec2 varTexCoord;\n"
    "layout(location = 0, index = 0) out vec4 outColor;\n"
    "void main()\n"
    "{\n"
    "  vec3 hdrColor = texture(samplerHDR, varTexCoord).rgb;\n"
    "  vec3 ldrColor = invWhitePoint * colorBalance * hdrColor;\n"
    "  ldrColor *= (ldrColor * burnHighlights + 1.0) / (ldrColor + 1.0);\n"
    "  float luminance = dot(ldrColor, vec3(0.3, 0.59, 0.11));\n"
    "  ldrColor = max(mix(vec3(luminance), ldrColor, saturation), 0.0);\n"
    "  luminance = dot(ldrColor, vec3(0.3, 0.59, 0.11));\n"
    "  if (luminance < 1.0)\n"
    "  {\n"
    "    ldrColor = max(mix(pow(ldrColor, vec3(crushBlacks)), ldrColor, sqrt(luminance)), 0.0);\n"
    "  }\n"
    "  ldrColor = pow(ldrColor, vec3(invGamma));\n"
    "  outColor = vec4(ldrColor, 1.0);\n"
    "}\n";

  GLint vsCompiled = 0;
  GLint fsCompiled = 0;
    
  m_glslVS = glCreateShader(GL_VERTEX_SHADER);
  if (m_glslVS)
  {
    GLsizei len = (GLsizei) vsSource.size();
    const GLchar *vs = vsSource.c_str();
    glShaderSource(m_glslVS, 1, &vs, &len);
    glCompileShader(m_glslVS);
    checkInfoLog(vs, m_glslVS);

    glGetShaderiv(m_glslVS, GL_COMPILE_STATUS, &vsCompiled);
    MY_ASSERT(vsCompiled);
  }

  m_glslFS = glCreateShader(GL_FRAGMENT_SHADER);
  if (m_glslFS)
  {
    GLsizei len = (GLsizei) fsSource.size();
    const GLchar *fs = fsSource.c_str();
    glShaderSource(m_glslFS, 1, &fs, &len);
    glCompileShader(m_glslFS);
    checkInfoLog(fs, m_glslFS);

    glGetShaderiv(m_glslFS, GL_COMPILE_STATUS, &fsCompiled);
    MY_ASSERT(fsCompiled);
  }

  m_glslProgram = glCreateProgram();
  if (m_glslProgram)
  {
    GLint programLinked = 0;

    if (m_glslVS && vsCompiled)
    {
      glAttachShader(m_glslProgram, m_glslVS);
    }
    if (m_glslFS && fsCompiled)
    {
      glAttachShader(m_glslProgram, m_glslFS);
    }

    glLinkProgram(m_glslProgram);
    checkInfoLog("m_glslProgram", m_glslProgram);

    glGetProgramiv(m_glslProgram, GL_LINK_STATUS, &programLinked);
    MY_ASSERT(programLinked);

    if (programLinked)
    {
      glUseProgram(m_glslProgram);

      m_positionLocation = glGetAttribLocation(m_glslProgram, "attrPosition");
      MY_ASSERT(m_positionLocation != -1);

      m_texCoordLocation = glGetAttribLocation(m_glslProgram, "attrTexCoord");
      MY_ASSERT(m_texCoordLocation != -1);
      
      glUniform1i(glGetUniformLocation(m_glslProgram, "samplerHDR"), 0); // Always using texture image unit 0 for the display texture.

      glUniform1f(glGetUniformLocation(m_glslProgram, "invGamma"),       1.0f / m_gamma);
      glUniform3f(glGetUniformLocation(m_glslProgram, "colorBalance"),   m_colorBalance.x, m_colorBalance.y, m_colorBalance.z);
      glUniform1f(glGetUniformLocation(m_glslProgram, "invWhitePoint"),  m_brightness / m_whitePoint);
      glUniform1f(glGetUniformLocation(m_glslProgram, "burnHighlights"), m_burnHighlights);
      glUniform1f(glGetUniformLocation(m_glslProgram, "crushBlacks"),    m_crushBlacks + m_crushBlacks + 1.0f);
      glUniform1f(glGetUniformLocation(m_glslProgram, "saturation"),     m_saturation);

      glUseProgram(0);
    }
  }
}


void Application::updateTonemapper()
{
  glUseProgram(m_glslProgram);

  //glUniform1i(glGetUniformLocation(m_glslProgram, "samplerHDR"), 0); // Always using texture image unit 0 for the display texture.
  glUniform1f(glGetUniformLocation(m_glslProgram, "invGamma"),       1.0f / m_gamma);
  glUniform3f(glGetUniformLocation(m_glslProgram, "colorBalance"),   m_colorBalance.x, m_colorBalance.y, m_colorBalance.z);
  glUniform1f(glGetUniformLocation(m_glslProgram, "invWhitePoint"),  m_brightness / m_whitePoint);
  glUniform1f(glGetUniformLocation(m_glslProgram, "burnHighlights"), m_burnHighlights);
  glUniform1f(glGetUniformLocation(m_glslProgram, "crushBlacks"),    m_crushBlacks + m_crushBlacks + 1.0f);
  glUniform1f(glGetUniformLocation(m_glslProgram, "saturation"),     m_saturation);

  glUseProgram(0);
}


void Application::guiWindow()
{
  if (!m_isVisibleGUI) // Use SPACE to toggle the display of the GUI window.
  {
    return;
  }

  ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiCond_FirstUseEver);

  ImGuiWindowFlags window_flags = 0;
  if (!ImGui::Begin("GLTF_renderer", nullptr, window_flags)) // No bool flag to omit the close button.
  {
    // Early out if the window is collapsed, as an optimization.
    ImGui::End();
    return;
  }

  ImGui::PushItemWidth(-170); // Right-aligned, keep pixels for the labels.

  if (ImGui::CollapsingHeader("System"))
  {
    if (ImGui::Checkbox("Benchmark", &m_benchmark))
    {
      // Next render() call will change behaviour.
    }
    if (ImGui::InputInt("Launches", &m_launches, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue)) // This requires RETURN to apply a new value.
    {
      m_launches = std::max(1, std::min(m_launches, 1000));
      m_iterations.resize(m_launches); // The size of this vector must always match m_launches.
    }
    if (ImGui::DragInt2("Path Length (min, max)", reinterpret_cast<int*>(&m_launchParameters.pathLengths), 1.0f, 0, 100))
    {
      m_launchParameters.iteration = 0u; // Restart accumulation.
    }
    if (ImGui::DragFloat("Scene Epsilon", &m_epsilonFactor, 1.0f, 0.0f, 10000.0f))
    {
      m_launchParameters.sceneEpsilon = m_epsilonFactor * SCENE_EPSILON_SCALE;
      m_launchParameters.iteration = 0u; // Restart accumulation.
    }
    if (ImGui::Checkbox("Direct Lighting", &m_useDirectLighting))
    {
      m_launchParameters.directLighting = (m_useDirectLighting) ? 1 : 0;
      m_launchParameters.iteration = 0u; // Restart accumulation.
    }
    // Allow disabling all occlusionTexture effects globally.
    if (ImGui::Checkbox("Ambient Occlusion", &m_useAmbientOcclusion))
    {
      m_launchParameters.ambientOcclusion = (m_useAmbientOcclusion) ? 1 : 0;
      m_launchParameters.iteration = 0u; // Restart accumulation.
    }
    if (ImGui::Checkbox("Show Environment", &m_showEnvironment))
    {
      m_launchParameters.showEnvironment = (m_showEnvironment) ? 1 : 0;
      m_launchParameters.iteration = 0u; // Restart accumulation.
    }
    if (ImGui::DragFloat("Mouse Ratio", &m_mouseSpeedRatio, 0.01f, 0.01f, 10000.0f, "%.2f"))
    {
      if (m_mouseSpeedRatio < 0.01f)
      {
        m_mouseSpeedRatio = 0.01f;
      }
      else if (10000.0f < m_mouseSpeedRatio)
      {
        m_mouseSpeedRatio = 10000.0f;
      }
      m_trackball.setSpeedRatio(m_mouseSpeedRatio);
    }
  }

  if (ImGui::CollapsingHeader("Tonemapper"))
  {
    bool changed = false;
    if (ImGui::ColorEdit3("Balance", (float*) &m_colorBalance))
    {
      changed = true;
    }
    if (ImGui::DragFloat("Gamma", &m_gamma, 0.01f, 0.01f, 10.0f)) // Must not get 0.0f
    {
      changed = true;
    }
    if (ImGui::DragFloat("White Point", &m_whitePoint, 0.01f, 0.01f, 255.0f, "%.2f")) // Must not get 0.0f
    {
      changed = true;
    }
    if (ImGui::DragFloat("Burn Lights", &m_burnHighlights, 0.01f, 0.0f, 10.0f, "%.2f"))
    {
      changed = true;
    }
    if (ImGui::DragFloat("Crush Blacks", &m_crushBlacks, 0.01f, 0.0f, 1.0f, "%.2f"))
    {
      changed = true;
    }
    if (ImGui::DragFloat("Saturation", &m_saturation, 0.01f, 0.0f, 10.0f, "%.2f"))
    {
      changed = true;
    }
    if (ImGui::DragFloat("Brightness", &m_brightness, 0.01f, 0.0f, 100.0f, "%.2f"))
    {
      changed = true;
    }
    if (changed)
    {
      updateTonemapper(); // This doesn't need a renderer restart.
    }
  }

  // Only show the Scenes pane when there is more than one scene inside the asset.
  if (1 < m_asset.scenes.size()) 
  {
    MY_ASSERT(m_indexScene < m_asset.scenes.size())

    if (ImGui::CollapsingHeader("Scenes"))
    {
      std::string labelCombo = std::to_string(m_indexScene) + std::string(") ") + std::string(m_asset.scenes[m_indexScene].name); // The name of the currently selected scene.

      if (ImGui::BeginCombo("Scene", labelCombo.c_str()))
      {
        // Add selectable scenes to the combo box.
        for (size_t i = 0; i < m_asset.scenes.size(); ++i)
        {
          bool isSelected = (i == m_indexScene);

          std::string label = std::to_string(i) + std::string(") ") + std::string(m_asset.scenes[i].name);

          if (ImGui::Selectable(label.c_str(), isSelected))
          {
            if (m_indexScene != i)
            {
              m_indexScene = i; 
              // Here the scene has changed and the IAS needs to be rebuild for the selected scene.
              m_isDirtyScene = true;
            }
          }
          if (isSelected)
          {
            ImGui::SetItemDefaultFocus();
          }
        }
        ImGui::EndCombo();
      }
    }
  } // End of Scenes pane.

  // Only show the Cameras pane when there is more than one camere inside the asset.
  if (1 < m_asset.cameras.size())
  {
    MY_ASSERT(m_indexCamera < m_asset.cameras.size());
    
    if (ImGui::CollapsingHeader("Cameras"))
    {
      std::string labelCombo = std::to_string(m_indexCamera) + std::string(") ") + std::string(m_asset.cameras[m_indexCamera].name); // The name of the currently selected camera.

      if (ImGui::BeginCombo("Camera", labelCombo.c_str()))
      {
        // Add selectable cameras to the combo box.
        for (size_t i = 0; i < m_asset.cameras.size(); ++i)
        {
          bool isSelected = (i == m_indexCamera);

          std::string label = std::to_string(i) + std::string(") ") + std::string(m_asset.cameras[i].name);

          if (ImGui::Selectable(label.c_str(), isSelected))
          {
            if (m_indexCamera != i)
            {
              m_indexCamera = i; 
              // Here the scene has changed and the IAS needs to be rebuilt for the selected scene.
              m_isDirtyCamera = true;
            }
          }
          if (isSelected)
          {
            ImGui::SetItemDefaultFocus();
          }
        }
        ImGui::EndCombo();
      }
    }
  }

  // Only show the Variants pane when there are material variants inside the scene.
  if (!m_asset.materialVariants.empty())
  {
    MY_ASSERT(m_indexVariant < m_asset.materialVariants.size());

    if (ImGui::CollapsingHeader("Variants"))
    {
      const size_t previousVariant = m_indexVariant;

      // The name of the currently selected material variant
      std::string labelCombo = std::to_string(m_indexVariant) + std::string(") ") + m_asset.materialVariants[m_indexVariant];

      if (ImGui::BeginCombo("Variant", labelCombo.c_str()))
      {
        for (size_t i = 0; i < m_asset.materialVariants.size(); ++i)
        {
          bool isSelected = (i == m_indexVariant);

          std::string label = std::to_string(i) + std::string(") ") + m_asset.materialVariants[i];

          if (ImGui::Selectable(label.c_str(), isSelected))
          {
            if (m_indexVariant != i)
            {
              m_indexVariant = i;
            }
          }
          if (isSelected)
          {
            ImGui::SetItemDefaultFocus();
          }
        }
        ImGui::EndCombo();
      }
      
      if (previousVariant != m_indexVariant)
      {
        updateVariant();
      }
    }
  }

  // Only show the Materials pane when there are materials inside the asset. Actually when not, there will be default materials.
  if (!m_asset.materials.empty()) // Make sure there is at least one material inside the asset.
  {
    MY_ASSERT(m_indexMaterial < m_asset.materials.size())

    if (ImGui::CollapsingHeader("Materials"))
    {
      // The name of the currently selected material
      std::string labelCombo = std::to_string(m_indexMaterial) + std::string(") ") + std::string(m_asset.materials[m_indexMaterial].name); 

      if (ImGui::BeginCombo("Material", labelCombo.c_str()))
      {
        // Add selectable materials to the combo box.
        for (size_t i = 0; i < m_asset.materials.size(); ++i)
        {
          bool isSelected = (i == m_indexMaterial);

          std::string label = std::to_string(i) + std::string(") ") + std::string(m_asset.materials[i].name);

          if (ImGui::Selectable(label.c_str(), isSelected))
          {
            if (m_indexMaterial != i)
            {
              m_indexMaterial = i; 
            }
          }
          if (isSelected)
          {
            ImGui::SetItemDefaultFocus();
          }
        }
        ImGui::EndCombo();
      }

      // Now display all editable material parameters.
      // (There is a lot of repeated code in here, because otherwise ImGui didn't build unique widgets!)
      const MaterialData& org = m_materialsOrg[m_indexMaterial];
      MaterialData&       cur = m_materials[m_indexMaterial];

      bool changed = false; // Material changed, update the SBT hit records.
      bool rebuild = false; // Material changed in a way which requires to rebuild the AS of all primitives using that material.

      if (ImGui::Button("Reset"))
      {
        // The face culling state is affected by both the doubleSided and the volume state.
        // The only case when face culling is enabled is when the material is not doubleSided and not using the volume extension.
        const bool orgCull = (!org.doubleSided && (org.flags & FLAG_KHR_MATERIALS_VOLUME) == 0);
        const bool curCull = (!cur.doubleSided && (cur.flags & FLAG_KHR_MATERIALS_VOLUME) == 0);
        
        // If the alphaMode changes, the anyhit program invocation for primitives changes.
        rebuild = (curCull != orgCull) || (cur.alphaMode != org.alphaMode);
          
        cur = org; // Reset all material changes to the original values inside the asset.

        changed = true;
      }

      // Generic settings. 
      ImGui::Separator();
      changed |= ImGui::Checkbox("unlit", &cur.unlit);

      // Note that doubleSided and alphaMode changes can trigger AS rebuilds!
      ImGui::Separator();
      if (ImGui::Checkbox("doubleSided", &cur.doubleSided))
      {
        changed = true;
        // If the doubleSided flag changes on materials which are not using the KHR_materials_volume extension,
        // the AS needs to be rebuilt to change the face culling state.
        rebuild = ((cur.flags & FLAG_KHR_MATERIALS_VOLUME) == 0); 
      }

      ImGui::Separator();
      MaterialData::AlphaMode alphaMode = cur.alphaMode;
      if (ImGui::Combo("alphaMode", reinterpret_cast<int*>(&cur.alphaMode), "OPAQUE\0MASK\0BLEND\0\0"))
      {
        changed = true;
        // Any alphaMode change requires a rebuild because each alpha mode handles anyhit programs differently.
        rebuild = true; 
      }

      // baseColor alpha and alphaCutoff have no effect if for ALPHA_MODE_OPAQUE.
      if (cur.alphaMode != MaterialData::ALPHA_MODE_OPAQUE)
      {
        // This is only one of three factors defining the opacity.
        // There is also the color.w and the baseColorTexture.w.
        changed |= ImGui::SliderFloat("baseAlpha", &cur.baseColorFactor.w, 0.0f, 1.0f); 
      }

      // alphaCutoff is only used with alphaMode == ALPHA_MODE_MASK.
      if (cur.alphaMode == MaterialData::ALPHA_MODE_MASK)
      {
        changed |= ImGui::SliderFloat("alphaCutoff", &cur.alphaCutoff, 0.0f, 1.0f);
      }

      ImGui::Separator();
      changed |= ImGui::ColorEdit3("baseColor", reinterpret_cast<float*>(&cur.baseColorFactor));
      // Only display the texture GUI when the original material defines it.
      if (org.baseColorTexture.object != 0) 
      {
        bool isEnabled = (cur.baseColorTexture.object != 0);
        if (ImGui::Checkbox("baseColorTexture", &isEnabled))
        {
          cur.baseColorTexture.object = (isEnabled) ? org.baseColorTexture.object : 0;
          changed = true;
        }
        // DEBUG If the KHR_texture_transform element should be shown inside the GUI, this code would need to be replicated for all textures.
        // (Moving that into a function requires some item tracking with ImGui::PushId/PopId.)
        // Manipulating the rotation is rather non-intuitive anyway, so keep the clutter out of the GUI
        // and don't offer the texture transforms as editable parameters.
        // Also mind that when using KHR_mesh_quantization with unnormalized texture coordinates,
        // the transform scale is used to normalize them by multiplication with 1.0f/255.0f or 1.0f/65535.0f

        //changed |= ImGui::DragFloat2("baseColorTexture.scale", reinterpret_cast<float*>(&cur.baseColorTexture.scale), 0.01f, -128.0f, 128.0, "%.2f", 1.0f);
        //if (ImGui::SliderFloat("baseColorTexture.rotation", &cur.baseColorTexture.angle, 0.0f, 2.0f * M_PIf)) or with:
        //if (ImGui::SliderAngle("baseColorTexture.rotation", &cur.baseColorTexture.angle, 0.0f, 360.0f)) // While the value is in radians, the display is in degrees.
        //{
        //  cur.baseColorTexture.rotation.x = sinf(cur.baseColorTexture.angle);
        //  cur.baseColorTexture.rotation.y = cosf(cur.baseColorTexture.angle);
        //  changed = true;
        //}
        //changed |= ImGui::DragFloat2("baseColorTexture.translation", reinterpret_cast<float*>(&cur.baseColorTexture.translation), 0.01f, -128.0f, 128.0, "%.2f", 1.0f);
      }

      ImGui::Separator();
      changed |= ImGui::SliderFloat("roughness", &cur.roughnessFactor, 0.0f, 1.0f);
      changed |= ImGui::SliderFloat("metallic", &cur.metallicFactor, 0.0f, 1.0f);
      if (org.metallicRoughnessTexture.object != 0) 
      {
        bool isEnabled = (cur.metallicRoughnessTexture.object != 0);
        if (ImGui::Checkbox("metallicRoughnessTexture", &isEnabled))
        {
          cur.metallicRoughnessTexture.object = (isEnabled) ? org.metallicRoughnessTexture.object : 0;
          changed = true;
        }
      }

      ImGui::Separator();
      changed |= ImGui::SliderFloat("ior", &cur.ior, 1.0f, 5.0f); // Don't allow values below 1.0f here.

      ImGui::Separator();
      changed |= ImGui::SliderFloat("specular", &cur.specularFactor, 0.0f, 1.0f);
      if (org.specularTexture.object != 0) 
      {
        bool isEnabled = (cur.specularTexture.object != 0);
        if (ImGui::Checkbox("specularTexture", &isEnabled))
        {
          cur.specularTexture.object = (isEnabled) ? org.specularTexture.object : 0;
          changed = true;
        }
      }
      changed |= ImGui::ColorEdit3("specularColor", reinterpret_cast<float*>(&cur.specularColorFactor));
      if (org.specularColorTexture.object != 0) 
      {
        bool isEnabled = (cur.specularColorTexture.object != 0);
        if (ImGui::Checkbox("specularColorTexture", &isEnabled))
        {
          cur.specularColorTexture.object = (isEnabled) ? org.specularColorTexture.object : 0;
          changed = true;
        }
      }

      ImGui::Separator();
      changed |= ImGui::SliderFloat("transmission", &cur.transmissionFactor, 0.0f, 1.0f);
      if (org.transmissionTexture.object != 0) 
      {
        bool isEnabled = (cur.transmissionTexture.object != 0);
        if (ImGui::Checkbox("transmissionTexture", &isEnabled))
        {
          cur.transmissionTexture.object = (isEnabled) ? org.transmissionTexture.object : 0;
          changed = true;
        }
      }

      if (org.normalTexture.object != 0) 
      {
        ImGui::Separator();
        bool isEnabled = (cur.normalTexture.object != 0);
        if (ImGui::Checkbox("normalTexture", &isEnabled))
        {
          cur.normalTexture.object = (isEnabled) ? org.normalTexture.object : 0;
          changed = true;
        }
        // normalTextureScale has no effect when there is no normalTexture.
        // Always show the normalTextureScale slider when the original material has a normalTexture
        // because that could be used as cleatcoatNormalTexture with the GUI below and 
        // I don't want the GUI elements to shift when toggling texture enables.
        changed |= ImGui::SliderFloat("normalScale", &cur.normalTextureScale, -10.0f, 10.0f); // Default is 1.0f. What is a suitable range?
      }

      if (m_useAmbientOcclusion)
      {
        ImGui::Separator();
        if (org.occlusionTexture.object != 0) 
        {
          bool isEnabled = (cur.occlusionTexture.object != 0);
          if (ImGui::Checkbox("occlusionTexture", &isEnabled))
          {
            cur.occlusionTexture.object = (isEnabled) ? org.occlusionTexture.object : 0;
            changed = true;
          }
          changed |= ImGui::SliderFloat("occlusionTextureStrength", reinterpret_cast<float*>(&cur.occlusionTextureStrength), 0.0f, 1.0f);
        }
      }

      ImGui::Separator();
      changed |= ImGui::SliderFloat("anisotropy", &cur.anisotropyStrength, 0.0f, 1.0f);
      changed |= ImGui::SliderFloat("anisotropyRotation", &cur.anisotropyRotation, 0.0f, 2.0f * M_PIf);
      if (org.anisotropyTexture.object != 0) 
      {
        bool isEnabled = (cur.anisotropyTexture.object != 0);
        if (ImGui::Checkbox("anisotropyTexture", &isEnabled))
        {
          cur.anisotropyTexture.object = (isEnabled) ? org.anisotropyTexture.object : 0;
          changed = true;
        }
      }

      ImGui::Separator();
      changed |= ImGui::SliderFloat("emissiveStrength", &cur.emissiveStrength, 0.0f, 1000.0f); // Default is 1.0f. Modulates emissiveFactor
      changed |= ImGui::ColorEdit3("emissiveColor", reinterpret_cast<float*>(&cur.emissiveFactor));
      if (org.emissiveTexture.object != 0) 
      {
        bool isEnabled = (cur.emissiveTexture.object != 0);
        if (ImGui::Checkbox("emissiveTexture", &isEnabled))
        {
          cur.emissiveTexture.object = (isEnabled) ? org.emissiveTexture.object : 0;
          changed = true;
        }
      }

      ImGui::Separator();
      bool useVolume = ((cur.flags & FLAG_KHR_MATERIALS_VOLUME) != 0);
      if (ImGui::Checkbox("volume", &useVolume))
      {
        if (useVolume)
        {
          cur.flags |= FLAG_KHR_MATERIALS_VOLUME; // Set the volume extension flag inside the material.
        }
        else
        {
          cur.flags &= ~FLAG_KHR_MATERIALS_VOLUME; // Clear the volume extension flag inside the material.
        }

        changed = true;
        
        // If the geometry is not doubleSided then toggling the volume state needs to rebuild the GAS to disable/enable face culling.
        rebuild = !cur.doubleSided;
      }

      // Only show the volume absorption parameters when the volume extension is enabled.
      if (cur.flags & FLAG_KHR_MATERIALS_VOLUME)
      {
        changed |= ImGui::SliderFloat("attenuationDistance", &cur.attenuationDistance, 0.001f, 2.0f * m_sceneExtent); // Must not be 0.0f!
        if (ImGui::ColorEdit3("attenuationColor", reinterpret_cast<float*>(&cur.attenuationColor)))
        {
          // Make sure the logf() for the volume absorption coefficient is never used on zero color components.
          cur.attenuationColor = fmaxf(make_float3(0.001f), cur.attenuationColor);
          changed = true;
        }
        
        // HACK The renderer only evaluates thicknessFactor == 0.0f as thinwalled. It's not using it for the absorption calculation!
        changed |= ImGui::SliderFloat("thickness", &cur.thicknessFactor, 0.0f, 1.0f);
      }

      ImGui::Separator();
      changed |= ImGui::SliderFloat("clearcoat", &cur.clearcoatFactor, 0.0f, 1.0f);
      if (org.clearcoatTexture.object != 0) 
      {
        bool isEnabled = (cur.clearcoatTexture.object != 0);
        if (ImGui::Checkbox("clearcoatTexture", &isEnabled))
        {
          cur.clearcoatTexture.object = (isEnabled) ? org.clearcoatTexture.object : 0;
          changed = true;
        }
      }
      changed |= ImGui::SliderFloat("clearcoatRoughness", &cur.clearcoatRoughnessFactor, 0.0f, 1.0f);
      if (org.clearcoatRoughnessTexture.object != 0) 
      {
        bool isEnabled = (cur.clearcoatRoughnessTexture.object != 0);
        if (ImGui::Checkbox("clearcoatRoughnessTexture", &isEnabled))
        {
          cur.clearcoatRoughnessTexture.object = (isEnabled) ? org.clearcoatRoughnessTexture.object : 0;
          changed = true;
        }
      }
      if (org.clearcoatNormalTexture.object != 0) 
      {
        bool isEnabled = (cur.clearcoatNormalTexture.object != 0);
        if (ImGui::Checkbox("clearcoatNormalTexture", &isEnabled))
        {
          cur.clearcoatNormalTexture.object = (isEnabled) ? org.clearcoatNormalTexture.object : 0;
          changed = true;
        }
      }
      // If the material is not using a clearcoatNormalTexture, but has a normalTexture,
      // allow the user to apply the normalTexture on the clearcoat as well.
      else if (org.normalTexture.object != 0)
      {
        bool useNormalTexture = (cur.clearcoatNormalTexture.object != 0);
        if (ImGui::Checkbox("use normaTexture on clearcoat", &useNormalTexture))
        {
          if (useNormalTexture)
          {
            cur.clearcoatNormalTexture = org.normalTexture; // Use base normalTexture as clearcoatNormalTexture.
            cur.isClearcoatNormalBaseNormal = true;
          }
          else
          {
            cur.clearcoatNormalTexture = org.clearcoatNormalTexture; // clearcoatNormalTexture off.
          }
          changed = true;
        }
      }

      ImGui::Separator();
      changed |= ImGui::ColorEdit3("sheenColor", reinterpret_cast<float*>(&cur.sheenColorFactor));
      if (org.sheenColorTexture.object != 0) 
      {
        bool isEnabled = (cur.sheenColorTexture.object != 0);
        if (ImGui::Checkbox("sheenColorTexture", &isEnabled))
        {
          cur.sheenColorTexture.object = (isEnabled) ? org.sheenColorTexture.object : 0;
          changed = true;
        }
      }
      changed |= ImGui::SliderFloat("sheenRoughness", &cur.sheenRoughnessFactor, 0.0f, 1.0f);
      if (org.sheenRoughnessTexture.object != 0) 
      {
        bool isEnabled = (cur.sheenRoughnessTexture.object != 0);
        if (ImGui::Checkbox("sheenRoughnessTexture", &isEnabled))
        {
          cur.sheenRoughnessTexture.object = (isEnabled) ? org.sheenRoughnessTexture.object : 0;
          changed = true;
        }
      }

      ImGui::Separator();
      changed |= ImGui::SliderFloat("iridescence", &cur.iridescenceFactor, 0.0f, 1.0f);
      if (org.iridescenceTexture.object != 0) 
      {
        bool isEnabled = (cur.iridescenceTexture.object != 0);
        if (ImGui::Checkbox("iridescenceTexture", &isEnabled))
        {
          cur.iridescenceTexture.object = (isEnabled) ? org.iridescenceTexture.object : 0;
          changed = true;
        }
      }
      changed |= ImGui::SliderFloat("iridescenceIor", &cur.iridescenceIor, 1.0f, 5.0f);
      changed |= ImGui::SliderFloat("iridescenceThicknessMin", &cur.iridescenceThicknessMinimum, 0.0f, cur.iridescenceThicknessMaximum);
      changed |= ImGui::SliderFloat("iridescenceThicknessMax", &cur.iridescenceThicknessMaximum, cur.iridescenceThicknessMinimum, 2000.0f);
      if (org.iridescenceThicknessTexture.object != 0) 
      {
        bool isEnabled = (cur.iridescenceThicknessTexture.object != 0);
        if (ImGui::Checkbox("iridescenceThicknessTexture", &isEnabled))
        {
          cur.iridescenceThicknessTexture.object = (isEnabled) ? org.iridescenceThicknessTexture.object : 0;
          changed = true;
        }
      }

      if (changed)
      {
        //debugDumpMaterial(m); // DEBUG
        updateMaterial(m_indexMaterial, rebuild);
      }
    }
  }

  if (!m_lightDefinitions.empty())
  {
    if (ImGui::CollapsingHeader("Lights"))
    {
      // If there is an environment light, it's always inside the first element.
      if (m_lightDefinitions[0].typeLight == TYPE_LIGHT_ENV_CONST ||
          m_lightDefinitions[0].typeLight == TYPE_LIGHT_ENV_SPHERE)
      {
        if (ImGui::ColorEdit3("env color", m_colorEnv))
        {
          m_lightDefinitions[0].emission = make_float3(m_colorEnv[0], m_colorEnv[1], m_colorEnv[2]) * m_intensityEnv;
          m_isDirtyLights = true; // Next render() call will update the device side data.
        }
        if (ImGui::DragFloat("env intensity", &m_intensityEnv, 0.001f, 0.0f, 10000.0f))
        {
          m_lightDefinitions[0].emission = make_float3(m_colorEnv[0], m_colorEnv[1], m_colorEnv[2]) * m_intensityEnv;
          m_isDirtyLights = true;
        }
        // If it's a spherical HDR texture environment light, show the environment rotation Euler angles.
        if (m_lightDefinitions[0].typeLight == TYPE_LIGHT_ENV_SPHERE)
        {

          if (ImGui::DragFloat3("env rotation", m_rotationEnv, 1.0f, 0.0f, 360.0f))
          {
            glm::vec3 euler(glm::radians(m_rotationEnv[0]),
                            glm::radians(m_rotationEnv[1]),
                            glm::radians(m_rotationEnv[2]));
          
            glm::quat quatRotation(euler);

            glm::mat4 matRotation    = glm::toMat4(quatRotation);
            glm::mat4 matRotationInv = glm::inverse(matRotation);
          
            for (int i = 0; i < 3; ++i)
            {
              glm::vec4 row = glm::row(matRotation, i);
              m_lightDefinitions[0].matrix[i]    = make_float4(row.x, row.y, row.z, row.w);
              row = glm::row(matRotationInv, i);
              m_lightDefinitions[0].matrixInv[i] = make_float4(row.x, row.y, row.z, row.w);
            }
            m_isDirtyLights = true;
          }
        }
        ImGui::Separator();
      }
      
      if (!m_lights.empty()) // KHR_lights_punctual.
      {
        // For all other lights defined by the KHR_lights_punctual show only the currently selected one.
        // FIXME Implement interactive manipulation of the position and orientation of the current light inside the viewport via the trackball.
        std::string labelCombo = std::to_string(m_indexLight) + std::string(") ") + std::string(m_lights[m_indexLight]->name); 

        if (ImGui::BeginCombo("Light", labelCombo.c_str()))
        {
          // Add selectable lights to the combo box.
          for (size_t i = 0; i < m_lights.size(); ++i)
          {
            bool isSelected = (i == m_indexLight);

            std::string label = std::to_string(i) + std::string(") ") + std::string(m_lights[i]->name);

            if (ImGui::Selectable(label.c_str(), isSelected))
            {
              if (m_indexLight != i)
              {
                m_indexLight = i; 
              }
            }
            if (isSelected)
            {
              ImGui::SetItemDefaultFocus();
            }
          }
          ImGui::EndCombo();
        }

        // Now show the light parameters of the currently selected light. 
        dev::Light* light = m_lights[m_indexLight];

        if (ImGui::ColorEdit3("color", &light->color.x))
        {
          m_isDirtyLights = true; // Next render() call will update the device side data.
        }
        if (ImGui::DragFloat("intensity", &light->intensity, 0.001f, 0.0f, 10000.0f))
        {
          m_isDirtyLights = true;
        }

        if (light->type != 2) // point or spot
        {
          // Pick a maximum range for the GUI  which is well below the RT_DEFAULT_MAX.
          if (ImGui::DragFloat("range", &light->range, 0.001f, 0.0f, 10.0f * m_sceneExtent))
          {
            m_isDirtyLights = true;
          }
        }
        if (light->type == 1) // spot
        {
          bool isDirtyCone = false;

          //if (ImGui::SliderAngle("inner cone angle", &light->innerConeAngle, 0.0f, glm::degrees(light->outerConeAngle))) // These show only full degrees.
          if (ImGui::SliderFloat("inner cone angle", &light->innerConeAngle, 0.0f, light->outerConeAngle))
          {
            isDirtyCone = true;
            m_isDirtyLights = true;
          }
          //if (ImGui::SliderAngle("outer cone angle", &light->outerConeAngle, glm::degrees(light->innerConeAngle), 90.0f))
          if (ImGui::SliderFloat("outer cone angle", &light->outerConeAngle, light->innerConeAngle, 0.5f * M_PIf))
          {
            isDirtyCone = true;
            m_isDirtyLights = true;
          }
        
          // innerConeAngle must be less than outerConeAngle!
          if (isDirtyCone && light->innerConeAngle >= light->outerConeAngle)
          {
            const float delta = 0.001f;
            if (light->innerConeAngle + delta <= 0.5f * M_PIf) // Room to increase outer angle?
            {
              light->outerConeAngle = light->innerConeAngle + delta;
            }
            else // inner angle to near to maximum cone angle.
            {
              light->innerConeAngle = light->outerConeAngle - delta; // Shrink inner cone angle.
            }
          }
        }
      } // End of m_lights.
    }
  }

  ImGui::PopItemWidth();

  ImGui::End();
}


void Application::guiEventHandler()
{
  const ImGuiIO& io = ImGui::GetIO();

  if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) // SPACE key toggles the GUI window display.
  {
    m_isVisibleGUI = !m_isVisibleGUI;
  }
  if (ImGui::IsKeyPressed(ImGuiKey_P, false)) // Key P: Save the current output buffer with tonemapping into a *.png file.
  {
    MY_VERIFY( screenshot(true) );
  }
  if (ImGui::IsKeyPressed(ImGuiKey_H, false)) // Key H: Save the current linear output buffer into a *.hdr file.
  {
    MY_VERIFY( screenshot(false) );
  }

  // Client-relative mouse coordinates when ImGuiConfigFlags_ViewportsEnable is off.
  ImVec2 mousePosition = ImGui::GetMousePos();
  // With ImGuiConfigFlags_ViewportsEnable set, mouse coordinates are relative to the primary OS monitor!
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
  {
    // Subtract the main window's client position from the OS mouse position to get the client relative position again.
    mousePosition -= ImGui::GetMainViewport()->Pos;
  }
  const int x = int(mousePosition.x);
  const int y = int(mousePosition.y);
  
  switch (m_guiState)
  {
    case GUI_STATE_NONE:
      if (!io.WantCaptureMouse) // Only allow camera interactions to begin when not interacting with the GUI.
      {
        if (ImGui::IsMouseDown(0)) // LMB down event?
        {
          if (io.KeyCtrl)
          {
            // Any picking.x position >= 0.0f will trigger the material picking inside the next render() call.
            m_launchParameters.picking = make_float2(float(x) + 0.5f, float(m_height - y) - 0.5f);
          }
          else
          {
            m_trackball.startTracking(x, y);
            m_guiState = GUI_STATE_ORBIT;
          }
        }
        else if (ImGui::IsMouseDown(1)) // RMB down event?
        {
          m_trackball.startTracking(x, y);
          m_guiState = GUI_STATE_DOLLY;
        }
        else if (ImGui::IsMouseDown(2)) // MMB down event?
        {
          m_trackball.startTracking(x, y);
          m_guiState = GUI_STATE_PAN;
        }
        else if (io.MouseWheel != 0.0f) // Mouse wheel event?
        {
          m_trackball.zoom(io.MouseWheel);
          m_isDirtyCamera = true;
        }
      }
      break;

    case GUI_STATE_ORBIT:
      if (ImGui::IsMouseReleased(0)) // LMB released? End of orbit mode.
      {
        m_guiState = GUI_STATE_NONE;
      }
      else
      {
        m_trackball.setViewMode(dev::Trackball::LookAtFixed);
        m_trackball.orbit(x, y);
        m_isDirtyCamera = true;
      }
      break;

    case GUI_STATE_DOLLY:
      if (ImGui::IsMouseReleased(1)) // RMB released? End of dolly mode.
      {
        m_guiState = GUI_STATE_NONE;
      }
      else
      {
        m_trackball.dolly(x, y);
        m_isDirtyCamera = true;
      }
      break;

    case GUI_STATE_PAN:
      if (ImGui::IsMouseReleased(2)) // MMB released? End of pan mode.
      {
        m_guiState = GUI_STATE_NONE;
      }
      else
      {
        m_trackball.pan(x, y);
        m_isDirtyCamera = true;
      }
      break;
  }
}


std::vector<char> Application::readData(std::string const& filename)
{
  std::ifstream fileStream(filename, std::ios::binary);

  if (fileStream.fail())
  {
    std::cerr << "ERROR: readData() Failed to open file " << filename << '\n';
    return std::vector<char>();
  }

  // Get the size of the file in bytes.
  fileStream.seekg(0, fileStream.end);
  std::streamsize size = fileStream.tellg();
  fileStream.seekg (0, fileStream.beg);

  if (size <= 0)
  {
    std::cerr << "ERROR: readData() File size of " << filename << " is <= 0.\n";
    return std::vector<char>();
  }

  std::vector<char> data(size);

  fileStream.read(data.data(), size);

  if (fileStream.fail())
  {
    std::cerr << "ERROR: readData() Failed to read file " << filename << '\n';
    return std::vector<char>();
  }

  return data;
}


template<typename T>
void parseTextureInfo(const std::vector<cudaTextureObject_t>& samplers, const T& textureInfo, MaterialData::Texture& texture)
{
  size_t texCoordIndex = textureInfo.texCoordIndex;

  // KHR_texture_transform extension data.
  float2 scale       = make_float2(1.0f);
  float  rotation    = 0.0f;
  float2 translation = make_float2(0.0f);

  // Optional KHR_texture_transform extension data.
  if (textureInfo.transform != nullptr)
  {
    scale.x = textureInfo.transform->uvScale[0];
    scale.y = textureInfo.transform->uvScale[1];

    rotation = textureInfo.transform->rotation; 

    translation.x = textureInfo.transform->uvOffset[0];
    translation.y = textureInfo.transform->uvOffset[1];

    // KHR_texture_transform can override the texture coordinate index.
    if (textureInfo.transform->texCoordIndex.has_value())
    {
      texCoordIndex = textureInfo.transform->texCoordIndex.value();
    }
  }

  if (NUM_ATTR_TEXCOORDS <= texCoordIndex)
  {
    std::cerr << "ERROR: parseTextureInfo() Maximum supported texture coordinate index exceeded, using 0.\n";
    texCoordIndex = 0; // PERF This means the device code doesn't need to check if the texcoord index is in the valid range!
  }

  MY_ASSERT(0 <= textureInfo.textureIndex && textureInfo.textureIndex < samplers.size());

  texture.index       = static_cast<int>(texCoordIndex);
  texture.angle       = rotation; // Need to store the original rotation in radians to be able to recalculate sin and cos below.
  texture.object      = samplers[textureInfo.textureIndex];
  texture.scale       = scale;
  texture.rotation    = make_float2(sinf(rotation), cosf(rotation));
  texture.translation = translation;
}


static cudaTextureAddressMode getTextureAddressMode(fastgltf::Wrap wrap)
{
  switch (wrap)
  {
    case fastgltf::Wrap::Repeat:
      return cudaAddressModeWrap;

    case fastgltf::Wrap::ClampToEdge:
      return cudaAddressModeClamp;

    case fastgltf::Wrap::MirroredRepeat:
      return cudaAddressModeMirror;

    default:
      std::cerr << "WARNING: getTextureAddressMode() Unexpected texture wrap mode = " << static_cast<std::uint16_t>(wrap) << '\n';
      return cudaAddressModeWrap;
  }
}


static std::string getPrimitiveTypeName(fastgltf::PrimitiveType type)
{
  switch (type)
  {
    case fastgltf::PrimitiveType::Points:
      return std::string("POINTS");
    case fastgltf::PrimitiveType::Lines:
      return std::string("LINES");
    case fastgltf::PrimitiveType::LineLoop:
      return std::string("LINE_LOOP");
    case fastgltf::PrimitiveType::LineStrip:
      return std::string("LINE_STRIP");
    case fastgltf::PrimitiveType::Triangles:
      return std::string("TRIANGLES");
    case fastgltf::PrimitiveType::TriangleStrip:
      return std::string("TRIANGLE_STRIP");
    case fastgltf::PrimitiveType::TriangleFan:
      return std::string("TRIANGLE_FAN");
    default:
      return std::string("UNKNOWN");
  }
}


void Application::loadGLTF(const std::filesystem::path& path)
{
  std::cout << "loadGTF(" << path << ")\n"; // DEBUG

  if (!std::filesystem::exists(path))
  {
    std::cerr << "ERROR: loadGLTF() filename " << path << " not found.\n";
    throw std::runtime_error("loadGLTF() File not found");
  }

  // Only the material extensions which are enabled inside the parser are actually filled
  // inside the fastgltf::Material and then transferred to the dev::Material inside initMaterials().
  fastgltf::Extensions extensions =
    fastgltf::Extensions::KHR_materials_anisotropy | 
    fastgltf::Extensions::KHR_materials_clearcoat |
    fastgltf::Extensions::KHR_materials_emissive_strength |
    fastgltf::Extensions::KHR_materials_ior |
    fastgltf::Extensions::KHR_materials_iridescence |
    fastgltf::Extensions::KHR_materials_sheen |
    fastgltf::Extensions::KHR_materials_specular |
    fastgltf::Extensions::KHR_materials_transmission |
    fastgltf::Extensions::KHR_materials_unlit |
    fastgltf::Extensions::KHR_materials_variants |
    fastgltf::Extensions::KHR_materials_volume |
    fastgltf::Extensions::KHR_mesh_quantization |
    fastgltf::Extensions::KHR_texture_transform;

  // The command line parameter --punctual (-p) <int> allows selecting support for the KHR_lights_punctual extension.
  if (m_punctual)
  {
    extensions |= fastgltf::Extensions::KHR_lights_punctual;
  }

  fastgltf::Parser parser(extensions);

  constexpr auto gltfOptions = fastgltf::Options::None 
    | fastgltf::Options::DontRequireValidAssetMember
    | fastgltf::Options::LoadGLBBuffers
    | fastgltf::Options::LoadExternalBuffers
    //| fastgltf::Options::DecomposeNodeMatrices // FIXME I'll want this for animations later.
    | fastgltf::Options::LoadExternalImages;

  fastgltf::GltfDataBuffer data;
    
  data.loadFromFile(path);

  const auto type = fastgltf::determineGltfFileType(&data);

  fastgltf::Expected<fastgltf::Asset> asset(fastgltf::Error::None);

  std::filesystem::path pathParent = path.parent_path();

  if (pathParent.empty())
  {
    pathParent = std::filesystem::path("./");
  }

  if (type == fastgltf::GltfType::glTF)
  {
    asset = parser.loadGltf(&data, pathParent, gltfOptions);
  }
  else if (type == fastgltf::GltfType::GLB)
  {
    asset = parser.loadGltfBinary(&data, pathParent, gltfOptions);
  }
  else // if (type == Invalid)
  {
    std::cerr << "ERROR: determineGltfFileType returned Invalid\n";
    throw std::runtime_error("loadGLTF() Invalid file type");
  }

  if (asset.error() != fastgltf::Error::None)
  {
    std::cerr << "ERROR: loadGLTF() failed with error '" << fastgltf::getErrorMessage(asset.error()) << "'\n";
    throw std::runtime_error("loadGLTF() Failed");
  }

  m_asset = std::move(asset.get());
}


// FIXME This only supports 8 bit component images!
void Application::initImages()
{
  // Images. Load all up-front for simplicity.
  for (const fastgltf::Image& image : m_asset.images)
  {
    std::visit(fastgltf::visitor {
      [](const auto& arg) {
      },
      
      [&](const fastgltf::sources::URI& filePath) {
        MY_ASSERT(filePath.fileByteOffset == 0); // No offsets supported with stbi.
        MY_ASSERT(filePath.uri.isLocalPath());   // Loading only local files.
        int width;
        int height;
        int components;

        const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());

        unsigned char* data = stbi_load(path.c_str(), &width, &height, &components, 4);

        if (data != nullptr)
        {
          addImage(width, height, 8, 4, data);
        }
        else
        {
          std::cout << "ERROR: stbi_load() returned nullptr on image " << image.name << '\n';
          const unsigned char texel[4] = { 0xFF, 0x00, 0xFF, 0xFF };
          addImage(1, 1, 8, 4, texel); // DEBUG Error image is 1x1 RGBA8 magenta opaque.
        }
        
        stbi_image_free(data);
      },
      
      [&](const fastgltf::sources::Array& vector) {
        int width;
        int height;
        int components;

        unsigned char* data = stbi_load_from_memory(vector.bytes.data(), static_cast<int>(vector.bytes.size()), &width, &height, &components, 4);

        if (data != nullptr)
        {
          addImage(width, height, 8, 4, data);
        }
        else
        {
          std::cout << "ERROR: stbi_load() returned nullptr on image " << image.name << '\n';
          const unsigned char texel[4] = { 0xFF, 0x00, 0xFF, 0xFF };
          addImage(1, 1, 8, 4, texel); // DEBUG Error image is 1x1 RGBA8 magenta opaque.
        }

        stbi_image_free(data);
      },

      [&](const fastgltf::sources::Vector& vector) {
        int width;
        int height;
        int components;

        unsigned char* data = stbi_load_from_memory(vector.bytes.data(), static_cast<int>(vector.bytes.size()), &width, &height, &components, 4);

        if (data != nullptr)
        {
          addImage(width, height, 8, 4, data);
        }
        else
        {
          std::cout << "ERROR: stbi_load() returned nullptr on image " << image.name << '\n';
          const unsigned char texel[4] = { 0xFF, 0x00, 0xFF, 0xFF };
          addImage(1, 1, 8, 4, texel); // DEBUG Error image is 1x1 RGBA8 magenta opaque.
        }

        stbi_image_free(data);
      },

      [&](const fastgltf::sources::BufferView& view) {
        const auto& bufferView = m_asset.bufferViews[view.bufferViewIndex];
        const auto& buffer     = m_asset.buffers[bufferView.bufferIndex];

        std::visit(fastgltf::visitor {
          // We only care about VectorWithMime here, because we specify LoadExternalBuffers, meaning all buffers are already loaded into a vector.
          [](const auto& arg) {
          },

          [&](const fastgltf::sources::Array& vector) {
            int width;
            int height;
            int components;
            
            unsigned char* data = stbi_load_from_memory(vector.bytes.data() + bufferView.byteOffset, static_cast<int>(bufferView.byteLength), &width, &height, &components, 4);
       
            if (data != nullptr)
            {
              addImage(width, height, 8, 4, data);
            }
            else
            {
              std::cout << "ERROR: stbi_load() returned nullptr on image " << image.name << '\n';
              const unsigned char texel[4] = { 0xFF, 0x00, 0xFF, 0xFF };
              addImage(1, 1, 8, 4, texel); // DEBUG Error image is 1x1 RGBA8 magenta opaque.
            }
            
            stbi_image_free(data);
          },

          [&](const fastgltf::sources::Vector& vector) {
            int width;
            int height;
            int components;
            
            unsigned char* data = stbi_load_from_memory(vector.bytes.data() + bufferView.byteOffset, static_cast<int>(bufferView.byteLength), &width, &height, &components, 4);
       
            if (data != nullptr)
            {
              addImage(width, height, 8, 4, data);
            }
            else
            {
              std::cout << "ERROR: stbi_load() returned nullptr on image " << image.name << '\n';
              const unsigned char texel[4] = { 0xFF, 0x00, 0xFF, 0xFF };
              addImage(1, 1, 8, 4, texel); // DEBUG Error image is 1x1 RGBA8 magenta opaque.
            }
            
            stbi_image_free(data);
          }
        }, buffer.data);
      },
    }, image.data);
  }
}


void Application::initTextures()
{
  if (m_asset.textures.empty())
  {
    return;
  }

  // glTF requires sRGB for baseColor, specularColor, sheenColor and emissive textures inside the tetxure interpolation.
  // Doing sRGB adjustments with pow(rgb, 2.2) inside the shader after the texture lookup is too late.
  // TextureLinearInterpolationTest.gltf will only pass with sRGB done inside the tetxure object itself.
  std::vector<int> sRGB(m_asset.textures.size(), 0); 

  // Run over all materials inside the asset and set the sRGB flag for all textures
  // which are used as baseColorTexture, emissiveTexture or specularColorTexture.
  for (const fastgltf::Material& material : m_asset.materials)
  {
    if (material.pbrData.baseColorTexture.has_value())
    {
      const fastgltf::TextureInfo& textureInfo = material.pbrData.baseColorTexture.value();
      sRGB[textureInfo.textureIndex] = 1;
    }
    if (material.emissiveTexture.has_value())
    {
      const fastgltf::TextureInfo& textureInfo = material.emissiveTexture.value();
      sRGB[textureInfo.textureIndex] = 1;
    }
    if (material.specular != nullptr && material.specular->specularColorTexture.has_value())
    {
      const fastgltf::TextureInfo& textureInfo = material.specular->specularColorTexture.value();
      sRGB[textureInfo.textureIndex] = 1;
    }
    if (material.sheen != nullptr && material.sheen->sheenColorTexture.has_value())
    {
      const fastgltf::TextureInfo& textureInfo = material.sheen->sheenColorTexture.value();
      sRGB[textureInfo.textureIndex] = 1;
    }
  }

  // Textures. These refer to previously loaded images.
  for (size_t i = 0; i < m_asset.textures.size(); ++i)
  {
    const fastgltf::Texture& texture = m_asset.textures[i];

    // Default to wrap repeat and linear filtering when there is no sampler.
    cudaTextureAddressMode address_s = cudaAddressModeWrap;
    cudaTextureAddressMode address_t = cudaAddressModeWrap;
    cudaTextureFilterMode  filter    = cudaFilterModeLinear;

    if (texture.samplerIndex.has_value())
    {
      MY_ASSERT(texture.samplerIndex.value() < m_asset.samplers.size());
      const auto& sampler = m_asset.samplers[texture.samplerIndex.value()];

      address_s = getTextureAddressMode(sampler.wrapS);
      address_t = getTextureAddressMode(sampler.wrapT);

      if (sampler.minFilter.has_value())
      {
        fastgltf::Filter minFilter = sampler.minFilter.value();

        switch (minFilter)
        {
          // This renderer is not downloading mipmaps. 
          // Pick the filter depending on the 2D filtering which is the first.
          case fastgltf::Filter::Nearest:
          case fastgltf::Filter::NearestMipMapNearest:
          case fastgltf::Filter::NearestMipMapLinear:
            filter = cudaFilterModePoint;
            break;

          case fastgltf::Filter::Linear:
          case fastgltf::Filter::LinearMipMapNearest:
          case fastgltf::Filter::LinearMipMapLinear:
          default:
            filter = cudaFilterModeLinear;
            break;
        }
      }
    }
    
    MY_ASSERT(texture.imageIndex.has_value());
    addSampler(address_s, address_t, filter, texture.imageIndex.value(), sRGB[i]);
  }
}


void Application::initMaterials()
{
  // Materials
  for (size_t index = 0; index < m_asset.materials.size(); ++index)
  {
    //std::cout << "Processing glTF material: '" << material.name << "'\n";

    const fastgltf::Material& material = m_asset.materials[index];

    MaterialData mtl;

    mtl.index = index; // To be able to identify the material during picking.

    mtl.doubleSided = material.doubleSided;

    switch (material.alphaMode)
    {
      case fastgltf::AlphaMode::Opaque:
        mtl.alphaMode = MaterialData::ALPHA_MODE_OPAQUE;
        break;

      case fastgltf::AlphaMode::Mask:
        mtl.alphaMode   = MaterialData::ALPHA_MODE_MASK;
        mtl.alphaCutoff = material.alphaCutoff;
        break;

      case fastgltf::AlphaMode::Blend:
        mtl.alphaMode = MaterialData::ALPHA_MODE_BLEND;
        break;

      default:
        std::cerr << "ERROR: Invalid material alpha mode. Using opaque\n";
        mtl.alphaMode = MaterialData::ALPHA_MODE_OPAQUE;
        break;
    }

    mtl.baseColorFactor = make_float4(material.pbrData.baseColorFactor[0], 
                                      material.pbrData.baseColorFactor[1], 
                                      material.pbrData.baseColorFactor[2], 
                                      material.pbrData.baseColorFactor[3]);
    if (material.pbrData.baseColorTexture.has_value())
    {
      parseTextureInfo(m_samplers, material.pbrData.baseColorTexture.value(), mtl.baseColorTexture);
    }

    mtl.metallicFactor  = material.pbrData.metallicFactor;
    mtl.roughnessFactor = material.pbrData.roughnessFactor;
    if (material.pbrData.metallicRoughnessTexture.has_value())
    {
      parseTextureInfo(m_samplers, material.pbrData.metallicRoughnessTexture.value(), mtl.metallicRoughnessTexture);
    }

    if (material.normalTexture.has_value())
    {
      const auto& normalTextureInfo = material.normalTexture.value();
      
      mtl.normalTextureScale = normalTextureInfo.scale;
      parseTextureInfo(m_samplers, normalTextureInfo, mtl.normalTexture);
    }  

    // Ambient occlusion should not really be required with a global illumination renderer,
    // but many glTF models are very low-resolution geometry and details are baked into normal and occlusion maps.
    if (material.occlusionTexture.has_value())
    {
      const auto& occlusionTextureInfo = material.occlusionTexture.value();

      mtl.occlusionTextureStrength = occlusionTextureInfo.strength;
      parseTextureInfo(m_samplers, occlusionTextureInfo, mtl.occlusionTexture);
    }  
    
    mtl.emissiveStrength = material.emissiveStrength; // KHR_materials_emissive_strength
    mtl.emissiveFactor = make_float3(material.emissiveFactor[0],
                                     material.emissiveFactor[1],
                                     material.emissiveFactor[2]);
    if (material.emissiveTexture.has_value())
    {
      parseTextureInfo(m_samplers, material.emissiveTexture.value(), mtl.emissiveTexture);
    }  

    // Set material.flags bits to indicate which Khronos material extension is used and has data.
    // This is only evaluated for the KHH_materials_volume so far because that affects the face culling
    // Volumes require double-sided geometry even when the glTF file didn't specify it.
    mtl.flags = 0;

    // KHR_materials_ior
    // Not handled as optional extension inside fastgltf.
    // It's always present and defaults to 1.5 when not set inside the asset.
    mtl.ior = material.ior;

    // KHR_materials_specular
    if (material.specular != nullptr) 
    {
      mtl.flags |= FLAG_KHR_MATERIALS_SPECULAR;

      mtl.specularFactor = material.specular->specularFactor;
      if (material.specular->specularTexture.has_value())
      {
        parseTextureInfo(m_samplers, material.specular->specularTexture.value(), mtl.specularTexture);
      }
      mtl.specularColorFactor = make_float3(material.specular->specularColorFactor[0],
                                            material.specular->specularColorFactor[1],
                                            material.specular->specularColorFactor[2]);
      if (material.specular->specularColorTexture.has_value())
      {
        parseTextureInfo(m_samplers, material.specular->specularColorTexture.value(), mtl.specularColorTexture);
      }
    }

    // KHR_materials_transmission
    if (material.transmission != nullptr) 
    {
      mtl.flags |= FLAG_KHR_MATERIALS_TRANSMISSION;

      mtl.transmissionFactor = material.transmission->transmissionFactor;
      if (material.transmission->transmissionTexture.has_value())
      {
        parseTextureInfo(m_samplers, material.transmission->transmissionTexture.value(), mtl.transmissionTexture);
      }
    }

    // KHR_materials_volume
    if (material.volume != nullptr) 
    {
      mtl.flags |= FLAG_KHR_MATERIALS_VOLUME;

      // Some glTF models like the IridescenceLamp.gltf use only the thicknessFactor to define the volume.
      // HACK The ray tracer ignores the thicknessFactor except when it's 0.0f,
      // which is one condition for thin-walled materials even when FLAG_KHR_MATERIALS_VOLUME is set.
      mtl.thicknessFactor = material.volume->thicknessFactor;
      //if (material.volume->thicknessTexture.has_value())
      //{
      //  parseTextureInfo(m_samplers, material.volume->thicknessTexture.value(), mtl.thicknessTexture);
      //}
      // The attenuationDistance default is +inf which effectively disables volume absorption.
      // The raytracer only enables volume absorption for attenuationDistance values less than RT_DEFAULT_MAX.
      mtl.attenuationDistance = material.volume->attenuationDistance;
      mtl.attenuationColor = make_float3(material.volume->attenuationColor[0],
                                         material.volume->attenuationColor[1],
                                         material.volume->attenuationColor[2]);
    }

    // KHR_materials_clearcoat
    if (material.clearcoat != nullptr) 
    {
      mtl.flags |= FLAG_KHR_MATERIALS_CLEARCOAT;

      mtl.clearcoatFactor = material.clearcoat->clearcoatFactor;
      if (material.clearcoat->clearcoatTexture.has_value())
      {
        parseTextureInfo(m_samplers, material.clearcoat->clearcoatTexture.value(), mtl.clearcoatTexture);
      }
      mtl.clearcoatRoughnessFactor = material.clearcoat->clearcoatRoughnessFactor;
      if (material.clearcoat->clearcoatRoughnessTexture.has_value())
      {
        parseTextureInfo(m_samplers, material.clearcoat->clearcoatRoughnessTexture.value(), mtl.clearcoatRoughnessTexture);
      }
      if (material.clearcoat->clearcoatNormalTexture.has_value())
      {
        parseTextureInfo(m_samplers, material.clearcoat->clearcoatNormalTexture.value(), mtl.clearcoatNormalTexture);
        
        // If the clearcoatNormalTexture is the same as the normalTexture, then let the shader apply
        // the same normalTextureScale to match the clearcoat normal to the material normal.
        // (The Texture fields are all default initialized, so this comparison always works with valid data.)
        mtl.isClearcoatNormalBaseNormal = (mtl.clearcoatNormalTexture == mtl.normalTexture);
      }
    }

    // KHR_materials_sheen
    if (material.sheen != nullptr) 
    {
      mtl.flags |= FLAG_KHR_MATERIALS_SHEEN;

      mtl.sheenColorFactor = make_float3(material.sheen->sheenColorFactor[0],
                                         material.sheen->sheenColorFactor[1],
                                         material.sheen->sheenColorFactor[2]);
      if (material.sheen->sheenColorTexture.has_value())
      {
        parseTextureInfo(m_samplers, material.sheen->sheenColorTexture.value(), mtl.sheenColorTexture);
      }
      mtl.sheenRoughnessFactor = material.sheen->sheenRoughnessFactor;
      if (material.sheen->sheenRoughnessTexture.has_value())
      {
        parseTextureInfo(m_samplers, material.sheen->sheenRoughnessTexture.value(), mtl.sheenRoughnessTexture);
      }
    }

    // KHR_materials_anisotropy
    if (material.anisotropy != nullptr) 
    {
      mtl.flags |= FLAG_KHR_MATERIALS_ANISOTROPY;

      mtl.anisotropyStrength = material.anisotropy->anisotropyStrength;
      mtl.anisotropyRotation = material.anisotropy->anisotropyRotation;
      if (material.anisotropy->anisotropyTexture.has_value())
      {
        parseTextureInfo(m_samplers, material.anisotropy->anisotropyTexture.value(), mtl.anisotropyTexture);
      }
    }

    // KHR_materials_iridescence
    if (material.iridescence != nullptr) 
    {
      mtl.flags |= FLAG_KHR_MATERIALS_IRIDESCENCE;

      mtl.iridescenceFactor = material.iridescence->iridescenceFactor;
      if (material.iridescence->iridescenceTexture.has_value())
      {
        parseTextureInfo(m_samplers, material.iridescence->iridescenceTexture.value(), mtl.iridescenceTexture);
      }
      mtl.iridescenceIor = material.iridescence->iridescenceIor;
      mtl.iridescenceThicknessMinimum = material.iridescence->iridescenceThicknessMinimum;
      mtl.iridescenceThicknessMaximum = material.iridescence->iridescenceThicknessMaximum;
      if (material.iridescence->iridescenceThicknessTexture.has_value())
      {
        parseTextureInfo(m_samplers, material.iridescence->iridescenceThicknessTexture.value(), mtl.iridescenceThicknessTexture);
      }
    }

    // KHR_materials_unlit
    mtl.unlit = material.unlit;

    //debugDumpMaterial(mtl); // DEBUG 

    m_materialsOrg.push_back(mtl); // The original data inside the asset.
    m_materials.push_back(mtl);    // The materials changed by the GUI.
  }
}


void Application::initMeshes()
{
  for (const fastgltf::Mesh& gltf_mesh : m_asset.meshes)
  {
    //std::cout << "Processing glTF mesh: '" << gltf_mesh.name << "'\n";

    dev::Mesh* mesh = new dev::Mesh();

    mesh->name = gltf_mesh.name;

    for (const fastgltf::Primitive& primitive : gltf_mesh.primitives)
    {
      // FIXME Implement all polygonal primitive modes and convert them to independent triangles.
      // FIXME Implement all primitive modes (points and lines) as well and convert them to spheres and linear curves.
      // (That wouldn't handle the "lines render as single pixel" in screen space GLTF specs.)
      if (primitive.type != fastgltf::PrimitiveType::Triangles) // Ignore non-triangle meshes
      {
        std::cerr << "WARNING: Unsupported non-triangle primitive " << getPrimitiveTypeName(primitive.type) << " skipped.\n";
        continue;
      }

      // POSITION attribute must be present!
      auto itPosition = primitive.findAttribute("POSITION");
      if (itPosition == primitive.attributes.end()) // Meshes MUST have a position attribute.
      {
        std::cerr << "ERROR: primitive has no POSITION attribute, skipped.\n";
        continue;
      }
      
      dev::Primitive prim;
      
      int indexAccessor = static_cast<int>(itPosition->second); // Type integer to allow -1 in createDeviceBuffer() for an empty DeviceBuffer() .
      prim.positions = createDeviceBuffer(m_asset, indexAccessor, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float);

      indexAccessor = (primitive.indicesAccessor.has_value()) ? static_cast<int>(primitive.indicesAccessor.value()) : -1;
      prim.indices = createDeviceBuffer(m_asset, indexAccessor, fastgltf::AccessorType::Scalar, fastgltf::ComponentType::UnsignedInt);

      auto itColor = primitive.findAttribute("COLOR_0"); // Only supporting one color attribute.
      indexAccessor = (itColor != primitive.attributes.end()) ? static_cast<int>(itColor->second) : -1;
      // This also handles alpha expansion of Vec3 colors to Vec4.
      prim.colors = createDeviceBuffer(m_asset, indexAccessor, fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float);

      // "When normals are not specified, client implementations MUST calculate flat normals and the provided tangents (if present) MUST be ignored."
      auto itNormal = primitive.findAttribute("NORMAL");
      indexAccessor = (itNormal != primitive.attributes.end()) ? static_cast<int>(itNormal->second) : -1;
      const bool allowTangents = (0 <= indexAccessor);
      prim.normals = createDeviceBuffer(m_asset, indexAccessor, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float);

      // "When tangents are not specified, client implementations SHOULD calculate tangents using default 
      // MikkTSpace algorithms with the specified vertex positions, normals, and texture coordinates associated with the normal texture."
      auto itTangent = primitive.findAttribute("TANGENT");
      indexAccessor = (itTangent != primitive.attributes.end() && allowTangents) ? static_cast<int>(itTangent->second) : -1;
      prim.tangents = createDeviceBuffer(m_asset, indexAccessor, fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float);

      for (size_t j = 0; j < NUM_ATTR_TEXCOORDS; ++j)
      {
        std::string texcoord_str = std::string("TEXCOORD_") + std::to_string(j);
        auto itTexcoord = primitive.findAttribute(texcoord_str);
        indexAccessor = (itTexcoord != primitive.attributes.end()) ? static_cast<int>(itTexcoord->second) : -1;
        prim.texcoords[j] = createDeviceBuffer(m_asset, indexAccessor, fastgltf::AccessorType::Vec2, fastgltf::ComponentType::Float);
      }

      for (size_t j = 0; j < NUM_ATTR_JOINTS; ++j)
      {
        std::string joints_str = std::string("JOINTS_") + std::to_string(j);
        auto itJoints = primitive.findAttribute(joints_str);
        indexAccessor = (itJoints != primitive.attributes.end()) ? static_cast<int>(itJoints->second) : -1;
        prim.joints[j]= createDeviceBuffer(m_asset, indexAccessor, fastgltf::AccessorType::Vec4, fastgltf::ComponentType::UnsignedShort);
      }
      
      for (size_t j = 0; j < NUM_ATTR_WEIGHTS; ++j)
      {
        std::string weights_str = std::string("WEIGHTS_") + std::to_string(j);
        auto itWeights = primitive.findAttribute(weights_str);
        indexAccessor = (itWeights != primitive.attributes.end()) ? static_cast<int>(itWeights->second) : -1;
        prim.weights[j] = createDeviceBuffer(m_asset, indexAccessor, fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float);
      }
      
      prim.indexMaterial = (primitive.materialIndex.has_value()) ? static_cast<int32_t>(primitive.materialIndex.value()) : -1;

      // KHR_materials_variants
      for (size_t i = 0; i < primitive.mappings.size(); ++i)
      {
        const int32_t index = primitive.mappings[i].has_value() ? static_cast<int32_t>(primitive.mappings[i].value()) : prim.indexMaterial;
        
        prim.mappings.push_back(index);
      }
      
      // Derive the current material index.
      prim.currentMaterial = (primitive.mappings.empty()) ? prim.indexMaterial : prim.mappings[m_indexVariant];

      // DEBUG Check if the provided attributes are reasonable.
      // Wrong tangents are fixed up inside the renderer, but wrong normals cannot be easily corrected.
      //prim.checkTangents(); // DEBUG. Some models provide invalid tangents, collinear with the geometric normal.
      //prim.checkNormals();  // DEBUG. Some models provide normal attributes which are perpendicular to the geometry normal which can result in NaN shading space TBN vectors.

      // If we arrived here, the mesh contains at least one triangle primitive.
      mesh->primitives.push_back(prim);
    } // for primitive

    // Unconditionally push the mesh pointer to have the same index as into m_asset.meshes.
    m_meshes.push_back(mesh);
  } // for gltf_mesh
}


void Application::initCameras()
{
  // If there is no camera inside the scene, generate a default perspective camera.
  // That simplifies the whole GUI and scene handling.
  if (m_asset.cameras.empty())
  {
    fastgltf::Camera::Perspective perspective;

    perspective.aspectRatio = 1.0f;
    perspective.yfov = 45.0f * M_PIf / 180.0f; // In radians.
    perspective.znear = 0.01;

    fastgltf::Camera camera;

    camera.camera = perspective;

    m_asset.cameras.push_back(camera);

    m_isDefaultCamera = true; // This triggers an initialization of the default camera position and lookat insde initTrackball()
  }

  for (const fastgltf::Camera& gltf_camera : m_asset.cameras)
  {
    dev::Camera* camera = new dev::Camera();

    // At this time, the camera transformation matrix is unknown.
    // Just initialize position and up vectors with defaults and update them during scene node traversal later.
    if (const auto* pPerspective = std::get_if<fastgltf::Camera::Perspective>(&gltf_camera.camera))
    {
      const glm::vec3 pos(0.0f, 0.0f, 0.0f);
      const glm::vec3 up(0.0f, 1.0f, 0.0f);

      // yfov should be less than PI and must be > 0.0f to work. 
      MY_ASSERT(0.0f < pPerspective->yfov && pPerspective->yfov < M_PIf);
      const float yfov = pPerspective->yfov * 180.0f / M_PIf;

      // This value isn't used anyway because for perspective cameras the viewport defines the aspect ratio.
      float aspectRatio = (pPerspective->aspectRatio.has_value() && pPerspective->aspectRatio.value() != 0.0f) 
                        ? pPerspective->aspectRatio.value() 
                        : 1.0f;
      
      camera->setPosition(pos);
      camera->setUp(up);
      camera->setFovY(yfov); // In degrees.
      camera->setAspectRatio(aspectRatio);
    }
    else if (const auto* pOrthograhpic = std::get_if<fastgltf::Camera::Orthographic>(&gltf_camera.camera))
    {
      const glm::vec3 pos(0.0f, 0.0f, 0.0f);
      const glm::vec3 up(0.0f, 1.0f, 0.0f);

      // The orthographic projection is always finite inside GLTF because znear and zfar are required.
      // This defines an infinite projection from a plane at the position.
      camera->setPosition(pos);
      camera->setUp(up);
      camera->setFovY(-1.0f); // <= 0.0f means orthographic projection.
      camera->setMagnification(glm::vec2(pOrthograhpic->xmag, pOrthograhpic->ymag));
    }
    else
    {
      std::cerr << "ERROR: Unexpected camera type.\n";
    }

    // Just default initialize the camera inside the array to have the same indexing as m_asset.cameras.
    m_cameras.push_back(camera);
  }
}


void Application::initLights()
{
  for (const fastgltf::Light& gltf_light : m_asset.lights)
  {
    dev::Light* light = new dev::Light();

    // Shared values.
    light->name = gltf_light.name;
    // These types  match the renderer's order of light sampling callable programs.
    switch (gltf_light.type)
    {
      case fastgltf::LightType::Point:
        light->type = 0;
        break;
      case fastgltf::LightType::Spot:
        light->type = 1;
        break;
      case fastgltf::LightType::Directional:
        light->type = 2;
        break;
    }
    light->color = make_float3(gltf_light.color[0], gltf_light.color[1], gltf_light.color[2]);
    light->intensity = gltf_light.intensity;

    switch (gltf_light.type)
    {
      case fastgltf::LightType::Point:
        light->range = (gltf_light.range.has_value()) ? gltf_light.range.value() : RT_DEFAULT_MAX;
        break;

      case fastgltf::LightType::Spot:
        light->range = (gltf_light.range.has_value()) ? gltf_light.range.value() : RT_DEFAULT_MAX;
        light->innerConeAngle = (gltf_light.innerConeAngle.has_value()) ? gltf_light.innerConeAngle.value() : 0.0f;
        light->outerConeAngle = (gltf_light.outerConeAngle.has_value()) ? gltf_light.outerConeAngle.value() : 0.25f * M_PIf;
        break;

      case fastgltf::LightType::Directional:
        // No additional data.
        break;
    }

    light->matrix = glm::mat4(1.0f); // Identity. Updated during traverse() of the scene nodes.

    m_lights.push_back(light);
  }
}

void Application::initScene(const int index)
{
  //std::cout << "initScene(" << index << ")\n"; // DEBUG

  // glTF specs: "A glTF asset that does not contain any scenes SHOULD be treated as a library of individual entities such as materials or meshes."
  // That would only make sense if the application would be able to mix assets from different files.
  // If there is no scene defined, this just creates one from all root nodes inside the asset to be able to continue.
  if (m_asset.scenes.empty())
  {
    // Find all root nodes.
    std::vector<bool> isRoot(m_asset.nodes.size(), true);
    
    // Root nodes are all nodes which do not appear inside any node's children.
    for (const fastgltf::Node& node : m_asset.nodes)
    {
      for (size_t child : node.children)
      {
        isRoot[child] = false;
      }
    }

    // Now build a scene which is just the vector of root node indices.
    fastgltf::Scene scene;

    scene.name = std::string("scene_root_nodes");

    for (size_t i = 0; i < isRoot.size(); ++i )
    {
      if (isRoot[i])
      {
        scene.nodeIndices.push_back(i);
      }
    }

    MY_ASSERT(!scene.nodeIndices.empty());

    m_asset.scenes.push_back(scene);
  }

  // Determine which scene inside the asset should be used.
  if (index < 0)
  {
    m_indexScene = (m_asset.defaultScene.has_value()) ? m_asset.defaultScene.value() : 0;
  }
  else if (index < m_asset.scenes.size())
  {
    m_indexScene = static_cast<size_t>(index);
  }
  // else m_indexScene unchanged.

  MY_ASSERT(m_indexScene < m_asset.scenes.size());

  const fastgltf::Scene& scene = m_asset.scenes[m_indexScene];

  for (size_t indexNode : scene.nodeIndices)
  {
    //std::cout << "===== ROOT NODE " << indexNode << " =====\n"; // DEBUG
    
    // This does the initialization of all resources which are reachable via the active scene's nodes.
    traverseNode(indexNode, glm::mat4(1.0f));
  }
}


void Application::updateScene()
{
  //std::cout << "updateScene()\n"; // DEBUG

  // Delete the previous instances.
  for (dev::Instance* instance : m_instances)
  {
    delete instance;
  }
  m_instances.clear();

  MY_ASSERT(m_indexScene < m_asset.scenes.size());
  const fastgltf::Scene& scene = m_asset.scenes[m_indexScene];

  for (size_t indexNode : scene.nodeIndices)
  {
    //std::cout << "===== ROOT NODE " << indexNode << " =====\n"; // DEBUG
    
    traverseNode(indexNode, glm::mat4(1.0f));
  }
}


static glm::mat4 getTransformMatrix(const fastgltf::Node& node, glm::mat4x4& base)
{
  // Matrix and TRS values are mutually exclusive according to the spec
  if (const fastgltf::Node::TransformMatrix* matrix = std::get_if<fastgltf::Node::TransformMatrix>(&node.transform))
  {
    return base * glm::mat4x4(glm::make_mat4x4(matrix->data()));
  }

  if (const fastgltf::TRS* transform = std::get_if<fastgltf::TRS>(&node.transform))
  {
    // Warning: The quaternion to mat4x4 conversion here is not correct with all versions of glm.
    // glTF provides the quaternion as (x, y, z, w), which is the same layout glm used up to version 0.9.9.8.
    // However, with commit 59ddeb7 (May 2021) the default order was changed to (w, x, y, z).
    // You could either define GLM_FORCE_QUAT_DATA_XYZW to return to the old layout,
    // or you could use the recently added static factory constructor glm::quat::wxyz(w, x, y, z),
    // which guarantees the parameter order.
    // => 
    // Using glm version 0.9.9.9 (or newer) and glm::quat::wxyz(w, x, y, z).
    // If this is not compiling your glm version is too old!
    return base * glm::translate(glm::mat4(1.0f), glm::make_vec3(transform->translation.data()))
                * glm::toMat4(glm::quat::wxyz(transform->rotation[3], transform->rotation[0], transform->rotation[1], transform->rotation[2]))
                * glm::scale(glm::mat4(1.0f), glm::make_vec3(transform->scale.data()));
  }

  return base;
}


void Application::traverseNode(const size_t nodeIndex, glm::mat4 matrix)
{
  //std::cout << "traverseNode(" << nodeIndex << ")\n"; // DEBUG

  const fastgltf::Node& node = m_asset.nodes[nodeIndex];
  
  //std::cout << "  node.name = " << node.name << "\n"; // DEBUG

  matrix = getTransformMatrix(node, matrix);

  if (node.meshIndex.has_value())
  {
    const size_t indexMesh = node.meshIndex.value();
    //std::cout << "  indexMesh = " << indexMesh << '\n'; // DEBUG 

    MY_ASSERT(indexMesh < m_meshes.size());
    dev::Mesh* mesh = m_meshes[indexMesh]; // This array has been initialized in initMeshes().

    // If the mesh contains triangle data, add an instance to the scene graph.
    if (!mesh->primitives.empty()) 
    {
      dev::Instance* instance = new dev::Instance();

      instance->transform = matrix;
      instance->indexMesh = static_cast<int>(indexMesh);

      m_instances.push_back(instance);
    }
  }

  // FIXME Implement skinning animation.
  //if (node.skinIndex.has_value())
  //{
  //  const size_t indexSkin = node.skinIndex.value();
  //  //std::cout << "  indexSkin = " << indexSkin << '\n'; // DEBUG 
  //}

  if (node.cameraIndex.has_value())
  {
    const size_t indexCamera = node.cameraIndex.value();
    //std::cout << "  indexCamera = " << indexCamera << '\n'; // DEBUG
    
    const fastgltf::Camera& gltf_camera = m_asset.cameras[indexCamera];
    dev::Camera* camera = m_cameras[indexCamera]; // The m_cameras array is already initialized with default perspective cameras.

    if (const fastgltf::Camera::Perspective* pPerspective = std::get_if<fastgltf::Camera::Perspective>(&gltf_camera.camera))
    {
      const glm::vec3 pos = glm::vec3(matrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
      const glm::vec3 up  = glm::vec3(matrix * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));
      const float yfov    = pPerspective->yfov * 180.0f / M_PIf;

      float aspectRatio = (pPerspective->aspectRatio.has_value() && pPerspective->aspectRatio.value() != 0.0f) 
                        ? pPerspective->aspectRatio.value() 
                        : 1.0f;

      camera->setPosition(pos);
      camera->setUp(up);
      camera->setFovY(yfov);
      camera->setAspectRatio(aspectRatio);
    }
    else if (const fastgltf::Camera::Orthographic* pOrthograhpic = std::get_if<fastgltf::Camera::Orthographic>(&gltf_camera.camera))
    {
      const glm::vec3 pos = glm::vec3(matrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
      const glm::vec3 up  = glm::vec3(matrix * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));

      camera->setPosition(pos);
      camera->setUp(up);
      camera->setFovY(-1.0f); // <= 0.0f means orthographic projection.
      camera->setMagnification(glm::vec2(pOrthograhpic->xmag, pOrthograhpic->ymag));
    }
    else
    {
      std::cerr << "ERROR: Unexpected camera type.\n";
    }
  }

  // KHR_lights_punctual
  if (node.lightIndex.has_value())
  {
    const size_t indexLight = node.lightIndex.value();
    std::cout << "  indexLight = " << indexLight << '\n'; // DEBUG

    MY_ASSERT(indexLight < m_lights.size());
    m_lights[indexLight]->matrix = matrix;

    m_isDirtyLights = true;
  }

  for (size_t child : node.children)
  {
    traverseNode(child, matrix);
  }
}


void Application::addImage(
  const int32_t width,
  const int32_t height,
  const int32_t bitsPerComponent,
  const int32_t numComponents,
  const void*   data)
{
  // Allocate CUDA array in device memory
  int32_t               pitch;
  cudaChannelFormatDesc channel_desc;

  if (bitsPerComponent == 8)
  {
    pitch = width * numComponents * sizeof(uint8_t);
    channel_desc = cudaCreateChannelDesc<uchar4>();
  }
  else if (bitsPerComponent == 16)
  {
    pitch = width * numComponents * sizeof(uint16_t);
    channel_desc = cudaCreateChannelDesc<ushort4>();
  }
  else
  {
    std::cerr << "ERROR: addImage() Unsupported bitsPerComponent " << bitsPerComponent << '\n';
    throw std::runtime_error("addImage() Unsupported bitsPerComponent");
  }

  cudaArray_t cuda_array = nullptr;

  CUDA_CHECK( cudaMallocArray(&cuda_array, &channel_desc, width, height) );
  CUDA_CHECK( cudaMemcpy2DToArray(cuda_array, 0, 0, data, pitch, pitch, height, cudaMemcpyHostToDevice) );

  m_images.push_back(cuda_array);
}


void Application::addSampler(
  cudaTextureAddressMode address_s,
  cudaTextureAddressMode address_t,
  cudaTextureFilterMode  filter,
  const size_t           image_idx, 
  const int              sRGB)
{
  cudaResourceDesc resDesc = {};

  resDesc.resType = cudaResourceTypeArray;
  MY_ASSERT(image_idx < m_images.size())
  resDesc.res.array.array = m_images[image_idx];

  cudaTextureDesc texDesc = {};

  texDesc.addressMode[0]      = address_s;
  texDesc.addressMode[1]      = address_t;
  texDesc.filterMode          = filter;
  texDesc.readMode            = cudaReadModeNormalizedFloat;
  texDesc.normalizedCoords    = 1;
  texDesc.maxAnisotropy       = 1;
  texDesc.maxMipmapLevelClamp = 0;
  texDesc.minMipmapLevelClamp = 0;
  texDesc.mipmapFilterMode    = cudaFilterModePoint; // No mipmap filtering.
  texDesc.borderColor[0]      = 1.0f; // DEBUG Why is the Khronos glTF-Sample Viewer using white for the border color?
  texDesc.borderColor[1]      = 1.0f;
  texDesc.borderColor[2]      = 1.0f;
  texDesc.borderColor[3]      = 1.0f;
  // glTF uses sRGB for baseColor, specularColor, sheenColor and emissive texture RGB values, all other texture data is linear.
  // TextureLinearInterpolationTest.gltf requires that the texture engine interpolates with sRGB enabled.
  // Doing sRGB adjustments with pow(rgb, 2.2) inside the shader is not producing the correct result because that is after linear texture interpolation.
  texDesc.sRGB                = sRGB;

  // Create texture object.
  cudaTextureObject_t cuda_tex = 0;

  CUDA_CHECK( cudaCreateTextureObject(&cuda_tex, &resDesc, &texDesc, nullptr) );

  m_samplers.push_back(cuda_tex);
}


void Application::initRenderer()
{
  buildMeshAccels();
  buildInstanceAccel(); // The top-level IAS build sets m_sceneAABB.

  createPipeline();
  createSBT();
}


void Application::updateRenderer()
{
  buildInstanceAccel(); // The top-level IAS build sets m_sceneAABB.

  updateSBT(); // Rewrite the hit records according to the m_instances of the current scene.
}


void Application::cleanup()
{
  // OptiX cleanup.
  if (m_pipeline)
  {
    OPTIX_CHECK( m_api.optixPipelineDestroy(m_pipeline) );
    m_pipeline = 0;
  }

  for (OptixProgramGroup programGroup : m_programGroups)
  {
    OPTIX_CHECK( m_api.optixProgramGroupDestroy(programGroup) );
  }
  m_programGroups.clear();

  for (OptixModule m : m_modules)
  {
    OPTIX_CHECK(m_api.optixModuleDestroy(m));
  }
  m_modules.clear();

  if (m_optixContext)
  {
    OPTIX_CHECK( m_api.optixDeviceContextDestroy(m_optixContext) );
    m_optixContext = 0;
  }

  // CUDA cleanup.
  if (m_cudaGraphicsResource != nullptr)
  {
    CU_CHECK( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
  }

  for (cudaTextureObject_t& sampler : m_samplers)
  {
    CUDA_CHECK( cudaDestroyTextureObject(sampler) );
  }
  m_samplers.clear();

  for (cudaArray_t& image : m_images)
  {
    CUDA_CHECK( cudaFreeArray(image) );
  }
  m_images.clear();

  if (m_d_ias)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_d_ias)) );
    m_d_ias = 0;
  }
  if (m_sbt.raygenRecord)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)) );
    m_sbt.raygenRecord = 0;
  }
  if (m_sbt.missRecordBase)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)) );
    m_sbt.missRecordBase = 0;
  }
  if (m_sbt.hitgroupRecordBase)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)) );
    m_sbt.hitgroupRecordBase = 0;
  }

  for (dev::Mesh* mesh : m_meshes)
  {
    for (dev::Primitive& primitive : mesh->primitives)
    {
      primitive.free();
    }
    
    if (mesh->d_gas)
    {
      CUDA_CHECK( cudaFree(reinterpret_cast<void*>(mesh->d_gas)) );
    }

    delete mesh;
  }
  m_meshes.clear();

  for (dev::Instance* instance : m_instances)
  {
    delete instance;
  }
  m_instances.clear();

  for (dev::Camera* camera : m_cameras)
  {
    delete camera;
  }
  m_cameras.clear();

  for (dev::Light* light : m_lights)
  {
    delete light;
  }
  m_lights.clear();

  if (m_launchParameters.bufferAccum != 0 && 
      m_interop != INTEROP_PBO) // For INTEROP_PBO: bufferAccum is the last PBO mapping, do not call cudaFree on that.
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferAccum)) );
  }

  if (m_launchParameters.bufferPicking != 0)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferPicking)) );
  }

  // OpenGL cleanup:
  if (m_pbo != 0)
  {
    glDeleteBuffers(1, &m_pbo);
  }
  if (m_hdrTexture != 0)
  {
    glDeleteTextures(1, &m_hdrTexture);
  }
  if (m_vboAttributes != 0)
  {
    glDeleteBuffers(1, &m_vboAttributes);
  }
  if (m_vboIndices != 0)
  {
    glDeleteBuffers(1, &m_vboIndices);
  }

  // Host side allocations.
  if (m_picSheenLUT != nullptr)
  {
    delete m_picSheenLUT;
  }
  if (m_texSheenLUT != nullptr)
  {
    delete m_texSheenLUT;
  }
  if (m_picEnv != nullptr)
  {
    delete m_picEnv;
  }
  if (m_texEnv != nullptr)
  {
    delete m_texEnv;
  }
}


// A very simplified version of the original code which is not doing any batching to keep this simple.
// Also because some material parameter changes (face-culling and alpha mode) require AS rebuilds for individual meshes,
// this makes it much easier to handle that.
// In the future that will happen even more often when supporting skinning and animation.
void Application::buildMeshAccels()
{
  // Build input flags depending on the different material configuration assigned to the individual dev::Primitive.
  // Each alphaMode has a different anyhit program handling!
  // Each element 0 has face culling enabled, and element 1 has face culling disabled. 
  //
  // Note that face-culling isn't really compatible with global illumination algorithms! 
  // Materials which are fully transparent on one side and fully opaque on the other aren't physically plausible.
  // Neither reflections nor shadows will look as expected in some test scenes (like NegativeScaleTest.gltf) which 
  // are explicitly built for local lighting in rasterizers.

  // ALPHA_MODE_OPAQUE materials do not need to call into anyhit programs!
  const unsigned int inputFlagsOpaque[2] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
                                             OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING };
  // ALPHA_MODE_MASK materials are either fully opaque or fully transparent which is tested 
  // inside the anyhit program by comparing the opacity against the alphaCutoff value.
  const unsigned int inputFlagsMask[2]   = { OPTIX_GEOMETRY_FLAG_NONE,
                                             OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING };
  // ALPHA_MODE_BLEND materials are using a stochastic opacity threshold which must be evaluated only once per primitive.
  const unsigned int inputFlagsBlend[2]  = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL,
                                             OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING };

  // Same build options for all meshes.
  OptixAccelBuildOptions accelBuildOptions = {};

  accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  accelBuildOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

  // This builds one GAS per Mesh but with build input and SBT hit record per dev::Primitive (with Triangles mode)
  // to be able to use different input flags and material indices.
  for (dev::Mesh* mesh : m_meshes)
  {
    if (!mesh->isDirty) // If the mesh doesn't need to be rebuilt, continue.
    {
      MY_ASSERT(mesh->gas != 0 && mesh->d_gas != 0);
      continue; // Nothing to do for this mesh.
    }

    // If this routine is called more than once for a mesh, free the d_gas of this mesh and rebuild it.
    if (mesh->d_gas)
    {
      CUDA_CHECK( cudaFree(reinterpret_cast<void*>(mesh->d_gas)) );

      mesh->d_gas = 0;
      mesh->gas   = 0;
    }

    std::vector<OptixBuildInput> buildInputs;
    
    for (const dev::Primitive& prim : mesh->primitives)
    {
      OptixBuildInput buildInput = {};

      buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

      buildInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
      buildInput.triangleArray.vertexStrideInBytes = sizeof(float3); // DeviceBuffer data is always tightly packed.
      buildInput.triangleArray.numVertices         = prim.positions.count;
      buildInput.triangleArray.vertexBuffers       = &(prim.positions.d_ptr);

      if (prim.indices.count != 0) // Indexed triangle mesh.
      {
        buildInput.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
        buildInput.triangleArray.numIndexTriplets   = prim.indices.count / 3;
        buildInput.triangleArray.indexBuffer        = prim.indices.d_ptr;
      }
      else // Triangle soup.
      {
        // PERF This is redundant with the initialization above. All values are zero.
        buildInput.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_NONE;
        buildInput.triangleArray.indexStrideInBytes = 0;
        buildInput.triangleArray.numIndexTriplets   = 0;
        buildInput.triangleArray.indexBuffer        = 0;
      }

      buildInput.triangleArray.numSbtRecords = 1; // GLTF Material assignment is per dev::Primitive!

      const int32_t indexMaterial = prim.currentMaterial;

      if (0 <= indexMaterial)
      {
        // This index switches between geometry flags without (0) and with (1) face culling enabled.
        int32_t indexFlags = 0; // Enable face culling by default.

        // If the material is double-sided (== not face culled) or has volume attenuation, disable face culling. 
        // Volume attenuation only works correctly when the backfaces of a volume can be intersected.
        if ( m_materials[indexMaterial].doubleSided || 
            (m_materials[indexMaterial].flags & FLAG_KHR_MATERIALS_VOLUME) != 0) 
        {
          indexFlags = 1;
        }

        switch (m_materials[indexMaterial].alphaMode)
        {
          case MaterialData::ALPHA_MODE_OPAQUE:
          default:
            buildInput.triangleArray.flags = &inputFlagsOpaque[indexFlags];
            break;

          case MaterialData::ALPHA_MODE_MASK:
            buildInput.triangleArray.flags = &inputFlagsMask[indexFlags];
            break;

          case MaterialData::ALPHA_MODE_BLEND:
            buildInput.triangleArray.flags = &inputFlagsBlend[indexFlags];
            break;
        };
      }
      else
      {
        buildInput.triangleArray.flags = &inputFlagsOpaque[0]; // Default is single-sided opaque.
      }

      buildInputs.push_back(buildInput);
    } // for mesh->primitives
    
    if (!buildInputs.empty())
    {
      OptixAccelBufferSizes accelBufferSizes = {};

      OPTIX_CHECK( m_api.optixAccelComputeMemoryUsage(m_optixContext,
                                                      &accelBuildOptions,
                                                      buildInputs.data(),
                                                      static_cast<unsigned int>(buildInputs.size()),
                                                      &accelBufferSizes) );

      CUdeviceptr d_gas; // Must be aligned to OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT.

      CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&d_gas), accelBufferSizes.outputSizeInBytes) ); 

      CUdeviceptr d_temp; // Must be aligned to OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT.

      CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&d_temp), accelBufferSizes.tempSizeInBytes) );

      OptixAccelEmitDesc accelEmit = {};

      CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&accelEmit.result), 8) ); // Room for size_t for the compacted size.
      accelEmit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

      OPTIX_CHECK( m_api.optixAccelBuild(m_optixContext,
                                         m_cudaStream,
                                         &accelBuildOptions,
                                         buildInputs.data(),
                                         static_cast<unsigned int>(buildInputs.size()),
                                         d_temp, accelBufferSizes.tempSizeInBytes,
                                         d_gas, accelBufferSizes.outputSizeInBytes, 
                                         &mesh->gas,
                                         &accelEmit, // Emitted property: compacted size
                                         1) );       // Number of emitted properties.

      size_t sizeCompact;

      CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(&sizeCompact), (const void*) accelEmit.result, sizeof(size_t), cudaMemcpyDeviceToHost) );

      CUDA_CHECK( cudaFree(reinterpret_cast<void*>(accelEmit.result)) );
      CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_temp)) );

      // Compact the AS only when possible. This can save more than half the memory on RTX boards.
      if (sizeCompact < accelBufferSizes.outputSizeInBytes)
      {
        CUdeviceptr d_gasCompact;
      
        CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&d_gasCompact), accelBufferSizes.outputSizeInBytes) );

        OPTIX_CHECK( m_api.optixAccelCompact(m_optixContext, 0, mesh->gas, d_gasCompact, sizeCompact, &mesh->gas) );

        CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_gas)) );

        mesh->d_gas = d_gasCompact;
      }
      else
      {
        mesh->d_gas = d_gas;
      }
    }

    mesh->isDirty = false;
  } // for m_meshes
}


static void setInstanceTransform(OptixInstance& instance, const glm::mat4x4& matrix)
{
  // GLM matrix indexing is column major: [column][row].
  // Instance matrix 12 floats for 3x4 row major matrix.
  // Copy the first three rows from the glm:mat4x4:
  instance.transform[ 0] = matrix[0][0];
  instance.transform[ 1] = matrix[1][0];
  instance.transform[ 2] = matrix[2][0];
  instance.transform[ 3] = matrix[3][0];
  instance.transform[ 4] = matrix[0][1];
  instance.transform[ 5] = matrix[1][1];
  instance.transform[ 6] = matrix[2][1];
  instance.transform[ 7] = matrix[3][1];
  instance.transform[ 8] = matrix[0][2];
  instance.transform[ 9] = matrix[1][2];
  instance.transform[10] = matrix[2][2];
  instance.transform[11] = matrix[3][2];
}


void Application::buildInstanceAccel()
{
  // If there already exist an IAS, delete it.
  if (m_d_ias)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_d_ias)) );
    m_d_ias = 0;
    m_ias   = 0;
  }

  // Invalid scene AABB.
  m_sceneAABB[0] = glm::vec3(1e37f);
  m_sceneAABB[1] = glm::vec3(-1e37f);
  
  const size_t numInstances = m_instances.size();

  std::vector<OptixInstance> optix_instances(numInstances);

  unsigned int sbt_offset = 0;

  for (size_t i = 0; i < m_instances.size(); ++i)
  {
    dev::Instance* instance = m_instances[i];

    OptixInstance& optix_instance = optix_instances[i];
    memset(&optix_instance, 0, sizeof(OptixInstance));

    optix_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
    optix_instance.instanceId        = static_cast<unsigned int>(i);
    optix_instance.sbtOffset         = sbt_offset;
    optix_instance.visibilityMask    = 1;
    optix_instance.traversableHandle = m_meshes[instance->indexMesh]->gas;
    
    setInstanceTransform(optix_instance, instance->transform);
 
    sbt_offset += static_cast<unsigned int>(m_meshes[instance->indexMesh]->primitives.size()) * NUM_RAY_TYPES; // One sbt record per GAS build input per RAY_TYPE.
  }

  const size_t instances_size_in_bytes = sizeof(OptixInstance) * numInstances;

  CUdeviceptr d_instances;

  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&d_instances), instances_size_in_bytes) );
  CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(d_instances), optix_instances.data(), instances_size_in_bytes, cudaMemcpyHostToDevice) );

  OptixBuildInput buildInput = {};

  buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;

  buildInput.instanceArray.instances    = d_instances;
  buildInput.instanceArray.numInstances = static_cast<unsigned int>(numInstances);

  OptixAccelBuildOptions accelBuildOptions = {};

  accelBuildOptions.buildFlags = /* OPTIX_BUILD_FLAG_NONE | */ OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  accelBuildOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes accelBufferSizes;

  OPTIX_CHECK( m_api.optixAccelComputeMemoryUsage(m_optixContext, &accelBuildOptions, &buildInput, 1, &accelBufferSizes) );

  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_d_ias), accelBufferSizes.outputSizeInBytes) );

  CUdeviceptr d_temp; // Must be aligned to OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT.

  // Make sure tempSizeInBytes is a multiple of four to place the AABB float data behind it on its correct CUDA memory alignment.
  accelBufferSizes.tempSizeInBytes = (accelBufferSizes.tempSizeInBytes + 3ull) & ~3ull; 

  // Temporary AS buffer + (4 byte aligned) 6 floats for emitted AABB 
  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&d_temp), accelBufferSizes.tempSizeInBytes + 6 * sizeof(float)) ); 

  OptixAccelEmitDesc emitDesc = {};
  
  // Emit the top-level AABB to know the scene size.
  emitDesc.type   = OPTIX_PROPERTY_TYPE_AABBS;
  emitDesc.result = d_temp + accelBufferSizes.tempSizeInBytes;

  OPTIX_CHECK( m_api.optixAccelBuild(m_optixContext,
                                     m_cudaStream,
                                     &accelBuildOptions,
                                     &buildInput,
                                     1, // num build inputs
                                     d_temp,  accelBufferSizes.tempSizeInBytes,
                                     m_d_ias, accelBufferSizes.outputSizeInBytes,
                                     &m_ias,
                                     &emitDesc,
                                     1));

  // Copy the emitted top-level IAS AABB data to m_sceneAABB.
  CUDA_CHECK( cudaMemcpy(m_sceneAABB, reinterpret_cast<const void*>(emitDesc.result), 6 * sizeof(float), cudaMemcpyDeviceToHost) );

  CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_temp)) );
  CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_instances)) );

  // Calculate derived values from the scene AABB.
  m_sceneCenter = 0.5f * (m_sceneAABB[0] + m_sceneAABB[1]);

  const glm::vec3 extent = m_sceneAABB[1] - m_sceneAABB[0];
  m_sceneExtent = fmaxf(fmaxf(extent.x, extent.y), extent.z);
}


void Application::createPipeline()
{
  // Set all module and pipeline options.

  // OptixModuleCompileOptions
  m_mco = {};

  m_mco.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#if USE_DEBUG_EXCEPTIONS
  m_mco.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0; // No optimizations.
  m_mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;     // Full debug. Never profile kernels with this setting!
#else
  m_mco.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3; // All optimizations, is the default.
  // Keep generated line info. (NVCC_OPTIONS use --generate-line-info in CMakeLists.txt)
  m_mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL; // PERF Must use OPTIX_COMPILE_DEBUG_LEVEL_MODERATE to profile code with Nsight Compute!
#endif // USE_DEBUG_EXCEPTIONS

  // OptixPipelineCompileOptions
  m_pco = {};

  m_pco.usesMotionBlur        = 0;
  m_pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  m_pco.numPayloadValues      = 2; // Need only two register for the payload pointer.
  m_pco.numAttributeValues    = 2; // For the two barycentric coordinates of built-in triangles. (The required minimum value.)
#ifndef NDEBUG // USE_DEBUG_EXCEPTIONS
  m_pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
                         OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                         OPTIX_EXCEPTION_FLAG_USER;
#else
  m_pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
  m_pco.pipelineLaunchParamsVariableName = "theLaunchParameters";
  m_pco.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE; // This renderer only supports built-in triangles at this time.

  // OptixPipelineLinkOptions
  m_plo = {};

  m_plo.maxTraceDepth = 2;

  // OptixProgramGroupOptions
  m_pgo = {}; // This is a just placeholder.

  // Build the module path names.

  const std::string path("./GLTF_renderer_core/");

#if defined(USE_OPTIX_IR)
  const std::string extension(".optixir");
#else
  const std::string extension(".ptx");
#endif

  m_moduleFilenames.resize(NUM_MODULE_IDENTIFIERS);

  // Starting with OptiX SDK 7.5.0 and CUDA 11.7 either PTX or OptiX IR input can be used to create modules.
  // Just initialize the m_moduleFilenames depending on the definition of USE_OPTIX_IR.
  // That is added to the project definitions inside the CMake script when OptiX SDK 7.5.0 and CUDA 11.7 or newer are found.
  m_moduleFilenames[MODULE_ID_RAYGENERATION]  = path + std::string("raygen") + extension;
  m_moduleFilenames[MODULE_ID_EXCEPTION]      = path + std::string("exception") + extension;
  m_moduleFilenames[MODULE_ID_MISS]           = path + std::string("miss") + extension;
  m_moduleFilenames[MODULE_ID_HIT]            = path + std::string("hit") + extension;
  m_moduleFilenames[MODULE_ID_LIGHT_SAMPLE]   = path + std::string("light_sample") + extension; // Direct callable programs.

  // Create all modules.

  MY_ASSERT(NUM_RAY_TYPES == 2); // The following code only works for two raytypes.

  m_modules.resize(NUM_MODULE_IDENTIFIERS);

  for (size_t i = 0; i < m_moduleFilenames.size(); ++i)
  {
    std::vector<char> programData = readData(m_moduleFilenames[i]);

    OPTIX_CHECK( m_api.optixModuleCreate(m_optixContext, &m_mco, &m_pco, programData.data(), programData.size(), nullptr, nullptr, &m_modules[i]) );
  }

  // Create the program groups descriptions.

  std::vector<OptixProgramGroupDesc> programGroupDescriptions(NUM_PROGRAM_GROUP_IDS);
  memset(programGroupDescriptions.data(), 0, sizeof(OptixProgramGroupDesc) * programGroupDescriptions.size());

  OptixProgramGroupDesc* pgd;

  if (m_interop != INTEROP_IMG)
  {
    pgd = &programGroupDescriptions[PGID_RAYGENERATION];
    pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->raygen.module = m_modules[MODULE_ID_RAYGENERATION];
    pgd->raygen.entryFunctionName = "__raygen__path_tracer";
  }
  else
  {
    pgd = &programGroupDescriptions[PGID_RAYGENERATION];
    pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->raygen.module = m_modules[MODULE_ID_RAYGENERATION];
    pgd->raygen.entryFunctionName = "__raygen__path_tracer_surface";
  }


  pgd = &programGroupDescriptions[PGID_EXCEPTION];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->exception.module            = m_modules[MODULE_ID_EXCEPTION];
  pgd->exception.entryFunctionName = "__exception__all";

  pgd = &programGroupDescriptions[PGID_MISS_RADIANCE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->miss.module = m_modules[MODULE_ID_MISS];
  switch (m_missID)
  {
    case 0:
    default: // Every other ID means there is no environment light.
      pgd->miss.entryFunctionName = "__miss__env_null";
      break;
    case 1:
      pgd->miss.entryFunctionName = "__miss__env_constant";
      break;
    case 2:
      pgd->miss.entryFunctionName = "__miss__env_sphere";
      break;
  }

  pgd = &programGroupDescriptions[PGID_MISS_SHADOW];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->miss.module            = m_modules[MODULE_ID_MISS];
  pgd->miss.entryFunctionName = "__miss__shadow"; // alphaMode OPAQUE is not using anyhit or closest hit programs for the shadow ray.

  // The hit records for the radiance ray.
  pgd = &programGroupDescriptions[PGID_HIT_RADIANCE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleCH            = m_modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pgd->hitgroup.moduleAH            = m_modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__radiance";

  // The hit records for the shadow ray.
  pgd = &programGroupDescriptions[PGID_HIT_SHADOW];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleAH            = m_modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__shadow";

  // Light Sampler
  // Only one of the environment callables will ever be used, but both are required
  // for the proper direct callable index calculation for BXDFs using NUM_LIGHT_TYPES.
  pgd = &programGroupDescriptions[PGID_LIGHT_ENV_CONSTANT];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = m_modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_env_constant";

  pgd = &programGroupDescriptions[PGID_LIGHT_ENV_SPHERE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = m_modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_env_sphere";

  pgd = &programGroupDescriptions[PGID_LIGHT_POINT];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = m_modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_point";

  pgd = &programGroupDescriptions[PGID_LIGHT_SPOT];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = m_modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_spot";

  pgd = &programGroupDescriptions[PGID_LIGHT_DIRECTIONAL];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = m_modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_directional";

  // Create the program groups.

  m_programGroups.resize(programGroupDescriptions.size());
  
  OPTIX_CHECK( m_api.optixProgramGroupCreate(m_optixContext, programGroupDescriptions.data(), (unsigned int) programGroupDescriptions.size(), &m_pgo, nullptr, nullptr, m_programGroups.data()) );
  
  // 3.) Create the pipeline.

  OPTIX_CHECK( m_api.optixPipelineCreate(m_optixContext, &m_pco, &m_plo, m_programGroups.data(), (unsigned int) m_programGroups.size(), nullptr, nullptr, &m_pipeline) );


  // 4.) Calculate the stack size. 
  // This is is always recommended and strictly required when using any direct or continuation callables.
  OptixStackSizes ssp = {}; // Whole pipeline.

  for (OptixProgramGroup pg: m_programGroups)
  {
    OptixStackSizes ss;

#if (OPTIX_VERSION >= 70700)
    OPTIX_CHECK( m_api.optixProgramGroupGetStackSize(pg, &ss, m_pipeline) );
#else
    OPTIX_CHECK( m_api.optixProgramGroupGetStackSize(pg, &ss) );
#endif

    ssp.cssRG = std::max(ssp.cssRG, ss.cssRG);
    ssp.cssMS = std::max(ssp.cssMS, ss.cssMS);
    ssp.cssCH = std::max(ssp.cssCH, ss.cssCH);
    ssp.cssAH = std::max(ssp.cssAH, ss.cssAH);
    ssp.cssIS = std::max(ssp.cssIS, ss.cssIS);
    ssp.cssCC = std::max(ssp.cssCC, ss.cssCC);
    ssp.dssDC = std::max(ssp.dssDC, ss.dssDC);
  }
  
  // Temporaries
  unsigned int cssCCTree           = ssp.cssCC; // Should be 0. No continuation callables in this pipeline. // maxCCDepth == 0
  unsigned int cssCHOrMSPlusCCTree = std::max(ssp.cssCH, ssp.cssMS) + cssCCTree;

  // Arguments
  unsigned int directCallableStackSizeFromTraversal = ssp.dssDC; // maxDCDepth == 1 // FromTraversal: DC is invoked from IS or AH.      // Possible stack size optimizations.
  unsigned int directCallableStackSizeFromState     = ssp.dssDC; // maxDCDepth == 1 // FromState:     DC is invoked from RG, MS, or CH. // Possible stack size optimizations.
  unsigned int continuationStackSize = ssp.cssRG + cssCCTree + cssCHOrMSPlusCCTree * (std::max(1u, m_plo.maxTraceDepth) - 1u) +
                                       std::min(1u, m_plo.maxTraceDepth) * std::max(cssCHOrMSPlusCCTree, ssp.cssAH + ssp.cssIS);
  unsigned int maxTraversableGraphDepth = 2;

  OPTIX_CHECK( m_api.optixPipelineSetStackSize(m_pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState, continuationStackSize, maxTraversableGraphDepth) );
}

#if 0 // FIXME Currently unused.
static unsigned int getAttributeFlags(const dev::Primitive& prim)
{
  // The below code is using hardcocded array indices.
  MY_ASSERT(NUM_ATTR_TEXCOORDS == 2 && 
            NUM_ATTR_JOINTS    == 2 && 
            NUM_ATTR_WEIGHTS   == 2);

  unsigned int flags = 0;
  
  flags |= (prim.indices.d_ptr)      ? ATTR_INDEX      : 0;
  flags |= (prim.colors.d_ptr)       ? ATTR_COLOR_0    : 0;
  flags |= (prim.tangents.d_ptr)     ? ATTR_TANGENT    : 0;
  flags |= (prim.normals.d_ptr)      ? ATTR_NORMAL     : 0;
  flags |= (prim.texcoords[0].d_ptr) ? ATTR_TEXCOORD_0 : 0;
  flags |= (prim.texcoords[1].d_ptr) ? ATTR_TEXCOORD_1 : 0;
  flags |= (prim.joints[0].d_ptr)    ? ATTR_JOINTS_0   : 0;
  flags |= (prim.joints[1].d_ptr)    ? ATTR_JOINTS_1   : 0;
  flags |= (prim.weights[0].d_ptr)   ? ATTR_WEIGHTS_0  : 0;
  flags |= (prim.weights[1].d_ptr)   ? ATTR_WEIGHTS_1  : 0;
  
  return flags;
}
#endif

void Application::createSBT()
{
  {
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_sbt.raygenRecord), sizeof(dev::EmptyRecord)) );

    dev::EmptyRecord rg_sbt;

    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_RAYGENERATION], &rg_sbt) );

    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_sbt.raygenRecord), &rg_sbt, sizeof(dev::EmptyRecord), cudaMemcpyHostToDevice) );
  }

  {
    const size_t miss_record_size = sizeof(dev::EmptyRecord);

    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_sbt.missRecordBase), miss_record_size * NUM_RAY_TYPES) );

    dev::EmptyRecord ms_sbt[NUM_RAY_TYPES];

    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_MISS_RADIANCE], &ms_sbt[0]) );
    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_MISS_SHADOW],   &ms_sbt[1]) );

    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_sbt.missRecordBase), ms_sbt, miss_record_size * NUM_RAY_TYPES, cudaMemcpyHostToDevice) );
    
    m_sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    m_sbt.missRecordCount = NUM_RAY_TYPES;
  }

  {
    std::vector<dev::HitGroupRecord> hitGroupRecords;

    for (const dev::Instance* instance : m_instances)
    {
      const dev::Mesh* mesh = m_meshes[instance->indexMesh];

      for (size_t i = 0; i < mesh->primitives.size(); ++i)
      {
        const dev::Primitive& prim = mesh->primitives[i];

        dev::HitGroupRecord rec = {};

        OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_RADIANCE], &rec) );
        
        GeometryData::TriangleMesh triangleMesh = {}; 
        
        // Indices
        triangleMesh.indices = reinterpret_cast<uint3*>(prim.indices.d_ptr);
        // Attributes
        triangleMesh.positions = reinterpret_cast<float3*>(prim.positions.d_ptr);
        triangleMesh.normals   = reinterpret_cast<float3*>(prim.normals.d_ptr);
        for (size_t j = 0; j < NUM_ATTR_TEXCOORDS; ++j)
        {
          triangleMesh.texcoords[j] = reinterpret_cast<float2*>(prim.texcoords[j].d_ptr);
        }
        triangleMesh.colors   = reinterpret_cast<float4*>(prim.colors.d_ptr);
        triangleMesh.tangents = reinterpret_cast<float4*>(prim.tangents.d_ptr);
        for (size_t j = 0; j < NUM_ATTR_JOINTS; ++j)
        {
          triangleMesh.joints[j] = reinterpret_cast<ushort4*>(prim.joints[j].d_ptr);
        }
        for (size_t j = 0; j < NUM_ATTR_WEIGHTS; ++j)
        {
          triangleMesh.weights[j] = reinterpret_cast<float4*>(prim.weights[j].d_ptr);
        }
        //triangleMesh.flagAttributes = getAttributeFlags(prim); // FIXME Currently unused.
        
        rec.data.geometryData.setTriangleMesh(triangleMesh);

        if (0 <= prim.currentMaterial)
        {
          rec.data.materialData = m_materials[prim.currentMaterial];
        }
        else
        {
          rec.data.materialData = MaterialData(); // These default materials cannot be edited!
        }
        
        hitGroupRecords.push_back(rec);

        OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_SHADOW], &rec) );

        hitGroupRecords.push_back(rec);
      }
    }

    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_sbt.hitgroupRecordBase), hitGroupRecords.size() * sizeof(dev::HitGroupRecord)) );
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase), hitGroupRecords.data(), hitGroupRecords.size() * sizeof(dev::HitGroupRecord), cudaMemcpyHostToDevice) );

    m_sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(dev::HitGroupRecord));
    m_sbt.hitgroupRecordCount         = static_cast<unsigned int>(hitGroupRecords.size());
  }

  {
    const size_t call_record_size = sizeof(dev::EmptyRecord);

    const int numCallables = NUM_PROGRAM_GROUP_IDS - PGID_LIGHT_ENV_CONSTANT;
    MY_ASSERT(numCallables == 5);

    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_sbt.callablesRecordBase), call_record_size * numCallables) );

    dev::EmptyRecord call_sbt[numCallables];

    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_LIGHT_ENV_CONSTANT], &call_sbt[0]) );
    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_LIGHT_ENV_SPHERE],   &call_sbt[1]) );
    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_LIGHT_POINT],        &call_sbt[2]) );
    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_LIGHT_SPOT],         &call_sbt[3]) );
    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_LIGHT_DIRECTIONAL],  &call_sbt[4]) );

    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_sbt.callablesRecordBase), call_sbt, call_record_size * numCallables, cudaMemcpyHostToDevice) );
    
    m_sbt.callablesRecordStrideInBytes = static_cast<uint32_t>(call_record_size);
    m_sbt.callablesRecordCount         = numCallables;
  }
}


void Application::updateSBT()
{
  if (m_sbt.hitgroupRecordBase)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)) );
    m_sbt.hitgroupRecordBase = 0;
  }

  {
    std::vector<dev::HitGroupRecord> hitGroupRecords;

    for (const dev::Instance* instance : m_instances)
    {
      const dev::Mesh* mesh = m_meshes[instance->indexMesh];

      for (size_t i = 0; i < mesh->primitives.size(); ++i)
      {
        const dev::Primitive& prim = mesh->primitives[i];

        dev::HitGroupRecord rec = {};

        OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_RADIANCE], &rec) );
        
        GeometryData::TriangleMesh triangleMesh = {}; 
        
        // Indices
        triangleMesh.indices = reinterpret_cast<uint3*>(prim.indices.d_ptr);
        // Attributes
        triangleMesh.positions = reinterpret_cast<float3*>(prim.positions.d_ptr);
        triangleMesh.normals   = reinterpret_cast<float3*>(prim.normals.d_ptr);
        for (size_t j = 0; j < NUM_ATTR_TEXCOORDS; ++j)
        {
          triangleMesh.texcoords[j] = reinterpret_cast<float2*>(prim.texcoords[j].d_ptr);
        }
        triangleMesh.colors   = reinterpret_cast<float4*>(prim.colors.d_ptr);
        triangleMesh.tangents = reinterpret_cast<float4*>(prim.tangents.d_ptr);
        for (size_t j = 0; j < NUM_ATTR_JOINTS; ++j)
        {
          triangleMesh.joints[j] = reinterpret_cast<ushort4*>(prim.joints[j].d_ptr);
        }
        for (size_t j = 0; j < NUM_ATTR_WEIGHTS; ++j)
        {
          triangleMesh.weights[j] = reinterpret_cast<float4*>(prim.weights[j].d_ptr);
        }
        //triangleMesh.flagAttributes = getAttributeFlags(prim); // FIXME Currently unused.
        
        rec.data.geometryData.setTriangleMesh(triangleMesh);

        if (0 <= prim.currentMaterial)
        {
          rec.data.materialData = m_materials[prim.currentMaterial];
        }
        else
        {
          rec.data.materialData = MaterialData(); // These default materials cannot be edited!
        }
        
        hitGroupRecords.push_back(rec);

        OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_SHADOW], &rec) );

        hitGroupRecords.push_back(rec);
      }
    }
    
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_sbt.hitgroupRecordBase), hitGroupRecords.size() * sizeof(dev::HitGroupRecord)) );
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase), hitGroupRecords.data(), hitGroupRecords.size() * sizeof(dev::HitGroupRecord), cudaMemcpyHostToDevice) );

    m_sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(dev::HitGroupRecord));
    m_sbt.hitgroupRecordCount         = static_cast<unsigned int>(hitGroupRecords.size());
  }
}


void Application::updateMaterial(const int index, const bool rebuild)
{
  MY_ASSERT(m_sbt.hitgroupRecordBase != 0);
  
  dev::HitGroupRecord* rec = reinterpret_cast<dev::HitGroupRecord*>(m_sbt.hitgroupRecordBase);

  for (const dev::Instance* instance : m_instances)
  {
    dev::Mesh* mesh = m_meshes[instance->indexMesh];

    for (size_t i = 0; i < mesh->primitives.size(); ++i)
    {
      const dev::Primitive& prim = mesh->primitives[i];

      if (index == prim.currentMaterial)
      {
        if (!rebuild) // Only update the SBT hhit record material data in place when not rebuilding everything anyway.
        {
          // Update the radiance ray hit record material data.
          CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(&rec->data.materialData), &m_materials[index], sizeof(MaterialData), cudaMemcpyHostToDevice) );
          ++rec;

          // Update the shadow ray hit record material data.
          CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(&rec->data.materialData), &m_materials[index], sizeof(MaterialData), cudaMemcpyHostToDevice) );
          ++rec;
        }
        else
        {
          rec += 2; // Skip two HitGroupRecords.
        }

        m_launchParameters.iteration = 0u; // Restart accumulation when any material in the currently active scene changed.   

        mesh->isDirty = rebuild; // Flag mesh GAS which need to be rebuild.
      }
      else
      {
        rec += 2; // Skip two HitGroupRecords which are not using the m_material[index].
      }
    }
  }

  // When doubleSided or alphaMode changed in a way which requires to rebuild any mesh,
  // update the respective GAS 
  if (rebuild)
  {
    buildMeshAccels();        // This rebuilds only the meshes with isDirty flags.
    buildInstanceAccel();     // This rebuilds the top-level IAS with the new mesh GAS.
    updateSBT();              // This rebuilds the SBT with all material records. Means the above copies aren't required.
    updateLaunchParameters(); // This sets the new root m_ias.
  }
}


// Only need to update the SBT hit records with the new material data.
void Application::updateSBTMaterialData()
{
  MaterialData defaultMaterialData = {}; // Default material data in case any primitive has no material assigned.

  MY_ASSERT(m_sbt.hitgroupRecordBase != 0);
  
  dev::HitGroupRecord* rec = reinterpret_cast<dev::HitGroupRecord*>(m_sbt.hitgroupRecordBase);

  for (const dev::Instance* instance : m_instances)
  {
    dev::Mesh* mesh = m_meshes[instance->indexMesh];

    for (const dev::Primitive& prim : mesh->primitives)
    {
      const MaterialData* src = (0 <= prim.currentMaterial) ? &m_materials[prim.currentMaterial] : &defaultMaterialData;

      // Update the radiance ray hit record material data.
      CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(&rec->data.materialData), src, sizeof(MaterialData), cudaMemcpyHostToDevice) );
      ++rec;
      // Update the shadow ray hit record material data.
      CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(&rec->data.materialData), src, sizeof(MaterialData), cudaMemcpyHostToDevice) );
      ++rec;
    }
  }
}


void Application::updateVariant()
{
  bool changed   = false;
  bool rebuildAS = false;

  for (dev::Mesh* mesh : m_meshes)
  {
    for (dev::Primitive& prim : mesh->primitives)
    {
      // Variants can only change on this primitive if there are material index mappings available 
      if (!prim.mappings.empty())
      {
        const int32_t indexMaterial = prim.mappings[m_indexVariant]; // m_indexVariant contains the new variant.

        if (indexMaterial != prim.currentMaterial) // If switching the variant changes the material index on this primitive.
        {
          changed = true; // Indicate that at least 

          // Check if the material switch requires a rebuild of the AS.
          const MaterialData& cur = m_materials[prim.currentMaterial];
          const MaterialData& var = m_materials[indexMaterial];

          // The face culling state is affected by both the doubleSided and the volume state.
          // The only case when face culling is enabled is when the material is not doubleSided and not using the volume extension.
          const bool curCull = (!cur.doubleSided && (cur.flags & FLAG_KHR_MATERIALS_VOLUME) == 0);
          const bool varCull = (!var.doubleSided && (var.flags & FLAG_KHR_MATERIALS_VOLUME) == 0);
        
          // If the alphaMode changes, the anyhit program invocation for primitives changes.
          const bool rebuild = (curCull != varCull) || (cur.alphaMode != var.alphaMode);

          // Now switch the primitive to the new material index.
          prim.currentMaterial = indexMaterial;

          if (rebuild)
          {
            mesh->isDirty = true;
            rebuildAS     = true;
          }
        }
      }
    }
  }
  
  // While the above code will change the isDirty flags for all valid meshes inside the asset, 
  // the current scene is potentially only using a subset of these.
  // All meshes reached by the currently active m_instances will be rebuilt here.
  // The others are rebuilt automatically when switching scenes.
  if (rebuildAS)
  {
    buildMeshAccels();        // This rebuilds only the meshes with isDirty flags.
    buildInstanceAccel();     // This rebuilds the top-level IAS with the new mesh GAS.
    updateSBT();              // This rebuilds the SBT with all material records. Means the above copies aren't required.
    updateLaunchParameters(); // This sets the new root m_ias.
  }
  else if (changed) // No rebuild required, just update all SBT hit records with the new MaterialData.
  {
    updateSBTMaterialData();
    
    m_launchParameters.iteration = 0u; // Restart accumulation when any material in the currently active scene changed.   
  }
}


void Application::initTrackball()
{
  if (m_isDefaultCamera)
  {
    dev::Camera* camera = m_cameras[0];

    camera->setPosition(m_sceneCenter + glm::vec3(0.0f, 0.0f, 1.75f * m_sceneExtent));
    camera->setLookat(m_sceneCenter);
  }

  m_isDirtyCamera = true;

  // The trackball does nothing when there is no camera assigned to it.
  m_trackball.setCamera(m_cameras[m_indexCamera]);
  
  //m_trackball.setMoveSpeed(10.0f);
  
  // This is required to initialize the current longitude and latitude values.
  m_trackball.setReferenceFrame(glm::vec3(1.0f, 0.0f, 0.0f),
                                glm::vec3(0.0f, 1.0f, 0.0f),
                                glm::vec3(0.0f, 0.0f, 1.0f));
  
  m_trackball.setGimbalLock(true); // This helps keeping models upright when orbiting the trackball.
}


LightDefinition Application::createConstantEnvironmentLight() const
{
  LightDefinition light = {}; // All unused fields are set to zero.

  light.typeLight = TYPE_LIGHT_ENV_CONST; 

  light.matrix[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrix[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrix[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
  
  light.matrixInv[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrixInv[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrixInv[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);

  light.emission = make_float3(1.0f); // White
 
  light.area        = 4.0f * M_PIf;
  light.invIntegral = 1.0f / light.area;

  return light;
}


LightDefinition Application::createSphericalEnvironmentLight()
{
  if (m_picEnv == nullptr)
  {
    m_picEnv = new Picture();
  }

  bool loadedEnv = false;
  if (!m_pathEnv.empty())
  {
    loadedEnv = m_picEnv->load(m_pathEnv, IMAGE_FLAG_2D);
  }
  if (!loadedEnv)
  {
    //m_picEnv->generateEnvironment(32, 16); // Dummy white environment.
    m_picEnv->generateEnvironmentSynthetic(512, 256); // Generated HDR environment with some light regions.
  }
  
  // Create a new texture to keep the old texture intact in case anything goes wrong.
  Texture *texture = new Texture(m_allocator);

  if (!texture->create(m_picEnv, IMAGE_FLAG_2D | IMAGE_FLAG_ENV))
  {
    delete texture;
    throw std::exception("createSphericalEnvironmentLight() environment map creation failed");
  }

  if (m_texEnv != nullptr)
  {
    delete m_texEnv;
    //m_texEnv = nullptr;
  }

  m_texEnv = texture;

  LightDefinition light = {}; // All unused fields are set to zero.

  light.matrix[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrix[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrix[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
  
  light.matrixInv[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrixInv[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrixInv[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);

  // Textured environment
  light.cdfU = m_texEnv->getCDF_U(); 
  light.cdfV = m_texEnv->getCDF_V();

  // Emisson texture. If not zero, scales emission.
  light.textureEmission = m_texEnv->getTextureObject();

  light.emission = make_float3(1.0f); // Modulates the texture.

  light.typeLight = TYPE_LIGHT_ENV_SPHERE; 
  
  light.area        = 4.0f * M_PIf; // Unused.
  light.invIntegral = 1.0f / m_texEnv->getIntegral();

  // Emission texture width and height. Used to index the CDFs, see above.
  // For mesh lights the width matches the number of triangles and the cdfU is over the triangle areas.
  light.width  = m_texEnv->getWidth(); 
  light.height = m_texEnv->getHeight();

  return light;
}


LightDefinition Application::createPointLight() const
{
  LightDefinition light = {};

  light.matrix[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrix[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrix[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
  
  light.matrixInv[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrixInv[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrixInv[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);

  // Set the point light emission depending on the scene size to cancel out the quadratic attenuation.
  light.emission = make_float3(1.0f); // * sqrtf(maxExtent) * 2.0f;

  light.typeLight = TYPE_LIGHT_POINT;
  
  light.area        = 1.0f; // Unused.
  light.invIntegral = 1.0f; // Unused.

  return light;
}

void Application::initLaunchParameters()
{
  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_d_launchParameters), sizeof(LaunchParameters)) );

  m_launchParameters.handle = m_ias; // Root traversable handle of the scene.
  
  // Output buffer for the rendered image (HDR linear color).
  // This is initialized inside updateBuffers() depending on the m_interop state.
  m_launchParameters.bufferAccum = nullptr;
  
  // Output buffer for the picked material index.
  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.bufferPicking), sizeof(int)) );

  m_launchParameters.resolution       = make_int2(m_width, m_height);
  m_launchParameters.picking          = make_float2(-1.0f); // No picking ray.
  m_launchParameters.pathLengths      = make_int2(2, 6);
  m_launchParameters.iteration        = 0u;              // Sub-frame number for the progressive accumulation of results.
  m_launchParameters.sceneEpsilon     = m_epsilonFactor * SCENE_EPSILON_SCALE;
  m_launchParameters.directLighting   = (m_useDirectLighting)   ? 1 : 0;
  m_launchParameters.ambientOcclusion = (m_useAmbientOcclusion) ? 1 : 0;
  m_launchParameters.showEnvironment  = (m_showEnvironment)     ? 1 : 0;
  m_launchParameters.textureSheenLUT  = (m_texSheenLUT != nullptr) ? m_texSheenLUT->getTextureObject() : 0;

  m_launchParameters.numLights = static_cast<int>(m_lights.size()) + ((m_missID == 0) ? 0 : 1);

  m_lightDefinitions.resize(m_launchParameters.numLights);

  switch (m_missID)
  {
  case 0: // No environment light. This only makes sense for scenes with emissive materials or KHR_lights_punctual.
    break;

  case 1: // Constant white environment light.
    m_lightDefinitions[0] = createConstantEnvironmentLight();
    break;

  case 2: // Sperical HDR environment light.
    m_lightDefinitions[0] = createSphericalEnvironmentLight();
    break;
  }

  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.lightDefinitions), m_lightDefinitions.size() * sizeof(LightDefinition)) );

  // All dirty flags are set here and the first render() call will take care to allocate and update all necessary resources.
}


void Application::updateLaunchParameters()
{
  // This is called after acceleration structures have been rebuilt.
  m_launchParameters.handle    = m_ias; // Update the top-level IAS handle.
  m_launchParameters.iteration = 0u;    // Restart accumulation.
}


void Application::updateLights()
{
  // When there exists an environment light, skip it and start the indexing of m_lightDefinitions at 1.
  int indexDefinition = (m_missID == 0) ? 0 : 1; 

  for (const dev::Light* light : m_lights)
  {
    LightDefinition& lightDefinition = m_lightDefinitions[indexDefinition];

    lightDefinition.emission = light->color * light->intensity;

    switch (light->type)
    {
      case 0: // Point
        lightDefinition.typeLight = TYPE_LIGHT_POINT;
        lightDefinition.range     = light->range;
        break;

      case 1: // Spot
        {
          lightDefinition.typeLight  = TYPE_LIGHT_SPOT;
          lightDefinition.range      = light->range;
          MY_ASSERT(light->innerConeAngle < light->outerConeAngle); // GLTF spec says these must not be equal.
          lightDefinition.cosInner   = cosf(light->innerConeAngle); // Minimum 0.0f, maximum < outerConeAngle.
          lightDefinition.cosOuter   = cosf(light->outerConeAngle); // Maximum M_PIf / 2.0f which is 90 degrees so this is the half-angle.
        }
        break;

      case 2: // Directional
        {
          lightDefinition.typeLight = TYPE_LIGHT_DIRECTIONAL;
          // Directional lights need to know the world size as area to be able to convert lux (lm/m^2).
          const float radius = m_sceneExtent * 0.5f;
          MY_ASSERT(DENOMINATOR_EPSILON < radius); 
          lightDefinition.area = radius * radius * M_PIf;
        }
        break;
    }

    glm::mat4 matInv = glm::inverse(light->matrix);
    for (int i = 0; i < 3; ++i)
    {
      glm::vec4 row = glm::row(light->matrix, i);
      m_lightDefinitions[indexDefinition].matrix[i] = make_float4(row.x, row.y, row.z, row.w);
      row = glm::row(matInv, i);
      m_lightDefinitions[indexDefinition].matrixInv[i] = make_float4(row.x, row.y, row.z, row.w);
    }

    ++indexDefinition;
  }

  // Update all light definition device data. This requires that initLaunchParameters() ran before.
  CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_launchParameters.lightDefinitions), m_lightDefinitions.data(), m_lightDefinitions.size() * sizeof(LightDefinition), cudaMemcpyHostToDevice) );

  m_launchParameters.iteration = 0; // Restart accumulation.
}


void Application::updateCamera()
{
  // Update host side copy of the launch parameters.
  dev::Camera* camera = m_cameras[m_indexCamera];

  if (m_trackball.getCamera() != camera)
  {
    m_trackball.setCamera(camera);

    // This is required to initialize the current longitude and latitude values.
    m_trackball.setReferenceFrame(glm::vec3(1.0f, 0.0f, 0.0f),
                                  glm::vec3(0.0f, 1.0f, 0.0f),
                                  glm::vec3(0.0f, 0.0f, 1.0f));

    // This helps keeping models upright when orbiting the trackball.
    m_trackball.setGimbalLock(true);
  }
    
  // This means the pPerspective->aspectRatio value doesn't matter at all.
  camera->setAspectRatio(static_cast<float>(m_width) / static_cast<float>(m_height));
    
  m_launchParameters.cameraType = (0.0f < camera->getFovY()) ? 1 : 0; // 0 == orthographic, 1 == perspective.

  glm::vec3 P = camera->getPosition();
    
  m_launchParameters.cameraP = make_float3(P.x, P.y, P.z);

  glm::vec3 U;
  glm::vec3 V;
  glm::vec3 W;
    
  camera->getUVW(U, V, W);

  // Convert to CUDA float3 vector types.
  m_launchParameters.cameraU = make_float3(U.x, U.y, U.z);
  m_launchParameters.cameraV = make_float3(V.x, V.y, V.z);
  m_launchParameters.cameraW = make_float3(W.x, W.y, W.z);

  m_launchParameters.iteration = 0; // Restart accumulation.
}


void Application::update()
{
  MY_ASSERT(!m_isDirtyResize && m_hdrTexture != 0);
  
  switch (m_interop)
  {
    case INTEROP_OFF:
      // Copy the GPU local render buffer into host and update the HDR texture image from there.
      MY_ASSERT(m_bufferHost != nullptr);
      CUDA_CHECK( cudaMemcpy((void*) m_bufferHost, m_launchParameters.bufferAccum, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToHost) );
      // Copy the host buffer to the OpenGL texture image (slowest path, most portable).
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei) m_width, (GLsizei) m_height, GL_RGBA, GL_FLOAT, m_bufferHost); // RGBA32F
      break;

    case INTEROP_PBO: 
      // The image was rendered into the linear PBO buffer directly. Just upload to the block-linear texture.
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei) m_width, (GLsizei) m_height, GL_RGBA, GL_FLOAT, (GLvoid*) 0); // RGBA32F from byte offset 0 in the pixel unpack buffer.
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      break;

  case INTEROP_TEX:
      {
        CUarray dstArray = nullptr;

        // Map the Texture object directly and copy the output buffer. 
        CU_CHECK( cuGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream )); // This is an implicit cuSynchronizeStream().
        CU_CHECK( cuGraphicsSubResourceGetMappedArray(&dstArray, m_cudaGraphicsResource, 0, 0) ); // arrayIndex = 0, mipLevel = 0

        CUDA_MEMCPY3D params = {};

        params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        params.srcDevice     = reinterpret_cast<CUdeviceptr>(m_launchParameters.bufferAccum);
        params.srcPitch      = m_launchParameters.resolution.x * sizeof(float4); // RGBA32F
        params.srcHeight     = m_launchParameters.resolution.y;

        params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        params.dstArray      = dstArray;
        params.WidthInBytes  = m_launchParameters.resolution.x * sizeof(float4);
        params.Height        = m_launchParameters.resolution.y;
        params.Depth         = 1;

        CU_CHECK( cuMemcpy3D(&params) ); // Copy from linear to array layout.

        CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
      }
      break;

  case INTEROP_IMG:
    // Nothing to do. Renders into the m_hdrTexture surface object directly.
    break;
  }
}


void Application::updateBufferHost()
{
  update();
  
  // After the update() call, the m_hdrTexture contains the linear HDR image.
  // When interop is off, the m_bufferHost already contains the current linear HDR image data as well.
  if (m_interop != INTEROP_OFF)
  {
    // Read the m_hdrTexture image into the m_bufferHost.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, (GLvoid*) m_bufferHost);
  }
}


std::string Application::getDateTime()
{
#if defined(_WIN32)
  SYSTEMTIME time;
  GetLocalTime(&time);
#elif defined(__linux__)
  time_t rawtime;
  struct tm* ts;
  time(&rawtime);
  ts = localtime(&rawtime);
#else
  #error "OS not supported."
#endif

  std::ostringstream oss;

#if defined( _WIN32 )
  oss << time.wYear;
  if (time.wMonth < 10)
  {
    oss << '0';
  }
  oss << time.wMonth;
  if (time.wDay < 10)
  {
    oss << '0';
  }
  oss << time.wDay << '_';
  if (time.wHour < 10)
  {
    oss << '0';
  }
  oss << time.wHour;
  if (time.wMinute < 10)
  {
    oss << '0';
  }
  oss << time.wMinute;
  if (time.wSecond < 10)
  {
    oss << '0';
  }
  oss << time.wSecond << '_';
  if (time.wMilliseconds < 100)
  {
    oss << '0';
  }
  if (time.wMilliseconds <  10)
  {
    oss << '0';
  }
  oss << time.wMilliseconds; 
#elif defined(__linux__)
  oss << ts->tm_year;
  if (ts->tm_mon < 10)
  {
    oss << '0';
  }
  oss << ts->tm_mon;
  if (ts->tm_mday < 10)
  {
    oss << '0';
  }
  oss << ts->tm_mday << '_';
  if (ts->tm_hour < 10)
  {
    oss << '0';
  }
  oss << ts->tm_hour;
  if (ts->tm_min < 10)
  {
    oss << '0';
  }
  oss << ts->tm_min;
  if (ts->tm_sec < 10)
  {
    oss << '0';
  }
  oss << ts->tm_sec << '_';
  oss << "000"; // No milliseconds available.
#else
  #error "OS not supported."
#endif

  return oss.str();
}


bool Application::screenshot(const bool tonemap)
{
  updateBufferHost(); // Make sure m_bufferHost contains the linear HDR image data.

  ILboolean hasImage = false;

  std::ostringstream path;
   
  path << "img_gltf_" << getDateTime();
  
  unsigned int imageID;

  ilGenImages(1, (ILuint *) &imageID);

  ilBindImage(imageID);
  ilActiveImage(0);
  ilActiveFace(0);

  ilDisable(IL_ORIGIN_SET);

  if (tonemap)
  {
    path << ".png"; // Store a tonemapped RGB8 *.png image

    if (ilTexImage(m_launchParameters.resolution.x, m_launchParameters.resolution.y, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, nullptr))
    {
      uchar3* dst = reinterpret_cast<uchar3*>(ilGetData());

      const float  invGamma       = 1.0f / m_gamma;
      const float3 colorBalance   = make_float3(m_colorBalance.x, m_colorBalance.y, m_colorBalance.z);
      const float  invWhitePoint  = m_brightness / m_whitePoint;
      const float  burnHighlights = m_burnHighlights;
      const float  crushBlacks    = m_crushBlacks + m_crushBlacks + 1.0f;
      const float  saturation     = m_saturation;

      for (int y = 0; y < m_launchParameters.resolution.y; ++y)
      {
        for (int x = 0; x < m_launchParameters.resolution.x; ++x)
        {
          const int idx = m_launchParameters.resolution.x * y + x;

          // Tonemapper. // PERF Add a native CUDA kernel doing this.
          float3 hdrColor = make_float3(m_bufferHost[idx]);

          float3 ldrColor = invWhitePoint * colorBalance * hdrColor;
          ldrColor       *= ((ldrColor * burnHighlights) + 1.0f) / (ldrColor + 1.0f);
          
          float luminance = dot(ldrColor, make_float3(0.3f, 0.59f, 0.11f));
          ldrColor = lerp(make_float3(luminance), ldrColor, saturation); // This can generate negative values for saturation > 1.0f!
          ldrColor = fmaxf(make_float3(0.0f), ldrColor); // Prevent negative values.

          luminance = dot(ldrColor, make_float3(0.3f, 0.59f, 0.11f));
          if (luminance < 1.0f)
          {
            const float3 crushed = powf(ldrColor, crushBlacks);
            ldrColor = lerp(crushed, ldrColor, sqrtf(luminance));
            ldrColor = fmaxf(make_float3(0.0f), ldrColor); // Prevent negative values.
          }
          ldrColor = clamp(powf(ldrColor, invGamma), 0.0f, 1.0f); // Saturate, clamp to range [0.0f, 1.0f].

          dst[idx] = make_uchar3((unsigned char) (ldrColor.x * 255.0f),
                                 (unsigned char) (ldrColor.y * 255.0f),
                                 (unsigned char) (ldrColor.z * 255.0f));
        }
      }
      hasImage = true;
    }
  }
  else
  {
    path << ".hdr"; // Store the float4 linear output buffer as *.hdr image.

    hasImage = ilTexImage(m_launchParameters.resolution.x, m_launchParameters.resolution.y, 1, 4, IL_RGBA, IL_FLOAT, (void*) m_bufferHost);
  }

  if (hasImage)
  {
    ilEnable(IL_FILE_OVERWRITE); // By default, always overwrite
    
    std::string filename = path.str();
    convertPath(filename);
	
    if (ilSaveImage((const ILstring) filename.c_str()))
    {
      ilDeleteImages(1, &imageID);

      std::cout << filename << '\n'; // Print out filename to indicate that a screenshot has been taken.
      return true;
    }
  }

  // There was an error when reaching this code.
  ILenum error = ilGetError(); // DEBUG 
  std::cerr << "ERROR: screenshot() failed with IL error " << error << '\n';

  while (ilGetError() != IL_NO_ERROR) // Clean up errors.
  {
  }

  // Free all resources associated with the DevIL image
  ilDeleteImages(1, &imageID);

  return false;
}
