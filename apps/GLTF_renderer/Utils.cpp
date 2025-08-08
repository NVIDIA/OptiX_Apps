//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "Utils.h"
#include "DeviceBuffer.h"
#include "HostBuffer.h"
#include "MyAssert.h"
#include "cuda/material_data.h"

#include <optix_types.h>
#include <iostream>

#if defined(_WIN32)

  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN 1
  #endif

  #include <windows.h>
  #include <memory>
  // The cfgmgr32 header is necessary for interrogating driver information in the
  // registry.
  #include <cfgmgr32.h>
  // For convenience the library is also linked in automatically using the #pragma
  // command.
  #pragma comment(lib, "Cfgmgr32.lib")

#else

  #include <dlfcn.h>

#endif //_WIN32

namespace utils
{
  #ifdef _WIN32
  // Code based on helper function in optix_stubs.h
  void* optixLoadWindowsDll()
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
    char* systemPath = (char*)malloc(pathSize);

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

    // If we didn't find it, go looking in the register store.  Since nvoptix.dll
    // doesn't have its own registry entry, we are going to look for the OpenGL
    // driver which lives next to nvoptix.dll. 0 (null) will be returned if any
    // errors occured.

    const char* deviceInstanceIdentifiersGUID =
        "{4d36e968-e325-11ce-bfc1-08002be10318}";
    const ULONG flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;
    ULONG deviceListSize = 0;

    if (CM_Get_Device_ID_List_SizeA(&deviceListSize,
                                    deviceInstanceIdentifiersGUID,
                                    flags) != CR_SUCCESS)
    {
      return NULL;
    }

    char* deviceNames = (char*)malloc(deviceListSize);

    if (CM_Get_Device_ID_ListA(deviceInstanceIdentifiersGUID, deviceNames,
                               deviceListSize, flags))
    {
      free(deviceNames);
      return NULL;
    }

    DEVINST devID = 0;

    // Continue to the next device if errors are encountered.
    for (char* deviceName = deviceNames; *deviceName;
         deviceName += strlen(deviceName) + 1)
    {
      if (CM_Locate_DevNodeA(&devID, deviceName, CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS)
      {
        continue;
      }

      HKEY regKey = 0;
      if (CM_Open_DevNode_Key(devID, KEY_QUERY_VALUE, 0,
                              RegDisposition_OpenExisting, &regKey,
                              CM_REGISTRY_SOFTWARE) != CR_SUCCESS)
      {
        continue;
      }

      const char* valueName = "OpenGLDriverName";
      DWORD valueSize = 0;

      LSTATUS ret = RegQueryValueExA(regKey, valueName, NULL, NULL, NULL, &valueSize);
      if (ret != ERROR_SUCCESS)
      {
        RegCloseKey(regKey);
        continue;
      }

      char* regValue = (char*)malloc(valueSize);
      ret = RegQueryValueExA(regKey, valueName, NULL, NULL, (LPBYTE)regValue, &valueSize);
      if (ret != ERROR_SUCCESS)
      {
        free(regValue);
        RegCloseKey(regKey);
        continue;
      }

      // Strip the OpenGL driver dll name from the string then create a new string
      // with the path and the nvoptix.dll name
      for (int i = valueSize - 1; i >= 0 && regValue[i] != '\\'; --i)
      {
        regValue[i] = '\0';
      }

      size_t newPathSize = strlen(regValue) + strlen(optixDllName) + 1;
      char* dllPath = (char*)malloc(newPathSize);
      strcpy(dllPath, regValue);
      strcat(dllPath, optixDllName);

      free(regValue);
      RegCloseKey(regKey);

      handle = LoadLibraryA((LPCSTR)dllPath);
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


  void debugDumpTexture(const std::string& name, const MaterialData::Texture& t)
  {
    std::cout << name << ": ( index = " << t.index << ", object = "
              << t.object
              // KHR_texture_transform
              << "), scale = (" << t.scale.x << ", " << t.scale.y
              << "), rotation = (" << t.rotation.x << ", " << t.rotation.y
              << "), translation = (" << t.translation.x << ", "
              << t.translation.y << ")\n";
  }


  void debugDumpMaterial(const MaterialData& m)
  {
    // PBR Metallic Roughness parameters:
    std::cout << "baseColorFactor = (" << m.baseColorFactor.x << ", "
              << m.baseColorFactor.y << ", " << m.baseColorFactor.z << ", "
              << m.baseColorFactor.w << ")\n";
    std::cout << "metallicFactor  = " << m.metallicFactor << "\n";
    std::cout << "roughnessFactor = " << m.roughnessFactor << "\n";

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

    std::cout << "alphaCutoff = " << m.alphaCutoff << "\n";
    std::cout << "normalTextureScale = " << m.normalTextureScale << "\n";
    debugDumpTexture("normalTexture", m.normalTexture);
    std::cout << "occlusionTextureStrength = " << m.occlusionTextureStrength
              << "\n";
    debugDumpTexture("occlusionTexture", m.occlusionTexture);
    std::cout << "emissiveStrength = " << m.emissiveStrength << "\n";
    std::cout << "emissiveFactor = (" << m.emissiveFactor.x << ", "
              << m.emissiveFactor.y << ", " << m.emissiveFactor.z << ")\n";
    debugDumpTexture("emissiveTexture", m.emissiveTexture);

    std::cout << "flags = 0 ";
    // if (m.flags & FLAG_KHR_MATERIALS_IOR)
    //{
    //   std::cout << " | FLAG_KHR_MATERIALS_IOR";
    // }
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
    std::cout << "ior = " << m.ior << "\n";
    // KHR_materials_specular
    std::cout << "specularFactor = " << m.specularFactor << "\n";

    debugDumpTexture("specularTexture", m.specularTexture);
    std::cout << "specularColorFactor = (" << m.specularColorFactor.x << ", "
              << m.specularColorFactor.y << ", " << m.specularColorFactor.z
              << ")\n";
    debugDumpTexture("specularColorTexture", m.specularColorTexture);

    // KHR_materials_transmission
    std::cout << "transmissionFactor = " << m.transmissionFactor << "\n";

    debugDumpTexture("transmissionTexture", m.transmissionTexture);

    //  // KHR_materials_volume
    std::cout << "thicknessFactor = " << m.thicknessFactor << "\n";

    // debugDumpTexture("thicknessTexture", m.thicknessTexture);
    std::cout << "attenuationDistance = " << m.attenuationDistance << "\n";

    std::cout << "attenuationColor = (" << m.attenuationColor.x << ", "
              << m.attenuationColor.y << ", " << m.attenuationColor.z << ")\n";

    // KHR_materials_clearcoat
    std::cout << "clearcoatFactor = " << m.clearcoatFactor << "\n";

    debugDumpTexture("clearcoatTexture", m.clearcoatTexture);
    std::cout << "clearcoatRoughnessFactor = " << m.clearcoatRoughnessFactor
              << "\n";

    debugDumpTexture("clearcoatRoughnessTexture", m.clearcoatRoughnessTexture);
    debugDumpTexture("clearcoatNormalTexture", m.clearcoatNormalTexture);

    // KHR_materials_sheen
    std::cout << "sheenColorFactor = (" << m.sheenColorFactor.x << ", "
              << m.sheenColorFactor.y << ", " << m.sheenColorFactor.z << ")\n";
    debugDumpTexture("sheenColorTexture", m.sheenColorTexture);
    std::cout << "sheenRoughnessFactor = " << m.sheenRoughnessFactor << "\n";

    debugDumpTexture("sheenRoughnessTexture", m.sheenRoughnessTexture);

    // KHR_materials_anisotropy
    std::cout << "anisotropyStrength = " << m.anisotropyStrength << "\n";
    std::cout << "anisotropyRotation = " << m.anisotropyRotation << "\n";

    debugDumpTexture("anisotropyTexture", m.anisotropyTexture);

    // KHR_materials_iridescence
    std::cout << "iridescenceFactor = " << m.iridescenceFactor << "\n";

    debugDumpTexture("iridescenceTexture", m.iridescenceTexture);
    std::cout << "iridescenceIor = " << m.iridescenceIor << "\n";

    std::cout << "iridescenceThicknessMinimum = " << m.iridescenceThicknessMinimum
              << "\n";

    std::cout << "iridescenceThicknessMaximum = " << m.iridescenceThicknessMaximum
              << "\n";

    debugDumpTexture("iridescenceThicknessTexture",
                     m.iridescenceThicknessTexture);

    // KHR_materials_unlit
    std::cout << "unlit = " << ((m.unlit) ? "true" : "false") << "\n";
  }

  // void context_log_cb( unsigned int level, const char* tag, const char*
  // message, void* /*cbdata */)
  //{
  //     std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) <<
  //     tag << "]: "
  //               << message << "\n";
  // }


  // Calculate the values which handle the access calculations.
  // This is used by all three conversion routines.
  void determineAccess(const ConversionArguments& args, size_t& bytesPerComponent,
                       size_t& strideInBytes)
  {
    bytesPerComponent =
        fastgltf::getComponentBitSize(args.srcComponentType) >> 3;  // Returned.
    MY_ASSERT(0 < bytesPerComponent);

    if (args.srcBufferView->byteStride.has_value())
    {
      // This assumes that the bufferView.byteStride adheres to the glTF data
      // alignment requirements!
      strideInBytes = args.srcBufferView->byteStride.value();  // Returned.
    }
    else
    {
      // BufferView has no byteStride value, means the data is tightly packed
      // according to the glTF alignment rules (vector types are 4 bytes aligned).
      const size_t numComponents = fastgltf::getNumComponents(args.srcType);
      MY_ASSERT(0 < numComponents);

      // This is the number of bytes per element inside the source buffer without
      // padding!
      size_t bytesPerElement = numComponents * bytesPerComponent;

      // Now it gets awkward:
      // The glTF specs "Data Alignment" chapter requires that start addresses of
      // vectors must align to 4-byte. That also affects the individual column
      // vectors of matrices! That means padding to 4-byte addresses of vectors is
      // required in the following four cases:
      if (args.srcType == fastgltf::AccessorType::Vec3 &&
          bytesPerComponent == 1)
      {
        bytesPerElement = 4;
      }
      else if (args.srcType == fastgltf::AccessorType::Mat2 &&
                 bytesPerComponent == 1)
      {
        bytesPerElement = 8;
      }
      else if (args.srcType == fastgltf::AccessorType::Mat3 &&
                 bytesPerComponent <= 2)
      {
        bytesPerElement = 12 * size_t(bytesPerComponent);  // Can be 12 or 24 bytes stride.
      }

      // The bytesPerElement value is only used when the bufferView doesn't
      // specify a byteStride.
      strideInBytes = bytesPerElement;  // Returned.
    }
  }


  unsigned short readComponentAsUshort(const ConversionArguments& args,
                                       const unsigned char* src)
  {
    // This is only ever called for JOINTS_n which can be uchar or ushort.
    switch (args.srcComponentType)
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


  unsigned int readComponentAsUint(const ConversionArguments& args,
                                   const unsigned char* src)
  {
    switch (args.srcComponentType)
    {
      case fastgltf::ComponentType::UnsignedByte:
        return (unsigned int)(*reinterpret_cast<const unsigned char*>(src));

      case fastgltf::ComponentType::UnsignedShort:
        return (unsigned int)(*reinterpret_cast<const unsigned short*>(src));

      case fastgltf::ComponentType::UnsignedInt:
        return *reinterpret_cast<const unsigned int*>(src);

      default:
        // This is only ever used for indices and they should only be uchar,
        // ushort, uint.
        MY_ASSERT(
            !"readComponentAsUint(): Illegal component type");  // Normalized
                                                                // values are only
                                                                // allowed for 8
                                                                // and 16 bit
                                                                // integers.
        return 0u;
    }
  }


  float readComponentAsFloat(const ConversionArguments& args,
                             const unsigned char* src)
  {
    float f;

    switch (args.srcComponentType)
    {
      case fastgltf::ComponentType::Byte:
        f = float(*reinterpret_cast<const int8_t*>(src));
        return (args.srcNormalized) ? std::max(-1.0f, f / 127.0f) : f;

      case fastgltf::ComponentType::UnsignedByte:
        f = float(*reinterpret_cast<const uint8_t*>(src));
        return (args.srcNormalized) ? f / 255.0f : f;

      case fastgltf::ComponentType::Short:
        f = float(*reinterpret_cast<const int16_t*>(src));
        return (args.srcNormalized) ? std::max(-1.0f, f / 32767.0f) : f;

      case fastgltf::ComponentType::UnsignedShort:
        f = float(*reinterpret_cast<const uint16_t*>(src));
        return (args.srcNormalized) ? f / 65535.0f : f;

      case fastgltf::ComponentType::Float:
        return *reinterpret_cast<const float*>(src);

      default:
        // None of the vertex attributes supports normalized int32_t or uint32_t
        // or double.
        MY_ASSERT(
            !"readComponentAsFloat() Illegal component type");  // Normalized
                                                                // values are only
                                                                // allowed for 8
                                                                // and 16 bit
                                                                // integers.
        return 0.0f;
    }
  }


  void convertToUshort(const ConversionArguments& args)
  {
    size_t bytesPerComponent;
    size_t strideInBytes;

    determineAccess(args, bytesPerComponent, strideInBytes);

    std::visit(
        fastgltf::visitor{
            [](auto& /* arg */)
            {
              // Covers FilePathWithOffset, BufferView, ... which are all not
              // possible
            },

            [&](fastgltf::sources::Array& vector)
            {
              const unsigned char* ptrBase =
                  reinterpret_cast<const unsigned char*>(vector.bytes.data()) +
                  args.srcBufferView->byteOffset +
                  args.srcByteOffset;  // FIXME std::byte
              unsigned short* ptr =
                  reinterpret_cast<unsigned short*>(args.dstPtr);

              // Check if the data can simply be memcpy'ed.
              if (args.srcType == fastgltf::AccessorType::Vec4 &&
                  args.srcComponentType ==
                      fastgltf::ComponentType::UnsignedShort &&
                  strideInBytes == 4 * sizeof(uint16_t))
              {
                memcpy(ptr, ptrBase, args.srcCount * strideInBytes);
              }
              else
              {
                switch (args.srcType)
                {
                  // This function will only ever be called for JOINTS_n which are
                  // uchar or ushort VEC4.
                  case fastgltf::AccessorType::Vec4:
                    for (size_t i = 0; i < args.srcCount; ++i)
                    {
                      const unsigned char* ptrElement =
                          ptrBase + i * strideInBytes;

                      ptr[0] = readComponentAsUshort(args, ptrElement);
                      ptr[1] = readComponentAsUshort(
                          args, ptrElement + bytesPerComponent);
                      ptr[2] = readComponentAsUshort(
                          args, ptrElement + bytesPerComponent * 2);
                      ptr[3] = readComponentAsUshort(
                          args, ptrElement + bytesPerComponent * 3);
                      ptr += 4;
                    }
                    break;

                  default:
                    MY_ASSERT(!"convertToUshort() Unexpected accessor type.")
                    break;
                }
              }
            }},
        args.srcBuffer->data);
  }


  void convertToUint(const ConversionArguments& args)
  {
    size_t bytesPerComponent;
    size_t strideInBytes;

    determineAccess(args, bytesPerComponent, strideInBytes);

    std::visit(
        fastgltf::visitor{
            [](auto& /* arg */)
            {
              // Covers FilePathWithOffset, BufferView, ... which are all not
              // possible
            },

            [&](fastgltf::sources::Array& vector)
            {
              const unsigned char* ptrBase =
                  reinterpret_cast<const unsigned char*>(vector.bytes.data()) +
                  args.srcBufferView->byteOffset +
                  args.srcByteOffset;  // FIXME std::byte
              unsigned int* ptr = reinterpret_cast<unsigned int*>(args.dstPtr);

              // Check if the data can simply be memcpy'ed.
              if (args.srcType == fastgltf::AccessorType::Scalar &&
                  args.srcComponentType == fastgltf::ComponentType::UnsignedInt &&
                  strideInBytes == sizeof(uint32_t))
              {
                memcpy(ptr, ptrBase, args.srcCount * strideInBytes);
              }
              else
              {
                switch (args.srcType)
                {
                  // This function will only ever be called for vertex indices
                  // which are uchar, ushort or uint scalars.
                  case fastgltf::AccessorType::Scalar:
                    for (size_t i = 0; i < args.srcCount; ++i)
                    {
                      const unsigned char* ptrElement =
                          ptrBase + i * strideInBytes;

                      *ptr++ = readComponentAsUint(args, ptrElement);
                    }
                    break;

                  default:
                    MY_ASSERT(!"convertToUint() Unexpected accessor type.")
                    break;
                }
              }
            }},
        args.srcBuffer->data);
  }


  void convertToFloat(const ConversionArguments& args)
  {
    size_t bytesPerComponent;
    size_t strideInBytes;

    determineAccess(args, bytesPerComponent, strideInBytes);

    const size_t numTargetComponents = fastgltf::getNumComponents(args.dstType);

    //visit args.srcBuffer->data
    std::visit(
        fastgltf::visitor{
            [](auto& /* arg */)
            {
              // Covers FilePathWithOffset, BufferView, ... which are all not
              // possible
            },

            [&](fastgltf::sources::Array& vector)
            {
              const unsigned char* ptrBase =
                  reinterpret_cast<const unsigned char*>(vector.bytes.data()) +
                  args.srcBufferView->byteOffset +
                  args.srcByteOffset;  // FIXME std::byte
              float* ptr = reinterpret_cast<float*>(args.dstPtr);

              // Check if the data can simply be memcpy'ed.
              if (args.srcType == args.dstType &&
                  args.srcComponentType == fastgltf::ComponentType::Float &&
                  strideInBytes == size_t(numTargetComponents) * sizeof(float))
              {
                memcpy(ptr, ptrBase, args.srcCount * strideInBytes);
              }
              else
              {
                for (size_t i = 0; i < args.srcCount; ++i)
                {
                  const unsigned char* ptrElement = ptrBase + i * strideInBytes;
                  switch (args.srcType)
                  {
                    case fastgltf::AccessorType::Scalar:
                      *ptr++ = readComponentAsFloat(args, ptrElement);
                      break;

                    case fastgltf::AccessorType::Vec2:
                      ptr[0] = readComponentAsFloat(args, ptrElement);
                      ptr[1] = readComponentAsFloat(
                          args, ptrElement + bytesPerComponent);
                      ptr += 2;
                      break;

                    case fastgltf::AccessorType::Vec3:
                      ptr[0] = readComponentAsFloat(args, ptrElement);
                      ptr[1] = readComponentAsFloat(args, ptrElement + bytesPerComponent);
                      ptr[2] = readComponentAsFloat(args, ptrElement + bytesPerComponent * 2);
                      ptr += 3;
                      // Special case for vec3f to vec4f conversion.
                      // Color attribute requires alpha = 1.0f, color morph target
                      // requires alpha == 0.0f.
                      if (args.dstType == fastgltf::AccessorType::Vec4)
                      {
                        *ptr++ = args.dstExpansion;  // Append the desired w-component.
                      }
                      break;

                    case fastgltf::AccessorType::Vec4:
                      ptr[0] = readComponentAsFloat(args, ptrElement);
                      ptr[1] = readComponentAsFloat(args, ptrElement + bytesPerComponent);
                      ptr[2] = readComponentAsFloat(args, ptrElement + bytesPerComponent * 2);
                      ptr[3] = readComponentAsFloat(args, ptrElement + bytesPerComponent * 3);
                      ptr += 4;
                      break;

                    case fastgltf::AccessorType::Mat2:  // DEBUG Are these
                                                        // actually used as source
                                                        // data in glTF anywhere?
                      if (1 < bytesPerComponent)  // Standard case, no padding to
                                                  // 4-byte vectors needed.
                      {
                        // glTF/OpenGL matrices are defined column-major!
                        ptr[0] = readComponentAsFloat(args, ptrElement);  // m00
                        ptr[1] = readComponentAsFloat(args, ptrElement + bytesPerComponent);  // m10
                        ptr[2] = readComponentAsFloat(args, ptrElement + bytesPerComponent * 2);  // m01
                        ptr[3] = readComponentAsFloat(args, ptrElement + bytesPerComponent * 3);  // m11
                      }
                      else  // mat2 with 1-byte components requires 2 bytes
                              // source data padding between the two vectors..
                      {
                        MY_ASSERT(bytesPerComponent == 1);
                        ptr[0] = readComponentAsFloat(args, ptrElement + 0);  // m00
                        ptr[1] = readComponentAsFloat(args, ptrElement + 1);  // m10
                        // 2 bytes padding
                        ptr[2] = readComponentAsFloat(args, ptrElement + 4);  // m01
                        ptr[3] = readComponentAsFloat(args, ptrElement + 5);  // m11
                      }
                      ptr += 4;
                      break;

                    case fastgltf::AccessorType::Mat3:  // DEBUG Are these
                                                        // actually used as source
                                                        // data in glTF anywhere?
                      if (2 < bytesPerComponent)  // Standard case, no padding to
                                                  // 4-byte vectors needed.
                     
                      {
                        // glTF/OpenGL matrices are defined column-major!
                        for (int element = 0; element < 9; ++element)
                        {
                          ptr[element] = readComponentAsFloat(args, ptrElement + bytesPerComponent * element);
                        }
                      }
                      else if (bytesPerComponent == 1)  // mat3 with 1-byte components requires 2
                                     // bytes source data padding between the two
                                     // vectors..
                      {
                        ptr[0] =
                            readComponentAsFloat(args, ptrElement + 0);  // m00
                        ptr[1] =
                            readComponentAsFloat(args, ptrElement + 1);  // m10
                        ptr[2] =
                            readComponentAsFloat(args, ptrElement + 2);  // m20
                        // 1 byte padding
                        ptr[3] =
                            readComponentAsFloat(args, ptrElement + 4);  // m01
                        ptr[4] =
                            readComponentAsFloat(args, ptrElement + 5);  // m11
                        ptr[5] =
                            readComponentAsFloat(args, ptrElement + 6);  // m21
                        // 1 byte padding
                        ptr[6] =
                            readComponentAsFloat(args, ptrElement + 8);  // m02
                        ptr[7] =
                            readComponentAsFloat(args, ptrElement + 9);  // m12
                        ptr[8] =
                            readComponentAsFloat(args, ptrElement + 10);  // m22
                      }
                      else if (bytesPerComponent == 2)  // mat3 with 2-byte components requires 2
                                     // bytes source data padding between the two
                                     // vectors..
                      {
                        ptr[0] =
                            readComponentAsFloat(args, ptrElement + 0);  // m00
                        ptr[1] =
                            readComponentAsFloat(args, ptrElement + 2);  // m10
                        ptr[2] =
                            readComponentAsFloat(args, ptrElement + 4);  // m20
                        // 2 bytes padding
                        ptr[3] =
                            readComponentAsFloat(args, ptrElement + 8);  // m01
                        ptr[4] =
                            readComponentAsFloat(args, ptrElement + 10);  // m11
                        ptr[5] =
                            readComponentAsFloat(args, ptrElement + 12);  // m21
                        // 2 bytes padding
                        ptr[6] =
                            readComponentAsFloat(args, ptrElement + 16);  // m02
                        ptr[7] =
                            readComponentAsFloat(args, ptrElement + 18);  // m12
                        ptr[8] =
                            readComponentAsFloat(args, ptrElement + 20);  // m22
                      }
                      ptr += 9;
                      break;

                    case fastgltf::AccessorType::Mat4:
                      // glTF/OpenGL matrices are defined column-major!
                      for (int element = 0; element < 16; ++element)
                      {
                        ptr[element] = readComponentAsFloat(
                            args, ptrElement + bytesPerComponent * element);
                      }
                      ptr += 16;
                      break;

                    default:
                      MY_ASSERT(!"convertToFloat() Unexpected accessor type.")
                      break;
                  }
                }
              }
            }},
        args.srcBuffer->data);
  }


  void convertSparse(fastgltf::Asset& asset,
                     const fastgltf::SparseAccessor& sparse,
                     const ConversionArguments& args)
  {
    // Allocate some memory for the sparse accessor indices.
    std::vector<unsigned int> indices(sparse.count);

    // Read the indices from the sparse accessor indices buffer and convert them
    // to uint.
    ConversionArguments argsIndices = {};

    argsIndices.srcByteOffset = sparse.indicesByteOffset;
    argsIndices.srcType = fastgltf::AccessorType::Scalar;
    argsIndices.srcComponentType = sparse.indexComponentType;
    argsIndices.srcCount = sparse.count;
    argsIndices.srcNormalized = false;
    argsIndices.srcBufferView = &asset.bufferViews[sparse.indicesBufferView];
    argsIndices.srcBuffer = &asset.buffers[argsIndices.srcBufferView->bufferIndex];
    argsIndices.dstType = fastgltf::AccessorType::Scalar;
    argsIndices.dstComponentType = fastgltf::ComponentType::UnsignedInt;
    argsIndices.dstExpansion = args.dstExpansion;
    argsIndices.dstPtr = reinterpret_cast<unsigned char*>(indices.data());

    convertToUint(argsIndices);

    // Read the values from the sparse accessor values buffer view and convert
    // them to the destination type.
    ConversionArguments argsValues = {};

    argsValues.srcByteOffset = sparse.valuesByteOffset;
    argsValues.srcType = args.srcType;
    argsValues.srcComponentType = args.srcComponentType;
    argsValues.srcCount = sparse.count;
    argsValues.srcNormalized = args.srcNormalized;
    argsValues.srcBufferView = &asset.bufferViews[sparse.valuesBufferView];
    argsValues.srcBuffer = &asset.buffers[argsValues.srcBufferView->bufferIndex];
    argsValues.dstType = args.dstType;
    argsValues.dstComponentType = args.dstComponentType;
    argsValues.dstExpansion = args.dstExpansion;
    argsValues.dstPtr = reinterpret_cast<unsigned char*>(indices.data());

    // Allocate the buffer to which the sparse values are converted.
    const size_t numTargetComponents = fastgltf::getNumComponents(argsValues.dstType);
    MY_ASSERT(0 < numTargetComponents);

    const size_t sizeTargetComponentInBytes = fastgltf::getComponentBitSize(argsValues.dstComponentType) >> 3;
    MY_ASSERT(0 < sizeTargetComponentInBytes);

    const size_t sizeTargetElementInBytes = numTargetComponents * sizeTargetComponentInBytes;
    const size_t sizeTargetBufferInBytes  = argsValues.srcCount * sizeTargetElementInBytes;

    argsValues.dstPtr = new unsigned char[sizeTargetBufferInBytes];  // Allocate the buffer which

    // The GLTF_renderer converts all attributes only to ushort, uint, or float
    // components.
    bool hasValues = true;
    switch (argsValues.dstComponentType)
    {
      case fastgltf::ComponentType::UnsignedShort:
        convertToUshort(argsValues);
        break;

      case fastgltf::ComponentType::UnsignedInt:
        convertToUint(argsValues);
        break;

      case fastgltf::ComponentType::Float:
        convertToFloat(argsValues);
        break;

      default:
        std::cerr
            << "ERROR: convertSparse() unexpected destination component type\n";
        hasValues = false;
        break;
    }

    if (hasValues)
    {
      unsigned char* src = argsValues.dstPtr;

      for (unsigned int index : indices)
      {
        // Calculate the destination address inside the original host buffer:
        unsigned char* dst = args.dstPtr + index * sizeTargetElementInBytes;
        memcpy(dst, src, sizeTargetElementInBytes);
        src += sizeTargetElementInBytes;
      }
    }

    delete[] argsValues.dstPtr;
  }


  uint32_t createHostBuffer(
    const char*             name,
    fastgltf::Asset&        asset,
    const int               indexAccessor,
    fastgltf::AccessorType  typeTarget,
    fastgltf::ComponentType typeTargetComponent,
    const float             expansion,
    HostBuffer&             hostBuffer)
  {
    // Negative accessor index means the data is optional and an empty HostBuffer
    // is returned.

    if (indexAccessor < 0)
    {
      //std::cerr << name << ": No data for accessor " << indexAccessor << ": the host buffer stays empty." << std::endl;
      return 0;  // HostBuffer stays empty!
    }

    // Accessor, BufferView, and Buffer together specify the glTF source data.
    MY_ASSERT(indexAccessor < static_cast<int>(asset.accessors.size()));
    const fastgltf::Accessor& accessor = asset.accessors[indexAccessor];

    if (accessor.count == 0)  // DEBUG Can there be accessors with count == 0?
    {
      std::cerr << "WARNING: createHostBuffer() accessor.count == 0\n";
      return 0;
    }

    if (!accessor.bufferViewIndex.has_value() && !accessor.sparse.has_value())
    {
      //std::cerr << "WARNING: " << name << " createHostBuffer() No buffer view and no sparse accessor -> the host buffer stays empty.\n";
      return 0;
    }

    //  std::cout << "Creating host buffer for " << name << ": " << accessor.count << " elements" << std::endl;

    // First calculate the size of the HostBuffer.
    const size_t numTargetComponents = fastgltf::getNumComponents(typeTarget);
    MY_ASSERT(0 < numTargetComponents);

    const size_t sizeTargetComponentInBytes = fastgltf::getComponentBitSize(typeTargetComponent) >> 3;
    MY_ASSERT(0 < sizeTargetComponentInBytes);

    // Number of elements inside the source accessor times the target element size
    // in bytes.
    const size_t sizeTargetBufferInBytes = accessor.count * numTargetComponents * sizeTargetComponentInBytes;

    // Host target buffer allocation.
    hostBuffer.h_ptr = new unsigned char[sizeTargetBufferInBytes];  // Allocate the host buffer.
    hostBuffer.size  = sizeTargetBufferInBytes;  // Size of the host and device buffers in bytes.
    hostBuffer.count = accessor.count;  // Number of elements of the actual vector type inside the buffer.
    hostBuffer.setName(name);

    // glTF: "When accessor.bufferView is undefined, the sparse accessor is initialized as an array of 
    // zeros of size (size of the accessor element) * (accessor.count) bytes."
    const bool hasBufferView = accessor.bufferViewIndex.has_value();
    if (!hasBufferView)
    {
      memset(hostBuffer.h_ptr, 0, sizeTargetBufferInBytes);
    }

    const bool hasSparse =  accessor.sparse.has_value();  // DEBUG Set this to false to disable sparse accessor support.

    ConversionArguments args = {};

    args.srcByteOffset = accessor.byteOffset;
    args.srcType = accessor.type;
    args.srcComponentType = accessor.componentType;
    args.srcCount = accessor.count;
    args.srcNormalized = accessor.normalized;
    args.srcBufferView = (hasBufferView) ? &asset.bufferViews[accessor.bufferViewIndex.value()] 
                                           : nullptr;
    args.srcBuffer = (hasBufferView) ? &asset.buffers[args.srcBufferView->bufferIndex]
                                           : nullptr;
    args.dstType = typeTarget;
    args.dstComponentType = typeTargetComponent;
    args.dstExpansion = expansion;
    args.dstPtr = hostBuffer.h_ptr;

    // Convert all elements inside the source data to the expected target data
    // format individually.
    switch (typeTargetComponent)
    {
      case fastgltf::ComponentType::UnsignedShort:  // JOINTS_n are converted to
                                                    // ushort.
        if (hasBufferView)
        {
          convertToUshort(args);
        }
        if (hasSparse)
        {
          convertSparse(asset, accessor.sparse.value(), args);
        }
        break;

      case fastgltf::ComponentType::UnsignedInt:  // Primitive indices are
                                                  // converted to uint.
        if (hasBufferView)
        {
          convertToUint(args);
        }
        if (hasSparse)
        {
          convertSparse(asset, accessor.sparse.value(), args);
        }
        break;

      case fastgltf::ComponentType::Float:  // Everything else is float.
        if (hasBufferView)
        {
          convertToFloat(args);
        }
        if (hasSparse)
        {
          convertSparse(asset, accessor.sparse.value(), args);
        }
        break;

      default:
        MY_ASSERT(!"createHostBuffer() Unexpected target component type.")
        break;
    }
    return static_cast<uint32_t>(accessor.count);
  }


  /// Create device buffer from host buffer.
  /// @param deviceBuffer OUT
  /// @param hostBuffer   Can have a null h_ptr.
  /// @return true if the buffer was created, false if the host buffer is null (this is legal, e.g. for a mesh without tangents).
  bool createDeviceBuffer(DeviceBuffer& deviceBuffer,
                          const HostBuffer& hostBuffer)
  {
    if (hostBuffer.h_ptr)
    {
      if (hostBuffer.size < 1)
        std::cerr << "WARNING creating a device buffer with 0 elements (" << hostBuffer.getName() << ")" << std::endl;

      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceBuffer.d_ptr),
                            hostBuffer.size));
      MY_ASSERT(deviceBuffer.d_ptr != (CUdeviceptr) 0);

      CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(deviceBuffer.d_ptr),
                            hostBuffer.h_ptr, hostBuffer.size,
                            cudaMemcpyHostToDevice));

      deviceBuffer.size = hostBuffer.size;
      deviceBuffer.count = hostBuffer.count;

      return true;//created
    }
    return false;//not created
  }


  // Convert between slashes and backslashes in paths depending on the operating
  // system.
  void convertPath(std::string& path)
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


  bool matchLUID(const char* cudaLUID, const unsigned int cudaNodeMask,
                 const char* glLUID, const unsigned int glNodeMask)
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


  bool matchUUID(const CUuuid& cudaUUID, const char* glUUID)
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


  cudaTextureAddressMode getTextureAddressMode(fastgltf::Wrap wrap)
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
        std::cerr
            << "WARNING: getTextureAddressMode() Unexpected texture wrap mode = "
            << static_cast<std::uint16_t>(wrap) << '\n';
        return cudaAddressModeWrap;
    }
  }


  std::string getPrimitiveTypeName(const fastgltf::PrimitiveType type)
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


  // DEBUG
  void printMat4(const std::string name, const glm::mat4& mat)
  {
    constexpr int W = 8;

    std::ostringstream stream;

    stream.precision(4);  // Precision is # digits in fraction part.
    // The mat[i] is a column-vector. Print the matrix in row.major layout!
    stream << std::fixed << std::setw(W) << mat[0].x << ", " << std::setw(W)
           << mat[1].x << ", " << std::setw(W) << mat[2].x << ", " << std::setw(W)
           << mat[3].x << '\n'
           << std::setw(W) << mat[0].y << ", " << std::setw(W) << mat[1].y << ", "
           << std::setw(W) << mat[2].y << ", " << std::setw(W) << mat[3].y << '\n'
           << std::setw(W) << mat[0].z << ", " << std::setw(W) << mat[1].z << ", "
           << std::setw(W) << mat[2].z << ", " << std::setw(W) << mat[3].z << '\n'
           << std::setw(W) << mat[0].w << ", " << std::setw(W) << mat[1].w << ", "
           << std::setw(W) << mat[2].w << ", " << std::setw(W) << mat[3].w
           << '\n';
    std::cout << name << '\n' << stream.str() << '\n';
  }


  void setInstanceTransform(OptixInstance& instance, const glm::mat4x4& matrix)
  {
    // GLM matrix indexing is column-major: [column][row].
    // Instance matrix 12 floats for 3x4 row-major matrix.
    // Copy the first three rows from the glm:mat4x4.
    instance.transform[0] = matrix[0][0];
    instance.transform[1] = matrix[1][0];
    instance.transform[2] = matrix[2][0];
    instance.transform[3] = matrix[3][0];
    instance.transform[4] = matrix[0][1];
    instance.transform[5] = matrix[1][1];
    instance.transform[6] = matrix[2][1];
    instance.transform[7] = matrix[3][1];
    instance.transform[8] = matrix[0][2];
    instance.transform[9] = matrix[1][2];
    instance.transform[10] = matrix[2][2];
    instance.transform[11] = matrix[3][2];
  }


  #if 0  // FIXME This function is currently unused. The defines are used for the
         // morph targets though.
  unsigned int getAttributeFlags(const dev::DevicePrimitive& devicePrim)
  {
    // The below code is using hardcocded array indices.
    MY_ASSERT(NUM_ATTR_TEXCOORDS == 2 && 
              NUM_ATTR_JOINTS    == 2 && 
              NUM_ATTR_WEIGHTS   == 2);

    unsigned int flags = 0;
    
    flags |= (devicePrim.indices.d_ptr)      ? ATTR_INDEX      : 0;
    flags |= (devicePrim.positions.d_ptr)    ? ATTR_POSITION   : 0;
    flags |= (devicePrim.tangents.d_ptr)     ? ATTR_TANGENT    : 0;
    flags |= (devicePrim.normals.d_ptr)      ? ATTR_NORMAL     : 0;
    flags |= (devicePrim.colors.d_ptr)       ? ATTR_COLOR_0    : 0;
    flags |= (devicePrim.texcoords[0].d_ptr) ? ATTR_TEXCOORD_0 : 0;
    flags |= (devicePrim.texcoords[1].d_ptr) ? ATTR_TEXCOORD_1 : 0;
    flags |= (devicePrim.joints[0].d_ptr)    ? ATTR_JOINTS_0   : 0;
    flags |= (devicePrim.joints[1].d_ptr)    ? ATTR_JOINTS_1   : 0;
    flags |= (devicePrim.weights[0].d_ptr)   ? ATTR_WEIGHTS_0  : 0;
    flags |= (devicePrim.weights[1].d_ptr)   ? ATTR_WEIGHTS_1  : 0;
    
    return flags;
  }
  #endif


  std::string getDateTime()
  {
    std::ostringstream oss;
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    const auto transformed = now.time_since_epoch().count() / 1000000;
    const auto millis = transformed % 1000;
    const std::tm* local_time = std::localtime(&now_time);
    const auto putTime = std::put_time(local_time, "%Y%m%d_%H%M%S");
    oss << putTime << "_" << millis;
    return oss.str();
  }


  float getFontScale()
  {
    const auto context = glfwGetCurrentContext();
    float xScale, yScale;
    glfwGetWindowContentScale(context, &xScale, &yScale);
    return xScale;  // arbitrary choice: X axis
  }


  void getSystemInformation()
  {
    int versionDriver = 0;
    CUDA_CHECK(cudaDriverGetVersion(&versionDriver));

    // The version is returned as (1000 * major + 10 * minor).
    int major = versionDriver / 1000;
    int minor = (versionDriver - major * 1000) / 10;
    std::cout << "Driver Version  = " << major << "." << minor << '\n';

    int versionRuntime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&versionRuntime));

    // The version is returned as (1000 * major + 10 * minor).
    major = versionRuntime / 1000;
    minor = (versionRuntime - major * 1000) / 10;
    std::cout << "Runtime Version = " << major << "." << minor << '\n';

    int countDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&countDevices));
    std::cout << "Device Count    = " << countDevices << '\n';

    for (int i = 0; i < countDevices; ++i)
    {
      cudaDeviceProp properties;

      CUDA_CHECK(cudaGetDeviceProperties(&properties, i));

      std::cout << "Device " << i << ": " << properties.name << '\n';
      if (i == 0)  // This single-GPU application selects the device 0 in initCUDA().
      {
        std::cout << "This GPU will be used for CUDA context creation!\n";
        if (nullptr == strstr(properties.name, "NVIDIA"))
        {
          std::cout << "WARNING the GPU doens't seem to be NVIDIA's"
                    << std::endl;
        }
      }
      
  #if 1  // Condensed information
      std::cout << "  SM " << properties.major << "." << properties.minor << '\n';
      std::cout << "  Total Mem = " << properties.totalGlobalMem << '\n';
#if CUDA_VERSION <= 12080
      std::cout << "  ClockRate [kHz] = " << properties.clockRate << '\n';
#endif
      std::cout << "  MaxThreadsPerBlock = " << properties.maxThreadsPerBlock
                << '\n';
      std::cout << "  SM Count = " << properties.multiProcessorCount << '\n';
#if CUDA_VERSION <= 12080
      std::cout << "  Timeout Enabled = " << properties.kernelExecTimeoutEnabled << '\n';
#endif
      std::cout << "  TCC Driver = " << properties.tccDriver << '\n';
  #else  // Dump every property.
      // std::cout << "name[256] = " << properties.name << '\n';
      std::cout << "uuid = " << properties.uuid.bytes << '\n';
      std::cout << "totalGlobalMem = " << properties.totalGlobalMem << '\n';
      std::cout << "sharedMemPerBlock = " << properties.sharedMemPerBlock << '\n';
      std::cout << "regsPerBlock = " << properties.regsPerBlock << '\n';
      std::cout << "warpSize = " << properties.warpSize << '\n';
      std::cout << "memPitch = " << properties.memPitch << '\n';
      std::cout << "maxThreadsPerBlock = " << properties.maxThreadsPerBlock
                << '\n';
      std::cout << "maxThreadsDim[3] = " << properties.maxThreadsDim[0] << ", "
                << properties.maxThreadsDim[1] << ", "
                << properties.maxThreadsDim[0] << '\n';
      std::cout << "maxGridSize[3] = " << properties.maxGridSize[0] << ", "
                << properties.maxGridSize[1] << ", " << properties.maxGridSize[2]
                << '\n';
#if CUDA_VERSION <= 12080
      std::cout << "clockRate = " << properties.clockRate << '\n';
#endif
      std::cout << "totalConstMem = " << properties.totalConstMem << '\n';
      std::cout << "major = " << properties.major << '\n';
      std::cout << "minor = " << properties.minor << '\n';
      std::cout << "textureAlignment = " << properties.textureAlignment << '\n';
      std::cout << "texturePitchAlignment = " << properties.texturePitchAlignment
                << '\n';
      std::cout << "deviceOverlap = " << properties.deviceOverlap << '\n';
      std::cout << "multiProcessorCount = " << properties.multiProcessorCount
                << '\n';
#if CUDA_VERSION <= 12080
      std::cout << "kernelExecTimeoutEnabled = " << properties.kernelExecTimeoutEnabled << '\n';
#endif
      std::cout << "integrated = " << properties.integrated << '\n';
      std::cout << "canMapHostMemory = " << properties.canMapHostMemory << '\n';
      std::cout << "computeMode = " << properties.computeMode << '\n';
      std::cout << "maxTexture1D = " << properties.maxTexture1D << '\n';
      std::cout << "maxTexture1DMipmap = " << properties.maxTexture1DMipmap
                << '\n';
      std::cout << "maxTexture1DLinear = " << properties.maxTexture1DLinear
                << '\n';
      std::cout << "maxTexture2D[2] = " << properties.maxTexture2D[0] << ", "
                << properties.maxTexture2D[1] << '\n';
      std::cout << "maxTexture2DMipmap[2] = " << properties.maxTexture2DMipmap[0]
                << ", " << properties.maxTexture2DMipmap[1] << '\n';
      std::cout << "maxTexture2DLinear[3] = " << properties.maxTexture2DLinear[0]
                << ", " << properties.maxTexture2DLinear[1] << ", "
                << properties.maxTexture2DLinear[2] << '\n';
      std::cout << "maxTexture2DGather[2] = " << properties.maxTexture2DGather[0]
                << ", " << properties.maxTexture2DGather[1] << '\n';
      std::cout << "maxTexture3D[3] = " << properties.maxTexture3D[0] << ", "
                << properties.maxTexture3D[1] << ", "
                << properties.maxTexture3D[2] << '\n';
      std::cout << "maxTexture3DAlt[3] = " << properties.maxTexture3DAlt[0]
                << ", " << properties.maxTexture3DAlt[1] << ", "
                << properties.maxTexture3DAlt[2] << '\n';
      std::cout << "maxTextureCubemap = " << properties.maxTextureCubemap << '\n';
      std::cout << "maxTexture1DLayered[2] = "
                << properties.maxTexture1DLayered[0] << ", "
                << properties.maxTexture1DLayered[1] << '\n';
      std::cout << "maxTexture2DLayered[3] = "
                << properties.maxTexture2DLayered[0] << ", "
                << properties.maxTexture2DLayered[1] << ", "
                << properties.maxTexture2DLayered[2] << '\n';
      std::cout << "maxTextureCubemapLayered[2] = "
                << properties.maxTextureCubemapLayered[0] << ", "
                << properties.maxTextureCubemapLayered[1] << '\n';
      std::cout << "maxSurface1D = " << properties.maxSurface1D << '\n';
      std::cout << "maxSurface2D[2] = " << properties.maxSurface2D[0] << ", "
                << properties.maxSurface2D[1] << '\n';
      std::cout << "maxSurface3D[3] = " << properties.maxSurface3D[0] << ", "
                << properties.maxSurface3D[1] << ", "
                << properties.maxSurface3D[2] << '\n';
      std::cout << "maxSurface1DLayered[2] = "
                << properties.maxSurface1DLayered[0] << ", "
                << properties.maxSurface1DLayered[1] << '\n';
      std::cout << "maxSurface2DLayered[3] = "
                << properties.maxSurface2DLayered[0] << ", "
                << properties.maxSurface2DLayered[1] << ", "
                << properties.maxSurface2DLayered[2] << '\n';
      std::cout << "maxSurfaceCubemap = " << properties.maxSurfaceCubemap << '\n';
      std::cout << "maxSurfaceCubemapLayered[2] = "
                << properties.maxSurfaceCubemapLayered[0] << ", "
                << properties.maxSurfaceCubemapLayered[1] << '\n';
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
      std::cout << "maxThreadsPerMultiProcessor = "
                << properties.maxThreadsPerMultiProcessor << '\n';
      std::cout << "streamPrioritiesSupported = "
                << properties.streamPrioritiesSupported << '\n';
      std::cout << "globalL1CacheSupported = "
                << properties.globalL1CacheSupported << '\n';
      std::cout << "localL1CacheSupported = " << properties.localL1CacheSupported
                << '\n';
      std::cout << "sharedMemPerMultiprocessor = "
                << properties.sharedMemPerMultiprocessor << '\n';
      std::cout << "regsPerMultiprocessor = " << properties.regsPerMultiprocessor
                << '\n';
      std::cout << "managedMemory = " << properties.managedMemory << '\n';
      std::cout << "isMultiGpuBoard = " << properties.isMultiGpuBoard << '\n';
      std::cout << "multiGpuBoardGroupID = " << properties.multiGpuBoardGroupID
                << '\n';
      std::cout << "singleToDoublePrecisionPerfRatio = "
                << properties.singleToDoublePrecisionPerfRatio << '\n';
      std::cout << "pageableMemoryAccess = " << properties.pageableMemoryAccess
                << '\n';
      std::cout << "concurrentManagedAccess = "
                << properties.concurrentManagedAccess << '\n';
      std::cout << "computePreemptionSupported = "
                << properties.computePreemptionSupported << '\n';
      std::cout << "canUseHostPointerForRegisteredMem = "
                << properties.canUseHostPointerForRegisteredMem << '\n';
      std::cout << "cooperativeLaunch = " << properties.cooperativeLaunch << '\n';
      std::cout << "cooperativeMultiDeviceLaunch = "
                << properties.cooperativeMultiDeviceLaunch << '\n';
      std::cout << "pageableMemoryAccessUsesHostPageTables = "
                << properties.pageableMemoryAccessUsesHostPageTables << '\n';
      std::cout << "directManagedMemAccessFromHost = "
                << properties.directManagedMemAccessFromHost << '\n';
  #endif
    }
  }


  void print3f(CUdeviceptr ptr, size_t numVectors, const char* info)
  { 
    if (ptr && 0<numVectors)
    {
      std::cout << (info ? info : "");
      float* data = new float[numVectors*3];
      // read into data[]
      CUDA_CHECK(cudaMemcpy((void*)data, (const void*)ptr, sizeof(float) * 3 * numVectors, cudaMemcpyDeviceToHost));
      // print data[]
      for (auto pf = data; numVectors--; pf += 3)
      {
        std::cout << "[" << pf[0] << " " << pf[1] << " " << pf[2] << "]\n";
      }
      std::cout << std::endl;
      delete[]data;
    }
  }
}  // namespace utils