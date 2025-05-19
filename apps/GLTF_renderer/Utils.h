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

#pragma once

#include "ConversionArguments.h"
#include "DeviceBuffer.h"
#include "cuda/material_data.h"

#include <fastgltf/core.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>  // mat4

#include <GL/glew.h>    // GL_LUID_SIZE_EXT
#include <GLFW/glfw3.h>

#include <chrono>

struct OptixInstance;

class HostBuffer;

namespace utils
{
  void* optixLoadWindowsDll();

  void debugDumpTexture(const std::string& name, const MaterialData::Texture& t);

  void debugDumpMaterial(const MaterialData& m);

  // void context_log_cb( unsigned int level, const char* tag, const char*
  // message, void* /*cbdata */)
  //{
  //     std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 )
  //     << tag << "]: "
  //               << message << "\n";
  // }

  // Calculate the values which handle the access calculations.
  // This is used by all three conversion routines.
  void determineAccess(const ConversionArguments& args,
                       size_t& bytesPerComponent,
                       size_t& strideInBytes);

  unsigned short readComponentAsUshort(const ConversionArguments& args, const unsigned char* src);

  unsigned int readComponentAsUint(const ConversionArguments& args, const unsigned char* src);

  float readComponentAsFloat(const ConversionArguments& args, const unsigned char* src);

  void convertToUshort(const ConversionArguments& args);

  void convertToUint(const ConversionArguments& args);

  void convertToFloat(const ConversionArguments& args);

  void convertSparse(fastgltf::Asset& asset, const fastgltf::SparseAccessor& sparse, const ConversionArguments& args);

  /// @return Total count of elements created, can be zero.
  uint32_t createHostBuffer(
      const char*             dbgName,            // POSITION, NORMAL, ... for debugging.
      fastgltf::Asset&        asset,              // The asset contains all source data (Accessor, BufferView, Buffer)
      const int               indexAccessor,      // The accessor index defines the source data. -1 means no data.
      fastgltf::AccessorType  typeTarget,         // One of Scalar, Vec2, Vec3, Vec4.
      fastgltf::ComponentType typeTargetComponent,// One of UnsignedInt primitive indices, UnsignedShort JOINTS_n, everything else Float
      const float             expansion,          // 1.0f or 0.0f. Vec3 to Vec4 expansion of color attributes uses 1.0f, but color morph targets require 0.0f!
      HostBuffer&             hostBuffer          // OUT
  );

  /// Alloc and copy data, host-to-device.
  /// @returns true iff buffer created
  bool createDeviceBuffer(DeviceBuffer& deviceBuffer, const HostBuffer& hostBuffer);

  // Convert between slashes and backslashes in paths depending on the operating
  // system.
  void convertPath(std::string& path);

  bool matchLUID(const char* cudaLUID, const unsigned int cudaNodeMask,
                 const char* glLUID, const unsigned int glNodeMask);

  bool matchUUID(const CUuuid& cudaUUID, const char* glUUID);

  cudaTextureAddressMode getTextureAddressMode(fastgltf::Wrap wrap);

  std::string getPrimitiveTypeName(fastgltf::PrimitiveType type);

  // DEBUG
  void printMat4(const std::string name, const glm::mat4& mat);

  void setInstanceTransform(OptixInstance& instance, const glm::mat4x4& matrix);

  #if 0  
         // FIXME This function is currently unused. The defines are used for the
         // morph targets though.
   unsigned int getAttributeFlags(const dev::DevicePrimitive& devicePrim);
  #endif

  // yyyymmdd_hhmmss_mil. mil are milliseconds on 3 digits.
  std::string getDateTime();

  // Get (current : default) ratio, for both axis. This value is 2.5 for 4K screens. PERFO?
  float getFontScale();

  // Optionally dumps system information.
  void getSystemInformation();

  /// Print 3D vectors.
  void print3f(CUdeviceptr ptr, size_t numVectors, const char* infoToPrint);

  struct Timer
  {
    Timer()
    {
      start();
    }

    void start()
    {
      m_tStart = std::chrono::high_resolution_clock::now();
    }

    float getElapsedMilliseconds() const
    {
      auto tEnd = std::chrono::high_resolution_clock::now();
      auto timeRender = tEnd - m_tStart;

      return std::chrono::duration<float, std::milli>(timeRender).count();
    }

  private:

    std::chrono::time_point<std::chrono::high_resolution_clock> m_tStart;
  };
}  // namespace utils
