/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "Mesh.h"
#include "Utils.h"

namespace dev
{
  static CUdeviceptr d_radii = (CUdeviceptr)0; // radius of all the spheres. Not elegant but works for now.

  const unsigned int* DevicePrimitive::getBuildInputFlags(const std::vector<MaterialData>& materials,
                                                          const unsigned int* inputFlagsOpaque,
                                                          const unsigned int* inputFlagsMask,
                                                          const unsigned int* inputFlagsBlend) const
  {
    if (currentMaterial >= 0)
    {
      // This index switches between geometry flags without (0) and with (1) face culling enabled.
      // If the material is double-sided (== not face culled) or has volume attenuation, disable face culling. 
      // Volume attenuation only works correctly when the backfaces of a volume can be intersected.
      const uint8_t indexFlagsCulling =
        (materials[currentMaterial].doubleSided ||
         (materials[currentMaterial].flags & FLAG_KHR_MATERIALS_VOLUME) != 0) ? 1 : 0;

      switch (materials[currentMaterial].alphaMode)
      {
        case MaterialData::ALPHA_MODE_OPAQUE:
        default:
        return &inputFlagsOpaque[indexFlagsCulling];

        case MaterialData::ALPHA_MODE_MASK:
        return &inputFlagsMask[indexFlagsCulling];

        case MaterialData::ALPHA_MODE_BLEND:
        return &inputFlagsBlend[indexFlagsCulling];
      }
    }
    return &inputFlagsOpaque[0]; // Default is single-sided opaque.
  }

  bool DevicePrimitive::setupBuildInput(OptixBuildInput& buildInput,
                                        const std::vector<MaterialData>& materials,
                                        const unsigned int * inputFlagsOpaque,
                                        const unsigned int * inputFlagsMask,
                                        const unsigned int * inputFlagsBlend,
                                        const float          sphereRadius) const
  {
    if (getPrimitiveType() == dev::PrimitiveType::Triangles)
    {
      //
      // Build input for TRIANGLES
      //
      buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

      // GEOMETRY
      buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
      buildInput.triangleArray.vertexStrideInBytes = sizeof(float3); // DeviceBuffer data is always tightly packed.
      buildInput.triangleArray.numVertices = static_cast<unsigned int>(positions.count);
      buildInput.triangleArray.vertexBuffers = &vertexBuffer; // This is the cached CUdeviceptr to the final position data
                                                              // (precedence is: skinned, morphed, base).

      if (indices.count != 0) // Indexed triangle mesh.
      {
        // INDICES
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
        buildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(indices.count / 3);
        buildInput.triangleArray.indexBuffer = indices.d_ptr;
      }
      else                    // Triangle soup.
      {
        // PERF This is redundant with the initialization above. All values are zero.
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
        buildInput.triangleArray.indexStrideInBytes = 0;
        buildInput.triangleArray.numIndexTriplets = 0;
        buildInput.triangleArray.indexBuffer = 0;
      }

      // SBT
      buildInput.triangleArray.numSbtRecords = 1; // glTF Material assignment is per Primitive (think: OpenGL draw call)!

      //
      // FACE CULLING, OPACITY
      //
      buildInput.triangleArray.flags = getBuildInputFlags(materials, inputFlagsOpaque, inputFlagsMask, inputFlagsBlend);
      return true;
    }
    else if (getPrimitiveType() == dev::PrimitiveType::Points)
    {
      //
      // Build input for SPHERES
      //
      MY_ASSERT(sphereRadius > 0.0f);

      // First allocate the radius
      
      const size_t       numRadii = 1;                   // TODO move to a new SpherePrimitive class.
      const size_t       sizeRadiiBytes = numRadii * sizeof(float);

      if (d_radii == (CUdeviceptr)0)
      {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radii), sizeRadiiBytes));
      }

      //Set radius for all the spheres.
      CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_radii),
                            //numRadii == 1 ? &sphereRadius : radii,
                            &sphereRadius,
                            sizeRadiiBytes,
                            cudaMemcpyHostToDevice));

      buildInput.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
      buildInput.sphereArray.numSbtRecords = 1;

      /*DEBUG*/ //utils::print3f(vertexBuffer, 1, "\tSphere Positions: ");

      buildInput.sphereArray.vertexBuffers = &vertexBuffer;
      buildInput.sphereArray.vertexStrideInBytes = 0;
      buildInput.sphereArray.numVertices = static_cast<unsigned>(positions.count);

      buildInput.sphereArray.radiusBuffers = &d_radii;
      buildInput.sphereArray.radiusStrideInBytes = 0;
      buildInput.sphereArray.singleRadius = (numRadii == 1) ? 1 : 0;

      buildInput.sphereArray.flags = getBuildInputFlags(materials, inputFlagsOpaque, inputFlagsMask, inputFlagsBlend);
      return true;
    }
    else // TODO if (getPrimitiveType() == dev::PrimitiveType::Curves)
    {
      // buildInput.type = OPTIX_BUILD_INPUT_TYPE_CURVES;
      //...
      std::cerr << "ERROR Unknown primitive type in setupBuildInput(): " << getPrimitiveType() << std::endl;
    }
    return false;
  }
} // namespace dev


