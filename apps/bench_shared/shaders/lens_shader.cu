/* 
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "config.h"

#include <optix.h>

#include "system_data.h"
#include "shader_common.h"

extern "C" __constant__ SystemData sysData;

// Note that all these lens shaders return the primary ray origin and direction in world space!

extern "C" __device__ void __direct_callable__pinhole(const float2 screen, const float2 pixel, const float2 sample, 
                                                      float3& origin, float3& direction)
{
  const float2 fragment = pixel + sample;                    // Jitter the sub-pixel location
  const float2 ndc      = (fragment / screen) * 2.0f - 1.0f; // Normalized device coordinates in range [-1, 1].

  const CameraDefinition camera = sysData.cameraDefinitions[0];

  origin    = camera.P;
  direction = normalize(camera.U * ndc.x +
                        camera.V * ndc.y +
                        camera.W);
}


extern "C" __device__ void __direct_callable__fisheye(const float2 screen, const float2 pixel, const float2 sample, 
                                                      float3& origin, float3& direction)
{
  const float2 fragment = pixel + sample; // x, y
  
  // Implement a fisheye projection with 180 degrees angle across the image diagonal (=> all pixels rendered, not a circular fisheye).
  const float2 center = screen * 0.5f;
  const float2 uv     = (fragment - center) / length(center); // uv components are in the range [0, 1]. Both 1 in the corners of the image!
  const float z       = cosf(length(uv) * 0.7071067812f * 0.5f * M_PIf); // Scale by 1.0f / sqrtf(2.0f) to get length into the range [0, 1]

  const CameraDefinition camera = sysData.cameraDefinitions[0];

  const float3 U = normalize(camera.U);
  const float3 V = normalize(camera.V);
  const float3 W = normalize(camera.W);

  origin    = camera.P;
  direction = normalize(uv.x * U + uv.y * V + z * W);
}


extern "C" __device__ void __direct_callable__sphere(const float2 screen, const float2 pixel, const float2 sample, 
                                                     float3& origin, float3& direction)
{
  const float2 uv = (pixel + sample) / screen; // "texture coordinates"

  // Convert the 2D index into a direction.
  const float phi   = uv.x * 2.0f * M_PIf;
  const float theta = uv.y * M_PIf;

  const float sinTheta = sinf(theta);

  const float3 v = make_float3(-sinf(phi) * sinTheta,
                               -cosf(theta),
                               -cosf(phi) * sinTheta);

  const CameraDefinition camera = sysData.cameraDefinitions[0];

  const float3 U = normalize(camera.U);
  const float3 V = normalize(camera.V);
  const float3 W = normalize(camera.W);

  origin    = camera.P;
  direction = normalize(v.x * U + v.y * V + v.z * W);
}
