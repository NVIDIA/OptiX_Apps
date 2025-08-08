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

#include <cuda_runtime.h>

#include <Mesh.h>

#include "vector_math.h"

__global__ void kernel_morphing_float2(
  const float* weights,
  const unsigned int numAttributes,
  const int numTargets,
  const float2** targets,
  const float2* srcAttr,
  float2* dstAttr)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < numAttributes)
  {
    float2 v = srcAttr[idx];

    for (int n = 0; n < numTargets; ++n)
    {
      v += weights[n] * targets[n][idx];
    }

    dstAttr[idx] = v;
  }
}

__global__ void kernel_morphing_float3(
  const float* weights,
  const unsigned int numAttributes,
  const int numTargets,
  const float3** targets,
  const float3* srcAttr,
  float3* dstAttr)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < numAttributes)
  {
    float3 v = srcAttr[idx];

    for (int n = 0; n < numTargets; ++n)
    {
      v += weights[n] * targets[n][idx];
    }

    dstAttr[idx] = v;
  }
}

__global__ void kernel_morphing_float4(
  const float* weights,
  const unsigned int numAttributes,
  const int numTargets,
  const float4** targets,
  const float4* srcAttr,
  float4* dstAttr)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < numAttributes)
  {
    float4 v = srcAttr[idx];

    for (int n = 0; n < numTargets; ++n)
    {
      v += weights[n] * targets[n][idx];
    }

    dstAttr[idx] = v;
  }
}

// Special case for tangents because the attributes are float4  but the morph targets only float3
// because the handedness inside the w.component is not morphed and the vector needs to be normalized.
__global__ void kernel_morphing_tangent(
  const float* weights,
  const unsigned int numAttributes,
  const int numTargets,
  const float3** targets,
  const float4* srcAttr,
  float4* dstAttr)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < numAttributes)
  {
    const float4 v = srcAttr[idx];
    float3 t = make_float3(v); // Morph targets for tangents are only float3.

    for (int n = 0; n < numTargets; ++n)
    {
      t += weights[n] * targets[n][idx];
    }

    dstAttr[idx] = make_float4(normalize(t), v.w);
  }
}

// Special case for normals. Same as float3 but the vector needs to be normalized.
__global__ void kernel_morphing_normal(
  const float* weights,
  const unsigned int numAttributes,
  const int numTargets,
  const float3** targets,
  const float3* srcAttr,
  float3* dstAttr)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < numAttributes)
  {
    float3 v = srcAttr[idx];

    for (int n = 0; n < numTargets; ++n)
    {
      v += weights[n] * targets[n][idx];
    }

    dstAttr[idx] = normalize(v);
  }
}

__host__ void device_morphing(cudaStream_t cudaStream,
                              const float* d_weights,
                              dev::DevicePrimitive& devicePrim)
{
  const unsigned int numAttributes = static_cast<unsigned int>(devicePrim.positions.count);
  
  dim3 threadsPerBlock(128, 1, 1); // This should be good enough. Let CUDA figure out how to load-balance blocks
  dim3 numBlocks((numAttributes + 127) / threadsPerBlock.x, 
                 1,  // == (1 + 0) / threadsPerBlock.y, 
                 1); // == (1 + 0) / threadsPerBlock.z);

  // "Attributes present in the base mesh primitive but not included in a given morph target MUST retain their original values for the morph target.
  // Client implementations SHOULD support at least three attributes — POSITION, NORMAL, and TANGENT — for morphing. 
  // Client implementations MAY optionally support morphed TEXCOORD_n and/or COLOR_n attributes.
  // Note that the W component for handedness is omitted when targeting TANGENT data since handedness cannot be displaced."

  CUdeviceptr targets = devicePrim.targetPointers.d_ptr;

  if (devicePrim.maskTargets & ATTR_POSITION)
  {
    kernel_morphing_float3<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(
      d_weights,
      numAttributes,
      devicePrim.numTargets, 
      reinterpret_cast<const float3**>(targets),
      reinterpret_cast<const float3*>(devicePrim.positions.d_ptr),
      reinterpret_cast<float3*>(devicePrim.positionsMorphed.d_ptr));

    targets += devicePrim.numTargets * sizeof(CUdeviceptr);
  }

  if (devicePrim.maskTargets & ATTR_TANGENT)
  {
    kernel_morphing_tangent<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(
      d_weights,
      numAttributes,
      devicePrim.numTargets, 
      reinterpret_cast<const float3**>(targets),
      reinterpret_cast<const float4*>(devicePrim.tangents.d_ptr),
      reinterpret_cast<float4*>(devicePrim.tangentsMorphed.d_ptr));

    targets += devicePrim.numTargets * sizeof(CUdeviceptr);
  }

  if (devicePrim.maskTargets & ATTR_NORMAL)
  {
    kernel_morphing_normal<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(
      d_weights,
      numAttributes,
      devicePrim.numTargets, 
      reinterpret_cast<const float3**>(targets),
      reinterpret_cast<const float3*>(devicePrim.normals.d_ptr),
      reinterpret_cast<float3*>(devicePrim.normalsMorphed.d_ptr));

    targets += devicePrim.numTargets * sizeof(CUdeviceptr);
  }

  if (devicePrim.maskTargets & ATTR_COLOR_0)
  {
    kernel_morphing_float4<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(
      d_weights,
      numAttributes,
      devicePrim.numTargets, 
      reinterpret_cast<const float4**>(targets),
      reinterpret_cast<const float4*>(devicePrim.normals.d_ptr),
      reinterpret_cast<float4*>(devicePrim.normalsMorphed.d_ptr));

    targets += devicePrim.numTargets * sizeof(CUdeviceptr);
  }

  for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j)
  {
    if (devicePrim.maskTargets & (ATTR_TEXCOORD_0 << j))
    {
      kernel_morphing_float2<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(
        d_weights,
        numAttributes,
        devicePrim.numTargets, 
        reinterpret_cast<const float2**>(targets),
        reinterpret_cast<const float2*>(devicePrim.texcoords[j].d_ptr),
        reinterpret_cast<float2*>(devicePrim.texcoordsMorphed[j].d_ptr));

      targets += devicePrim.numTargets * sizeof(CUdeviceptr);
    }
  }
}

