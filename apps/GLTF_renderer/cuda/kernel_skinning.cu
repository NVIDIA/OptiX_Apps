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


// All matrix transforms here work on row-major matrices, means transposed glm::mat4 data!

// Row-major Matrix3x4 * point. (v.w == 1.0f)
__forceinline__ __device__ float3 transformPoint(const float4* m, const float3 v)
{
  float3 r;

  r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z + m[0].w;
  r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z + m[1].w;
  r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z + m[2].w;

  return r;
}


// Row-major Matrix3x4 * vector. (v.w == 0.0f)
// (Used for normals as well because then matrices are inverse transpose.)
__forceinline__ __device__ float3 transformVector(const float4* m, const float3 v)
{
  float3 r;

  r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z;
  r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z;
  r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z;

  return r;
}


__forceinline__ __device__ void get_skin_matrix(const float4* __restrict__ skinMatrices,
                                                const ushort4 joint, 
                                                const float4 weight,
                                                float4* __restrict__ matrix)
{
  // PERF Row-major matrix allows reading less data!
  const float4* skinMatrix = skinMatrices + (int(joint.x) << 2);

  const float4 sx0 = skinMatrix[0];
  const float4 sx1 = skinMatrix[1];
  const float4 sx2 = skinMatrix[2];
  //const float4 sx3 = skinMatrix[3];

  skinMatrix = skinMatrices + (int(joint.y) << 2);

  const float4 sy0 = skinMatrix[0];
  const float4 sy1 = skinMatrix[1];
  const float4 sy2 = skinMatrix[2];
  //const float4 sy3 = skinMatrix[3];
  
  skinMatrix = skinMatrices + (int(joint.z) << 2);

  const float4 sz0 = skinMatrix[0];
  const float4 sz1 = skinMatrix[1];
  const float4 sz2 = skinMatrix[2];
  //const float4 sz3 = skinMatrix[3];
  
  skinMatrix = skinMatrices + (int(joint.w) << 2);

  const float4 sw0 = skinMatrix[0];
  const float4 sw1 = skinMatrix[1];
  const float4 sw2 = skinMatrix[2];
  //const float4 sw3 = skinMatrix[3];

  matrix[0] = weight.x * sx0 + weight.y * sy0 + weight.z * sz0 + weight.w * sw0;
  matrix[1] = weight.x * sx1 + weight.y * sy1 + weight.z * sz1 + weight.w * sw1;
  matrix[2] = weight.x * sx2 + weight.y * sy2 + weight.z * sz2 + weight.w * sw2;
  //matrix[3] = weight.x * sx3 + weight.y * sy3 + weight.z * sz3 + weight.w * sw3;
}


__forceinline__ __device__ void get_skin_matrix(const float4* __restrict__ skinMatrices,
                                                const ushort4 joint0,
                                                const float4 weight0,
                                                const ushort4 joint1,
                                                const float4 weight1,
                                                float4* __restrict__ matrix)
{
  get_skin_matrix(skinMatrices, joint0, weight0, matrix);

  float4 m[3]; // Only calculating 3x4 matrices here.

  get_skin_matrix(skinMatrices, joint1, weight1, m);

  matrix[0] += m[0];
  matrix[1] += m[1];
  matrix[2] += m[2];
  //matrix[3] += m[3];
}


// Calculate the skinned positions for joints_0 and weight_0
__global__ void kernel_skinning_0(const unsigned int numAttributes,
                                  const float4* skinMatrices,
                                  const ushort4* joints0,
                                  const float4* weights0,
                                  const float3* srcPositions,
                                  float3* dstPositions)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < numAttributes)
  {
    float4 matrix[3]; // Result only calculating 3x4 matrices here.

    get_skin_matrix(skinMatrices, joints0[idx], weights0[idx], matrix);

    dstPositions[idx] = transformPoint(matrix, srcPositions[idx]);
  }
}


// Calculate the skinned positions for joints_0 and weight_0 and joints_1, weight_1.
__global__ void kernel_skinning_1(const unsigned int numAttributes,
                                  const float4* skinMatrices,
                                  const ushort4* joints0,
                                  const float4* weights0,
                                  const ushort4* joints1,
                                  const float4* weights1,
                                  const float3* srcPositions,
                                  float3* dstPositions)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < numAttributes)
  {
    float4 matrix[3]; // Only calculating 3x4 matrices here.

    get_skin_matrix(skinMatrices, joints0[idx], weights0[idx], joints1[idx], weights1[idx], matrix);

    dstPositions[idx] = transformPoint(matrix, srcPositions[idx]);
  }
}


// Calculate the skinned positions and normals for joints_0 and weight_0
__global__ void kernel_skinning_2(const unsigned int numAttributes,
                                  const float4* skinMatrices,
                                  const float4* skinMatricesIT,
                                  const ushort4* joints0,
                                  const float4* weights0,
                                  const float3* srcPositions,
                                  const float3* srcNormals,
                                  float3* dstPositions,
                                  float3* dstNormals)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < numAttributes)
  {
    const ushort4 joint0  = joints0[idx]; 
    const float4  weight0 = weights0[idx]; 

    float4 matrix[3]; // Only calculating 3x4 matrices here.

    get_skin_matrix(skinMatrices, joint0, weight0, matrix);
    
    dstPositions[idx] = transformPoint(matrix, srcPositions[idx]);

    get_skin_matrix(skinMatricesIT, joint0, weight0, matrix);
    
    dstNormals[idx] = normalize(transformVector(matrix, srcNormals[idx])); 
  }
}


// Calculate the skinned positions and normals for joints_0, weight_0 and joints_1, weight_1.
__global__ void kernel_skinning_3(const unsigned int numAttributes,
                                  const float4* skinMatrices,
                                  const float4* skinMatricesIT,
                                  const ushort4* joints0,
                                  const float4* weights0,
                                  const ushort4* joints1,
                                  const float4* weights1,
                                  const float3* srcPositions,
                                  const float3* srcNormals,
                                  float3* dstPositions,
                                  float3* dstNormals)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < numAttributes)
  {
    const ushort4 joint0  = joints0[idx]; 
    const ushort4 joint1  = joints1[idx];
    const float4  weight0 = weights0[idx]; 
    const float4  weight1 = weights1[idx]; 

    float4 matrix[3]; // Only using 3x4 row-major matrices!

    get_skin_matrix(skinMatrices, joint0, weight0, joint1, weight1, matrix);

    dstPositions[idx] = transformPoint(matrix, srcPositions[idx]);

    get_skin_matrix(skinMatricesIT, joint0, weight0, joint1, weight1, matrix);

    dstNormals[idx] = normalize(transformVector(matrix, srcNormals[idx])); 
  }
}


// Calculate the skinned positions, tangents, normals for joints_0 and weight_0
__global__ void kernel_skinning_4(const unsigned int numAttributes,
                                  const float4* skinMatrices,
                                  const float4* skinMatricesIT,
                                  const ushort4* joints0,
                                  const float4*  weights0,
                                  const float3* srcPositions,
                                  const float4* srcTangents,
                                  const float3* srcNormals,
                                  float3* dstPositions,
                                  float4* dstTangents,
                                  float3* dstNormals)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < numAttributes)
  {
    const ushort4 joint0  = joints0[idx]; 
    const float4  weight0 = weights0[idx]; 

    float4 matrix[3]; // Only using 3x4 row-major matrices!

    get_skin_matrix(skinMatrices, joint0, weight0, matrix);

    dstPositions[idx] = transformPoint(matrix, srcPositions[idx]);

    const float4 srcTangent = srcTangents[idx]; // Handedness in .w is not skinned.

    dstTangents[idx] = make_float4(normalize(transformVector(matrix, make_float3(srcTangent))), srcTangent.w);

    get_skin_matrix(skinMatricesIT, joint0, weight0, matrix);

    dstNormals[idx] = normalize(transformVector(matrix, srcNormals[idx])); 
  }
}


// Calculate the skinned positions, tangents, normals for joints_0, weight_0 and joints_1, weight_1.
__global__ void kernel_skinning_5(const unsigned int numAttributes,
                                  const float4* skinMatrices,
                                  const float4* skinMatricesIT,
                                  const ushort4* joints0,
                                  const float4* weights0,
                                  const ushort4* joints1,
                                  const float4* weights1,
                                  const float3* srcPositions,
                                  const float4* srcTangents,
                                  const float3* srcNormals,
                                  float3* dstPositions,
                                  float4* dstTangents,
                                  float3* dstNormals)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < numAttributes)
  {
    const ushort4 joint0  = joints0[idx]; 
    const ushort4 joint1  = joints1[idx]; 
    const float4  weight0 = weights0[idx]; 
    const float4  weight1 = weights1[idx]; 

    float4 matrix[3]; // Only using 3x4 row-major matrices!

    get_skin_matrix(skinMatrices, joint0, weight0, joint1, weight1, matrix);
    dstPositions[idx] = transformPoint(matrix, srcPositions[idx]);
    
    const float4 srcTangent = srcTangents[idx]; // Handedness in .w is not skinned.

    dstTangents[idx] = make_float4(normalize(transformVector(matrix, make_float3(srcTangent))), srcTangent.w);

    get_skin_matrix(skinMatricesIT, joint0, weight0, joint1, weight1, matrix);
    
    dstNormals[idx] = normalize(transformVector(matrix, srcNormals[idx])); 
  }
}


__host__ void device_skinning(cudaStream_t cudaStream,
                              const float4* d_skinMatrices,
                              const unsigned int numSkinMatrices,
                              dev::DevicePrimitive& devicePrim)
{
  const unsigned int numAttributes = static_cast<unsigned int>(devicePrim.positions.count);

  const ushort4* d_joints0  = reinterpret_cast<const ushort4*>(devicePrim.joints[0].d_ptr);
  const float4*  d_weights0 = reinterpret_cast<const float4*>(devicePrim.weights[0].d_ptr);
  const ushort4* d_joints1  = reinterpret_cast<const ushort4*>(devicePrim.joints[1].d_ptr);
  const float4*  d_weights1 = reinterpret_cast<const float4*>(devicePrim.weights[1].d_ptr);

  const float3* d_srcPositions = (devicePrim.positionsMorphed.d_ptr) 
                               ? reinterpret_cast<const float3*>(devicePrim.positionsMorphed.d_ptr)
                               : reinterpret_cast<const float3*>(devicePrim.positions.d_ptr);
  const float4* d_srcTangents  = (devicePrim.tangentsMorphed.d_ptr) 
                               ? reinterpret_cast<const float4*>(devicePrim.tangentsMorphed.d_ptr)
                               : reinterpret_cast<const float4*>(devicePrim.tangents.d_ptr);
  const float3* d_srcNormals   = (devicePrim.normalsMorphed.d_ptr) 
                               ? reinterpret_cast<const float3*>(devicePrim.normalsMorphed.d_ptr)
                               : reinterpret_cast<const float3*>(devicePrim.normals.d_ptr);

  float3* d_dstPositions = reinterpret_cast<float3*>(devicePrim.positionsSkinned.d_ptr);
  float4* d_dstTangents  = reinterpret_cast<float4*>(devicePrim.tangentsSkinned.d_ptr);
  float3* d_dstNormals   = reinterpret_cast<float3*>(devicePrim.normalsSkinned.d_ptr);

  dim3 threadsPerBlock(128, 1, 1); // This should be good enough. Let CUDA figure out how to load-balance blocks
  dim3 numBlocks((numAttributes + 127) / threadsPerBlock.x, 
                 1,  // == (1 + 0) / threadsPerBlock.y, 
                 1); // == (1 + 0) / threadsPerBlock.z);

  // d_skinMatrices holds both the skin.matrices and skin.matricesIT.
  // Get the pointer to the optional skin.matricesIT. Only set and used when there are normal attributes.
  const float4* d_skinMatricesIT = d_skinMatrices + numSkinMatrices * 4;

  MY_ASSERT(d_srcPositions != nullptr && d_dstPositions != nullptr); // Position attributes must always be present.

  const int whichWeights = (d_joints1 == nullptr || d_weights1 == nullptr) ? 0 : 1;

  int whichAttributes = 0; // positions

  if (d_srcNormals != nullptr)
  {
    MY_ASSERT(d_dstNormals != nullptr);
    ++whichAttributes; // positions + normals

    if (d_srcTangents != nullptr)
    {
      MY_ASSERT(d_dstTangents != nullptr);
      ++whichAttributes; // positions + normals + tangents
    }
  }
  
  const int whichKernel = whichAttributes * 2 + whichWeights;

  switch (whichKernel)
  {
    case 0: // kernel_skinning_0() is positions, joints0, weights0
      kernel_skinning_0<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(
        numAttributes,
        d_skinMatrices,
        d_joints0,
        d_weights0,
        d_srcPositions,
        d_dstPositions);
      break;

    case 1: // kernel_skinning_1() is positions, joints0, weights0, joints1, weights1
      kernel_skinning_1<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(
        numAttributes,
        d_skinMatrices,
        d_joints0,
        d_weights0,
        d_joints1,
        d_weights1,
        d_srcPositions,
        d_dstPositions);
      break;

    case 2: // kernel_skinning_2() is positions + normals, joints0, weights0
      kernel_skinning_2<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(
        numAttributes,
        d_skinMatrices,
        d_skinMatricesIT,
        d_joints0,
        d_weights0,
        d_srcPositions,
        d_srcNormals,
        d_dstPositions,
        d_dstNormals);
      break;

    case 3: // kernel_skinning_3() is positions + normals, joints0, weights0 and joints1, weights1
      kernel_skinning_3<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(
        numAttributes,
        d_skinMatrices,
        d_skinMatricesIT,
        d_joints0,
        d_weights0,
        d_joints1,
        d_weights1,
        d_srcPositions,
        d_srcNormals,
        d_dstPositions,
        d_dstNormals);
      break;

    case 4: // kernel_skinning_4() is positions + normals + tangents, joints0, weights0
      kernel_skinning_4<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(
        numAttributes,
        d_skinMatrices,
        d_skinMatricesIT,
        d_joints0,
        d_weights0,
        d_srcPositions,
        d_srcTangents,
        d_srcNormals,
        d_dstPositions,
        d_dstTangents,
        d_dstNormals);
      break;

    case 5: // kernel_skinning_5() is positions + normals + tangents, joints0, weights0 and joints1, weights1
      kernel_skinning_5<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(
        numAttributes,
        d_skinMatrices,
        d_skinMatricesIT,
        d_joints0,
        d_weights0,
        d_joints1,
        d_weights1,
        d_srcPositions,
        d_srcTangents,
        d_srcNormals,
        d_dstPositions,
        d_dstTangents,
        d_dstNormals);
      break;
  }
}

