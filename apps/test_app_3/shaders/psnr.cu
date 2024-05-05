/* 
 * Copyright (c) 2013-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "psnr_data.h"
#include "vector_math.h"
#include "half_common.h"


// kernel lifted from https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
// (Not the most optimal version!)

// template <unsigned int blockSize>
// __device__ void warpReduce(volatile int *sdata, unsigned int tid) {
//     if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//     if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//     if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//     if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//     if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//     if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
// }
// template <unsigned int blockSize>
// __global__ void reduce6(float *g_idata, float *g_odata, unsigned int n) {
//     extern __shared__ int sdata[];
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*(blockSize*2) + tid;
//     unsigned int gridSize = blockSize*2*gridDim.x;
//     sdata[tid] = 0;
//     while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
//     __syncthreads();
//     if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
//     if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
//     if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
//     if (tid < 32) warpReduce<blockSize>(sdata, tid);
//     if (tid == 0) g_odata[blockIdx.x] = sdata[0];

// }

// // lifted from: https://stackoverflow.com/questions/17371275/implementing-max-reduce-in-cuda
// __device__ float atomicMaxf(float* address, float val)
// {
//     int *address_as_int =(int*)address;
//     int old = *address_as_int, assumed;
//     while (val > __int_as_float(old)) {
//         assumed = old;
//         old = atomicCAS(address_as_int, assumed,
//                         __float_as_int(val));
//     }
//     return __int_as_float(old);
// }

extern "C" __global__ void compute_psnr_stats(PsnrData* args)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t idx = blockDim.x * blockIdx.x + tid;

    extern __shared__ float sdata[];

    if (idx < args->num_pixels)
    {
        const float4* ref = reinterpret_cast<float4*>(args->outputBuffer_ref);
        const float4* res = reinterpret_cast<float4*>(args->outputBuffer);

        // dst[yLaunch * args->resolution.x + xLaunch] = make_float4(0,0.5,1,1);

        const float4 ref_rgba = ref[idx];
        const float4 res_rgba = res[idx];

        float4 diff_rgba = ref_rgba - res_rgba;
        const float luminance = (0.299f * diff_rgba.x + 0.587f * diff_rgba.y + 0.114f * diff_rgba.z);
        float lum_2 = luminance * luminance;

        sdata[tid] = lum_2;
        sdata[blockDim.x + tid] = luminance;

        __syncthreads();

        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && (idx + s) < args->num_pixels) {
                sdata[tid] += sdata[tid + s];
                float a = sdata[blockDim.x + tid];
                float b = sdata[blockDim.x + tid + s];
                sdata[blockDim.x + tid] = max(a,b);
            }
            __syncthreads();
        }

        if (tid == 0) {
            float* workspace = reinterpret_cast<float*>(args->workspace);
            workspace[blockIdx.x] = sdata[0];
            workspace[gridDim.x + blockIdx.x] = sdata[blockDim.x];
        }
    }
}

extern "C" __global__ void compute_psnr_stats_mid(PsnrData* args)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t idx = blockDim.x * blockIdx.x + tid;

    extern __shared__ float sdata[];

    if (idx < args->gridDimX_start)
    {
        const float* workspace = reinterpret_cast<float*>(args->workspace);
        sdata[tid] = workspace[tid];
        sdata[tid + blockDim.x] = workspace[tid + blockDim.x];

        __syncthreads();

        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && (idx + s) < args->num_pixels) {
                sdata[tid] += sdata[tid + s];
                float a = sdata[blockDim.x + tid];
                float b = sdata[blockDim.x + tid + s];
                sdata[blockDim.x + tid] = max(a,b);
            }
            __syncthreads();
        }

        if (tid == 0) {
            float* workspace = reinterpret_cast<float*>(args->workspace);
            workspace[blockIdx.x] = sdata[0];
            workspace[gridDim.x + blockIdx.x] = sdata[blockDim.x];
        }
    }
}

// Basic tree-reduction-based PSNR implementation
extern "C" __global__ void compute_psnr(PsnrData* args)
{
    const uint32_t tid = threadIdx.x;

    extern __shared__ float sdata[];

    const float* workspace = reinterpret_cast<float*>(args->workspace);
    sdata[tid] = workspace[tid];
    sdata[tid + blockDim.x] = workspace[tid + blockDim.x];

    __syncthreads();

    for (uint32_t s = 512; s > 0; s >>= 1) {    // HACK: block size shouldn't be > 1024
        if (tid < s && (tid + s < blockDim.x)) {
            sdata[tid] += sdata[tid + s];
            float a = sdata[blockDim.x + tid];
            float b = sdata[blockDim.x + tid + s];
            sdata[blockDim.x + tid] = max(a,b);

        }
        __syncthreads();
    }

    if (tid == 0) {
        float mse_sum             = sdata[0];
        float max_luminance       = sdata[blockDim.x];
        const uint32_t num_pixels = args->num_pixels;

        float mse = mse_sum / num_pixels;
        printf("summse = %f\tMSE = %f\tmax luminance = %f\t num_pixels = %u", mse_sum, mse, max_luminance, num_pixels);
        if (mse == 0.f) {
            printf("\n");
            return;
        }
        float psnr = 20.f * log10(max_luminance / sqrt(mse));
        printf("\tPSNR = %f\n", psnr);
    }
}
