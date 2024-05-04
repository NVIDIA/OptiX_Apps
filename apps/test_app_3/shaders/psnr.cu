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

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}
template <unsigned int blockSize>
__global__ void reduce6(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}

// lifted from: https://stackoverflow.com/questions/17371275/implementing-max-reduce-in-cuda
__device__ float atomicMaxf(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
    }
    return __int_as_float(old);
}

// Naive PSNR implementation (refine by reducing smarter)
extern "C" __global__ void compute_psnr(PsnrData* args)
{
    const uint32_t xLaunch = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t yLaunch = blockDim.y * blockIdx.y + threadIdx.y;

    const uint32_t max_idx = args->resolution.x * args->resolution.y - 1;

    // extern __shared__ int sdata[];
    // uint32_t total_block_size = blockDim.x * blockDim.y;
    // uint32_t local_tid = threadIdx.y * blockDim.x + threadIdx.x;

    float mse_sum   =  0.f;
    float max_pixel = -INFINITY;

    if (yLaunch < args->resolution.y && xLaunch < args->resolution.x)
    {
        const float4* ref = reinterpret_cast<float4*>(args->outputBuffer_ref);
        const float4* res = reinterpret_cast<float4*>(args->outputBuffer);

        // dst[yLaunch * args->resolution.x + xLaunch] = make_float4(0,0.5,1,1);
        int32_t idx = yLaunch * args->resolution.x + xLaunch;

        float4 ref_rgba = ref[idx];
        float4 res_rgba = res[idx];

        float4 diff_rgba = ref_rgba - res_rgba;
        const float luminance = (0.299f * diff_rgba.x + 0.587f * diff_rgba.y + 0.114f * diff_rgba.z);
        const float lum_2 = luminance * luminance;

        atomicAdd(&mse_sum, lum_2);
        atomicMaxf(&max_pixel, luminance);
        //sdata[local_tid] = luminance;

        //     // First calculate block coordinates of this launch index.
        //     // That is the launch index divided by the tile dimensions. (No operator>>() on vectors?)
        //     const unsigned int xBlock = xLaunch >> args->tileShift.x;
        //     const unsigned int yBlock = yLaunch >> args->tileShift.y;

        //     // Each device needs to start at a different column and each row should start with a different device.
        //     const unsigned int xTile = xBlock * args->deviceCount + ((args->deviceIndex + yBlock) % args->deviceCount);

        //     // The horizontal pixel coordinate is: tile coordinate * tile width + launch index % tile width.
        //     const unsigned int xPixel = xTile * args->tileSize.x + (xLaunch & (args->tileSize.x - 1)); // tileSize needs to be power-of-two for this modulo operation.

        //     if (xPixel < args->resolution.x)
        //     {
        // #if USE_FP32_OUTPUT
        //       const float4 *src = reinterpret_cast<float4*>(args->tileBuffer);
        //       float4       *dst = reinterpret_cast<float4*>(args->outputBuffer);
        // #else
        //       const Half4 *src = reinterpret_cast<Half4*>(args->tileBuffer);
        //       Half4       *dst = reinterpret_cast<Half4*>(args->outputBuffer);
        // #endif
        //       // The src location needs to be calculated with the original launch width, because gridDim.x * blockDim.x might be different.
        //       dst[yLaunch * args->resolution.x + xPixel] = src[yLaunch * args->launchWidth + xLaunch]; // Copy one pixel per launch index.
        //     }


        if (idx == max_idx) {
            float mse = mse_sum / (max_idx + 1);
            printf("MSE = %f\nmax luminance = %f", mse, max_pixel);
            if (mse == 0.f) {
                printf("\n");
                return;
            }
            float psnr = 20.f * log10(max_pixel / sqrt(mse));
            printf("\tPSNR = %f\n");
        }
    }
}
