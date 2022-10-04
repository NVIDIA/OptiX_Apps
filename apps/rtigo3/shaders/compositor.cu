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

#include "compositor_data.h"
#include "vector_math.h"

// Compositor kernel to copy the tiles in the texelBuffer into the final outputBuffer location.
extern "C" __global__ void compositor(CompositorData* args)
{
  const unsigned int xLaunch = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int yLaunch = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (yLaunch < args->resolution.y)
  {
    // First calculate block coordinates of this launch index.
    // That is the launch index divided by the tile dimensions. (No operator>>() on vectors?)
    const unsigned int xBlock = xLaunch >> args->tileShift.x;
    const unsigned int yBlock = yLaunch >> args->tileShift.y;
  
    // Each device needs to start at a different column and each row should start with a different device.
    const unsigned int xTile = xBlock * args->deviceCount + ((args->deviceIndex + yBlock) % args->deviceCount);

    // The horizontal pixel coordinate is: tile coordinate * tile width + launch index % tile width.
    const unsigned int xPixel = xTile * args->tileSize.x + (xLaunch & (args->tileSize.x - 1)); // tileSize needs to be power-of-two for this modulo operation.

    if (xPixel < args->resolution.x)
    {
      const float4 *src = reinterpret_cast<float4*>(args->tileBuffer);
      float4       *dst = reinterpret_cast<float4*>(args->outputBuffer);

      // The src location needs to be calculated with the original launch width, because gridDim.x * blockDim.x might be different.
      dst[yLaunch * args->resolution.x + xPixel] = src[yLaunch * args->launchWidth + xLaunch]; // Copy one float4 per launch index.
    }
  }
}
