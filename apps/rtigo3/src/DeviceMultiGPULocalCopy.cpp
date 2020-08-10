/* 
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "inc/DeviceMultiGPULocalCopy.h"

#include "inc/CheckMacros.h"

#include "shaders/compositor_data.h"

#include <GL/glew.h>
#if defined( _WIN32 )
#include <GL/wglew.h>
#endif

// CUDA Driver API version of the OpenGL interop header. 
#include <cudaGL.h>

#include <algorithm>
#include <string.h>

DeviceMultiGPULocalCopy::DeviceMultiGPULocalCopy(const RendererStrategy strategy,
                                                 const int ordinal,
                                                 const int index,
                                                 const int count,
                                                 const int miss,
                                                 const int interop,
                                                 const unsigned int tex,
                                                 const unsigned int pbo)
: Device(strategy, ordinal, index, count, miss, interop, tex, pbo)
, m_d_compositorData(0)
, m_cudaGraphicsResource(nullptr)
{
  CU_CHECK( cuModuleLoad(&m_moduleCompositor, "./rtigo3_core/compositor.ptx") ); // FIXME Only load this on the primary device!
  CU_CHECK( cuModuleGetFunction(&m_functionCompositor, m_moduleCompositor, "compositor") );
}

DeviceMultiGPULocalCopy::~DeviceMultiGPULocalCopy()
{
  CU_CHECK_NO_THROW( cuCtxSetCurrent(m_cudaContext) );
  CU_CHECK_NO_THROW( cuCtxSynchronize() );

  if (m_cudaGraphicsResource != nullptr)
  {
    CU_CHECK_NO_THROW( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
  }

  if (m_ownsSharedBuffer)
  {
    CU_CHECK_NO_THROW( cuMemFree(m_systemData.outputBuffer) ); 
    CU_CHECK_NO_THROW( cuMemFree(m_d_compositorData) );
  }

  CU_CHECK_NO_THROW( cuModuleUnload(m_moduleCompositor) );
}

void DeviceMultiGPULocalCopy::setState(DeviceState const& state)
{
  if (m_systemData.resolution != state.resolution ||
      m_systemData.tileSize   != state.tileSize)
  {
    // Calculate the new launch width for the tiled rendering.
    // It must be a multiple of the tileSize width, otherwise the right-most tiles will not get filled correctly.
    const int width = (state.resolution.x + m_count - 1) / m_count;
    const int mask  = state.tileSize.x - 1;
    m_launchWidth = (width + mask) & ~mask; // == ((width + (tileSize - 1)) / tileSize.x) * tileSize.x;
  }

  Device::setState(state); // Call the base class to track the state.
}

void DeviceMultiGPULocalCopy::activateContext()
{
  CU_CHECK( cuCtxSetCurrent(m_cudaContext) ); 
}

void DeviceMultiGPULocalCopy::synchronizeStream()
{
  CU_CHECK( cuStreamSynchronize(m_cudaStream) );
}

void DeviceMultiGPULocalCopy::render(const unsigned int iterationIndex, void** buffer)
{
  activateContext();

  m_systemData.iterationIndex = iterationIndex;

  if (m_isDirtyOutputBuffer)
  {
    MY_ASSERT(buffer != nullptr);
    if (*buffer == nullptr) // The buffer is nullptr for the device which is should allocate the full resolution buffers. This device is called first!
    {
      // Only allocate the host buffer on one device.
      m_bufferHost.resize(m_systemData.resolution.x * m_systemData.resolution.y);

      // These are synchronous.
      // Note that this requires that all other devices have finished accessing this buffer, but that is automatically the case
      // after calling Device::setState() which is the only place which can change the resolution.
      CU_CHECK( cuMemFree(m_systemData.outputBuffer) );
      CU_CHECK( cuMemAlloc(&m_systemData.outputBuffer, sizeof(float4) * m_systemData.resolution.x * m_systemData.resolution.y) );

      *buffer = reinterpret_cast<void*>(m_systemData.outputBuffer); // Set the pointer, so that other devices don't allocate it. It's not shared!

      // This is a temporary buffer on the primary board which is used by the compositor. The texelBuffer needs to stay intact for the accumulation.
      CU_CHECK( cuMemFree(m_systemData.tileBuffer) );
      CU_CHECK( cuMemAlloc(&m_systemData.tileBuffer, sizeof(float4) * m_launchWidth * m_systemData.resolution.y) );

      CU_CHECK( cuMemAlloc(&m_d_compositorData, sizeof(CompositorData)) );

      m_ownsSharedBuffer = true; // Indicate which device owns the m_systemData.outputBuffer so that only that frees it again.

      if (m_cudaGraphicsResource != nullptr) // Need to unregister texture or PBO before resizing it.
      {
        CU_CHECK( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
      }

      switch (m_interop)
      {
        case INTEROP_MODE_OFF:
          break;

        case INTEROP_MODE_TEX:
          // Let the device which is called first resize the OpenGL texture.
          glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_systemData.resolution.x, (GLsizei) m_systemData.resolution.y, 0, GL_RGBA, GL_FLOAT, (GLvoid*) m_bufferHost.data()); // RGBA32F
          glFinish(); // Synchronize with following CUDA operations.

          CU_CHECK( cuGraphicsGLRegisterImage(&m_cudaGraphicsResource, m_tex, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) );
          break;

        case INTEROP_MODE_PBO:
          glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
          glBufferData(GL_PIXEL_UNPACK_BUFFER, m_systemData.resolution.x * m_systemData.resolution.y * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
          glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

          CU_CHECK( cuGraphicsGLRegisterBuffer(&m_cudaGraphicsResource, m_pbo, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) ); 
          break;
      }
    }
    // Allocate a GPU local buffer in the per-device launch size. This is where the accumulation happens.
    CU_CHECK( cuMemFree(m_systemData.texelBuffer) );
    CU_CHECK( cuMemAlloc(&m_systemData.texelBuffer, sizeof(float4) * m_launchWidth * m_systemData.resolution.y) );

    m_isDirtyOutputBuffer = false; // Buffer is allocated with new size.
    m_isDirtySystemData   = true;  // Now the sysData on the device needs to be updated, and that needs a sync!
  }

  if (m_isDirtySystemData) // Update the whole SystemData block because more than the iterationIndex changed. This normally means a GUI interaction. Just sync.
  {
    synchronizeStream();

    CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(m_d_systemData), &m_systemData, sizeof(SystemData), m_cudaStream) );
    m_isDirtySystemData = false;
  }
  else // Just copy the new iterationIndex.
  {
    synchronizeStream();

    // FIXME Then for really asynchronous copies of the iteration indices multiple source pointers are required. Good that I know the number of iterations upfront!
    CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(&m_d_systemData->iterationIndex), &m_systemData.iterationIndex, sizeof(unsigned int), m_cudaStream) );
  }

  // Note the launch width per device to render in tiles.
  OPTIX_CHECK( m_api.optixLaunch(m_pipeline, m_cudaStream, reinterpret_cast<CUdeviceptr>(m_d_systemData), sizeof(SystemData), &m_sbt, m_launchWidth, m_systemData.resolution.y, /* depth */ 1) );
}


void DeviceMultiGPULocalCopy::updateDisplayTexture()
{
  activateContext();

  // Only allow this on the device which owns the shared peer-to-peer buffer which also resized the host buffer to copy this to the host.
  MY_ASSERT(!m_isDirtyOutputBuffer && m_ownsSharedBuffer && m_tex != 0);

  switch (m_interop)
  {
    case INTEROP_MODE_OFF:
      // Copy the GPU local render buffer into host and update the HDR texture image from there.
      CU_CHECK( cuMemcpyDtoHAsync(m_bufferHost.data(), m_systemData.outputBuffer, sizeof(float4) * m_systemData.resolution.x * m_systemData.resolution.y, m_cudaStream) );
      synchronizeStream(); // Wait for the buffer to arrive on the host.

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, m_tex);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_systemData.resolution.x, (GLsizei) m_systemData.resolution.y, 0, GL_RGBA, GL_FLOAT, m_bufferHost.data()); // RGBA32F from host buffer data.
      break;
      
    case INTEROP_MODE_TEX:
      {
        // Map the Texture object directly and copy the output buffer. 
        CU_CHECK( cuGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream )); // This is an implicit cuSynchronizeStream().

        CUarray dstArray = nullptr;

        CU_CHECK( cuGraphicsSubResourceGetMappedArray(&dstArray, m_cudaGraphicsResource, 0, 0) ); // arrayIndex = 0, mipLevel = 0

        CUDA_MEMCPY3D params = {};

        params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        params.srcDevice     = m_systemData.outputBuffer;
        params.srcPitch      = m_systemData.resolution.x * sizeof(float4);
        params.srcHeight     = m_systemData.resolution.y;

        params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        params.dstArray      = dstArray;
        params.WidthInBytes  = m_systemData.resolution.x * sizeof(float4);
        params.Height        = m_systemData.resolution.y;
        params.Depth         = 1;

        CU_CHECK( cuMemcpy3D(&params) ); // Copy from linear to array layout.

        CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
      }
      break;

    case INTEROP_MODE_PBO: // This contains two device-to-device copies and is just for demonstration. Use INTEROP_MODE_TEX when possible.
      {
        size_t size = 0;
        CUdeviceptr d_ptr;
  
        CU_CHECK( cuGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
        CU_CHECK( cuGraphicsResourceGetMappedPointer(&d_ptr, &size, m_cudaGraphicsResource) ); // The pointer can change on every map!
        MY_ASSERT(m_systemData.resolution.x * m_systemData.resolution.y * sizeof(float4) <= size);
        CU_CHECK( cuMemcpyDtoDAsync(d_ptr, m_systemData.outputBuffer, m_systemData.resolution.x * m_systemData.resolution.y * sizeof(float4), m_cudaStream) ); // PERF PBO interop is kind of moot with a direct texture access.
        CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_tex);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_systemData.resolution.x, (GLsizei) m_systemData.resolution.y, 0, GL_RGBA, GL_FLOAT, (GLvoid*) 0); // RGBA32F from byte offset 0 in the pixel unpack buffer.
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      }
      break;
  }
}


const void* DeviceMultiGPULocalCopy::getOutputBufferHost()
{
  activateContext();

  MY_ASSERT(!m_isDirtyOutputBuffer && m_ownsSharedBuffer); // Only allow this on the device which owns the shared peer-to-peer buffer and resized the host buffer to copy this to the host.
  
  // Note that the caller takes care to sync the other devices before calling into here or this image might not be complete!
  CU_CHECK( cuMemcpyDtoHAsync(m_bufferHost.data(), m_systemData.outputBuffer, sizeof(float4) * m_systemData.resolution.x * m_systemData.resolution.y, m_cudaStream) );
    
  synchronizeStream(); // Wait for the buffer to arrive on the host.

  return m_bufferHost.data();
}


void DeviceMultiGPULocalCopy::compositor(Device* other)
{
  MY_ASSERT(!m_isDirtyOutputBuffer && m_ownsSharedBuffer);

  // The compositor sources the tileBuffer, which is only allocated on the primary device. 
  // The texelBuffer is a GPU local buffer on all devices and contains the accumulation.
  if (this == other)
  {
    activateContext();

    CU_CHECK( cuMemcpyDtoDAsync(m_systemData.tileBuffer, m_systemData.texelBuffer,
                                sizeof(float4) * m_launchWidth * m_systemData.resolution.y, m_cudaStream) );
  }
  else
  {
    // Make sure the other device has finished rendering! Otherwise there can be checkerboard corruption visible.
    other->activateContext();
    other->synchronizeStream();
  
    activateContext();

    CU_CHECK( cuMemcpyPeerAsync(m_systemData.tileBuffer, m_cudaContext, other->m_systemData.texelBuffer, other->m_cudaContext,
                                sizeof(float4) * m_launchWidth * m_systemData.resolution.y, m_cudaStream) );
  }

  CompositorData compositorData; // DAR FIXME This needs to be persistent per Device to allow async copies!

  compositorData.outputBuffer = m_systemData.outputBuffer;
  compositorData.tileBuffer   = m_systemData.tileBuffer;
  compositorData.resolution   = m_systemData.resolution;
  compositorData.tileSize     = m_systemData.tileSize;
  compositorData.tileShift    = m_systemData.tileShift;
  compositorData.launchWidth  = m_launchWidth;
  compositorData.deviceCount  = m_systemData.deviceCount;
  compositorData.deviceIndex  = other->m_systemData.deviceIndex; // This is the only value which changes per device. 

  // Need a synchronous copy here to not overwrite or delete the compositorData above.
  CU_CHECK( cuMemcpyHtoD(m_d_compositorData, &compositorData, sizeof(CompositorData)) );
 
  void* args[1] = { &m_d_compositorData };

  const int blockDimX = std::min(compositorData.tileSize.x, 16);
  const int blockDimY = std::min(compositorData.tileSize.y, 16);

  const int gridDimX  = (m_launchWidth               + blockDimX - 1) / blockDimX;
  const int gridDimY  = (compositorData.resolution.y + blockDimY - 1) / blockDimY;

  MY_ASSERT(gridDimX <= m_deviceAttribute.maxGridDimX && 
            gridDimY <= m_deviceAttribute.maxGridDimY);

  // Reduction kernel with launch dimension of height blocks with 32 threads.
  CU_CHECK( cuLaunchKernel(m_functionCompositor,    // CUfunction f,
                                       gridDimX,    // unsigned int gridDimX,
                                       gridDimY,    // unsigned int gridDimY,
                                              1,    // unsigned int gridDimZ,
                                      blockDimX,    // unsigned int blockDimX,
                                      blockDimY,    // unsigned int blockDimY,
                                              1,    // unsigned int blockDimZ,
                                              0,    // unsigned int sharedMemBytes,
                                   m_cudaStream,    // CUstream hStream,
                                           args,    // void **kernelParams,
                                        nullptr) ); // void **extra

  synchronizeStream();
}
