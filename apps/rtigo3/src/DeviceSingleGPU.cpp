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

#include "inc/DeviceSingleGPU.h"

#include "inc/CheckMacros.h"

#include <GL/glew.h>
#if defined( _WIN32 )
#include <GL/wglew.h>
#endif

// CUDA Driver API version of the OpenGL interop header. 
#include <cudaGL.h>

#include <string.h>

DeviceSingleGPU::DeviceSingleGPU(const RendererStrategy strategy,
                                 const int ordinal,
                                 const int index,
                                 const int count,
                                 const int miss,
                                 const int interop,
                                 const unsigned int tex,
                                 const unsigned int pbo)
: Device(strategy, ordinal, index, count, miss, interop, tex, pbo)
, m_cudaGraphicsResource(nullptr)
{
  switch (m_interop) // Just keep interop resources registered to the single active device.
  {
    case INTEROP_MODE_OFF:
      break;
    case INTEROP_MODE_TEX:
      MY_ASSERT(m_tex != 0);
      CU_CHECK( cuGraphicsGLRegisterImage(&m_cudaGraphicsResource, m_tex, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) );
      break;
    case INTEROP_MODE_PBO:
      MY_ASSERT(m_pbo != 0);
      CU_CHECK( cuGraphicsGLRegisterBuffer(&m_cudaGraphicsResource, m_pbo, CU_GRAPHICS_REGISTER_FLAGS_NONE) );
      break;
  }
}

DeviceSingleGPU::~DeviceSingleGPU()
{
  //CU_CHECK_NO_THROW( cuCtxSetCurrent(m_cudaContext) ); // Redundant because there is only one device in this strategy.
  CU_CHECK_NO_THROW( cuCtxSynchronize() );

  switch (m_interop)
  {
    case INTEROP_MODE_OFF:
      CU_CHECK_NO_THROW( cuMemFree(m_systemData.outputBuffer) ); 
      break;
    case INTEROP_MODE_TEX:
      CU_CHECK_NO_THROW( cuMemFree(m_systemData.outputBuffer) ); 
      CU_CHECK_NO_THROW( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
      break;
    case INTEROP_MODE_PBO:
      CU_CHECK_NO_THROW( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
      break;
  }
}


void DeviceSingleGPU::activateContext()
{
  // Persistent CUDA context in single GPU renderers! No need to set it current after initial creation.
  // CU_CHECK( cuCtxSetCurrent(m_cudaContext) ); 
}

void DeviceSingleGPU::synchronizeStream()
{
  CU_CHECK( cuStreamSynchronize(m_cudaStream) );
}

void DeviceSingleGPU::render(const unsigned int iterationIndex, void** /* buffer */)
{
  // activateContext();

  m_systemData.iterationIndex = iterationIndex;

  if (m_isDirtyOutputBuffer)
  {
    // Required for getOutputBufferHost() which is still called in the screenshot() function.
    m_bufferHost.resize(m_systemData.resolution.x * m_systemData.resolution.y); 

    switch (m_interop)
    {
      case INTEROP_MODE_OFF:
        CU_CHECK( cuMemFree(m_systemData.outputBuffer) );
        CU_CHECK( cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&m_systemData.outputBuffer), sizeof(float4) * m_systemData.resolution.x * m_systemData.resolution.y) );
        break;

      case INTEROP_MODE_TEX:
        CU_CHECK( cuMemFree(m_systemData.outputBuffer) );
        CU_CHECK( cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&m_systemData.outputBuffer), sizeof(float4) * m_systemData.resolution.x * m_systemData.resolution.y) );
        // Resize the target texture as well for the cuMemcpy3D.
        CU_CHECK( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_systemData.resolution.x, (GLsizei) m_systemData.resolution.y, 0, GL_RGBA, GL_FLOAT, (GLvoid*) m_bufferHost.data()); // RGBA32F
        glFinish();
        CU_CHECK( cuGraphicsGLRegisterImage(&m_cudaGraphicsResource, m_tex, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) );
        break;

      case INTEROP_MODE_PBO:
        CU_CHECK( cuGraphicsUnregisterResource(m_cudaGraphicsResource) ); // No flags for read-write access during accumulation.
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, m_systemData.resolution.x * m_systemData.resolution.y * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        CU_CHECK( cuGraphicsGLRegisterBuffer(&m_cudaGraphicsResource, m_pbo, CU_GRAPHICS_REGISTER_FLAGS_NONE) );
        break;
    }

    m_isDirtyOutputBuffer = false; // Buffer is allocated with new size,
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
    synchronizeStream(); // FIXME For some render strategy "final frame" there should be no synchronizeStream() at all here.

    // FIXME Then for really asynchronous copies of the iteration indices multiple source pointers are required. Good that I know the number of iterations upfront!
    CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(&m_d_systemData->iterationIndex), &m_systemData.iterationIndex, sizeof(unsigned int), m_cudaStream) );
  }

  switch (m_interop)
  {
    case INTEROP_MODE_OFF:
    case INTEROP_MODE_TEX:
      OPTIX_CHECK( m_api.optixLaunch(m_pipeline, m_cudaStream, reinterpret_cast<CUdeviceptr>(m_d_systemData), sizeof(SystemData), &m_sbt, m_systemData.resolution.x, m_systemData.resolution.y, /* depth */ 1) );
      break;

    case INTEROP_MODE_PBO: // Rendering directly into the PBO.
      {
        size_t size;

        CU_CHECK( cuGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
    
        CU_CHECK( cuGraphicsResourceGetMappedPointer(&m_systemData.outputBuffer, &size, m_cudaGraphicsResource) ); // The pointer can change on every map!
        CU_CHECK( cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(&m_d_systemData->outputBuffer), &m_systemData.outputBuffer, sizeof(void*), m_cudaStream) ); // This will render directly into the PBO.

        OPTIX_CHECK( m_api.optixLaunch(m_pipeline, m_cudaStream, reinterpret_cast<CUdeviceptr>(m_d_systemData), sizeof(SystemData), &m_sbt, m_systemData.resolution.x, m_systemData.resolution.y, /* depth */ 1) );
      
        CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
      }
      break;
  }
}

void DeviceSingleGPU::updateDisplayTexture()
{
  // activateContext();
  
  MY_ASSERT(!m_isDirtyOutputBuffer && m_tex != 0);

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
        // Map the Texture object directly and copy the output buffer 
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

        CU_CHECK( cuMemcpy3D(&params) ); 

        CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
      }
      break;

    case INTEROP_MODE_PBO:
      synchronizeStream(); // Wait for rendering to finish writing into the PBO.

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, m_tex);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_systemData.resolution.x, (GLsizei) m_systemData.resolution.y, 0, GL_RGBA, GL_FLOAT, (GLvoid*) 0); // RGBA32F from byte offset 0 in the pixel unpack buffer.
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      break;
  }
}

const void* DeviceSingleGPU::getOutputBufferHost() 
{
  // activateContext();
  
  MY_ASSERT(!m_isDirtyOutputBuffer);

  switch (m_interop)
  {
    case INTEROP_MODE_OFF:
    case INTEROP_MODE_TEX:
      CU_CHECK( cuMemcpyDtoHAsync(m_bufferHost.data(), m_systemData.outputBuffer, sizeof(float4) * m_systemData.resolution.x * m_systemData.resolution.y, m_cudaStream) );
      synchronizeStream(); // Wait for the buffer to arrive on the host. Context is created with CU_CTX_SCHED_SPIN.
      break;

    case INTEROP_MODE_PBO:
      {
        size_t size;

        CU_CHECK( cuGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
        CU_CHECK( cuGraphicsResourceGetMappedPointer(&m_systemData.outputBuffer, &size, m_cudaGraphicsResource) ); // The pointer can change on every map!
        CU_CHECK( cuMemcpyDtoHAsync(m_bufferHost.data(), m_systemData.outputBuffer, sizeof(float4) * m_systemData.resolution.x * m_systemData.resolution.y, m_cudaStream) );
        CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
      }
      break;
  }

  return m_bufferHost.data();
}
