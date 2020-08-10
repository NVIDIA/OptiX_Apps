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

#include "inc/DeviceMultiGPUZeroCopy.h"

#include "inc/CheckMacros.h"

#include <GL/glew.h>
#if defined( _WIN32 )
#include <GL/wglew.h>
#endif

DeviceMultiGPUZeroCopy::DeviceMultiGPUZeroCopy(const RendererStrategy strategy,
                                               const int ordinal,
                                               const int index,
                                               const int count,
                                               const int miss,
                                               const int interop,
                                               const unsigned int tex,
                                               const unsigned int pbo)
: Device(strategy, ordinal, index, count, miss, interop, tex, pbo)
{
  if (m_deviceAttribute.canMapHostMemory == 0)
  {
    std::cout << "ERROR: DeviceMultiGPUZeroCopy() Device ordinal " << ordinal << " canMapHostMemory attribute is false.\n";
  }
}

DeviceMultiGPUZeroCopy::~DeviceMultiGPUZeroCopy()
{
  CU_CHECK_NO_THROW( cuCtxSetCurrent(m_cudaContext) );
  CU_CHECK_NO_THROW( cuCtxSynchronize() );

  if (m_ownsSharedBuffer) // This destruction order requires that all other devices cannot touch this shared buffer anymore.
  {
    CU_CHECK_NO_THROW( cuCtxSetCurrent(m_cudaContext) ); 

    CU_CHECK_NO_THROW( cuMemFreeHost(reinterpret_cast<void*>(m_systemData.outputBuffer)) );
  }
}

void DeviceMultiGPUZeroCopy::setState(DeviceState const& state)
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

void DeviceMultiGPUZeroCopy::activateContext()
{
  CU_CHECK( cuCtxSetCurrent(m_cudaContext) ); 
}

void DeviceMultiGPUZeroCopy::synchronizeStream()
{
  CU_CHECK( cuStreamSynchronize(m_cudaStream) );
}

void DeviceMultiGPUZeroCopy::render(const unsigned int iterationIndex, void** buffer)
{
  activateContext();

  m_systemData.iterationIndex = iterationIndex;

  if (m_isDirtyOutputBuffer)
  {
    MY_ASSERT(buffer != nullptr);
    if (*buffer == nullptr) // The first device called handles the reallocation of the shared pinned memory buffer.
    {
      // Allocate zero-copy pinned memory on the host.
      CU_CHECK( cuMemFreeHost(reinterpret_cast<void*>(m_systemData.outputBuffer)) );
      CU_CHECK( cuMemHostAlloc(reinterpret_cast<void**>(&m_systemData.outputBuffer), sizeof(float4) * m_systemData.resolution.x * m_systemData.resolution.y, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP) );
      
      *buffer = reinterpret_cast<void*>(m_systemData.outputBuffer); // Fill the shared buffer pointer.

      m_ownsSharedBuffer = true; // This device will destruct it.
    }
    else
    {
      // Use the same zero copy pinned memory buffer for all devices. 
      // m_systemData.outputBuffer = reinterpret_cast<CUdeviceptr>(*buffer); 
      // This call results in the same pointer because of CU_MEMHOSTALLOC_PORTABLE.
      CU_CHECK( cuMemHostGetDevicePointer(&m_systemData.outputBuffer, *buffer, 0) ); 
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

  // Note the launch width per device to render in tiles.
  OPTIX_CHECK( m_api.optixLaunch(m_pipeline, m_cudaStream, reinterpret_cast<CUdeviceptr>(m_d_systemData), sizeof(SystemData), &m_sbt, m_launchWidth, m_systemData.resolution.y, /* depth */ 1) );
}

void DeviceMultiGPUZeroCopy::updateDisplayTexture()
{
  // All other devices have been synced by the RaytracerMultiGPUZeroCopy caller.
  activateContext();
  synchronizeStream(); // Wait for the buffer to arrive on the host. 
  
  MY_ASSERT(!m_isDirtyOutputBuffer && m_ownsSharedBuffer && m_tex != 0);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_tex);

  // RGBA32F from shared pinned memory host buffer data.
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_systemData.resolution.x, (GLsizei) m_systemData.resolution.y, 0, GL_RGBA, GL_FLOAT, reinterpret_cast<GLvoid*>(m_systemData.outputBuffer));
}

const void* DeviceMultiGPUZeroCopy::getOutputBufferHost()
{
  // All other devices have been synced by the RaytracerMultiGPUZeroCopy caller.
  activateContext();
  synchronizeStream(); // Wait for the buffer to arrive on the host. 

  MY_ASSERT(!m_isDirtyOutputBuffer && m_ownsSharedBuffer);

  return reinterpret_cast<void*>(m_systemData.outputBuffer); // This buffer is in pinned memory on the host. Just return it.
}
