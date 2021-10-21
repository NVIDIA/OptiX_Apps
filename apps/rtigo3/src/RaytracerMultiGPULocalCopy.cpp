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

#include "inc/RaytracerMultiGPULocalCopy.h"

#include "inc/CheckMacros.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>

RaytracerMultiGPULocalCopy::RaytracerMultiGPULocalCopy(const int devicesMask,
                                                       const int miss,
                                                       const int interop,
                                                       const unsigned int tex,
                                                       const unsigned int pbo)
: Raytracer(RS_INTERACTIVE_MULTI_GPU_LOCAL_COPY, interop, tex, pbo)
{
  int count   = 0; // Need to determine the number of active devices first to have it available as constructor argument.
  int ordinal = 0;
  while (ordinal < m_visibleDevices) // Don't try to enable more devices than visible.
  {
    unsigned int mask = (1 << ordinal);
    if (devicesMask & mask)
    {
      // Track which and how many devices have actually been enabled.
      m_activeDevicesMask |= mask; 
      ++count;
    }
    ++ordinal;
  }

  // Now really construct the Device objects. 
  ordinal = 0;
  while (ordinal < m_visibleDevices)
  {
    unsigned int mask = (1 << ordinal);
    if (m_activeDevicesMask & mask)
    {
      const int index = static_cast<int>(m_activeDevices.size());

      DeviceMultiGPULocalCopy* device = new DeviceMultiGPULocalCopy(m_strategy, ordinal, index, count, miss, interop, tex, pbo);

      m_activeDevices.push_back(device);

      std::cout << "RaytracerMultiGPULocalCopy() Using device " << ordinal << ": " << device->m_deviceName << '\n';
    }
    ++ordinal;
  }

  // RS_INTERACTIVE_MULTI_GPU_LOCAL_COPY doesn't strictly require NVLINK peer-to-peer access.
  // It's using cuMemcpyPeer to copy the data to the device doing the compositing and will work either way,
  // just that copy gets faster if it can be done via an NVLINK connection in a peer-to-peer island.
  (void) enablePeerAccess();

  m_isValid = !m_activeDevices.empty();
}

RaytracerMultiGPULocalCopy::~RaytracerMultiGPULocalCopy()
{
  try
  {
    // This function contains throw() calls.
    disablePeerAccess(); // Just for cleanliness, the Devices are destroyed anyway after this.
  }
  catch (std::exception const& e)
  {
    std::cerr << e.what() << '\n';
  }
}


// The public function which does the multi-GPU wrapping.
// Returns the count of renderered iterations (m_iterationIndex after it has been incremented.
unsigned int RaytracerMultiGPULocalCopy::render()
{
  // Continue manual accumulation rendering if the samples per pixel have not been reached.
  if (m_iterationIndex < m_samplesPerPixel)
  {
    void* buffer = nullptr;
    
    // Make sure the OpenGL device is allocating the full resolution backing storage.
    if (m_deviceOGL != -1)
    {
      // This is the device which needs to allocate the peer-to-peer buffer to reside on the same device as the PBO or Texture
      m_activeDevices[m_deviceOGL]->render(m_iterationIndex, &buffer); // Interactive rendering. All devices work on the same iteration index.
    }

    for (size_t i = 0; i < m_activeDevices.size(); ++i)
    {
      if (m_deviceOGL != i)
      {
        // If buffer is still nullptr here, the first device will allocate the full resolution buffer.
        m_activeDevices[i]->render(m_iterationIndex, &buffer);
      }
    }
    
    ++m_iterationIndex;
  }  
  return m_iterationIndex;
}

void RaytracerMultiGPULocalCopy::updateDisplayTexture()
{
  const int index = (m_deviceOGL != -1) ? m_deviceOGL : 0; // Destination device.

  // First, copy the texelBuffer of the primary device into its tileBuffer and then place the tiles into the outputBuffer.
  m_activeDevices[index]->compositor(m_activeDevices[index]);

  // Now copy the other devices' texelBuffers over to the main tileBuffer and repeat the compositing for that other device.
  // The cuMemcpyPeerAsync done in that case is fast when the devices are in the same peer island, otherwise it's copied via PCI-E, but only N-1 copies of 1/N size are done
  // The saving here is no peer-to-peer read-modify-write when rendering, because everything is happening in GPU local buffers, which are also tightly packed.
  // The final compositing is just a kernel implementing a tiled memcpy. 
  // PERF If all tiles are copied to the main device at once, such kernel would only need to be called once.
  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    if (index != static_cast<int>(i))
    {
      m_activeDevices[index]->compositor(m_activeDevices[i]);
    }
  }

  // Finally copy the primary device outputBuffer to the display texture. 
  // FIXME DEBUG Does that work when m_deviceOGL is not in the list of active devices?
  m_activeDevices[index]->updateDisplayTexture();
}

const void* RaytracerMultiGPULocalCopy::getOutputBufferHost()
{
  // Same initial steps to fill the outputBuffer on the primary device as in updateDisplayTexture() 
  const int index = (m_deviceOGL != -1) ? m_deviceOGL : 0; // Destination device.

  // First, copy the texelBuffer of the primary device into its tileBuffer and then place the tiles into the outputBuffer.
  m_activeDevices[index]->compositor(m_activeDevices[index]);

  // Now copy the other devices' texelBuffers over to the main tileBuffer and repeat the compositing for that other device.
  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    if (index != static_cast<int>(i))
    {
      m_activeDevices[index]->compositor(m_activeDevices[i]);
    }
  }
  
  // The full outputBuffer resides on device "index" and the host buffer is also only resized by that device.
  return m_activeDevices[index]->getOutputBufferHost();
}
