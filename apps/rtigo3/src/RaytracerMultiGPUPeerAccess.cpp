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

#include "inc/RaytracerMultiGPUPeerAccess.h"

#include "inc/CheckMacros.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>

RaytracerMultiGPUPeerAccess::RaytracerMultiGPUPeerAccess(const int devicesMask,
                                                         const int miss,
                                                         const int interop,
                                                         const unsigned int tex,
                                                         const unsigned int pbo)
: Raytracer(RS_INTERACTIVE_MULTI_GPU_PEER_ACCESS, interop, tex, pbo)
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

      DeviceMultiGPUPeerAccess* device = new DeviceMultiGPUPeerAccess(m_strategy, ordinal, index, count, miss, interop, tex, pbo);

      m_activeDevices.push_back(device);

      std::cout << "RaytracerMultiGPUPeerAccess() Using device " << ordinal << ": " << device->m_deviceName << '\n';
    }
    ++ordinal;
  }

  // RS_INTERACTIVE_MULTI_GPU_PEER_ACCESS strategy is not supported for more than one peer-to-peer island.
  m_isValid = (!m_activeDevices.empty() && enablePeerAccess() && m_islands.size() == 1);

  if (m_islands.size() != 1)
  {
    std::cerr << "ERROR: enablePeerAccess() RS_INTERACTIVE_MULTI_GPU_PEER_ACCESS strategy is only supported with one peer-to-peer island.\n";
  }
}

RaytracerMultiGPUPeerAccess::~RaytracerMultiGPUPeerAccess()
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
unsigned int RaytracerMultiGPUPeerAccess::render()
{
  // Continue manual accumulation rendering if the samples per pixel have not been reached.
  if (m_iterationIndex < m_samplesPerPixel)
  {
    // This pointer is used to communicate the shared peer-to-peer memory pointer between devices.
    // The first device allocates it when dirty and returns the pointer, all others reuse the same address on the device.
    void *bufferPeer = nullptr; 

    // If there is OpenGL interop and the active devices contain the device running the OpenGL implementation.
    if (m_deviceOGL != -1)
    {
      // This is the device which needs to allocate the peer-to-peer buffer to reside on the same device as the PBO.
      m_activeDevices[m_deviceOGL]->render(m_iterationIndex, &bufferPeer); // Interactive rendering. All devices work on the same iteration index.
    }

    // This works for all cases, including m_deviceOGL == -1; 
    const int size = static_cast<int>(m_activeDevices.size());

    for (int i = 0; i < size; ++i)
    {
      if (i != m_deviceOGL) // Call other devices. This works for the case m_deviceOGL == -1 as well.
      {
        m_activeDevices[i]->render(m_iterationIndex, &bufferPeer); // Interactive rendering. All devices work on the same iteration index.
      }
    }

    ++m_iterationIndex;
  }  
  return m_iterationIndex;
}

void RaytracerMultiGPUPeerAccess::updateDisplayTexture()
{
  // Sync all devices before getting the buffer from the device owning the shared buffer.
  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    m_activeDevices[i]->activateContext();
    m_activeDevices[i]->synchronizeStream();
  }

  // If there is OpenGL interop and the active devices contain the device running the OpenGL implementation
  // that device has allocated the shared buffer, otherwise the first active device.
  const int index = (m_deviceOGL != -1) ? m_deviceOGL : 0;

  m_activeDevices[index]->updateDisplayTexture();
}

const void* RaytracerMultiGPUPeerAccess::getOutputBufferHost()
{
  // Sync all devices before getting the buffer from the device owning the shared buffer.
  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    m_activeDevices[i]->activateContext();
    m_activeDevices[i]->synchronizeStream();
  }

  // If there is OpenGL interop and the active devices contain the device running the OpenGL implementation
  // that device has allocated the shared buffer, otherwise the first active device.
  const int index = (m_deviceOGL != -1) ? m_deviceOGL : 0;
  
  // The shared peer-to-peer buffer resides on device "index" and the host buffer is also only resized by that device.
  return m_activeDevices[index]->getOutputBufferHost();
}
