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

#include "inc/RaytracerMultiGPUZeroCopy.h"

#include "inc/CheckMacros.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>

RaytracerMultiGPUZeroCopy::RaytracerMultiGPUZeroCopy(const int devicesMask,
                                                     const int miss,
                                                     const int interop,
                                                     const unsigned int tex,
                                                     const unsigned int pbo)
: Raytracer(RS_INTERACTIVE_MULTI_GPU_ZERO_COPY, interop, tex, pbo)
{
  if (interop != INTEROP_MODE_OFF)
  {
    std::cout << "WARNING: RaytracerMultiGPUZeroCopy() doesn't implement OpenGL interop. The output buffer resides on the host.\n";
  }

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

      DeviceMultiGPUZeroCopy* device = new DeviceMultiGPUZeroCopy(m_strategy, ordinal, index, count, miss, interop, tex, pbo);

      m_activeDevices.push_back(device);

      std::cout << "RaytracerMultiGPUZeroCopy() Using device " << ordinal << ": " << device->m_deviceName << '\n';
    }
    ++ordinal;
  }


  m_isValid = !m_activeDevices.empty();
}


// The public function which does the multi-GPU wrapping.
// Returns the count of renderered iterations (m_iterationIndex after it has been incremented.
unsigned int RaytracerMultiGPUZeroCopy::render()
{
  // Continue manual accumulation rendering if the samples per pixel have not been reached.
  if (m_iterationIndex < m_samplesPerPixel)
  {
    // This pointer is used to communicate the shared pinned memory pointer between devices.
    // The first device calls allocates it when dirty and returns the pointer, all others reuse the same address on the host
    void *bufferZeroCopy = nullptr; 

    for (size_t i = 0; i < m_activeDevices.size(); ++i)
    {
      m_activeDevices[i]->render(m_iterationIndex, &bufferZeroCopy); // Interactive rendering. All devices work on the same iteration index.
    }
    ++m_iterationIndex;
  }  
  return m_iterationIndex;
}

void RaytracerMultiGPUZeroCopy::updateDisplayTexture()
{
  // Finish rendering on all other devices before accessing the shared pinned memory buffer.
  for (size_t i = 1; i < m_activeDevices.size(); ++i)
  {
    m_activeDevices[i]->activateContext();
    m_activeDevices[i]->synchronizeStream();
  }

  m_activeDevices[0]->updateDisplayTexture();
}

const void* RaytracerMultiGPUZeroCopy::getOutputBufferHost()
{
  // Finish rendering on all other devices before accessing the shared pinned memory buffer.
  for (size_t i = 1; i < m_activeDevices.size(); ++i)
  {
    m_activeDevices[i]->activateContext();
    m_activeDevices[i]->synchronizeStream();
  }

  return m_activeDevices[0]->getOutputBufferHost();
}
