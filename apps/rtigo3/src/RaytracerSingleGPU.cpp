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

#include "inc/RaytracerSingleGPU.h"

#include "inc/CheckMacros.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>

RaytracerSingleGPU::RaytracerSingleGPU(const int devicesMask, 
                                       const int miss, 
                                       const int interop,
                                       const unsigned int tex,
                                       const unsigned int pbo)
: Raytracer(RS_INTERACTIVE_SINGLE_GPU, interop, tex, pbo)
{
  int ordinal = 0;
  while (ordinal < m_visibleDevices) // Don't try to enable more devices than visible.
  {
    unsigned int mask = (1 << ordinal);
    if (devicesMask & mask)
    {
      DeviceSingleGPU* device = new DeviceSingleGPU(m_strategy, ordinal, 0, 1, miss, interop, tex, pbo); // Hardcoded device index 0 and count 1.
      
      m_activeDevices.push_back(device);

      m_activeDevicesMask |= mask; // Track which device has actually been enabled.

      std::cout << "RaytracerSingleGPU() Using device " << ordinal << ": " << device->m_deviceName << '\n';
      
      break; // Stop after the first device, since this RaytracerSingleGPU.
    }
    ++ordinal;
  }

  m_isValid = !m_activeDevices.empty();
}


void RaytracerSingleGPU::updateDisplayTexture()
{
  m_activeDevices[0]->updateDisplayTexture();
}


// The public function which does the multi-GPU wrapping.
// Returns the count of renderered iterations (m_iterationIndex after it has been incremented.
unsigned int RaytracerSingleGPU::render()
{
  // Continue manual accumulation rendering if the samples per pixel have not been reached.
  if (m_iterationIndex < m_samplesPerPixel)
  {
    m_activeDevices[0]->render(m_iterationIndex, nullptr); // Only one device in this implementation.

    ++m_iterationIndex;
  }  

  return m_iterationIndex;
}

const void* RaytracerSingleGPU::getOutputBufferHost() // Not called when using OpenGL interop.
{
  return m_activeDevices[0]->getOutputBufferHost(); // Only one device in this implementation.
}

