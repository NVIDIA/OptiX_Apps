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

#include "inc/Raytracer.h"

#include "inc/CheckMacros.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>

Raytracer::Raytracer(RendererStrategy strategy,
                     const int interop,
                     const unsigned int tex,
                     const unsigned int pbo)
: m_strategy(strategy)
, m_interop(interop)
, m_tex(tex)
, m_pbo(pbo)
, m_isValid(false)
, m_visibleDevices(0)
, m_deviceOGL(-1)
, m_activeDevicesMask(0)
, m_iterationIndex(0)
, m_samplesPerPixel(1)
{
  CU_CHECK( cuInit(0) ); // Initialize CUDA driver API.

  int versionDriver = 0;
  CU_CHECK( cuDriverGetVersion(&versionDriver) ); 
  
  // The version is returned as (1000 * major + 10 * minor).
  int major =  versionDriver / 1000;
  int minor = (versionDriver - 1000 * major) / 10;
  std::cout << "Driver Version = " << major << "." << minor << '\n';
  
  CU_CHECK( cuDeviceGetCount(&m_visibleDevices) );
  std::cout << "Device Count   = " << m_visibleDevices << '\n';
}

Raytracer::~Raytracer()
{
  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    delete m_activeDevices[i];
  }
}

int Raytracer::matchUUID(const char* uuid)
{
  const int size = static_cast<int>(m_activeDevices.size());

  for (int i = 0; i < size; ++i)
  {
    if (m_activeDevices[i]->matchUUID(uuid))
    {
      // Use the first device which matches with the OpenGL UUID.
      // DEBUG This might not be the right thing to do with multicast enabled.
      m_deviceOGL = i;
      break;
    }
  }
  return m_deviceOGL; // If this stays -1, the active devices do not contain the one running the OpenGL implementation.
}

int Raytracer::matchLUID(const char* luid, const unsigned int nodeMask)
{
  const int size = static_cast<int>(m_activeDevices.size());

  for (int i = 0; i < size; ++i)
  {
    if (m_activeDevices[i]->matchLUID(luid, nodeMask))
    {
      // Use the first device which matches with the OpenGL LUID and test of the node mask bit.
      // DEBUG This might not be the right thing to do with multicast enabled.
      m_deviceOGL = i;
      break;
    }
  }
  return m_deviceOGL; // If this stays -1, the active devices do not contain the one running the OpenGL implementation.
}

int Raytracer::findActiveDevice(const unsigned int domain, const unsigned int bus, const unsigned int device) const
{
  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    const DeviceAttribute& attribute = m_activeDevices[i]->m_deviceAttribute;

    if (attribute.pciDomainId == domain &&
        attribute.pciBusId    == bus    && 
        attribute.pciDeviceId == device)
    {
      return static_cast<int>(i);
    }
  }

  return -1;
}


bool Raytracer::activeNVLINK(const int home, const int peer) const
{
  // All NVML calls related to NVLINK are only supported by Pascal (SM 6.0) and newer.
  if (m_activeDevices[home]->m_deviceAttribute.computeCapabilityMajor < 6)
  {
    return false;
  }

  nvmlDevice_t deviceHome;

  if (m_nvml.m_api.nvmlDeviceGetHandleByPciBusId(m_activeDevices[home]->m_devicePciBusId.c_str(), &deviceHome) != NVML_SUCCESS)
  {
    return false;
  }

  // The NVML deviceHome is part of the active devices at index "home".
  for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; ++link)
  {
    // First check if this link is supported at all and if it's active.
    nvmlEnableState_t enableState = NVML_FEATURE_DISABLED;

    if (m_nvml.m_api.nvmlDeviceGetNvLinkState(deviceHome, link, &enableState) != NVML_SUCCESS)
    {
      continue;
    }
    if (enableState != NVML_FEATURE_ENABLED)
    {
      continue;
    }

    // Is peer-to-peer over NVLINK supported by this link?
    // The requirement for peer-to-peer over NVLINK under Windows is Windows 10 (WDDM2), 64-bit, SLI enabled.
    unsigned int capP2P = 0;

    if (m_nvml.m_api.nvmlDeviceGetNvLinkCapability(deviceHome, link, NVML_NVLINK_CAP_P2P_SUPPORTED, &capP2P) != NVML_SUCCESS)
    {
      continue;
    }
    if (capP2P == 0)
    {
      continue;
    }

    nvmlPciInfo_t pciInfoPeer;

    if (m_nvml.m_api.nvmlDeviceGetNvLinkRemotePciInfo(deviceHome, link, &pciInfoPeer) != NVML_SUCCESS)
    {
      continue;
    }

    // Check if the NVML remote device matches the desired peer devcice.
    if (peer == findActiveDevice(pciInfoPeer.domain, pciInfoPeer.bus, pciInfoPeer.device))
    {
      return true;
    }
  }
  
  return false;
}


bool Raytracer::enablePeerAccess()
{
  bool success = true;

  // Build a peer-to-peer connection matrix which only allows peer-to-peer access over NVLINK bridges.
  const int size = static_cast<int>(m_activeDevices.size());
  MY_ASSERT(size <= 32);
    
  // Peer-to-peer access is encoded in a bitfield of uint32 entries. 
  // Indexed by [home] device and peer devices are the bit indices, accessed with (1 << peer) masks. 
  m_peerConnections.resize(size);

  // Initialize the connection matrix diagonal with the trivial case (home == peer).
  // This let's building the islands still work if there are any exceptions.
  for (int home = 0; home < size; ++home)
  {
    m_peerConnections[home] = (1 << home);
  }

  // The NVML_CHECK and CU_CHECK macros can throw exceptions.
  // Keep them local in this routine because not having NVLINK islands with peer-to-peer access
  // is not a fatal condition for the renderer. It just won't be able to share resources.
  try 
  {
    if (m_nvml.initFunctionTable())
    {
      NVML_CHECK( m_nvml.m_api.nvmlInit() );

      for (int home = 0; home < size; ++home) // Home device index.
      {
        for (int peer = 0; peer < size; ++peer) // Peer device index.
        {
          if (home != peer && activeNVLINK(home, peer))
          {
            int canAccessPeer = 0;

            // This requires the ordinals of the visible CUDA devices!
            CU_CHECK( cuDeviceCanAccessPeer(&canAccessPeer,
                                            (CUdevice) m_activeDevices[home]->m_ordinal,    // If this current home context
                                            (CUdevice) m_activeDevices[peer]->m_ordinal) ); // can access the peer context's memory.
            if (canAccessPeer != 0)
            {
              // Note that this function changes the current context!
              CU_CHECK( cuCtxSetCurrent(m_activeDevices[home]->m_cudaContext) );

              CUresult result = cuCtxEnablePeerAccess(m_activeDevices[peer]->m_cudaContext, 0);  // Flags must be 0!
              if (result == CUDA_SUCCESS)
              {
                m_peerConnections[home] |= (1 << peer); // Set the connection bit if the enable succeeded.
              }
              else
              {
                std::cerr << "WARNING: cuCtxEnablePeerAccess() between devices (" << m_activeDevices[home]->m_ordinal << ", " << m_activeDevices[peer]->m_ordinal << ") failed with CUresult " << result << '\n';
              }
            }
          }
        }
      }

      NVML_CHECK( m_nvml.m_api.nvmlShutdown() );
    }
  }
  catch (const std::exception& e)
  {
    // DAR FIXME Reaching this from CU_CHECK macros above means nvmlShutdown() hasn't been called.
    std::cerr << e.what() << '\n';
    // No return here. Always build the m_islands from the existing connection matrix information.
    success = false;
  }

  // Now use the peer-to-peer connection matrix to build peer-to-peer islands.
  // First fill a vector with all device indices which have not been assigned to an island.
  std::vector<int> unassigned(size);

  for (int i = 0; i < size; ++i)
  {
    unassigned[i] = i;
  }

  while (!unassigned.empty())
  {
    std::vector<int> island;
    std::vector<int>::const_iterator it = unassigned.begin();

    island.push_back(*it);
    unassigned.erase(it); // This device has been assigned to an island.

    it = unassigned.begin(); // The next unassigned device.
    while (it != unassigned.end())
    {
      bool isAccessible = true;

      const int peer = *it;

      // Check if this peer device is accessible by all other devices in the island.
      for (size_t i = 0; i < island.size(); ++i)
      {
        const int home = island[i];

        if ((m_peerConnections[home] & (1 << peer)) == 0 ||
            (m_peerConnections[peer] & (1 << home)) == 0) 
        {
          isAccessible = false;
        }
      }

      if (isAccessible)
      {
        island.push_back(*it);
        unassigned.erase(it); // This device has been assigned to an island.

        it = unassigned.begin(); // The next unassigned device.
      }
      else
      {
        ++it; // The next unassigned device, without erase in between.
      }
    }
    m_islands.push_back(island);
  }

  std::ostringstream text;

  text << m_islands.size() << " peer-to-peer island";
  if (1 < m_islands.size())
  {
	  text << 's';
  }
  text << ": "; 
  for (size_t i = 0; i < m_islands.size(); ++i)
  {
    const std::vector<int>& island = m_islands[i];

    text << "(";
    for (size_t j = 0; j < island.size(); ++j)
    {
      text << m_activeDevices[island[j]]->m_ordinal;
      if (j + 1 < island.size())
      {
        text << ", ";
      }
    }
    text << ")";
    if (i + 1 < m_islands.size())
    {
      text << " + ";
    }
  }
  std::cout << text.str() << '\n';

  return success;
}

void Raytracer::disablePeerAccess()
{
  const int size = static_cast<int>(m_activeDevices.size());
  MY_ASSERT(size <= 32); 
  
  // Peer-to-peer access is encoded in a bitfield of uint32 entries.
  for (int home = 0; home < size; ++home) // Home device index.
  {
    for (int peer = 0; peer < size; ++peer) // Peer device index.
    {
      if (home != peer && (m_peerConnections[home] & (1 << peer)) != 0)
      {
        // Note that this function changes the current context.
        CU_CHECK( cuCtxSetCurrent(m_activeDevices[home]->m_cudaContext) );        // Home context.
        CU_CHECK( cuCtxDisablePeerAccess(m_activeDevices[peer]->m_cudaContext) ); // Peer context.
        
        m_peerConnections[home] &= ~(1 << peer);
      }
    }
  }

  m_islands.clear(); // No peer-to-peer islands anymore.
  
  // Each device is its own island now.
  for (int i = 0; i < size; ++i)
  {
    std::vector<int> island;

    island.push_back(i);

    m_islands.push_back(island);

    m_peerConnections[i] |= (1 << i); // Should still be set from above.
  }
}

void Raytracer::synchronize()
{
  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    m_activeDevices[i]->activateContext();
    m_activeDevices[i]->synchronizeStream();
  }
}

// HACK Hardcocded textures.
void Raytracer::initTextures(std::map<std::string, Picture*> const& mapOfPictures)
{
  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    m_activeDevices[i]->initTextures(mapOfPictures);
  }
}

void Raytracer::initCameras(std::vector<CameraDefinition> const& cameras)
{
  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    m_activeDevices[i]->initCameras(cameras);
  }
}

void Raytracer::initLights(std::vector<LightDefinition> const& lights)
{
  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    m_activeDevices[i]->initLights(lights);
  }
}

void Raytracer::initMaterials(std::vector<MaterialGUI> const& materialsGUI)
{
  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    m_activeDevices[i]->initMaterials(materialsGUI);
  }
}

// Traverse the SceneGraph and store Groups, Instances and Triangles nodes in the raytracer representation.
void Raytracer::initScene(std::shared_ptr<sg::Group> root, const unsigned int numGeometries)
{
  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    m_activeDevices[i]->initScene(root, numGeometries);
  }
}

void Raytracer::initState(DeviceState const& state)
{
  m_samplesPerPixel = (unsigned int)(state.samplesSqrt * state.samplesSqrt);

  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    m_activeDevices[i]->setState(state);
  }
}

void Raytracer::updateCamera(const int idCamera, CameraDefinition const& camera)
{
  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    m_activeDevices[i]->updateCamera(idCamera, camera);
  }
  m_iterationIndex = 0; // Restart accumulation.
}

void Raytracer::updateLight(const int idLight, LightDefinition const& light)
{
  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    m_activeDevices[i]->updateLight(idLight, light);
  }
  m_iterationIndex = 0; // Restart accumulation.
}

void Raytracer::updateMaterial(const int idMaterial, MaterialGUI const& materialGUI)
{
  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    m_activeDevices[i]->updateMaterial(idMaterial, materialGUI);
  }
  m_iterationIndex = 0; // Restart accumulation.
}

void Raytracer::updateState(DeviceState const& state)
{
  m_samplesPerPixel = (unsigned int)(state.samplesSqrt * state.samplesSqrt);

  for (size_t i = 0; i < m_activeDevices.size(); ++i)
  {
    m_activeDevices[i]->setState(state);
  }
  m_iterationIndex = 0; // Restart accumulation.
}
