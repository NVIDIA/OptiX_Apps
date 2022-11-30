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
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

Raytracer::Raytracer(const int maskDevices,
                     const TypeLight typeEnv,
                     const int interop,
                     const unsigned int tex,
                     const unsigned int pbo,
                     const size_t sizeArena,
                     const int p2p)
: m_maskDevices(maskDevices)
, m_typeEnv(typeEnv)
, m_interop(interop)
, m_tex(tex)
, m_pbo(pbo)
, m_sizeArena(sizeArena)
, m_peerToPeer(p2p)
, m_isValid(false)
, m_numDevicesVisible(0)
, m_indexDeviceOGL(-1)
, m_maskDevicesActive(0)
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
  
  CU_CHECK( cuDeviceGetCount(&m_numDevicesVisible) );
  std::cout << "Device Count   = " << m_numDevicesVisible << '\n';

  // Match user defined m_maskDevices with the number of visible devices.
  // Builds m_maskActiveDevices and fills m_devicesActive which defines the device count.
  selectDevices();

  // This Raytracer is all about sharing data in peer-to-peer islands on multi-GPU setups.
  // While that can be individually enabled for texture array and/or GAS and vertex attribute data sharing,
  // the compositing of the final image is also done with peer-to-peer copies.
  (void) enablePeerAccess();

  m_isValid = !m_devicesActive.empty();
}


Raytracer::~Raytracer()
{
  try
  {
    // This function contains throw() calls.
    disablePeerAccess(); // Just for cleanliness, the Devices are destroyed anyway after this.

    // The GeometryData is either created on each device or only on one device of an NVLINK island.
    // In any case GeometryData is unique and must only be destroyed by the device owning the data.
    for (auto& data : m_geometryData)
    {
      m_devicesActive[data.owner]->destroyGeometry(data);
    }

    for (size_t i = 0; i < m_devicesActive.size(); ++i)
    {
      delete m_devicesActive[i];
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
  }

}

int Raytracer::matchUUID(const char* uuid)
{
  const int size = static_cast<int>(m_devicesActive.size());

  for (int i = 0; i < size; ++i)
  {
    if (m_devicesActive[i]->matchUUID(uuid))
    {
      // Use the first device which matches with the OpenGL UUID.
      // DEBUG This might not be the right thing to do with multicast enabled.
      m_indexDeviceOGL = i;
      break;
    }
  }

  std::cout << "OpenGL on active device " << m_indexDeviceOGL << '\n'; // DEBUG 

  return m_indexDeviceOGL; // If this stays -1, the active devices do not contain the one running the OpenGL implementation.
}

int Raytracer::matchLUID(const char* luid, const unsigned int nodeMask)
{
  const int size = static_cast<int>(m_devicesActive.size());

  for (int i = 0; i < size; ++i)
  {
    if (m_devicesActive[i]->matchLUID(luid, nodeMask))
    {
      // Use the first device which matches with the OpenGL LUID and test of the node mask bit.
      // DEBUG This might not be the right thing to do with multicast enabled.
      m_indexDeviceOGL = i;
      break;
    }
  }
  
  std::cout << "OpenGL on active device " << m_indexDeviceOGL << '\n'; // DEBUG 

  return m_indexDeviceOGL; // If this stays -1, the active devices do not contain the one running the OpenGL implementation.
}


int Raytracer::findActiveDevice(const unsigned int domain, const unsigned int bus, const unsigned int device) const
{
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    const DeviceAttribute& attribute = m_devicesActive[i]->m_deviceAttribute;

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
  if (m_devicesActive[home]->m_deviceAttribute.computeCapabilityMajor < 6)
  {
    return false;
  }

  nvmlDevice_t deviceHome;

  if (m_nvml.m_api.nvmlDeviceGetHandleByPciBusId(m_devicesActive[home]->m_devicePciBusId.c_str(), &deviceHome) != NVML_SUCCESS)
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
  const int size = static_cast<int>(m_devicesActive.size());
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

  // Check if the system configuration option "peerToPeer" allowed peer-to-peer via PCI-E irrespective of the NVLINK topology.
  // In that case the activeNVLINK() function is not called below. 
  // PERF In that case NVML wouldn't be needed at all.
  const bool allowPCI = ((m_peerToPeer & P2P_PCI) != 0);

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
          if (home != peer && (allowPCI || activeNVLINK(home, peer)))
          {
            int canAccessPeer = 0;

            // This requires the ordinals of the visible CUDA devices!
            CU_CHECK( cuDeviceCanAccessPeer(&canAccessPeer,
                                            (CUdevice) m_devicesActive[home]->m_ordinal,    // If this current home context
                                            (CUdevice) m_devicesActive[peer]->m_ordinal) ); // can access the peer context's memory.
            if (canAccessPeer != 0)
            {
              // Note that this function changes the current context!
              CU_CHECK( cuCtxSetCurrent(m_devicesActive[home]->m_cudaContext) );

              CUresult result = cuCtxEnablePeerAccess(m_devicesActive[peer]->m_cudaContext, 0);  // Flags must be 0!
              if (result == CUDA_SUCCESS)
              {
                m_peerConnections[home] |= (1 << peer); // Set the connection bit if the enable succeeded.
              }
              else
              {
                std::cerr << "WARNING: cuCtxEnablePeerAccess() between devices (" << m_devicesActive[home]->m_ordinal << ", " << m_devicesActive[peer]->m_ordinal << ") failed with CUresult " << result << '\n';
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
    // FIXME Reaching this from CU_CHECK macros above means nvmlShutdown() hasn't been called.
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
      text << m_devicesActive[island[j]]->m_ordinal;
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
  const int size = static_cast<int>(m_devicesActive.size());
  MY_ASSERT(size <= 32); 
  
  // Peer-to-peer access is encoded in a bitfield of uint32 entries.
  for (int home = 0; home < size; ++home) // Home device index.
  {
    for (int peer = 0; peer < size; ++peer) // Peer device index.
    {
      if (home != peer && (m_peerConnections[home] & (1 << peer)) != 0)
      {
        // Note that this function changes the current context!
        CU_CHECK( cuCtxSetCurrent(m_devicesActive[home]->m_cudaContext) );        // Home context.
        CU_CHECK( cuCtxDisablePeerAccess(m_devicesActive[peer]->m_cudaContext) ); // Peer context.
        
        m_peerConnections[home] &= ~(1 << peer);
      }
    }
  }

  m_islands.clear(); // No peer-to-peer islands anymore.
  
  // Each device is its own island now.
  for (int device = 0; device < size; ++device)
  {
    std::vector<int> island;

    island.push_back(device);

    m_islands.push_back(island);

    m_peerConnections[device] |= (1 << device); // Should still be set from above.
  }
}

void Raytracer::synchronize()
{
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->activateContext();
    m_devicesActive[i]->synchronizeStream();
  }
}

// FIXME This cannot handle cases where the same Picture would be used for different texture objects, but that is not happening in this example.
void Raytracer::initTextures(const std::map<std::string, Picture*>& mapPictures)
{
  const bool allowSharingTex = ((m_peerToPeer & P2P_TEX) != 0); // Material texture sharing (very cheap).
  const bool allowSharingEnv = ((m_peerToPeer & P2P_ENV) != 0); // HDR Environment and CDF sharing (CDF binary search is expensive).
  
  for (std::map<std::string, Picture*>::const_iterator it = mapPictures.begin(); it != mapPictures.end(); ++it)
  {
    const Picture* picture = it->second;

    const bool isEnv = ((picture->getFlags() & IMAGE_FLAG_ENV) != 0);

    if ((allowSharingTex && !isEnv) || (allowSharingEnv && isEnv))
    {
      for (const auto& island : m_islands) // Resource sharing only works across devices inside a peer-to-peer island.
      {
        const int deviceHome = getDeviceHome(island);

        const Texture* texture = m_devicesActive[deviceHome]->initTexture(it->first, picture, picture->getFlags());

        for (auto device : island)
        {
          if (device != deviceHome)
          {
            m_devicesActive[device]->shareTexture(it->first, texture);
          }
        }
      }
    }
    else
    {
      const unsigned int numDevices = static_cast<unsigned int>(m_devicesActive.size());

      for (unsigned int device = 0; device < numDevices; ++device)
      {
        (void) m_devicesActive[device]->initTexture(it->first, picture, picture->getFlags());
      }
    }
  }
}


void Raytracer::initCameras(const std::vector<CameraDefinition>& cameras)
{
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->initCameras(cameras);
  }
}

// For mesh lights this needs to be aware of the GAS sharing which results in different sizes of the m_geometryData built in initScene()!
void Raytracer::initLights(const std::vector<LightGUI>& lightsGUI, const std::vector<MaterialGUI>& materialsGUI)
{
  const bool allowSharingGas = ((m_peerToPeer & P2P_GAS) != 0); // GAS and vertex attribute sharing (GAS sharing is very expensive).
  
  if (allowSharingGas)
  {
    const unsigned int numIslands  = static_cast<unsigned int>(m_islands.size());

    for (unsigned int indexIsland = 0; indexIsland < numIslands; ++indexIsland)
    {
      const auto& island = m_islands[indexIsland]; // Vector of device indices.

      for (auto device : island) // Device index in this island.
      {
        m_devicesActive[device]->initLights(lightsGUI, materialsGUI, m_geometryData, numIslands, indexIsland);
      }
    }
  }
  else
  {
    const unsigned int numDevices = static_cast<unsigned int>(m_devicesActive.size());

    for (unsigned int device = 0; device < numDevices; ++device)
    {
      m_devicesActive[device]->initLights(lightsGUI, materialsGUI, m_geometryData, numDevices, device);
    }
  }
}

void Raytracer::initMaterials(const std::vector<MaterialGUI>& materialsGUI)
{
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->initMaterials(materialsGUI);
  }
}

// Traverse the SceneGraph and store Groups, Instances and Triangles nodes in the raytracer representation.
void Raytracer::initScene(std::shared_ptr<sg::Group> root, const unsigned int numGeometries)
{
  const bool allowSharingGas = ((m_peerToPeer & P2P_GAS) != 0); // GAS and vertex attribute sharing (GAS sharing is very expensive).

  if (allowSharingGas)
  {
    // Allocate the number of GeometryData per island.
    m_geometryData.resize(numGeometries * m_islands.size()); // Sharing data per island.
  }
  else
  {
    // Allocate the number of GeometryData per active device.
    m_geometryData.resize(numGeometries * m_devicesActive.size()); // Not sharing, all devices hold all geometry data.
  }

  InstanceData instanceData(~0u, -1, -1);

  float matrix[12];

  // Set the affine matrix to identity by default.
  memset(matrix, 0, sizeof(float) * 12);
  matrix[ 0] = 1.0f;
  matrix[ 5] = 1.0f;
  matrix[10] = 1.0f;

  traverseNode(root, instanceData, matrix);

  if (allowSharingGas)
  {
    const unsigned int numIslands  = static_cast<unsigned int>(m_islands.size());

    for (unsigned int indexIsland = 0; indexIsland < numIslands; ++indexIsland)
    {
      const auto& island = m_islands[indexIsland]; // Vector of device indices.
    
      for (auto device : island) // Device index in this island.
      {
        // The IAS and SBT are not shared in this example.
        m_devicesActive[device]->createTLAS(); 
        m_devicesActive[device]->createGeometryInstanceData(m_geometryData, matrix, numIslands, indexIsland);
      }
    }
  }
  else
  {
    const unsigned int numDevices = static_cast<unsigned int>(m_devicesActive.size());

    for (unsigned int device = 0; device < numDevices; ++device)
    {
      m_devicesActive[device]->createTLAS();
      m_devicesActive[device]->createGeometryInstanceData(m_geometryData, matrix, numDevices, device);
    }
  }
}


void Raytracer::initState(const DeviceState& state)
{
  m_samplesPerPixel = (unsigned int)(state.samplesSqrt * state.samplesSqrt);
   
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->setState(state);
  }
}

void Raytracer::updateCamera(const int idCamera, const CameraDefinition& camera)
{
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->updateCamera(idCamera, camera);
  }
  m_iterationIndex = 0; // Restart accumulation.
}

void Raytracer::updateLight(const int idLight, const MaterialGUI& materialGUI)
{
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->updateLight(idLight, materialGUI);
  }
  m_iterationIndex = 0; // Restart accumulation.
}

//void Raytracer::updateLight(const int idLight, const LightDefinition& light)
//{
//  for (size_t i = 0; i < m_devicesActive.size(); ++i)
//  {
//    m_devicesActive[i]->updateLight(idLight, light);
//  }
//  m_iterationIndex = 0; // Restart accumulation.
//}

void Raytracer::updateMaterial(const int idMaterial, const MaterialGUI& materialGUI)
{
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->updateMaterial(idMaterial, materialGUI);
  }
  m_iterationIndex = 0; // Restart accumulation.
}

void Raytracer::updateState(const DeviceState& state)
{
  m_samplesPerPixel = (unsigned int)(state.samplesSqrt * state.samplesSqrt);

  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->setState(state);
  }
  m_iterationIndex = 0; // Restart accumulation.
}

void Raytracer::updateTLAS()
{
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->updateTLAS();
  }
  m_iterationIndex = 0; // Restart accumulation.
}

// The public function which does the multi-GPU wrapping.
// Returns the count of renderered iterations (m_iterationIndex after it has been incremented).
unsigned int Raytracer::render(const int mode)
{
  // Continue manual accumulation rendering if the samples per pixel have not been reached.
  if (m_iterationIndex < m_samplesPerPixel)
  {
    void* buffer = nullptr;

    // Make sure the OpenGL device is allocating the full resolution backing storage.
    const int index = (m_indexDeviceOGL != -1) ? m_indexDeviceOGL : 0; // Destination device.

    // This is the device which needs to allocate the peer-to-peer buffer to reside on the same device as the PBO or Texture
    m_devicesActive[index]->render(m_iterationIndex, &buffer, mode); // Interactive rendering. All devices work on the same iteration index.

    for (size_t i = 0; i < m_devicesActive.size(); ++i)
    {
      if (index != static_cast<int>(i))
      {
        // If buffer is still nullptr here, the first device will allocate the full resolution buffer.
        m_devicesActive[i]->render(m_iterationIndex, &buffer, mode);
      }
    }
    
    ++m_iterationIndex;
  }  
  return m_iterationIndex;
}

void Raytracer::updateDisplayTexture()
{
  const int index = (m_indexDeviceOGL != -1) ? m_indexDeviceOGL : 0; // Destination device.

  // Only need to composite the resulting frame when using multiple decvices.
  // Single device renders directly into the full resolution output buffer.
  if (1 < m_devicesActive.size()) 
  {
    // First, copy the texelBuffer of the primary device into its tileBuffer and then place the tiles into the outputBuffer.
    m_devicesActive[index]->compositor(m_devicesActive[index]);

    // Now copy the other devices' texelBuffers over to the main tileBuffer and repeat the compositing for that other device.
    // The cuMemcpyPeerAsync done in that case is fast when the devices are in the same peer island, otherwise it's copied via PCI-E, but only N-1 copies of 1/N size are done.
    // The saving here is no peer-to-peer read-modify-write when rendering, because everything is happening in GPU local buffers, which are also tightly packed.
    // The final compositing is just a kernel implementing a tiled memcpy. 
    // PERF If all tiles are copied to the main device at once, such kernel would only need to be called once.
    for (size_t i = 0; i < m_devicesActive.size(); ++i)
    {
    if (index != static_cast<int>(i))
      {
        m_devicesActive[index]->compositor(m_devicesActive[i]);
      }
    }
  }
  // Finally copy the primary device outputBuffer to the display texture. 
  // FIXME DEBUG Does that work when m_indexDeviceOGL is not in the list of active devices?
  m_devicesActive[index]->updateDisplayTexture();
}

const void* Raytracer::getOutputBufferHost()
{
  // Same initial steps to fill the outputBuffer on the primary device as in updateDisplayTexture() 
  const int index = (m_indexDeviceOGL != -1) ? m_indexDeviceOGL : 0; // Destination device.

  // Only need to composite the resulting frame when using multiple decvices.
  // Single device renders  directly into the full resolution output buffer.
  if (1 < m_devicesActive.size()) 
  {
    // First, copy the texelBuffer of the primary device into its tileBuffer and then place the tiles into the outputBuffer.
    m_devicesActive[index]->compositor(m_devicesActive[index]);

    // Now copy the other devices' texelBuffers over to the main tileBuffer and repeat the compositing for that other device.
    for (size_t i = 0; i < m_devicesActive.size(); ++i) 
    {
    if (index != static_cast<int>(i))
      {
        m_devicesActive[index]->compositor(m_devicesActive[i]);
      }
    }
  }  
  // The full outputBuffer resides on device "index" and the host buffer is also only resized by that device.
  return m_devicesActive[index]->getOutputBufferHost();
}

// Private functions.

void Raytracer::selectDevices()
{
  // Need to determine the number of active devices first to have it available as device constructor argument.
  int count   = 0; 
  int ordinal = 0;

  while (ordinal < m_numDevicesVisible) // Don't try to enable more devices than visible to CUDA.
  {
    const unsigned int mask = (1 << ordinal);

    if (m_maskDevices & mask)
    {
      // Track which and how many devices have actually been enabled.
      m_maskDevicesActive |= mask; 
      ++count;
    }

    ++ordinal;
  }

  // Now really construct the Device objects.
  ordinal = 0;

  while (ordinal < m_numDevicesVisible)
  {
    const unsigned int mask = (1 << ordinal);

    if (m_maskDevicesActive & mask)
    {
      const int index = static_cast<int>(m_devicesActive.size());

      Device* device = new Device(ordinal, index, count, m_typeEnv, m_interop, m_tex, m_pbo, m_sizeArena);

      m_devicesActive.push_back(device);

      std::cout << "Device " << ordinal << ": " << device->m_deviceName << " selected\n";
    }

    ++ordinal;
  }
}

#if 1
// This implementation does not consider the actually free amount of VRAM on the individual devices in an island, but assumes they are equally loaded.
// This method works more fine grained with the arena allocator.
int Raytracer::getDeviceHome(const std::vector<int>& island) const
{
  // Find the device inside each island which has the least amount of allocated memory.
  size_t sizeMin    = ~0ull; // Biggest unsigned 64-bit number.
  int    deviceHome = 0;     // Default to zero if all devices are OOM. That will fail in CU_CHECK later.

  for (auto device : island)
  {
    const size_t size = m_devicesActive[device]->getMemoryAllocated();

    if (size < sizeMin)
    {
      sizeMin    = size;
      deviceHome = device;
    }
  }

  //std::cout << "deviceHome = " << deviceHome << ", allocated [MiB] = " << double(sizeMin) / (1024.0 * 1024.0) << '\n'; // DEBUG 

  return deviceHome;
}

#else

// This implementation uses the actual free amount of VRAM on the individual devices in an NVLINK island.
// With the arena allocator this will result in less fine grained distribution of resources because the free memory only changes when a new arena is allocated.
// Using a smaller arena size would switch allocations between devices more often in this case.
int Raytracer::getDeviceHome(const std::vector<int>& island) const
{
  // Find the device inside each island which has the most free memory.
  size_t sizeMax    = 0;
  int    deviceHome = 0; // Default to zero if all devices are OOM. That will fail in CU_CHECK later.

  for (auto device : island)
  {
    const size_t size = m_devicesActive[device]->getMemoryFree(); // Actual free VRAM overall.

    if (sizeMax < size)
    {
      sizeMax    = size;
      deviceHome = device;
    }
  }
  
  //std::cout << "deviceHome = " << deviceHome << ", free [MiB] = " << double(sizeMax) / (1024.0 * 1024.0) << '\n'; // DEBUG 

  return deviceHome;
}
#endif

// m = a * b;
static void multiplyMatrix(float* m, const float* a, const float* b)
{
  m[ 0] = a[0] * b[0] + a[1] * b[4] + a[ 2] * b[ 8]; // + a[3] * 0
  m[ 1] = a[0] * b[1] + a[1] * b[5] + a[ 2] * b[ 9]; // + a[3] * 0
  m[ 2] = a[0] * b[2] + a[1] * b[6] + a[ 2] * b[10]; // + a[3] * 0
  m[ 3] = a[0] * b[3] + a[1] * b[7] + a[ 2] * b[11] + a[3]; // * 1
  
  m[ 4] = a[4] * b[0] + a[5] * b[4] + a[ 6] * b[ 8]; // + a[7] * 0
  m[ 5] = a[4] * b[1] + a[5] * b[5] + a[ 6] * b[ 9]; // + a[7] * 0
  m[ 6] = a[4] * b[2] + a[5] * b[6] + a[ 6] * b[10]; // + a[7] * 0
  m[ 7] = a[4] * b[3] + a[5] * b[7] + a[ 6] * b[11] + a[7]; // * 1

  m[ 8] = a[8] * b[0] + a[9] * b[4] + a[10] * b[ 8]; // + a[11] * 0
  m[ 9] = a[8] * b[1] + a[9] * b[5] + a[10] * b[ 9]; // + a[11] * 0
  m[10] = a[8] * b[2] + a[9] * b[6] + a[10] * b[10]; // + a[11] * 0
  m[11] = a[8] * b[3] + a[9] * b[7] + a[10] * b[11] + a[11]; // * 1
}


// Depth-first traversal of the scene graph to flatten all unique paths to a geometry node to one-level instancing inside the OptiX render graph.
void Raytracer::traverseNode(std::shared_ptr<sg::Node> node, InstanceData instanceData, float matrix[12])
{
  switch (node->getType())
  {
    case sg::NodeType::NT_GROUP:
    {
      std::shared_ptr<sg::Group> group = std::dynamic_pointer_cast<sg::Group>(node);

      for (size_t i = 0; i < group->getNumChildren(); ++i)
      {
        traverseNode(group->getChild(i), instanceData, matrix);
      }
    }
    break;

    case sg::NodeType::NT_INSTANCE:
    {
      std::shared_ptr<sg::Instance> instance = std::dynamic_pointer_cast<sg::Instance>(node);

      // Track the assigned material and light indices. Only the bottom-most instance node matters.
      instanceData.idMaterial = instance->getMaterial();
      instanceData.idLight    = instance->getLight();

      // Concatenate the transformations along the path.
      float trafo[12];

      multiplyMatrix(trafo, matrix, instance->getTransform());

      traverseNode(instance->getChild(), instanceData, trafo);
    }
    break;

    case sg::NodeType::NT_TRIANGLES:
    {
      std::shared_ptr<sg::Triangles> geometry = std::dynamic_pointer_cast<sg::Triangles>(node);
      
      instanceData.idGeometry = geometry->getId();

      const bool allowSharingGas = ((m_peerToPeer & P2P_GAS) != 0);

      if (allowSharingGas)
      {
        const unsigned int numIslands  = static_cast<unsigned int>(m_islands.size());

        for (unsigned int indexIsland = 0; indexIsland < numIslands; ++indexIsland)
        {
          const auto& island = m_islands[indexIsland]; // Vector of device indices.

          const int deviceHome = getDeviceHome(island);

          // GeometryData is always shared and tracked per island.
          GeometryData& geometryData = m_geometryData[instanceData.idGeometry * numIslands + indexIsland];

          if (geometryData.traversable == 0) // If there is no traversable handle for this geometry in this island, try to create one on the home device.
          {
            geometryData = m_devicesActive[deviceHome]->createGeometry(geometry); 
          }
          else
          {
            std::cout << "traverseNode() Geometry " << instanceData.idGeometry << " reused\n"; // DEBUG
          }
        
          m_devicesActive[deviceHome]->createInstance(geometryData, instanceData, matrix);
        
          // Now share the GeometryData on the other devices in this island.
          for (const auto device : island)
          {
            if (device != deviceHome)
            {
              // Create the instance referencing the shared GAS traversable on the peer device in this island.
              // This is only host data. The IAS is created after gathering all flattened instances in the scene.
              m_devicesActive[device]->createInstance(geometryData, instanceData, matrix); 
            }
          }
        }
      }
      else
      {
        const unsigned int numDevices = static_cast<unsigned int>(m_devicesActive.size());

        for (unsigned int device = 0; device < numDevices; ++device)
        {
          GeometryData& geometryData = m_geometryData[instanceData.idGeometry * numDevices + device];

          if (geometryData.traversable == 0) // If there is no traversable handle for this geometry on this device, try to create one.
          {
            geometryData = m_devicesActive[device]->createGeometry(geometry);
          }

          m_devicesActive[device]->createInstance(geometryData, instanceData, matrix);
        }
      }
    }
    break;
  }
}
