/* 
 * Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
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
#include "inc/CompileResult.h"

#include "inc/CheckMacros.h"

#include "shaders/config.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>


bool static saveString(const std::string& filename, const std::string& text)
{
  std::ofstream outputStream(filename);

  if (!outputStream)
  {
    std::cerr << "ERROR: saveString() Failed to open file " << filename << '\n';
    return false;
  }

  outputStream << text;

  if (outputStream.fail())
  {
    std::cerr << "ERROR: saveString() Failed to write file " << filename << '\n';
    return false;
  }

  return true;
}

static std::string getDateTime()
{
#if defined(_WIN32)
  SYSTEMTIME time;
  GetLocalTime(&time);
#elif defined(__linux__)
  time_t rawtime;
  struct tm* ts;
  time(&rawtime);
  ts = localtime(&rawtime);
#else
  #error "OS not supported."
#endif

  std::ostringstream oss;

#if defined( _WIN32 )
  oss << time.wYear;
  if (time.wMonth < 10)
  {
    oss << '0';
  }
  oss << time.wMonth;
  if (time.wDay < 10)
  {
    oss << '0';
  }
  oss << time.wDay << '_';
  if (time.wHour < 10)
  {
    oss << '0';
  }
  oss << time.wHour;
  if (time.wMinute < 10)
  {
    oss << '0';
  }
  oss << time.wMinute;
  if (time.wSecond < 10)
  {
    oss << '0';
  }
  oss << time.wSecond << '_';
  if (time.wMilliseconds < 100)
  {
    oss << '0';
  }
  if (time.wMilliseconds <  10)
  {
    oss << '0';
  }
  oss << time.wMilliseconds; 
#elif defined(__linux__)
  oss << ts->tm_year;
  if (ts->tm_mon < 10)
  {
    oss << '0';
  }
  oss << ts->tm_mon;
  if (ts->tm_mday < 10)
  {
    oss << '0';
  }
  oss << ts->tm_mday << '_';
  if (ts->tm_hour < 10)
  {
    oss << '0';
  }
  oss << ts->tm_hour;
  if (ts->tm_min < 10)
  {
    oss << '0';
  }
  oss << ts->tm_min;
  if (ts->tm_sec < 10)
  {
    oss << '0';
  }
  oss << ts->tm_sec << '_';
  oss << "000"; // No milliseconds available.
#else
  #error "OS not supported."
#endif

  return oss.str();
}





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
void Raytracer::initLights(const std::vector<LightGUI>& lightsGUI)
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
        m_devicesActive[device]->initLights(lightsGUI, m_geometryData, numIslands, indexIsland);
      }
    }
  }
  else
  {
    const unsigned int numDevices = static_cast<unsigned int>(m_devicesActive.size());

    for (unsigned int device = 0; device < numDevices; ++device)
    {
      m_devicesActive[device]->initLights(lightsGUI, m_geometryData, numDevices, device);
    }
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

  InstanceData instanceData(~0u, -1, -1, -1);

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
        m_devicesActive[device]->createGeometryInstanceData(m_geometryData, numIslands, indexIsland);
      }
    }
  }
  else
  {
    const unsigned int numDevices = static_cast<unsigned int>(m_devicesActive.size());

    for (unsigned int device = 0; device < numDevices; ++device)
    {
      m_devicesActive[device]->createTLAS();
      m_devicesActive[device]->createGeometryInstanceData(m_geometryData, numDevices, device);
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

void Raytracer::updateLight(const int idLight, const LightGUI& lightGUI)
{
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->updateLight(idLight, lightGUI);
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

void Raytracer::updateMaterial(const int idMaterial, const MaterialMDL* materialMDL)
{
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->updateMaterial(idMaterial, materialMDL);
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
      instanceData.idObject   = instance->getId();

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

    case sg::NodeType::NT_CURVES:
    {
      std::shared_ptr<sg::Curves> geometry = std::dynamic_pointer_cast<sg::Curves>(node);
      
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


// MDL Material specific functions.

static std::string replace(const std::string& source, const std::string& from, const std::string& to)
{
  if (source.empty())
  {
    return source;
  }

  std::string result;
  result.reserve(source.length());

  std::string::size_type lastPos = 0;
  std::string::size_type findPos;

  while (std::string::npos != (findPos = source.find(from, lastPos)))
  {
    result.append(source, lastPos, findPos - lastPos);
    result.append(to);

    lastPos = findPos + from.length();
  }

  //result += source.substr(lastPos);
  result.append(source, lastPos, source.length() - lastPos);

  return result;
}


std::string buildModuleName(const std::string& path)
{
  if (path.empty())
  {
    return path;
  }
  
  // Build an MDL name. This assumes the path starts with a backslash (or slash on Linux).
  std::string name = path;

#if defined(_WIN32)
  if (name[0] != '\\')
  {
    name = std::string("\\") + path;
  }
  name = replace(name, "\\", "::");
#elif defined(__linux__)
  if (name[0] != '/')
  {
    name = std::string("/") + path;
  }
  name = replace(name, "/", "::");
#endif

  return name;
}


std::string add_missing_material_signature(const mi::neuraylib::IModule* module,
                                           const std::string& material_name)
{
  // Return input if it already contains a signature.
  if (material_name.back() == ')')
  {
    return material_name;
  }

  mi::base::Handle<const mi::IArray> result(module->get_function_overloads(material_name.c_str()));

  // Not supporting multiple function overloads with the same name but different signatures.
  if (!result || result->get_length() != 1)
  {
    return std::string();
  }

  mi::base::Handle<const mi::IString> overloads(result->get_element<mi::IString>(static_cast<mi::Size>(0)));

  return overloads->get_c_str();
}


bool isValidDistribution(mi::neuraylib::IExpression const* expr)
{
  if (expr == nullptr)
  {
    return false;
  }

  if (expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
  {
    mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_constant(expr->get_interface<mi::neuraylib::IExpression_constant>());
    mi::base::Handle<mi::neuraylib::IValue const> value(expr_constant->get_value());

    if (value->get_kind() == mi::neuraylib::IValue::VK_INVALID_DF)
    {
      return false;
    }
  }

  return true;
}


// Returns a string-representation of the given message category
const char* message_kind_to_string(mi::neuraylib::IMessage::Kind message_kind)
{
  switch (message_kind)
  {
    case mi::neuraylib::IMessage::MSG_INTEGRATION:
      return "MDL SDK";
    case mi::neuraylib::IMessage::MSG_IMP_EXP:
      return "Importer/Exporter";
    case mi::neuraylib::IMessage::MSG_COMILER_BACKEND:
      return "Compiler Backend";
    case mi::neuraylib::IMessage::MSG_COMILER_CORE:
      return "Compiler Core";
    case mi::neuraylib::IMessage::MSG_COMPILER_ARCHIVE_TOOL:
      return "Compiler Archive Tool";
    case mi::neuraylib::IMessage::MSG_COMPILER_DAG:
      return "Compiler DAG generator";
    default:
      break;
  }
  return "";
}


// Returns a string-representation of the given message severity
const char* message_severity_to_string(mi::base::Message_severity severity)
{
  switch (severity)
  {
    case mi::base::MESSAGE_SEVERITY_ERROR:
      return "ERROR";
    case mi::base::MESSAGE_SEVERITY_WARNING:
      return "WARNING";
    case mi::base::MESSAGE_SEVERITY_INFO:
      return "INFO";
    case mi::base::MESSAGE_SEVERITY_VERBOSE:
      return "VERBOSE";
    case mi::base::MESSAGE_SEVERITY_DEBUG:
      return "DEBUG";
    default:
      break;
  }
  return "";
}


class Default_logger: public mi::base::Interface_implement<mi::base::ILogger>
{
public:
  void message(mi::base::Message_severity level,
               const char* /* module_category */,
               const mi::base::Message_details& /* details */,
               const char* message) override
  {
    const char* severity = 0;

    switch (level)
    {
      case mi::base::MESSAGE_SEVERITY_FATAL:
        severity = "FATAL: ";
        MY_ASSERT(!"Default_logger() fatal error.");
        break;
      case mi::base::MESSAGE_SEVERITY_ERROR:
        severity = "ERROR: ";
        MY_ASSERT(!"Default_logger() error.");
        break;
      case mi::base::MESSAGE_SEVERITY_WARNING:
        severity = "WARN:  ";
        break;
      case mi::base::MESSAGE_SEVERITY_INFO:
        //return; // DEBUG No info messages.
        severity = "INFO:  ";
        break;
      case mi::base::MESSAGE_SEVERITY_VERBOSE:
        return; // DEBUG No verbose messages.
      case mi::base::MESSAGE_SEVERITY_DEBUG:
        return; // DEBUG No debug messages.
      case mi::base::MESSAGE_SEVERITY_FORCE_32_BIT:
        return;
    }

    std::cerr << severity << message << '\n';
  }

  void message(mi::base::Message_severity level,
               const char* module_category,
               const char* message) override
  {
    this->message(level, module_category, mi::base::Message_details(), message);
  }
};


/// Callback that notifies the application about new resources when generating an
/// argument block for an existing target code.
class Resource_callback
  : public mi::base::Interface_implement<mi::neuraylib::ITarget_resource_callback>
{
public:
  /// Constructor.
  Resource_callback(mi::neuraylib::ITransaction* transaction,
                    mi::neuraylib::ITarget_code const* target_code,
                    Compile_result& compile_result)
    : m_transaction(mi::base::make_handle_dup(transaction))
    , m_target_code(mi::base::make_handle_dup(target_code))
    , m_compile_result(compile_result)
  {
  }

  /// Destructor.
  virtual ~Resource_callback() = default;

  /// Returns a resource index for the given resource value usable by the target code
  /// resource handler for the corresponding resource type.
  ///
  /// \param resource  the resource value
  ///
  /// \returns a resource index or 0 if no resource index can be returned
  mi::Uint32 get_resource_index(mi::neuraylib::IValue_resource const* resource) override
  {
    // check whether we already know the resource index
    auto it = m_resource_cache.find(resource);
    if (it != m_resource_cache.end())
    {
      return it->second;
    }

    // handle resources already known by the target code
    mi::Uint32 res_idx = m_target_code->get_known_resource_index(m_transaction.get(), resource);
    if (res_idx != 0)
    {
      // only accept body resources
      switch (resource->get_kind())
      {
        case mi::neuraylib::IValue::VK_TEXTURE:
          if (res_idx < m_target_code->get_body_texture_count())
            return res_idx;
          break;
        case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
          if (res_idx < m_target_code->get_body_light_profile_count())
            return res_idx;
          break;
        case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
          if (res_idx < m_target_code->get_body_bsdf_measurement_count())
            return res_idx;
          break;
        default:
          return 0u;  // invalid kind
      }
    }

    switch (resource->get_kind())
    {
      case mi::neuraylib::IValue::VK_TEXTURE:
      {
        mi::base::Handle<mi::neuraylib::IValue_texture const> val_texture(resource->get_interface<mi::neuraylib::IValue_texture const>());
        if (!val_texture)
        {
          return 0u;  // unknown resource
        }

        mi::base::Handle<const mi::neuraylib::IType_texture> texture_type(val_texture->get_type());

        mi::neuraylib::ITarget_code::Texture_shape shape = mi::neuraylib::ITarget_code::Texture_shape(texture_type->get_shape());

        m_compile_result.textures.emplace_back(resource->get_value(), shape);
        res_idx = m_compile_result.textures.size() - 1;
        break;
      }

      case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
        m_compile_result.light_profiles.emplace_back(resource->get_value());
        res_idx = m_compile_result.light_profiles.size() - 1;
        break;

      case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
        m_compile_result.bsdf_measurements.emplace_back(resource->get_value());
        res_idx = m_compile_result.bsdf_measurements.size() - 1;
        break;

      default:
        return 0u;  // invalid kind
    }

    m_resource_cache[resource] = res_idx;
    return res_idx;
  }

  /// Returns a string identifier for the given string value usable by the target code.
  ///
  /// The value 0 is always the "not known string".
  ///
  /// \param s  the string value
  mi::Uint32 get_string_index(mi::neuraylib::IValue_string const* s) override
  {
    char const* str_val = s->get_value();
    if (str_val == nullptr)
      return 0u;

    for (mi::Size i = 0, n = m_target_code->get_string_constant_count(); i < n; ++i)
    {
      if (strcmp(m_target_code->get_string_constant(i), str_val) == 0)
      {
        return mi::Uint32(i);
      }
    }

    // string not known by code
    return 0u;
  }

private:
  mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
  mi::base::Handle<const mi::neuraylib::ITarget_code> m_target_code;

  std::map<mi::neuraylib::IValue_resource const*, mi::Uint32> m_resource_cache;
  Compile_result& m_compile_result;
};


mi::neuraylib::INeuray* Raytracer::load_and_get_ineuray(const char* filename)
{
  if (!filename)
  {
//#ifdef IRAY_SDK
//    filename = "libneuray" MI_BASE_DLL_FILE_EXT;
//#else
    filename = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;
//#endif
  }

#ifdef MI_PLATFORM_WINDOWS

  HMODULE handle = LoadLibraryA(filename);
  //if (!handle)
  //{
  //  // fall back to libraries in a relative lib folder, relevant for install targets
  //  std::string fallback = std::string("../../../lib/") + filename;
  //  handle = LoadLibraryA(fallback.c_str());
  //}
  if (!handle)
  {
    DWORD error_code = GetLastError();
    std::cerr << "ERROR: LoadLibraryA(" << filename << ") failed with error code " << error_code << '\n';
    return 0;
  }

  void* symbol = GetProcAddress(handle, "mi_factory");
  if (!symbol)
  {
    DWORD error_code = GetLastError();
    std::cerr << "ERROR: GetProcAddress(handle, \"mi_factory\") failed with error " << error_code << '\n';
    return 0;
  }

#else // MI_PLATFORM_WINDOWS

  void* handle = dlopen(filename, RTLD_LAZY);
  //if (!handle)
  //{
  //  // fall back to libraries in a relative lib folder, relevant for install targets
  //  std::string fallback = std::string("../../../lib/") + filename;
  //  handle = dlopen(fallback.c_str(), RTLD_LAZY);
  //}
  if (!handle)
  {
    std::cerr << "ERROR: dlopen(" << filename << " , RTLD_LAZY) failed with error code " << dlerror() << '\n';
    return 0;
  }
  
  void* symbol = dlsym(handle, "mi_factory");
  if (!symbol)
  {
    std::cerr << "ERROR: dlsym(handle, \"mi_factory\") failed with error " << dlerror() << '\n';
    return 0;
  }

#endif // MI_PLATFORM_WINDOWS
  
  m_dso_handle = handle;

  mi::neuraylib::INeuray* neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol);
  if (!neuray)
  {
    mi::base::Handle<mi::neuraylib::IVersion> version(mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(symbol));
    if (!version)
    {
      std::cerr << "ERROR: Incompatible library. Could not determine version.\n";
    }
    else
    {
      std::cerr << "ERROR: Library version " << version->get_product_version() << " does not match header version " << MI_NEURAYLIB_PRODUCT_VERSION_STRING << '\n';
    }
    return 0;
  }

//#ifdef IRAY_SDK
//  if (authenticate(neuray) != 0)
//  {
//    std::cerr << "ERROR: Iray SDK Neuray Authentication failed.\n";
//    unload();
//    return 0;
//  }
//#endif

  return neuray;
}


mi::Sint32 Raytracer::load_plugin(mi::neuraylib::INeuray* neuray, const char* path)
{
  mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_conf(neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());

  // Try loading the requested plugin before adding any special handling
  mi::Sint32 res = plugin_conf->load_plugin_library(path);
  if (res == 0)
  {
    //std::cerr << "load_plugin(" << path << ") succeeded.\n"; // DEBUG The logger prints this.
    return 0;
  }

  // Special handling for freeimage in the open source release.
  // In the open source version of the plugin we are linking against a dynamic vanilla freeimage library.
  // In the binary release, you can download from the MDL website, freeimage is linked statically and 
  // thereby requires no special handling.
#if defined(MI_PLATFORM_WINDOWS) && defined(MDL_SOURCE_RELEASE)
  if (strstr(path, "nv_freeimage" MI_BASE_DLL_FILE_EXT) != nullptr)
  {
    // Load the freeimage (without nv_ prefix) first.
    std::string freeimage_3rd_party_path = replace(path, "nv_freeimage" MI_BASE_DLL_FILE_EXT, "freeimage" MI_BASE_DLL_FILE_EXT);
    HMODULE handle_tmp = LoadLibraryA(freeimage_3rd_party_path.c_str());
    if (!handle_tmp)
    {
      DWORD error_code = GetLastError();
      std::cerr << "ERROR: load_plugin(" << freeimage_3rd_party_path << " failed with error " << error_code << '\n';
    }
    else
    {
      std::cerr << "Pre-loading library " << freeimage_3rd_party_path << " succeeded\n";
    }

    // Try to load the plugin itself now
    res = plugin_conf->load_plugin_library(path);
    if (res == 0)
    {
      std::cerr << "load_plugin(" << path << ") succeeded.\n"; // DAR FIXME The logger prints this as info anyway.
      return 0;
    }
  }
#endif

  // return the failure code
  std::cerr << "ERROR: load_plugin(" << path << ") failed with error " << res << '\n';
 
  return res;
}


bool Raytracer::initMDL(const std::vector<std::string>& searchPaths)
{
  // Load MDL SDK library and create a Neuray handle.
  m_neuray = load_and_get_ineuray(nullptr);
  if (!m_neuray.is_valid_interface())
  {
    std::cerr << "ERROR: Initialization of MDL SDK failed: libmdl_sdk" MI_BASE_DLL_FILE_EXT " not found or wrong version.\n";
    return false;
  }

  // Create the MDL compiler.
  m_mdl_compiler = m_neuray->get_api_component<mi::neuraylib::IMdl_compiler>();
  if (!m_mdl_compiler)
  {
    std::cerr << "ERROR: Initialization of MDL compiler failed.\n";
    return false;
  }

  // Configure Neuray.
  m_mdl_config = m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>();
  if (!m_mdl_config)
  {
    std::cerr << "ERROR: Retrieving MDL configuration failed.\n";
    return false;
  }

  m_logger = mi::base::make_handle(new Default_logger());
  m_mdl_config->set_logger(m_logger.get());

  // Convenient default search paths for the NVIDIA MDL vMaterials!

  // Environment variable MDL_SYSTEM_PATH.
  // Defaults to "C:\ProgramData\NVIDIA Corporation\mdl\" under Windows.
  // Required to find ::nvidia::core_definitions imports used inside the vMaterials *.mdl files.
  m_mdl_config->add_mdl_system_paths();
  
  // Environment variable MDL_USER_PATH.
  // Defaults to "C:\Users\<username>\Documents\mdl\" under Windows.
  // Required to find the vMaterials *.mdl files and their resources.
  m_mdl_config->add_mdl_user_paths();

  // Add all additional MDL and resource search paths defined inside the system description file as well.
  for (auto const& path : searchPaths)
  {
    mi::Sint32 result = m_mdl_config->add_mdl_path(path.c_str());
    if  (result != 0)
    {
      std::cerr << "WARNING: add_mdl_path( " << path << ") failed with " << result << '\n';
    }

    result = m_mdl_config->add_resource_path(path.c_str());
    if (result != 0)
    {
      std::cerr << "WARNING: add_resource_path( " << path << ") failed with " << result << '\n';
    }
  }

  // Load plugins.
  if (load_plugin(m_neuray.get(), "nv_freeimage" MI_BASE_DLL_FILE_EXT) != 0)
  {
    std::cerr << "FATAL: Failed to load nv_freeimage plugin\n";
    return false;
  }

  if (load_plugin(m_neuray.get(), "dds" MI_BASE_DLL_FILE_EXT) != 0)
  {
    std::cerr << "FATAL: Failed to load dds plugin\n";
    return false;
  }

  if (m_neuray->start() != 0)
  {
    std::cerr << "FATAL: Starting MDL SDK failed.\n";
    return false;
  }

  m_database = m_neuray->get_api_component<mi::neuraylib::IDatabase>();

  m_global_scope = m_database->get_global_scope();

  m_mdl_factory = m_neuray->get_api_component<mi::neuraylib::IMdl_factory>();

  // Configure the execution context.
  // Used for various configurable operations and for querying warnings and error messages.
  // It is possible to have more than one, in order to use different settings.
  m_execution_context = m_mdl_factory->create_execution_context();

  m_execution_context->set_option("internal_space", "coordinate_world");  // equals default
  m_execution_context->set_option("bundle_resources", false);             // equals default
  m_execution_context->set_option("meters_per_scene_unit", 1.0f);         // equals default
  m_execution_context->set_option("mdl_wavelength_min", 380.0f);          // equals default
  m_execution_context->set_option("mdl_wavelength_max", 780.0f);          // equals default
  // If true, the "geometry.normal" field will be applied to the MDL state prior to evaluation of the given DF.
  m_execution_context->set_option("include_geometry_normal", true);       // equals default 

  mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(m_neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());

  m_mdl_backend = mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX);

  // Hardcoded values!
  MY_STATIC_ASSERT(NUM_TEXTURE_SPACES == 1 || NUM_TEXTURE_SPACES == 2);
  // The renderer only supports one or two texture spaces.
  // The hair BSDF requires two texture coordinates! 
  // If you do not use the hair BSDF, NUM_TEXTURE_SPACES should be set to 1 for performance reasons.

  if (m_mdl_backend->set_option("num_texture_spaces", std::to_string(NUM_TEXTURE_SPACES).c_str()) != 0)
  {
    return false;
  }
  
  if (m_mdl_backend->set_option("num_texture_results", std::to_string(NUM_TEXTURE_RESULTS).c_str()) != 0)
  {
    return false;
  }
  
  // Use SM 5.0 for Maxwell and above.
  if (m_mdl_backend->set_option("sm_version", "50") != 0)
  {
    return false;
  }
  
  if (m_mdl_backend->set_option("tex_lookup_call_mode", "direct_call") != 0)
  {
    return false;
  }

  //if (enable_derivatives) // == false. Not supported in this renderer
  //{
  //  // Option "texture_runtime_with_derivs": Default is disabled.
  //  // We enable it to get coordinates with derivatives for texture lookup functions.
  //  if (m_mdl_backend->set_option("texture_runtime_with_derivs", "on") != 0)
  //  {
  //    return false;
  //  }
  //}

  if (m_mdl_backend->set_option("inline_aggressively", "on") != 0)
  {
    return false;
  }

  // FIXME Determine what scene data the renderer needs to provide here.
  // FIXME scene_data_names is not a supported option anymore!
  //if (m_mdl_backend->set_option("scene_data_names", "*") != 0)
  //{
  //  return false;
  //}

  m_image_api = m_neuray->get_api_component<mi::neuraylib::IImage_api>();

  return true;
}


void Raytracer::shutdownMDL()
{
  m_shaderConfigurations.clear();
  m_shaders.clear(); // Code handles must be destroyed or there will be memory leaks indicated by MDL.

  m_mapMaterialHashToShaderIndex.clear();

  m_image_api.reset();
  m_mdl_backend.reset();
  m_execution_context.reset();
  m_mdl_factory.reset();
  m_global_scope.reset();
  m_database.reset();
  m_mdl_config.reset();
  m_mdl_compiler.reset();

  m_neuray->shutdown();
}


bool Raytracer::log_messages(mi::neuraylib::IMdl_execution_context* context)
{
  m_last_mdl_error.clear();

  for (mi::Size i = 0; i < context->get_messages_count(); ++i)
  {
    mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(i));
    m_last_mdl_error += message_kind_to_string(message->get_kind());
    m_last_mdl_error += " ";
    m_last_mdl_error += message_severity_to_string(message->get_severity());
    m_last_mdl_error += ": ";
    m_last_mdl_error += message->get_string();
    m_last_mdl_error += "\n";
  }
  return context->get_error_messages_count() == 0;
}


// Query expressions inside the compiled material to determine which direct callable functions need to be generated and
// what the closest hit program needs to call to fully render this material.
void Raytracer::determineShaderConfiguration(const Compile_result& res, ShaderConfiguration& config)
{
  config.is_thin_walled_constant = false;
  config.thin_walled             = false;

  mi::base::Handle<mi::neuraylib::IExpression const> thin_walled_expr(res.compiled_material->lookup_sub_expression("thin_walled"));
  if (thin_walled_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
  {
    config.is_thin_walled_constant = true;

    mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_const(thin_walled_expr->get_interface<mi::neuraylib::IExpression_constant const>());
    mi::base::Handle<mi::neuraylib::IValue_bool const> value_bool(expr_const->get_value<mi::neuraylib::IValue_bool>());

    config.thin_walled = value_bool->get_value();
  }

  mi::base::Handle<mi::neuraylib::IExpression const> surface_scattering_expr(res.compiled_material->lookup_sub_expression("surface.scattering"));
      
  config.is_surface_bsdf_valid = isValidDistribution(surface_scattering_expr.get()); // True if surface.scattering != bsdf().

  config.is_backface_bsdf_valid = false;

  // The backface scattering is only used for thin-walled materials.
  if (!config.is_thin_walled_constant || config.thin_walled)
  {
    // When backface == bsdf() MDL uses the surface scattering on both sides, irrespective of the thin_walled state.
    mi::base::Handle<mi::neuraylib::IExpression const> backface_scattering_expr(res.compiled_material->lookup_sub_expression("backface.scattering"));

    config.is_backface_bsdf_valid = isValidDistribution(backface_scattering_expr.get()); // True if backface.scattering != bsdf().

    if (config.is_backface_bsdf_valid)
    {
      // Only use the backface scattering when it's valid and different from the surface scattering expression.
      config.is_backface_bsdf_valid = (res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_SCATTERING) !=
                                       res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_SCATTERING));
    }
  }

  // Surface EDF.
  mi::base::Handle<mi::neuraylib::IExpression const> surface_edf_expr(res.compiled_material->lookup_sub_expression("surface.emission.emission"));

  config.is_surface_edf_valid = isValidDistribution(surface_edf_expr.get());

  config.is_surface_intensity_constant      = true;
  config.surface_intensity                  = mi::math::Color(0.0f, 0.0f, 0.0f);
  config.is_surface_intensity_mode_constant = true;
  config.surface_intensity_mode             = 0; // == intensity_radiant_exitance;

  if (config.is_surface_edf_valid)
  {
    // Surface emission intensity.
    mi::base::Handle<mi::neuraylib::IExpression const> surface_intensity_expr(res.compiled_material->lookup_sub_expression("surface.emission.intensity"));

    config.is_surface_intensity_constant = false;

    if (surface_intensity_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
    {
      mi::base::Handle<mi::neuraylib::IExpression_constant const> intensity_const(surface_intensity_expr->get_interface<mi::neuraylib::IExpression_constant const>());
      mi::base::Handle<mi::neuraylib::IValue_color const> intensity_color(intensity_const->get_value<mi::neuraylib::IValue_color>());

      if (get_value(intensity_color.get(), config.surface_intensity) == 0)
      {
        config.is_surface_intensity_constant = true;
      }
    }

    // Surface emission mode. This is a uniform and normally the default intensity_radiant_exitance
    mi::base::Handle<mi::neuraylib::IExpression const> surface_intensity_mode_expr(res.compiled_material->lookup_sub_expression("surface.emission.mode"));

    config.is_surface_intensity_mode_constant = false;

    if (surface_intensity_mode_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
    {
      mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_const(surface_intensity_mode_expr->get_interface<mi::neuraylib::IExpression_constant const>());
      mi::base::Handle<mi::neuraylib::IValue_enum const> value_enum(expr_const->get_value<mi::neuraylib::IValue_enum>());

      config.surface_intensity_mode = value_enum->get_value();
          
      config.is_surface_intensity_mode_constant = true;
    }
  }

  // Backface EDF.
  config.is_backface_edf_valid               = false;
  // DEBUG Is any of this needed at all or is the BSDF init() function handling all this?
  config.is_backface_intensity_constant      = true;
  config.backface_intensity                  = mi::math::Color(0.0f, 0.0f, 0.0f);
  config.is_backface_intensity_mode_constant = true;
  config.backface_intensity_mode             = 0; // == intensity_radiant_exitance;
  config.use_backface_edf                    = false;
  config.use_backface_intensity              = false;
  config.use_backface_intensity_mode         = false;

  // A backface EDF is only used on thin-walled materials with a backface.emission.emission != edf()
  if (!config.is_thin_walled_constant || config.thin_walled)
  {
    mi::base::Handle<mi::neuraylib::IExpression const> backface_edf_expr(res.compiled_material->lookup_sub_expression("backface.emission.emission"));

    config.is_backface_edf_valid = isValidDistribution(backface_edf_expr.get());

    if (config.is_backface_edf_valid)
    {
      // Backface emission intensity.
      mi::base::Handle<mi::neuraylib::IExpression const> backface_intensity_expr(res.compiled_material->lookup_sub_expression("backface.emission.intensity"));

      config.is_backface_intensity_constant = false;

      if (backface_intensity_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
      {
        mi::base::Handle<mi::neuraylib::IExpression_constant const> intensity_const(backface_intensity_expr->get_interface<mi::neuraylib::IExpression_constant const>());
        mi::base::Handle<mi::neuraylib::IValue_color const> intensity_color(intensity_const->get_value<mi::neuraylib::IValue_color>());

        if (get_value(intensity_color.get(), config.backface_intensity) == 0)
        {
          config.is_backface_intensity_constant = true;
        }
      }

      // Backface emission mode. This is a uniform and normally the default intensity_radiant_exitance.
      mi::base::Handle<mi::neuraylib::IExpression const> backface_intensity_mode_expr(res.compiled_material->lookup_sub_expression("backface.emission.mode"));

      config.is_backface_intensity_mode_constant = false;

      if (backface_intensity_mode_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
      {
        mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_const(backface_intensity_mode_expr->get_interface<mi::neuraylib::IExpression_constant const>());
        mi::base::Handle<mi::neuraylib::IValue_enum const> value_enum(expr_const->get_value<mi::neuraylib::IValue_enum>());

        config.backface_intensity_mode = value_enum->get_value();

        config.is_backface_intensity_mode_constant = true;
      }

      // When surface and backface expressions are identical, reuse the surface expression to generate less code.
      config.use_backface_edf = (res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_EDF_EMISSION) !=
                                 res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_EDF_EMISSION));

      // If the surface and backface emission use different intensities then use the backface emission intensity.
      config.use_backface_intensity = (res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_INTENSITY) !=
                                       res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_INTENSITY));

      // If the surface and backface emission use different modes (radiant exitance vs. power) then use the backface emission intensity mode.
      config.use_backface_intensity_mode = (res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_MODE) !=
                                            res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_MODE));
    }
  }

  config.is_ior_constant = true;
  config.ior             = mi::math::Color(1.0f, 1.0f, 1.0f);

  mi::base::Handle<mi::neuraylib::IExpression const> ior_expr(res.compiled_material->lookup_sub_expression("ior"));
  if (ior_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
  {
    mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_const(ior_expr->get_interface<mi::neuraylib::IExpression_constant const>());
    mi::base::Handle<mi::neuraylib::IValue_color const> value_color(expr_const->get_value<mi::neuraylib::IValue_color>());

    if (get_value(value_color.get(), config.ior) == 0)
    {
      config.is_ior_constant = true;
    }
  }
  else
  {
    config.is_ior_constant = false;
  }

  // If the VDF is valid, it is the df::anisotropic_vdf(). ::vdf() is not a valid VDF.
  // Though there aren't any init, sample, eval or pdf functions genereted for a VDF.
  mi::base::Handle<mi::neuraylib::IExpression const> volume_vdf_expr(res.compiled_material->lookup_sub_expression("volume.scattering"));

  config.is_vdf_valid = isValidDistribution(volume_vdf_expr.get());

  // Absorption coefficient. Can be used without valid VDF.
  config.is_absorption_coefficient_constant = true;  // Default to constant and no absorption.
  config.use_volume_absorption              = false; // If there is no abosorption, the absorption coefficient is constant zero.
  config.absorption_coefficient             = mi::math::Color(0.0f, 0.0f, 0.0f); // No absorption.

  mi::base::Handle<mi::neuraylib::IExpression const> volume_absorption_coefficient_expr(res.compiled_material->lookup_sub_expression("volume.absorption_coefficient"));

  if (volume_absorption_coefficient_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
  {
    mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_const(volume_absorption_coefficient_expr->get_interface<mi::neuraylib::IExpression_constant const>());
    mi::base::Handle<mi::neuraylib::IValue_color const> value_color(expr_const->get_value<mi::neuraylib::IValue_color>());

    if (get_value(value_color.get(), config.absorption_coefficient) == 0)
    {
      config.is_absorption_coefficient_constant = true;

      if (config.absorption_coefficient[0] != 0.0f || config.absorption_coefficient[1] != 0.0f || config.absorption_coefficient[2] != 0.0f)
      {
        config.use_volume_absorption = true;
      }
    }
  }
  else
  {
    config.is_absorption_coefficient_constant = false;
    config.use_volume_absorption              = true;
  }

  // Scattering coefficient. Only used when there is a valid VDF. 
  config.is_scattering_coefficient_constant = true; // Default to constant and no scattering. Assumes invalid VDF.
  config.use_volume_scattering              = false;
  config.scattering_coefficient             = mi::math::Color(0.0f, 0.0f, 0.0f); // No scattering
  
  // Directional bias (Henyey_Greenstein g factor.) Only used when there is a valid VDF and volume scattering coefficient not zero.
  config.is_directional_bias_constant = true; 
  config.directional_bias             = 0.0f; 

  // The anisotropic_vdf() is the only valid VDF. 
  // The scattering_coefficient, directional_bias (and emission_intensity) are only needed when there is a valid VDF.
  if (config.is_vdf_valid)
  {
    mi::base::Handle<mi::neuraylib::IExpression const> volume_scattering_coefficient_expr(res.compiled_material->lookup_sub_expression("volume.scattering_coefficient"));

    if (volume_scattering_coefficient_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
    {
      mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_const(volume_scattering_coefficient_expr->get_interface<mi::neuraylib::IExpression_constant const>());
      mi::base::Handle<mi::neuraylib::IValue_color const> value_color(expr_const->get_value<mi::neuraylib::IValue_color>());

      if (get_value(value_color.get(), config.scattering_coefficient) == 0)
      {
        config.is_scattering_coefficient_constant = true;

        if (config.scattering_coefficient[0] != 0.0f || config.scattering_coefficient[1] != 0.0f || config.scattering_coefficient[2] != 0.0f)
        {
          config.use_volume_scattering = true;
        }
      }
    }
    else
    {
      config.is_scattering_coefficient_constant = false;
      config.use_volume_scattering              = true;
    }

    mi::base::Handle<mi::neuraylib::IExpression const> volume_directional_bias_expr(res.compiled_material->lookup_sub_expression("volume.scattering.directional_bias"));

    if (volume_directional_bias_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
    {
      config.is_directional_bias_constant = true;

      mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_const(volume_directional_bias_expr->get_interface<mi::neuraylib::IExpression_constant const>());
      mi::base::Handle<mi::neuraylib::IValue_float const> value_float(expr_const->get_value<mi::neuraylib::IValue_float>());

      // 0.0f is isotropic. No need to distinguish. The sampleHenyeyGreenstein() function takes this as parameter anyway.
      config.directional_bias = value_float->get_value(); 
    }
    else
    {
      config.is_directional_bias_constant = false;
    }

    // volume.scattering.emission_intensity is not supported by this renderer.
    // Also the volume absorption and scattering coefficients are assumed to be homogeneous in this renderer.
  }

  // geometry.displacement is not supported by this renderer.

  // geometry.normal is automatically handled because of set_option("include_geometry_normal", true);

  config.cutout_opacity             = 1.0f; // Default is fully opaque.
  config.is_cutout_opacity_constant = res.compiled_material->get_cutout_opacity(&config.cutout_opacity); // This sets cutout opacity to -1.0 when it's not constant!
  config.use_cutout_opacity         = !config.is_cutout_opacity_constant || config.cutout_opacity < 1.0f;

  mi::base::Handle<mi::neuraylib::IExpression const> hair_bsdf_expr(res.compiled_material->lookup_sub_expression("hair"));
  
  config.is_hair_bsdf_valid = isValidDistribution(hair_bsdf_expr.get()); // True if hair != hair_bsdf().
}


void Raytracer::initMaterialsMDL(std::vector<MaterialMDL*>& materialsMDL)
{
  // This will compile the material to OptiX PTX code and build the OptiX program and texture data on all devices
  // and track the material configuration and parameters stored inside the Application class to be able to build the GUI.
  for (MaterialMDL* materialMDL : materialsMDL)
  {
    initMaterialMDL(materialMDL);
  }

  // After all MDL material references have been handled and the device side data has been allocated, upload the necessary data to the GPU.
  for (size_t i = 0; i < m_devicesActive.size(); ++i)
  {
    m_devicesActive[i]->initTextureHandler(materialsMDL);
  }
}


void Raytracer::initMaterialMDL(MaterialMDL* material)
{
  // This function is called per unique material reference.
  // No need to check for duplicate reference definitions.

  mi::base::Handle<mi::neuraylib::ITransaction> handleTransaction = mi::base::make_handle<mi::neuraylib::ITransaction>(m_global_scope->create_transaction());
  mi::neuraylib::ITransaction* transaction = handleTransaction.get();

  // Local scope for all handles used inside the Compile_result.
  {
    Compile_result res;

    // Split into separate functions to make the Neuray handles and transaction scope lifetime handling automatic.
    // When the function was successful, the Compile_result contains all information required to setup the device resources.
    const bool valid = compileMaterial(transaction, material, res);
  
    material->setIsValid(valid);

    if (valid)
    {
      // Create the OptiX programs on all devices.
      for (size_t device = 0; device < m_devicesActive.size(); ++device)
      {
        m_devicesActive[device]->compileMaterial(transaction, material, res, m_shaderConfigurations[material->getShaderIndex()]);
      }

      // Prepare 2D and 3D textures.
      const bool allowSharingTex = ((m_peerToPeer & P2P_TEX) != 0); // Material texture sharing (very cheap).
      
      // Create the CUDA texture arrays on the devices with peer-to-peer sharing when enabled.
      for (mi::Size idxRes = 1; idxRes < res.textures.size(); ++idxRes) // The zeroth index is the invalid texture.
      {
        bool first = true; // Only append each texture index to the MaterialMDL m_indicesToTextures vector once.

        if (allowSharingTex)
        {
          for (const auto& island : m_islands) // Resource sharing only works across devices inside a peer-to-peer island.
          {
            const int deviceHome = getDeviceHome(island);

            const TextureMDLHost* shared = m_devicesActive[deviceHome]->prepareTextureMDL(transaction,
                                                                                          m_image_api,
                                                                                          res.textures[idxRes].db_name.c_str(),
                                                                                          res.textures[idxRes].shape);
            
            // Update the MaterialMDL vector of texture indices into the per-device cache only once!
            // This is per material reference and all caches are the same size on the all devices, shared CUarrays or not.
            if (first && shared != nullptr)
            {
              material->m_indicesToTextures.push_back(shared->m_index);
              first = false;
            }

            for (auto device : island)
            {
              if (device != deviceHome)
              {
                m_devicesActive[device]->shareTextureMDL(shared,
                                                         res.textures[idxRes].db_name.c_str(),
                                                         res.textures[idxRes].shape);
              }
            }
          }
        }
        else
        {
          for (size_t device = 0; device < m_devicesActive.size(); ++device)
          {
            const TextureMDLHost* texture = m_devicesActive[device]->prepareTextureMDL(transaction,
                                                                                       m_image_api,
                                                                                       res.textures[idxRes].db_name.c_str(),
                                                                                       res.textures[idxRes].shape);
            if (texture == nullptr)
            {
              std::cerr << "ERROR: initMaterialMDL(): prepareTextureMDL() failed for " << res.textures[idxRes].db_name << '\n';
            }
            else if (device == 0) // Only store the index once into the vector at the MaterialMDL.
            {
              material->m_indicesToTextures.push_back(texture->m_index);
            }
          }
        }
      }

      const bool allowSharingMBSDF = ((m_peerToPeer & P2P_MBSDF) != 0); // MBSDF texture and CDF sharing (medium expensive due to the memory traffic on the CDFs)

      // Prepare Bsdf_measurements.
      for (mi::Size idxRes = 1; idxRes < res.bsdf_measurements.size(); ++idxRes) // The zeroth index is the invalid Bsdf_measurement.
      {
        bool first = true;

        if (allowSharingMBSDF)
        {
          for (const auto& island : m_islands) // Resource sharing only works across devices inside a peer-to-peer island.
          {
            const int deviceHome = getDeviceHome(island);

            const MbsdfHost* shared = m_devicesActive[deviceHome]->prepareMBSDF(transaction, res.target_code.get(), idxRes);

            if (shared == nullptr)
            {
              std::cerr << "ERROR: initMaterialMDL(): prepareMBSDF() failed for BSDF measurement " << idxRes << '\n';
            }
            else if (first)
            {
              material->m_indicesToMBSDFs.push_back(shared->m_index);
              first = false;
            }

            for (auto device : island)
            {
              if (device != deviceHome)
              {
                m_devicesActive[device]->shareMBSDF(shared);
              }
            }
          }
        }
        else
        {
          for (size_t device = 0; device < m_devicesActive.size(); ++device)
          {
            const MbsdfHost* mbsdf = m_devicesActive[device]->prepareMBSDF(transaction, res.target_code.get(), idxRes);

            if (mbsdf == nullptr)
            {
              std::cerr << "ERROR: initMaterialMDL(): prepareMBSDF() failed for BSDF measurement " << idxRes << '\n';
            }
            else if (device == 0) // Only store the index once into the vector at the MaterialMDL.
            {
              material->m_indicesToMBSDFs.push_back(mbsdf->m_index);
            }
          }
        }
      }

      const bool allowSharingLightprofile = ((m_peerToPeer & P2P_IES) != 0); // IES texture and CDF sharing (medium expensive due to the memory traffic on the CDFs)

      // Prepare Light_profiles.
      for (mi::Size idxRes = 1; idxRes < res.light_profiles.size(); ++idxRes) // The zeroth index is the invalid light profile.
      {
        bool first = true;

        if (allowSharingLightprofile)
        {
          for (const auto& island : m_islands) // Resource sharing only works across devices inside a peer-to-peer island.
          {
            const int deviceHome = getDeviceHome(island);

            const LightprofileHost* shared = m_devicesActive[deviceHome]->prepareLightprofile(transaction, res.target_code.get(), idxRes);

            if (shared == nullptr)
            {
              std::cerr << "ERROR: initMaterialMDL(): prepareLightprofile() failed for light profile " << idxRes << '\n';
            }
            else if (first)
            {
              material->m_indicesToLightprofiles.push_back(shared->m_index);
              first = false;
            }

            for (auto device : island)
            {
              if (device != deviceHome)
              {
                m_devicesActive[device]->shareLightprofile(shared);
              }
            }
          }
        }
        else
        {
          for (size_t device = 0; device < m_devicesActive.size(); ++device)
          {
            const LightprofileHost* profile = m_devicesActive[device]->prepareLightprofile(transaction, res.target_code.get(), idxRes);

            if (profile == nullptr)
            {
              std::cerr << "ERROR: initMaterialMDL(): prepareLightprofile() failed for light profile " << idxRes << '\n';
            }
            else if (device == 0) // Only store the index once into the vector at the MaterialMDL.
            {
              material->m_indicesToLightprofiles.push_back(profile->m_index);
            }
          }
        }
      }
    }
  }

  transaction->commit();
}


bool Raytracer::compileMaterial(mi::neuraylib::ITransaction* transaction, MaterialMDL* materialMDL, Compile_result& res)
{
  // Build the fully qualified MDL module name.
  // The *.mdl file path has been converted to the proper OS format during input.
  std::string path = materialMDL->getPath();

  // Path needs to end with ".mdl" so any path with 4 or less characters cannot be a valid path name.
  if (path.size() <= 4)
  {
    std::cerr << "ERROR: compileMaterial() Path name " << path << " too short.\n";
    return false;
  }

  const std::string::size_type last = path.size() - 4;

  if (path.substr(last, path.size()) != std::string(".mdl"))
  {
    std::cerr << "ERROR: compileMaterial() Path name " << path << " not matching \".mdl\".\n";
    return false;
  }

  std::string module_name = buildModuleName(path.substr(0, last));

  // Get everything to load the module.
  mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(m_neuray->get_api_component<mi::neuraylib::IMdl_factory>()); // FIXME Redundant, could use m_mdl_factory.

  mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

  // Create an execution context for options and error message handling
  mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(mdl_factory->create_execution_context());

  mi::Sint32 reason = mdl_impexp_api->load_module(transaction, module_name.c_str(), context.get());
  if (reason < 0)
  {
    std::cerr << "ERROR: compileMaterial() Failed to load module " << module_name << '\n';
    switch (reason)
    {
      // case 1: // Success (module exists already, loading from file was skipped).
      // case 0: // Success (module was actually loaded from file).
      case -1:
        std::cerr << "The module name is invalid or a NULL pointer.\n";
        break;
      case -2:
        std::cerr << "Failed to find or to compile the module.\n";
        break;
      case -3:
        std::cerr << "The database name for an imported module is already in use but is not an MDL module,\n";
        std::cerr << "or the database name for a definition in this module is already in use.\n";
        break;
      case -4:
        std::cerr << "Initialization of an imported module failed.\n";
        break;
      default:
        std::cerr << "Unexpected return value " << reason << " from IMdl_impexp_api::load_module().\n";
        MY_ASSERT(!"Unexpected return value from IMdl_compiler::load_module()");
        break;
    }
  }

  if (!log_messages(context.get()))
  {
    return false;
  }

  // Get the database name for the module we loaded.
  mi::base::Handle<const mi::IString> module_db_name(mdl_factory->get_db_module_name(module_name.c_str()));

  // Note that the lifetime of this module handle must end before the transaction->commit() or there will be warnings.
  // This is automatically handled by placing the transaction into the caller.
  mi::base::Handle<const mi::neuraylib::IModule> module(transaction->access<mi::neuraylib::IModule>(module_db_name->get_c_str()));
  if (!module)
  {
      std::cerr << "ERROR: compileMaterial() Failed to access the loaded module " << module_db_name->get_c_str() << '\n';
      return false;
  }

  // Build the fully qualified data base name of the material.
  const std::string material_simple_name = materialMDL->getName();
    
  std::string material_db_name = std::string(module_db_name->get_c_str()) + "::" + material_simple_name;

  material_db_name = add_missing_material_signature(module.get(), material_db_name);

  if (material_db_name.empty())
  {
    std::cerr << "ERROR: compileMaterial() Failed to find the material " + material_simple_name + " in the module " + module_name + ".\n";
    return false;
  }

  // Compile the material.

  // Create a material instance from the material definition with the default arguments.
  mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(transaction->access<mi::neuraylib::IFunction_definition>(material_db_name.c_str()));
  if (!material_definition)
  {
    std::cerr << "ERROR: compileMaterial() Material definition could not be created for " << material_simple_name << '\n';
    return false;
  }

  mi::Sint32 ret = 0;
  mi::base::Handle<mi::neuraylib::IFunction_call> material_instance(material_definition->create_function_call(0, &ret));
  if (ret != 0)
  {
    std::cerr << "ERROR: compileMaterial() Instantiating material " + material_simple_name + " failed";
    return false;
  }

  // Create a compiled material.
  // DEBUG Experiment with instance compilation as well to see how the performance changes.
  mi::Uint32 flags = mi::neuraylib::IMaterial_instance::CLASS_COMPILATION;

  mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance2(material_instance->get_interface<mi::neuraylib::IMaterial_instance>());

  res.compiled_material = material_instance2->create_compiled_material(flags, context.get());
  if (!log_messages(context.get()))
  {
    std::cerr << "ERROR: compileMaterial() create_compiled_material() failed.\n";
    return false;
  }

  // Check if the target code has already been generated for another material reference name and reuse the existing target code.
  int indexShader = -1; // Invalid index.

  mi::base::Uuid material_hash = res.compiled_material->get_hash();

  std::map<mi::base::Uuid, int>::const_iterator it = m_mapMaterialHashToShaderIndex.find(material_hash);
  if (it != m_mapMaterialHashToShaderIndex.end())
  {
    indexShader = it->second;

    res.target_code = m_shaders[indexShader];

    // Initialize with body resources always being required.
    // Mind that the zeroth resource is the invalid resource.
    if (res.target_code->get_body_texture_count() > 0)
    {
      for (mi::Size i = 1, n = res.target_code->get_body_texture_count(); i < n; ++i)
      {
        res.textures.emplace_back(res.target_code->get_texture(i), res.target_code->get_texture_shape(i));
      }
    }

    if (res.target_code->get_body_light_profile_count() > 0)
    {
      for (mi::Size i = 1, n = res.target_code->get_body_light_profile_count(); i < n; ++i)
      {
        res.light_profiles.emplace_back(res.target_code->get_light_profile(i));
      }
    }

    if (res.target_code->get_body_bsdf_measurement_count() > 0)
    {
      for (mi::Size i = 1, n = res.target_code->get_body_bsdf_measurement_count(); i < n; ++i)
      {
        res.bsdf_measurements.emplace_back(res.target_code->get_bsdf_measurement(i));
      }
    }

    if (res.target_code->get_argument_block_count() > 0)
    {
      // Create argument block for the new compiled material and additional resources
      mi::base::Handle<Resource_callback> res_callback(new Resource_callback(transaction, res.target_code.get(), res));
          
      res.argument_block = res.target_code->create_argument_block(0, res.compiled_material.get(), res_callback.get());
    }
  }
  else
  {
    // Generate new target code. 
    indexShader = static_cast<int>(m_shaders.size()); // The amount of different shaders in the code cache gives the next index.

    // Determine the material configuration by checking which minimal amount of expressions need to be generated as direct callable programs.
    ShaderConfiguration config;
      
    determineShaderConfiguration(res, config);

    // Build the required function descriptions for the expression required by the material configuration.
    std::vector<mi::neuraylib::Target_function_description> descs;
      
    const std::string suffix = std::to_string(indexShader);

    // These are all expressions required for a material which does everything supported in this renderer.
    // The Target_function_description only stores the C-pointers to the base names!
    // Make sure these are not destroyed as long as the descs vector is used.
    std::string name_init                           = "__direct_callable__init"                          + suffix;
    std::string name_thin_walled                    = "__direct_callable__thin_walled"                   + suffix;
    std::string name_surface_scattering             = "__direct_callable__surface_scattering"            + suffix;
    std::string name_surface_emission_emission      = "__direct_callable__surface_emission_emission"     + suffix;
    std::string name_surface_emission_intensity     = "__direct_callable__surface_emission_intensity"    + suffix;
    std::string name_surface_emission_mode          = "__direct_callable__surface_emission_mode"         + suffix;
    std::string name_backface_scattering            = "__direct_callable__backface_scattering"           + suffix;
    std::string name_backface_emission_emission     = "__direct_callable__backface_emission_emission"    + suffix;
    std::string name_backface_emission_intensity    = "__direct_callable__backface_emission_intensity"   + suffix;
    std::string name_backface_emission_mode         = "__direct_callable__backface_emission_mode"        + suffix;
    std::string name_ior                            = "__direct_callable__ior"                           + suffix;
    std::string name_volume_absorption_coefficient  = "__direct_callable__volume_absorption_coefficient" + suffix;
    std::string name_volume_scattering_coefficient  = "__direct_callable__volume_scattering_coefficient" + suffix;
    std::string name_volume_directional_bias        = "__direct_callable__volume_directional_bias"       + suffix;
    std::string name_geometry_cutout_opacity        = "__direct_callable__geometry_cutout_opacity"       + suffix;
    std::string name_hair_bsdf                      = "__direct_callable__hair"                          + suffix;

    // Centralize the init functions in a single material init().
    // This will only save time when there would have been multiple init functions inside the shader.
    // Also for very complicated materials with cutout opacity this is most likely a loss,
    // because the geometry.cutout is only needed inside the anyhit program and 
    // that doesn't need additional evalations for the BSDFs, EDFs, or VDFs at that point.
    descs.push_back(mi::neuraylib::Target_function_description("init", name_init.c_str()));

    if (!config.is_thin_walled_constant)
    {
      descs.push_back(mi::neuraylib::Target_function_description("thin_walled", name_thin_walled.c_str()));
    }
    if (config.is_surface_bsdf_valid)
    {
      descs.push_back(mi::neuraylib::Target_function_description("surface.scattering", name_surface_scattering.c_str()));
    }
    if (config.is_surface_edf_valid)
    {
      descs.push_back(mi::neuraylib::Target_function_description("surface.emission.emission", name_surface_emission_emission.c_str()));
      if (!config.is_surface_intensity_constant)
      {
        descs.push_back(mi::neuraylib::Target_function_description("surface.emission.intensity", name_surface_emission_intensity.c_str()));
      }
      if (!config.is_surface_intensity_mode_constant)
      {
        descs.push_back(mi::neuraylib::Target_function_description("surface.emission.mode", name_surface_emission_mode.c_str()));
      }
    }
    if (config.is_backface_bsdf_valid)
    {
      descs.push_back(mi::neuraylib::Target_function_description("backface.scattering", name_backface_scattering.c_str()));
    }
    if (config.is_backface_edf_valid)
    {
      if (config.use_backface_edf)
      {
        descs.push_back(mi::neuraylib::Target_function_description("backface.emission.emission", name_backface_emission_emission.c_str()));
      }
      if (config.use_backface_intensity && !config.is_backface_intensity_constant)
      {
        descs.push_back(mi::neuraylib::Target_function_description("backface.emission.intensity", name_backface_emission_intensity.c_str()));
      }
      if (config.use_backface_intensity_mode && !config.is_backface_intensity_mode_constant)
      {
        descs.push_back(mi::neuraylib::Target_function_description("backface.emission.mode", name_backface_emission_mode.c_str()));
      }
    }
    if (!config.is_ior_constant)
    {
      descs.push_back(mi::neuraylib::Target_function_description("ior", name_ior.c_str()));
    }
    if (!config.is_absorption_coefficient_constant)
    {
      descs.push_back(mi::neuraylib::Target_function_description("volume.absorption_coefficient", name_volume_absorption_coefficient.c_str()));
    }
    if (config.is_vdf_valid)
    {
      // The MDL SDK is not generating functions for VDFs! This would fail in ILink_unit::add_material().
      //descs.push_back(mi::neuraylib::Target_function_description("volume.scattering", name_volume_scattering.c_str()));

      // The scattering coefficient and directional bias are not used when there is no valid VDF.
      if (!config.is_scattering_coefficient_constant)
      {
        descs.push_back(mi::neuraylib::Target_function_description("volume.scattering_coefficient", name_volume_scattering_coefficient.c_str()));
      }

      if (!config.is_directional_bias_constant)
      {
        descs.push_back(mi::neuraylib::Target_function_description("volume.scattering.directional_bias", name_volume_directional_bias.c_str()));
      }

      // volume.scattering.emission_intensity is not implemented.
    }

    // geometry.displacement is not implemented.
 
    // geometry.normal is automatically handled because of set_option("include_geometry_normal", true);

    if (config.use_cutout_opacity)
    {
      descs.push_back(mi::neuraylib::Target_function_description("geometry.cutout_opacity", name_geometry_cutout_opacity.c_str()));
    }
    if (config.is_hair_bsdf_valid)
    {
      descs.push_back(mi::neuraylib::Target_function_description("hair", name_hair_bsdf.c_str()));
    }

    // Generate target code for the compiled material.
    mi::base::Handle<mi::neuraylib::ILink_unit> link_unit(m_mdl_backend->create_link_unit(transaction, context.get()));

    mi::Sint32 reason = link_unit->add_material(res.compiled_material.get(), descs.data(), descs.size(), context.get());
    if (reason != 0)
    {
      std::cerr << "ERROR: compileMaterial() link_unit->add_material() returned " << reason << '\n';
    }
    if (!log_messages(context.get()))
    {
      std::cerr << "ERROR: compileMaterial() On link_unit->add_material()\n";
      return false;
    }

    res.target_code = mi::base::Handle<const mi::neuraylib::ITarget_code>(m_mdl_backend->translate_link_unit(link_unit.get(), context.get()));
    if (!log_messages(context.get()))
    {
      std::cerr << "ERROR: compileMaterial() On m_mdl_backend->translate_link_unit()\n";
      return false;
    }

    // Store the new shader index in the map.
    m_mapMaterialHashToShaderIndex[material_hash] = indexShader;
        
    // These two vectors are the same size:
    // Store the target code handle inside the shader cache.
    m_shaders.push_back(res.target_code);
    // Store the shader configuration to be able to build the required direct callables on the device later.
    m_shaderConfigurations.push_back(config);

    // Add all used resources. The zeroth entry is the invalid resource.
    for (mi::Size i = 1, n = res.target_code->get_texture_count(); i < n; ++i)
    {
      res.textures.emplace_back(res.target_code->get_texture(i), res.target_code->get_texture_shape(i));
    }

    if (res.target_code->get_light_profile_count() > 0)
    {
      for (mi::Size i = 1, n = res.target_code->get_light_profile_count(); i < n; ++i)
      {
        res.light_profiles.emplace_back(res.target_code->get_light_profile(i));
      }
    }

    if (res.target_code->get_bsdf_measurement_count() > 0)
    {
      for (mi::Size i = 1, n = res.target_code->get_bsdf_measurement_count(); i < n; ++i)
      {
        res.bsdf_measurements.emplace_back(res.target_code->get_bsdf_measurement(i));
      }
    }

    if (res.target_code->get_argument_block_count() > 0)
    {
      res.argument_block = res.target_code->get_argument_block(0);
    }

#if 0 // DEBUG Print or write the PTX code when a new shader has been generated.
    if (res.target_code)
    {
      std::string code = res.target_code->get_code();

      // Print generated PTX source code to the console.
      //std::cout << code << std::endl;

      // Dump generated PTX source code to a local folder for offline comparisons.
      const std::string filename = std::string("./mdl_ptx/") + material_simple_name + std::string("_") + getDateTime() + std::string(".ptx");

      saveString(filename, code);
    }
#endif // DEBUG 

  } // End of generating new target code.

  // Build the material information for this material reference.
  mi::base::Handle<mi::neuraylib::ITarget_value_layout const> arg_layout;

  if (res.target_code->get_argument_block_count() > 0)
  {
    arg_layout = res.target_code->get_argument_block_layout(0);
  }

  // Store the material information per reference inside the MaterialMDL structure.
  materialMDL->storeMaterialInfo(indexShader,
                                  material_definition.get(),
                                  res.compiled_material.get(),
                                  arg_layout.get(),
                                  res.argument_block.get());

  // Now that the code and the resources are setup as MDL handles,
  // call into the Device class to setup the CUDA and OptiX resources.

  return true;
}


bool Raytracer::isEmissiveShader(const int indexShader) const
{
  bool result = false;

  if (0 <= indexShader && indexShader < m_shaderConfigurations.size())
  {
    result = m_shaderConfigurations[indexShader].isEmissive();
  }
  else
  {
    std::cout << "ERROR: isEmissiveShader() called with invalid index " << indexShader << '\n';
  }

  return result;
}

