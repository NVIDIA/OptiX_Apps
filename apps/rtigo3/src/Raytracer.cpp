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
  std::cout << "Driver Version  = " << major << "." << minor << std::endl;
  
  CU_CHECK( cuDeviceGetCount(&m_visibleDevices) );
  std::cout << "Device Count    = " << m_visibleDevices << std::endl;
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

bool Raytracer::enablePeerAccess()
{
  // Build the peer-to-peer connection matrix.
  const int size = static_cast<int>(m_activeDevices.size());
  MY_ASSERT(size <= 32); 
  
  // Peer-to-peer access is encoded in a bitfield of uint32 entries.
  m_peerConnections.resize(size);

  for (int i = 0; i < size; ++i) // Home device i.
  {
    m_peerConnections[i] = 0;

    for (int j = 0; j < size; ++j) // Peer device j.
    {
      if (i != j)
      {
        int canAccessPeer = 0;
        CU_CHECK( cuDeviceCanAccessPeer(&canAccessPeer, (CUdevice) i, (CUdevice) j) );
        if (canAccessPeer != 0)
        {
          // FIXME Move into Device class?
          CU_CHECK( cuCtxSetCurrent(m_activeDevices[i]->m_cudaContext) );                // This current context
          CUresult result = cuCtxEnablePeerAccess(m_activeDevices[j]->m_cudaContext, 0); // can access the peer context's memory.
          if (result == CUDA_SUCCESS)
          {
            m_peerConnections[i] |= (1 << j); // Set the connection bit if the enable succeeded.
            std::cout << "enablePeerAccess(): Device " << i << " can access peer device " << j << std::endl;
          }
          else
          {
            std::cerr << "WARNING: cuCtxEnablePeerAccess() between devices (" << i << ", " << j << ") failed with CUresult " << result << std::endl;
          }
        }
      }
      else
      {
        // Trivial case (i == j) which is just the same memory.
        m_peerConnections[i] |= (1 << j); // Set the bit on the diagonal of the connection matrix.
      }
    }
  }
  // Note that this function has changed the current context!

  // Now use the peer-to-peer connection matrix to build peer-to-peer islands.
  // First fill a vector with all device indices which have not been assigned to an island.
  std::vector<int> unassigned;

  for (int i = 0; i < size; ++i)
  {
    unassigned.push_back(i);
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

      // Check if this peer device is accessible by all other devices in the island.
      for (size_t i = 0; i < island.size(); ++i)
      {
        const int home = island[i];
        const int peer = *it;

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
    m_peerIslands.push_back(island);
  }

  std::ostringstream text;

  text << m_peerIslands.size() << " peer-to-peer islands: ";
  for (size_t i = 0; i < m_peerIslands.size(); ++i)
  {
    std::vector<int> const& island = m_peerIslands[i];
    text << "(";
    for (size_t j = 0; j < island.size(); ++j)
    {
      text << island[j];
      if (j + 1 < island.size())
      {
        text << ", ";
      }
    }
    text << ")";
    if (i + 1 < m_peerIslands.size())
    {
      text << " + ";
    }
  }
  std::cout << text.str() << std::endl;

  // FIXME The RS_INTERACTIVE_MULTI_GPU_PEER_ACCESS strategy is not implemented to work with more than one island!
  if (m_strategy == RS_INTERACTIVE_MULTI_GPU_PEER_ACCESS && 1 < m_peerIslands.size())
  {
    std::cerr << "ERROR: enablePeerAccess() RS_INTERACTIVE_MULTI_GPU_PEER_ACCESS strategy is not supported with more than one peer-to-peer island." << std::endl;
    return false;
  }

  return true;
}

void Raytracer::disablePeerAccess()
{
  const int size = static_cast<int>(m_activeDevices.size());
  MY_ASSERT(size <= 32); 
  
  // Peer-to-peer access is encoded in a bitfield of uint32 entries.
  for (int i = 0; i < size; ++i) // Home device i.
  {
    for (int j = 0; j < size; ++j) // Peer device j.
    {
      if (i != j && (m_peerConnections[i] & (1 << j)) != 0)
      {
        CU_CHECK( cuCtxSetCurrent(m_activeDevices[i]->m_cudaContext) );        // Home context.
        CU_CHECK( cuCtxDisablePeerAccess(m_activeDevices[j]->m_cudaContext) ); // Peer context.
        
        m_peerConnections[i] &= ~(1 << j);
      }
    }
  }
  // Note that this function has changed the current context.

  m_peerIslands.clear();  // No peer-to-peer islands anymore,
  
  // Each device is its own island now. Set it like that for paranoia.
  for (int i = 0; i < size; ++i) // Home device i.
  {
    std::vector<int> island;

    island.push_back(i);

    m_peerIslands.push_back(island);
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
