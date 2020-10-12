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

#pragma once
 
#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "inc/Device.h"
#include "inc/MaterialGUI.h"
#include "inc/Picture.h"
#include "inc/SceneGraph.h"
#include "inc/Texture.h"
#include "inc/NVMLImpl.h"

#include "shaders/system_data.h"

#include <map>
#include <memory>
#include <vector>


// Virtual Raytracer base class for all derived Raytracer rendering strategies.
// The default behaviour of the implemented base class functions is to pass-through all calls to all devices.

class Raytracer
{
public:
  Raytracer(RendererStrategy strategy,
            const int interop,
            const unsigned int tex, 
            const unsigned int pbo);
  virtual ~Raytracer();

  int matchUUID(const char* uuid);
  int matchLUID(const char* luid, const unsigned int nodeMask);
  bool enablePeerAccess();   // Calculates peer-to-peer access bit matrix in m_peerConnections and the m_peerIslands. Returns false when more than one island is found!
  void disablePeerAccess();  // Clear the peer-to-peer islands. Afterwards each device is its own island.
  void synchronize();        // Needed for the benchmark to wait for all asynchronous rendering to have finished.

  virtual void initTextures(std::map<std::string, Picture*> const& mapOfPictures);
  virtual void initCameras(std::vector<CameraDefinition> const& cameras);
  virtual void initLights(std::vector<LightDefinition> const& lights);
  virtual void initMaterials(std::vector<MaterialGUI> const& materialsGUI);
  virtual void initScene(std::shared_ptr<sg::Group> root, const unsigned int numGeometries);
  virtual void initState(DeviceState const& state);

  // Update functions should be replaced with NOP functions in a derived batch renderer because the device functions are fully asynchronous then.
  virtual void updateCamera(const int idCamera, CameraDefinition const& camera);
  virtual void updateLight(const int idLight, LightDefinition const& light);
  virtual void updateMaterial(const int idMaterial, MaterialGUI const& src);
  virtual void updateState(DeviceState const& state);

  // Abstract functions must be implemented by each derived Raytracer per strategy individually.
  virtual unsigned int render() = 0;
  virtual void updateDisplayTexture() = 0;
  virtual const void* getOutputBufferHost() = 0;

private:
  bool activeNVLINK(const int home, const int peer) const;
  int findActiveDevice(const unsigned int domain, const unsigned int bus, const unsigned int device) const;

public:
  RendererStrategy m_strategy;  // Constructor arguments
  int              m_interop;
  unsigned int     m_tex;
  unsigned int     m_pbo;

  bool m_isValid;

  int                  m_visibleDevices;    // The number of visible CUDA devices. (What you can control via the CUDA_VISIBLE_DEVICES environment variable.)
  int                  m_deviceOGL;         // The first device which matches with the OpenGL LUID and node mask. -1 when there was no match.
  unsigned int         m_activeDevicesMask; // The bitmask marking the actually enabled devices.
  std::vector<Device*> m_activeDevices;

  unsigned int m_iterationIndex;  // Tracks which frame is currently raytraced.
  unsigned int m_samplesPerPixel; // This is samplesSqrt squared. Rendering end-condition is: m_iterationIndex == m_samplesPerPixel.

  std::vector<unsigned int>       m_peerConnections; // Bitfield indicating peer-to-peer access between devices. Indexing is m_peerConnections[home] & (1 << peer)
  std::vector< std::vector<int> > m_islands;         // Vector with vector of device indices (not ordinals) building a peer-to-peer island.

  NVMLImpl m_nvml;
};

#endif // RAYTRACER_H
