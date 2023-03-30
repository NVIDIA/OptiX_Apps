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

#pragma once
 
#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "inc/Device.h"
//#include "inc/MaterialGUI.h"
#include "inc/Picture.h"
#include "inc/SceneGraph.h"
#include "inc/Texture.h"
#include "inc/NVMLImpl.h"

#include "inc/MaterialMDL.h"
#include "inc/CompileResult.h"
#include "inc/ShaderConfiguration.h"

#include "shaders/system_data.h"

#include <mi/mdl_sdk.h>
#include <mi/base/config.h>



#include <map>
#include <memory>
#include <vector>

// Bitfield encoding for m_peerToPeer:
#define P2P_PCI    1
#define P2P_TEX    2
#define P2P_GAS    4
#define P2P_ENV    8
#define P2P_MBSDF 16
#define P2P_IES   32



class Raytracer
{
public:
  Raytracer(const int maskDevices,
            const TypeLight typeEnv,
            const int interop,
            const unsigned int tex,
            const unsigned int pbo,
            const size_t sizeArena,
            const int p2p);
  ~Raytracer();

  int matchUUID(const char* uuid);
  int matchLUID(const char* luid, const unsigned int nodeMask);

  bool enablePeerAccess();  // Calculates peer-to-peer access bit matrix in m_peerConnections and sets the m_islands.
  void disablePeerAccess(); // Clear the peer-to-peer islands. Afterwards each device is its own island.
  
  void synchronize();        // Needed for the benchmark to wait for all asynchronous rendering to have finished.

  void initTextures(const std::map<std::string, Picture*>& mapOfPictures);
  void initCameras(const std::vector<CameraDefinition>& cameras);
  void initLights(const std::vector<LightGUI>& lights);
  void initScene(std::shared_ptr<sg::Group> root, const unsigned int numGeometries);
  void initState(const DeviceState& state);

  // Update functions should be replaced with NOP functions in a derived batch renderer because the device functions are fully asynchronous then.
  void updateCamera(const int idCamera, const CameraDefinition& camera);
  void updateLight(const int idLight, const LightGUI& lightGUI);
  //void updateLight(const int idLight, const LightDefinition& light);
  void updateMaterial(const int idMaterial, const MaterialMDL* materialMDL);
  void updateState(const DeviceState& state);

  unsigned int render(const int mode = 0); // 0 = interactive, 1 = benchmark (fully asynchronous launches)
  void updateDisplayTexture();
  const void* getOutputBufferHost();

  bool initMDL(const std::vector<std::string>& searchPaths);
  void shutdownMDL();
  void initMaterialsMDL(std::vector<MaterialMDL*>& materialsMDL);

  bool isEmissiveShader(const int indexShader) const;

private:
  void selectDevices();
  int  getDeviceHome(const std::vector<int>& island) const;
  void traverseNode(std::shared_ptr<sg::Node> node, InstanceData instanceData, float matrix[12]);
  bool activeNVLINK(const int home, const int peer) const;
  int findActiveDevice(const unsigned int domain, const unsigned int bus, const unsigned int device) const;

  mi::neuraylib::INeuray* load_and_get_ineuray(const char* filename);
  mi::Sint32 load_plugin(mi::neuraylib::INeuray* neuray, const char* path);
  bool log_messages(mi::neuraylib::IMdl_execution_context* context);
  void determineShaderConfiguration(const Compile_result& res, ShaderConfiguration& config);

  void initMaterialMDL(MaterialMDL* materialMDL);
  bool compileMaterial(mi::neuraylib::ITransaction* transaction, MaterialMDL* materialMDL, Compile_result& res);

public:
  // Constructor arguments
  unsigned int m_maskDevices; // The bitmask with the devices the user selected.
  TypeLight    m_typeEnv;     // Controls the miss program selection.
  int          m_interop;     // Which CUDA-OpenGL interop to use.
  unsigned int m_tex;         // The OpenGL texture object used for display. 
  unsigned int m_pbo;         // The pixel buffer object when using INTEROP_MODE_PBO.
  size_t       m_sizeArena;   // The default Arena allocation size in mega-bytes, just routed through to Device class.
  int          m_peerToPeer;  // Bitfield encoding for which resources CUDA P2P sharing is allowed:
                              // Bit 0: Allow sharing via PCI-E, ignores the NVLINK topology, just checks cuDeviceCanAccessPeer().
                              // Bit 1: Share material textures (not the HDR environment).
                              // Bit 2: Share GAS and vertex attribute data.
                              // Bit 3: Share (optional) HDR environment and its CDF data.
  bool m_isValid;

  int                  m_numDevicesVisible; // The number of visible CUDA devices. (What you can control via the CUDA_VISIBLE_DEVICES environment variable.)
  int                  m_indexDeviceOGL;    // The first device which matches with the OpenGL LUID and node mask. -1 when there was no match.
  unsigned int         m_maskDevicesActive; // The bitmask marking the actually enabled devices.
  std::vector<Device*> m_devicesActive;

  unsigned int m_iterationIndex;  // Tracks which sub-frame is currently raytraced.
  unsigned int m_samplesPerPixel; // This is samplesSqrt squared. Rendering end-condition is: m_iterationIndex == m_samplesPerPixel.

  std::vector<unsigned int>       m_peerConnections; // Bitfield indicating peer-to-peer access between devices. Indexing is m_peerConnections[home] & (1 << peer)
  std::vector< std::vector<int> > m_islands;         // Vector with vector of active device indices (not ordinals) building a peer-to-peer island.

  std::vector<GeometryData> m_geometryData; // The geometry device data. (Either per P2P island when sharing GAS, or per device when not sharing.)

  NVMLImpl m_nvml;

private:
  // MDL specific things.

#ifdef MI_PLATFORM_WINDOWS
  HMODULE m_dso_handle = 0;
#else
  void* m_dso_handle = 0;
#endif

  mi::base::Handle<mi::base::ILogger> m_logger;

  // The last error message from MDL SDK.
  std::string m_last_mdl_error;

  mi::base::Handle<mi::neuraylib::INeuray>                m_neuray;
  mi::base::Handle<mi::neuraylib::IMdl_compiler>          m_mdl_compiler;
  mi::base::Handle<mi::neuraylib::IMdl_configuration>     m_mdl_config;
  mi::base::Handle<mi::neuraylib::IDatabase>              m_database;
  mi::base::Handle<mi::neuraylib::IScope>                 m_global_scope;
  mi::base::Handle<mi::neuraylib::IMdl_factory>           m_mdl_factory;
  mi::base::Handle<mi::neuraylib::IMdl_execution_context> m_execution_context;
  mi::base::Handle<mi::neuraylib::IMdl_backend>           m_mdl_backend;
  mi::base::Handle<mi::neuraylib::IImage_api>             m_image_api;

  // Maps a compiled material hash to a shader code cache index == shader configuration index.
  std::map<mi::base::Uuid, int> m_mapMaterialHashToShaderIndex;
  // These two vectors have the same size and implement shader reuse (references with the same MDL material).
  std::vector<mi::base::Handle<mi::neuraylib::ITarget_code const>> m_shaders;
  std::vector<ShaderConfiguration>                                 m_shaderConfigurations;
};

#endif // RAYTRACER_H
