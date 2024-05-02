/* 
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef APPLICATION_H
#define APPLICATION_H

// Always include this before any OptiX headers!
#include <cuda_runtime.h>

#include <optix.h>

// OptiX 7 function table structure.
#include <optix_function_table.h>

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif

#include <windows.h>
#endif

// IMGUI
#define IMGUI_DEFINE_MATH_OPERATORS 1
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>

#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

// GLEW
#ifndef __APPLE__
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#  endif
#endif

// CUDA Runtime API version. Needs to be included after OpenGL headers!
#include <cuda_gl_interop.h>

// GLFW
#include <GLFW/glfw3.h>

// GLM
// glm/gtx/component_wise.hpp doesn't compile when not setting GLM_ENABLE_EXPERIMENTAL.
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
//#include <glm/gtc/matrix_transform.hpp>

// FASTGLTF
#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>

#include "Options.h"
#include "Logger.h"

#include <string>
#include <vector>
#include <memory>

#include <stdint.h>

#include "Arena.h"

#include "Light.h"
#include "Camera.h"
#include "Trackball.h"

#include "DeviceBuffer.h"
#include "Mesh.h"

#include "Picture.h"
#include "Texture.h"

#include "cuda/config.h"
#include "cuda/launch_parameters.h"
#include "cuda/geometry_data.h"
#include "cuda/material_data.h"


#define APP_EXIT_SUCCESS          0

#define APP_ERROR_UNKNOWN        -1
#define APP_ERROR_CREATE_WINDOW  -2
#define APP_ERROR_GLFW_INIT      -3
#define APP_ERROR_GLEW_INIT      -4
#define APP_ERROR_APP_INIT       -5
#define APP_ERROR_EXCEPTION      -6


// OpenGL-CUDA interop modes of the renderer.
// Change with command line option --interop (-i) <0|1|2|3> (default is 0)
enum InteropMode
{
  INTEROP_OFF = 0, // Device-to-host copy from native CUDA buffer to m_bufferHost and then host-to-device copy to m_hdrTexture with glTexSubImage2D. 
                   // Slowest method but the default because it works with any OpenGL implementation supporting RGBA32F textures.
                   // No need to have OpenGL running on the selected CUDA device, so also works on CUDA devices in TCC mode and separate display device.
  INTEROP_PBO = 1, // Direct rendering into mapped linear PBO and device-to-device copy to m_hdrTexture with glTexSubImage2D.
  INTEROP_TEX = 2, // Device-to-device copy from native CUDA buffer to m_hdrTexture mapped as CUDA texture array.
  INTEROP_IMG = 3  // Direct rendering into the mapped CUDA texture array image surface object of m_hdrTexture.
};


enum GuiState
{
  GUI_STATE_NONE,
  GUI_STATE_ORBIT,
  GUI_STATE_PAN,
  GUI_STATE_DOLLY,
  GUI_STATE_FOCUS
};


enum ModuleIdentifier
{
  MODULE_ID_RAYGENERATION,
  MODULE_ID_EXCEPTION,
  MODULE_ID_MISS,
  MODULE_ID_HIT,
  MODULE_ID_LIGHT_SAMPLE,

  NUM_MODULE_IDENTIFIERS
};


enum ProgramGroupId
{
  PGID_RAYGENERATION,
  
  PGID_EXCEPTION,
  
  PGID_MISS_RADIANCE,
  PGID_MISS_SHADOW,
  
  // Hit records for triangles:
  PGID_HIT_RADIANCE,
  PGID_HIT_SHADOW,

  // Direct Callables (light sampling)
  // Area lights: 
  PGID_LIGHT_ENV_CONSTANT,
  PGID_LIGHT_ENV_SPHERE,
  // Singular lights:
  PGID_LIGHT_POINT,
  PGID_LIGHT_SPOT,
  PGID_LIGHT_DIRECTIONAL,

  // Number of all hardcoded program group entries. 
  NUM_PROGRAM_GROUP_IDS
};


class Application
{
public:
  Application(GLFWwindow* window, Options const& options);
  ~Application();

  void reshape(int width, int height);
  
  bool render(); // Returns true if a new texture image is available for display.
  void update();
  void display();

  void drop(const int countPaths, const char* paths[]);

  void guiNewFrame();
  void guiWindow();
  void guiEventHandler();
  void guiRender();

  void guiReferenceManual(); // The IMGUI "programming manual" in form of a live window.

private:

  void initOpenGL();

  void checkInfoLog(const char *msg, GLuint object);
  void initGLSL();
  void updateTonemapper();

  void getSystemInformation();
  void initCUDA();

  OptixResult initOptiXFunctionTable();
  void initOptiX();

  void loadGLTF(const std::filesystem::path& path);

  std::vector<char> readData(std::string const& filename);

  // ArenaAllocator versions of cudaMalloc and cudaFree. Attention: These are asynchronous to other cuda functions!
  CUdeviceptr memAlloc(const size_t size, const size_t alignment, const cuda::Usage usage);
  void memFree(const CUdeviceptr ptr);

  void updateCamera();
  void updateBuffers();

  void addImage(const int32_t width,
                const int32_t height,
                const int32_t bitsPerComponent,
                const int32_t numComponents,
                const void*   data);

  void addSampler(cudaTextureAddressMode address_s,
                  cudaTextureAddressMode address_t,
                  cudaTextureFilterMode  filter,
                  const size_t           image_idx,
                  const int              sRGB);

  void cleanup();
  void initRenderer();
  void updateRenderer();

  void buildMeshAccels();
  void buildInstanceAccel();

  void createPipeline();
  void createSBT();
  void updateSBT();
  void updateMaterial(const int index, const bool rebuild);
  void updateVariant();
  void updateSBTMaterialData();

  void initLaunchParameters();
  void updateLaunchParameters();

  void initImages();
  void initTextures();
  void initMaterials();
  void initMeshes();
  void initLights();
  void initCameras();

  void initScene(const int indexScene); // Process all nodes, meshes, cameras, materials, textures, images accessible by this scene's node hierarchy.
  void updateScene();

  void updateLights();

  void initTrackball();

  void traverseNode(const size_t nodeIndex, glm::mat4 matrix); // Function to visit and initialize all accessible nodes inside a scene.

  void initSheenLUT();

  LightDefinition createConstantEnvironmentLight() const;
  LightDefinition createSphericalEnvironmentLight(); // This initalizes m_picEnv and m_texEnv.
  LightDefinition createPointLight() const;
  
  void updateBufferHost();
  std::string Application::getDateTime();
  bool screenshot(const bool tonemap);

private:
  GLFWwindow* m_window;

  // Application command line parameters.
  std::filesystem::path m_pathAsset;
  int                   m_width;
  int                   m_height;
  int                   m_interop;  // 0 == off, 1 == pbo, 2 = copy to cudaArray, 3 = surface read/write
  bool                  m_punctual; // Support for KHR_lights_punctual, default true.
  int                   m_missID;   // 0 = null, 1 = constant, 2 = spherical environment (default).
  std::string           m_pathEnv;  // Command line option --env (-e) <path.hdr> sets m_pathEnv.

  fastgltf::Asset m_asset; // The glTF asset when the loading succeeded.
  
  size_t m_indexScene = 0; // The current scene is defined by m_asset.scenes[m_indexScene].
  bool   m_isDirtyScene = false;

  // GUI values for the environment lights.
  float m_colorEnv[3]    = { 1.0f,1.0f, 1.0f };
  float m_intensityEnv   = 1.0f;
  float m_rotationEnv[3] = { 0.0f, 0.0f, 0.0f }; // The Euler rotation angles for the spherical environment light.

  // OpenGL variables:
  GLuint m_pbo = 0;
  GLuint m_hdrTexture = 0;

  float4* m_bufferHost = nullptr;

  int  m_launches  = 1;     // The number of asynchronous launches per render() call. Can be set with command line option --launches (-l) <int>
  bool m_benchmark = false; // Print samples per second results of the launches per render call.

  // Data used to determine if the active CUDA device is also running the OpenGL implementation, otherwise no OpenGL-CUDA interop is possible.
  CUuuid m_cudaDeviceUUID;

  // GLSL shaders objects and program.
  GLuint m_glslVS;
  GLuint m_glslFS;
  GLuint m_glslProgram;

  // Tonemapper group:
  float  m_gamma;
  float3 m_colorBalance;
  float  m_whitePoint;
  float  m_burnHighlights;
  float  m_crushBlacks;
  float  m_saturation;
  float  m_brightness;

  GuiState m_guiState;
  
  bool m_isVisibleGUI; // Hide the GUI window completely with SPACE key.

  float m_mouseSpeedRatio = 100.0f; // Adjusts how many pixels the mouse needs to travel for 1 unit change for panning and dollying.
  
  float m_epsilonFactor = 1000.0f; // Self-intersection avoidance factor, multiplied with SCENE_EPSILON_SCALE to get final value.

  // Some renderer global settings.
  bool m_useDirectLighting   = true; // Switch between explicit direct lighting and brute force path tracing.
                                     // Singular lights only work with direct lighting! 
  bool m_useAmbientOcclusion = true; // Global illumination handles ambient occlusion automatically,
                                     // but many glTF models are low resolution with high resolution detail baked into the normal and occlusion textures which look better when applying the occlusion tetxure attenuation.
                                     // This affects only diffuse and glossy (metal) reflections inside the renderer and only environment lights.
  bool m_showEnvironment     = true; // Toggle the display of the environment for primary camera rays, shows black instead.

  // OpenGL resources used inside the VBO path.
  GLuint m_vboAttributes = 0;
  GLuint m_vboIndices    = 0;

  GLint m_positionLocation = -1;
  GLint m_texCoordLocation = -1;
    
  //std::vector<cudaDeviceProp> m_deviceProperties;

  // CUDA native types are prefixed with "cuda".
  CUdevice  m_cudaDevice  = 0; // The CUdevice handle of the CUDA device ordinal. (Usually identical to the ordinal number.)
  CUcontext m_cudaContext = nullptr;
  CUstream  m_cudaStream  = nullptr;

  // The handle for the registered OpenGL PBO when using interop.
  CUgraphicsResource m_cudaGraphicsResource = nullptr;

  // All others are OptiX types.
  OptixFunctionTable m_api;
  OptixDeviceContext m_optixContext = nullptr;

  Logger m_logger;

  cuda::ArenaAllocator* m_allocator = nullptr;
  size_t m_sizeArena = 128; // Default to 128 MiB Arenas when nothing is specified inside the system description.

  Picture* m_picSheenLUT = nullptr; // Loads the "sheen_lut.hdr" image in initSheenLUT().
  Texture* m_texSheenLUT = nullptr; // Lookup table for the sheen sampling weight estimation. 2D float texture with lookup sheenWeight == lut(dot(V, N), sheenRoughness);

  std::vector<dev::Camera*>        m_cameras;
  std::vector<dev::Light*>         m_lights;
  std::vector<dev::Instance*>      m_instances;
  std::vector<dev::Mesh*>          m_meshes;
  std::vector<MaterialData>        m_materialsOrg;  // The original material data inside the asset.
  std::vector<MaterialData>        m_materials;     // The material data changed by the GUI.
  std::vector<cudaArray_t>         m_images;        // Textures reference these images.
  std::vector<cudaTextureObject_t> m_samplers;      // Each texture has exactly one hardware sampler. 
                                                    // Sampler settings (wrap, filter) might be defined by glTF samplers (not sRGB though, see initTextures()).

  // These values are only valid after buildInstanceAccel().
  glm::vec3 m_sceneAABB[2]; // Top-level IAS emitted AABB result. 
  glm::vec3 m_sceneCenter;  // The center point of the scene AABB.
  float     m_sceneExtent;  // The maximum extent of the scene AABB.

  // Spherical HDR texture environment light.
  // Only used with command line option: --miss (-m) 2.
  // Supports drag-and-drop of *.hdr filenames then!
  Picture*    m_picEnv = nullptr;
  Texture*    m_texEnv = nullptr;

  std::vector<LightDefinition> m_lightDefinitions;

  OptixTraversableHandle m_ias   = 0; // This is the root traversable handle of the current scene.
  CUdeviceptr            m_d_ias = 0;

  // Module and Pipeline data.
  OptixModuleCompileOptions   m_mco = {};
  OptixPipelineCompileOptions m_pco = {};
  OptixPipelineLinkOptions    m_plo = {};
  OptixProgramGroupOptions    m_pgo = {}; // This is a just placeholder.

  std::vector<std::string>       m_moduleFilenames;
  std::vector<OptixModule>       m_modules;
  std::vector<OptixProgramGroup> m_programGroups;

  OptixPipeline m_pipeline = 0;

  OptixShaderBindingTable m_sbt = {};

  LaunchParameters  m_launchParameters   = {};
  LaunchParameters* m_d_launchParameters = nullptr;

  size_t m_indexCamera   = 0; // The currently selected camera. The GUI only offers to switch cameras if there are more than one.
  size_t m_indexVariant  = 0; // The currently selected material variant.
  size_t m_indexMaterial = 0; // The currently selected material inside the GUI.
  size_t m_indexLight    = 0; // The currently selected KHR_lights_punctual inside the GUI.
  
  // This is set to true in initCameras() when there was no camera inside the asset.
  // In that case a perspective camera had been added and that needs to be placed and centered
  // to the current scene's AABB which is only available later after the initScene() call.
  bool m_isDefaultCamera = false;

  // All true to invoke necessary camera updates in updateCamera(), updateBuffers(), and updateLights()
  bool m_isDirtyCamera = true;
  bool m_isDirtyResize = true;
  bool m_isDirtyLights = true;

  dev::Trackball m_trackball;

  // A vector which contains the asynchronously launched iteration values, filled inside the renderer() function.
  // Defining once here to not resize it on every render call but only when changing the launches value inside the GUI.
  std::vector<unsigned int> m_iterations;
};

#endif // APPLICATION_H

