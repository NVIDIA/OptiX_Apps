/* 
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include <GLFW/glfw3.h>

// GLM
// glm/gtx/component_wise.hpp doesn't compile when not setting GLM_ENABLE_EXPERIMENTAL.
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

// FASTGLTF
#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>

#include "Options.h"
#include "Logger.h"

#include <chrono>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <stdint.h>

#include "Arena.h"

#include "DeviceBuffer.h"

#include "Animation.h"
#include "Camera.h"
#include "Light.h"
#include "Mesh.h"
#include "Node.h"
#include "Skin.h"

#include "Trackball.h"
#include "SceneExtent.h"

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

// The maximum number of values inside the m_benchmarkValues vector over which a running average is built.
#define SIZE_BENCHMARK_VALUES 100

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

enum MiscConstants
{
  MAX_TRACE_DEPTH = 2
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
  // built-in modules don't have and ID (spheres, curves)
};


enum ProgramGroupId
{
  PGID_RAYGENERATION,
  
  PGID_EXCEPTION,
  
  PGID_MISS_RADIANCE,
  PGID_MISS_SHADOW,
  
  // Hit records for triangles:
  PGID_HIT_RADIANCE_TRIANGLES,
  PGID_HIT_SHADOW_TRIANGLES,

  // Hit records for spheres:
  PGID_HIT_RADIANCE_SPHERES,
  PGID_HIT_SHADOW_SPHERES,

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


namespace dev
{

  struct Instance
  {
    glm::mat4x4 transform;
    int         indexDeviceMesh; // Index into m_deviceMeshes.
  };
} // namespace dev;


class Application
{
public:

  enum BenchmarkMode : int
  {
    OFF,
    FPS,                // frames / second, render and display
    SAMPLES_PER_SECOND  // samples / second, pure raytracing performance
  };

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

  int  getBenchmarkMode() const;
  void setBenchmarkValue(const float value); 

private:

  void initOpenGL();

  void checkInfoLog(const char *msg, GLuint object);
  void initGLSL();

  void updateTonemapper();

  void updateProjectionMatrix();
  void updateVertexAttributes();
  float2 getPickingCoordinate(const int x, const int y);

  void getSystemInformation();
  void initCUDA();

  OptixResult initOptiXFunctionTable();
  void initOptiX();

  void loadGLTF(const std::filesystem::path& path);
  void initRenderer(const bool first);

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

  void cleanup(); // Complete destruction of all resources, called in ~Application().

  void buildDeviceMeshAccel(const int indexDeviceMesh, const bool rebuild);
  void buildDeviceMeshAccels(const bool rebuild);
  void buildInstanceAccel(const bool rebuild);

  void initPipeline();
  void initSBT();
  void updateSBT();
  void updateMaterial(const int index, const bool rebuild);
  void updateVariant();
  void updateSBTMaterialData();

  void initLaunchParameters();
  void updateLaunchParameters();

  void initNodes();
  void initSkins();
  void initImages();
  void initTextures();
  void initMaterials();
  void initMeshes();
  void initLights(const bool first);
  void initCameras();
  void initAnimations();
  void initSceneExtent();

  void updateRenderGraph();

  bool updateAnimations();

  void updateMorph(const int indexDeviceMesh, const size_t indexNode);
  void updateSkin(const size_t indexNode, const dev::KeyTuple key);

  void initScene(const int indexScene); // Process all nodes, meshes, cameras, materials, textures, images accessible by this scene's node hierarchy.
  void updateScene(const bool rebuild); // This can either init/switch scenes (rebuild == true), or just animate existing instances (rebuild == false).

  void updateLights();

  void initTrackball();

  // These recursive traverse functions visit and initialize all accessible nodes inside a scene.
  // traverseNodeTrafo() calculates all node.matrixGlobal values.
  // This must be done before updateSkin() to have valid joint matrices.
  void traverseNodeTrafo(const size_t indexNode, glm::mat4 matrix);

  /// Create or update device meshes.
  /// Create or update instances (with trafo and index), not OptixInstance-s.
  /// Recurse over node's children.
  ///
  /// @param indexNode      Index into m_nodes[]
  /// @param rebuild        Rebuild the AS for the mesh (new or existing).
  void traverseNode(const size_t indexNode, const bool rebuild);

  /// Recursive visit of all nodes (starting from scene's root nodes) to find out the world-space
  /// bounding box.
  /// Updates the scene extent.
  void traverseUpdateSceneExtent(size_t indexNode, const glm::mat4x4& mParent);

  void initSheenLUT();

  LightDefinition createConstantEnvironmentLight() const;
  LightDefinition createSphericalEnvironmentLight(); // This initalizes m_picEnv and m_texEnv.
  LightDefinition createPointLight() const;
  
  void updateBufferHost();
  bool screenshot(const bool tonemap);

  /// Init a device primitive from a host primitive. Also sets the primitive type.
  void createDevicePrimitive(dev::DevicePrimitive& devicePrim, const dev::HostPrimitive& hostPrim, const int skin);

  /// Create all device primitives for a given mesh.
  /// @param deviceMesh OUT
  /// @param hostKey    Index into the host meshes
  void createDeviceMesh(dev::DeviceMesh& deviceMesh, const dev::KeyTuple hostKey);

  void updateFonts();

  /// flags for accelBuild
  unsigned int getBuildFlags() const;

  // Move camera (on keyboard input)
  void cameraTranslate(float dx, float dy, float dz);
 
  // Get scene center (needs a valid scene extent!)
  glm::vec3 getSceneCenter() const;

private:
  GLFWwindow* m_window;

  // Application command line parameters.
  std::filesystem::path m_pathAsset;

  int         m_width;      // Client window width.
  int         m_height;     // Client window height.
  int2        m_resolution; // Render resolution, independent of the client window size (m_width, m_height).
  int         m_interop;    // 0 == off, 1 == pbo, 2 = copy to cudaArray, 3 = surface read/write
  bool        m_punctual;   // Support for KHR_lights_punctual, default true.
  int         m_missID;     // 0 = null, 1 = constant, 2 = spherical environment (default).
  std::string m_pathEnv;    // Command line option --env (-e) <path.hdr> sets m_pathEnv.


  fastgltf::Asset m_asset; // The glTF asset when the loading succeeded.
  
  size_t m_indexScene = 0; // The current scene is defined by m_asset.scenes[m_indexScene].

  bool   m_isDirtyScene = true;

  // GUI values for the environment lights.
  float m_colorEnv[3]    = { 1.0f, 1.0f, 1.0f };
  float m_intensityEnv   = 1.0f;
  float m_rotationEnv[3] = { 0.0f, 0.0f, 0.0f }; // The Euler rotation angles for the spherical environment light.

  // OpenGL variables:
  GLuint m_pbo = 0;
  GLuint m_hdrTexture = 0;

  float4* m_bufferHost = nullptr;

  int m_launches = 1; // The number of asynchronous launches per render() call. Can be set with command line option --launches (-l) <int>

  BenchmarkMode m_benchmarkMode = BenchmarkMode::OFF;

  int m_benchmarkEntries = 0;   // The current number of valid benchmark results inside the vector.
  int m_benchmarkCell    = 0;   // The next cell inside the m_benchmarkValues vector to be written to.
  std::vector<float> m_benchmarkValues;

  // Data used to determine if the active CUDA device is also running the OpenGL implementation, otherwise no OpenGL-CUDA interop is possible.
  CUuuid m_cudaDeviceUUID;

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

  ImFont* m_font = nullptr;
  float   m_fontScale = 0.0f;
  GLuint  m_fontTexture = 0;

  const float CameraSpeedFraction = 0.005f; // Camera translation (keyboard's A/W/S/D/... events)

  float m_mouseSpeedRatio = 100.0f; // Adjusts how many pixels the mouse needs to travel for 1 unit change for panning and dollying.
  bool  m_isLockedGimbal  = true;   // Toggle the gimbal lock on the trackball. true keeps the up-vector intact, false allows rolling.
  
  float m_epsilonFactor = 1000.0f; // Self-intersection avoidance factor, multiplied with SCENE_EPSILON_SCALE to get final value.

  // Some renderer global settings.
  bool m_useDirectLighting   = true;  // Switch between explicit direct lighting and brute force path tracing.
                                      // Singular lights only work with direct lighting! 
  bool m_useAmbientOcclusion = true;  // Global illumination handles ambient occlusion automatically,
                                      // but many glTF models are low resolution with high resolution detail baked into the normal and occlusion textures which look better when applying the occlusion tetxure attenuation.
                                      // This affects only diffuse and glossy (metal) reflections inside the renderer and only environment lights.
  bool m_showEnvironment     = true;  // Toggle the display of the environment for primary camera rays, shows black instead.
  bool m_forceUnlit          = false; // Force renderer to handle all materials as unlit. Useful in case the scene is not modeled correctly for global illumination (like VirtualCity.gltf).

  // OpenGL resources used inside the VBO path.
  GLuint m_vboAttributes = 0;
  GLuint m_vboIndices    = 0;

  // GLSL shaders objects and program.
  GLuint m_glslVS = 0;
  GLuint m_glslFS = 0;
  GLuint m_glslProgram = 0;

  GLint m_locAttrPosition = -1;
  GLint m_locAttrTexCoord = -1;
  GLint m_locProjection   = -1;
    
  //std::vector<cudaDeviceProp> m_deviceProperties;

  // CUDA native types are prefixed with "cuda".
  CUdevice  m_cudaDevice  = 0; // The CUdevice handle of the CUDA device ordinal. (Usually identical to the ordinal number.)
  CUcontext m_cudaContext = nullptr;
  CUstream  m_cudaStream  = nullptr;

  // The handle for the registered OpenGL PBO when using interop.
  CUgraphicsResource m_cudaGraphicsResource = nullptr;
  
  // Radius of the spheres for the glTF points.
  // Some datasets are tricky (huge bbox but most of the points are in a small
  // subvolume -> need to tweak the radius).
  // Could be a gui slider too.
  // Makes me think of yet another widget: clipping plane(s) to use with dense CT datasets.
  float              m_sphereRadiusFraction{ 0.005f };

  // All others are OptiX types.
  OptixFunctionTable m_api;
  OptixDeviceContext m_optixContext = nullptr;
  unsigned int m_visibilityMask = 255; // Instance Visibility Mask default. Queried and set in initOptiX().

  Logger m_logger;

  cuda::ArenaAllocator* m_allocator = nullptr;
  size_t m_sizeArena = 128; // Default to 128 MiB Arenas when nothing is specified inside the system description.

  Picture* m_picSheenLUT = nullptr; // Loads the "sheen_lut.hdr" image in initSheenLUT().
  Texture* m_texSheenLUT = nullptr; // Lookup table for the sheen sampling weight estimation. 2D float texture with lookup sheenWeight == lut(dot(V, N), sheenRoughness);

  std::vector<dev::Camera>         m_cameras;
  std::vector<dev::Light>          m_lights;
  std::vector<dev::Instance>       m_instances;
  std::vector<dev::HostMesh>       m_hostMeshes;
  std::vector<MaterialData>        m_materialsOrg;  // The original material data inside the asset.
  std::vector<MaterialData>        m_materials;     // The material data changed by the GUI.
  std::vector<cudaArray_t>         m_images;        // Textures reference these images.
  std::vector<cudaTextureObject_t> m_samplers;      // Each texture has exactly one hardware sampler. 
                                                    // Sampler settings (wrap, filter) might be defined by glTF samplers (not sRGB though, see initTextures()).
  std::vector<dev::Node>           m_nodes;         // A shadow of the asset nodes holding just the local transformation (matrix or translation, rotation, scale).
  std::vector<dev::Animation>      m_animations;    // All animations inside the asset, holding animation samplers and channels.
  std::vector<dev::Skin>           m_skins;         // All skins inside the asset.

  std::map<dev::KeyTuple, int> m_mapKeyTupleToDeviceMeshIndex; // For skinning and book-keeping.
  std::vector<dev::DeviceMesh> m_deviceMeshes;

  // Animation GUI handling.
  bool  m_isAnimated  = false; // True if at least one animation is enabled. (Enables the animation timeline GUI widgets.)
  bool  m_isPlaying   = false; // Tracks the Play/Stop button state. Clicking Play initializes m_timeBase.
  float m_timeMinimum = 0.0f;  // The minimum of all AnimationSampler timeMin values.
  float m_timeMaximum = 1.0f;  // The maximum of all AnimationSampler timeMax values.

  // Time based animation.
  bool  m_isTimeBased = true; // Switches the animation GUI between time and key-frames.
  float m_timeStart = 0.0f;
  float m_timeEnd   = 1.0f;
  float m_timeScale = 1.0f;   // Scale value for the m_timeCurrent calculation to be able to slow down time-based animations.
  std::chrono::steady_clock::time_point m_timeBase; // The time when the Play button has been pressed defines the real-time base time.

  // Key-framed animation:
  bool  m_isScrubbing = false;      // Set when m_frameCurrent has been changed by the slider. One-shot setting.
  int   m_frameMinimum    = 0;      // Derived from m_timeMinimum.
  int   m_frameMaximum    = 1;      // Derived from m_timeMaximum.
  int   m_frameStart      = 0;      // User defined start frame.
  int   m_frameEnd        = 1;      // User defined end frame (not reached)
  float m_framesPerSecond = 30.0f;  // Any positive value should work here.
  int   m_frameCurrent    = 0;      // The current frame in range [m_frameStart, m_frameEnd).

  // This is derived in either real-time or key-framed animation.
  float m_timeCurrent = 0.0f; // Current time of the animation, either from real-time timer or key frames.

  dev::SceneExtent m_sceneExtent;
  std::unordered_map<fastgltf::Primitive const*, HostBuffer const*> m_primitiveToHostBuffer; // rememeber the buffers to compute the extent after initMeshes()

  // Spherical HDR texture environment light.
  // Only used with command line option: --miss (-m) 2.
  // Supports drag-and-drop of *.hdr filenames then!
  Picture* m_picEnv = nullptr;
  Texture* m_texEnv = nullptr;

  std::vector<LightDefinition> m_lightDefinitions;
  CUdeviceptr m_d_lightDefinitions = 0;
  
  // Data which is kept intact for faster animations:
  size_t m_indexInstance = 0; // Global index into m_instances which is tracked in traverseNode() when just updating the instances.

  DeviceBuffer m_growInstances;

  OptixTraversableHandle m_ias = 0; // This is the root traversable handle of the current scene.
  DeviceBuffer m_growIas;     // The acceleration structure data of the root traversable IAS.
  DeviceBuffer m_growIasTemp; // The temporary memory required for rebuild or update of the IAS.

  CUdeviceptr m_d_iasAABB = 0; // Fixed allocation for the IAS AABB result.
  
  DeviceBuffer m_growMorphWeights; // Morph weights needed for GPU morphing. (Small, number of morph targets many floats.)

  // Skin matrices and skin matrices inverse transpose in a growing device buffer allocation.
  DeviceBuffer m_growSkinMatrices;

  // Module and Pipeline data.
  OptixModuleCompileOptions   m_mco = {};
  OptixPipelineCompileOptions m_pco = {};
  OptixPipelineLinkOptions    m_plo = {};
  OptixProgramGroupOptions    m_pgo = {}; // This is a just placeholder.

  std::vector<std::string>       m_moduleFilenames;
  std::vector<OptixModule>       m_modules;
  OptixModule                    m_moduleBuiltinISSphere;
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

  // All true to invoke necessary updateBuffers(), and updateLights().
  // Cameras track their isDirty state internally and updateCamera() clears it.
  bool m_isDirtyResolution = true;
  bool m_isDirtyLights     = true;

  dev::Trackball m_trackball;

  // A vector which contains the asynchronously launched iteration values, filled inside the renderer() function.
  // Defining once here to not resize it on every render call but only when changing the launches value inside the GUI.
  std::vector<unsigned int> m_iterations;
};

#endif // APPLICATION_H

