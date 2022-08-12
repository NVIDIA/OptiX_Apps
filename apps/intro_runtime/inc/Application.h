/* 
 * Copyright (c) 2013-2022, NVIDIA CORPORATION. All rights reserved.
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

#include "imgui.h"

#define IMGUI_DEFINE_MATH_OPERATORS 1
#include "imgui_internal.h"

#include "imgui_impl_glfw_gl3.h"

#ifndef __APPLE__
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#  endif
#endif

// Needs to be included after OpenGL headers!
// CUDA Runtime API version.
#include <cuda_gl_interop.h>

#include <GLFW/glfw3.h>

#include "inc/Options.h"
#include "inc/Logger.h"
#include "inc/PinholeCamera.h"
#include "inc/Timer.h"
#include "inc/Picture.h"
#include "inc/Texture.h"

#include "shaders/system_parameter.h"
#include "shaders/function_indices.h"
#include "shaders/light_definition.h"
#include "shaders/vertex_attributes.h"

#include <iostream>
#include <map>
#include <string>


#define APP_EXIT_SUCCESS          0

#define APP_ERROR_UNKNOWN        -1
#define APP_ERROR_CREATE_WINDOW  -2
#define APP_ERROR_GLFW_INIT      -3
#define APP_ERROR_GLEW_INIT      -4
#define APP_ERROR_APP_INIT       -5


// I don't have per-program data at all. Then the SBT only needs the header.
struct SbtRecordHeader
{
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// The hit group gets per instance data in addition.
template <typename T>
struct SbtRecordData
{
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

typedef SbtRecordData<GeometryInstanceData> SbtRecordGeometryInstanceData;


enum GuiState
{
  GUI_STATE_NONE,
  GUI_STATE_ORBIT,
  GUI_STATE_PAN,
  GUI_STATE_DOLLY,
  GUI_STATE_FOCUS
};


// Host side GUI material parameters 
struct MaterialParameterGUI
{
  FunctionIndex indexBSDF;  // BSDF index to use in the closest hit program
  float3        albedo;     // Tint, throughput change for specular materials
  bool          useAlbedoTexture;
  bool          useCutoutTexture;
  bool          thinwalled;
  float3        absorptionColor; // absorption color and distance scale together build the absorption coefficient
  float         volumeDistanceScale;
  float         ior;        // index of refraction
};


// The actual geometries are tracked in m_geometries.
struct GeometryData
{
  CUdeviceptr indices;
  CUdeviceptr attributes;
  size_t      numIndices;    // Count of unsigned ints, not triplets.
  size_t      numAttributes; // Count of VertexAttributes structs.
  CUdeviceptr gas;
};


enum ModuleIdentifier
{
  MODULE_ID_RAYGENERATION,
  MODULE_ID_EXCEPTION,
  MODULE_ID_MISS,
  MODULE_ID_CLOSESTHIT,
  MODULE_ID_ANYHIT,
  MODULE_ID_LENS_SHADER,
  MODULE_ID_LIGHT_SAMPLE,
  MODULE_ID_DIFFUSE_REFLECTION,
  MODULE_ID_SPECULAR_REFLECTION,
  MODULE_ID_SPECULAR_REFLECTION_TRANSMISSION,
  NUM_MODULE_IDENTIFIERS
};


enum ProgramIdentifier
{
  PROGRAM_ID_RAYGENERATION,
  PROGRAM_ID_EXCEPTION,
  PROGRAM_ID_MISS_RADIANCE,
  PROGRAM_ID_MISS_SHADOW,
  PROGRAM_ID_HIT_RADIANCE,
  PROGRAM_ID_HIT_SHADOW,
  PROGRAM_ID_HIT_RADIANCE_CUTOUT,
  PROGRAM_ID_HIT_SHADOW_CUTOUT,
  // Callables
  PROGRAM_ID_LENS_PINHOLE,
  PROGRAM_ID_LENS_FISHEYE,
  PROGRAM_ID_LENS_SPHERE,
  PROGRAM_ID_LIGHT_ENV,
  PROGRAM_ID_LIGHT_PARALLELOGRAM,
  PROGRAM_ID_BRDF_DIFFUSE_SAMPLE,
  PROGRAM_ID_BRDF_DIFFUSE_EVAL,
  PROGRAM_ID_BRDF_SPECULAR_SAMPLE,
  PROGRAM_ID_BRDF_SPECULAR_EVAL,
  PROGRAM_ID_BSDF_SPECULAR_SAMPLE,
  PROGRAM_ID_BSDF_SPECULAR_EVAL,
  NUM_PROGRAM_IDENTIFIERS
};


class Application
{
public:
  Application(GLFWwindow* window,
              Options const& options);
  ~Application();

  bool isValid() const;

  void reshape(int width, int height);
  bool render(); // Returns true if a new texture image is available for display.
  void display();

  void guiNewFrame();
  void guiWindow();
  void guiEventHandler();
  void guiRender();

  void guiReferenceManual(); // The IMGUI "programming manual" in form of a live window.

private:
  
  void getSystemInformation();

  void initOpenGL();

  void checkInfoLog(const char *msg, GLuint object);
  void initGLSL();

  OptixResult initOptiXFunctionTable();
  bool initOptiX();

  void initMaterials();
  void initPipeline();

  void initRenderer(); // All scene and renderer setup goes here.

  OptixTraversableHandle createBox();
  OptixTraversableHandle createPlane(const unsigned int tessU, const unsigned int tessV, const unsigned int upAxis);
  OptixTraversableHandle createSphere(const unsigned int tessU, const unsigned int tessV, const float radius, const float maxTheta);
  OptixTraversableHandle createTorus(const unsigned int tessU, const unsigned int tessV, const float innerRadius, const float outerRadius);
  OptixTraversableHandle createParallelogram(float3 const& position, float3 const& vecU, float3 const& vecV, float3 const& normal);

  OptixTraversableHandle createGeometry(std::vector<VertexAttributes> const& attributes, std::vector<unsigned int> const& indices);
  
  void createLights();
  
  void updateMaterialParameters();

  void restartAccumulation();

  std::vector<char> readData(std::string const& filename);

  void updateShaderBindingTable(const int instance);

private:
  GLFWwindow* m_window;

  int m_width;
  int m_height;
  bool m_interop;

  //int m_widthLaunch;
  //int m_heightLaunch;
  
  // Application command line parameters.
  //unsigned int m_devicesEncoding;
  int         m_lightID;
  int         m_missID;
  std::string m_environmentFilename;

  bool m_isValid;

  // Application GUI parameters.
  float m_sceneEpsilonFactor;  // Factor on 1e-7 used to offset ray origins along the path to reduce self intersections. 
  
  int   m_iterationIndex;
  
  // OpenGL variables:
  GLuint m_pbo;
  GLuint m_hdrTexture;

  float4* m_outputBuffer;

  // The material parameters exposed inside the GUI are slightly different than the resulting values for the device.
  // The GUI exposes an absorption color and a distance scale, and the thin-walled property as bool.
  // These are converted on the fly into the device side sysMaterialParameters buffer.
  std::vector<MaterialParameterGUI> m_guiMaterialParameters;

  bool   m_present;         // This controls if the texture image is updated per launch or only once a second.
  bool   m_presentNext;
  double m_presentAtSecond;

  int    m_frames; 

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

  float m_mouseSpeedRatio;

  PinholeCamera m_pinholeCamera;

  Timer m_timer;

  std::vector<LightDefinition> m_lightDefinitions;

  // Need to store these as pointers to be able to tear them down before the Application.
  Texture* m_textureEnvironment;
  Texture* m_textureAlbedo;
  Texture* m_textureCutout;

  // OpenGL resources used inside the VBO path.
  GLuint m_vboAttributes;
  GLuint m_vboIndices;

  GLint m_positionLocation;
  GLint m_texCoordLocation;
    
  std::vector<cudaDeviceProp> m_deviceProperties;

  // CUDA native types are prefixed with "cuda".
  CUcontext m_cudaContext;
  CUstream  m_cudaStream;

  // The handle for the registered OpenGL PBO when using interop.
  cudaGraphicsResource* m_cudaGraphicsResource;

  // All others are OptiX types.
  OptixFunctionTable m_api;
  OptixDeviceContext m_context;

  Logger m_logger;

  OptixTraversableHandle m_root;  // Scene root
  CUdeviceptr            m_d_ias; // Scene root's IAS (instance acceleration structure).

  std::vector<std::string> m_moduleFilenames;

  // API Reference sidenote on optixLaunch (doesn't apply for this example):
  // Concurrent launches to multiple streams require separate OptixPipeline objects. 
  OptixPipeline m_pipeline;
  
  SystemParameter  m_systemParameter;   // Host side of the system parameters, changed by the GUI directly.
  SystemParameter* m_d_systemParameter; // Device side CUdeviceptr of the system parameters.
  
  std::vector<GeometryData> m_geometries;

  std::vector<OptixInstance> m_instances;

  // The Shader Binding Table and data.
  OptixShaderBindingTable m_sbt;

  std::vector<SbtRecordGeometryInstanceData> m_sbtRecordGeometryInstanceData;

  CUdeviceptr m_d_sbtRecordRaygeneration;
  CUdeviceptr m_d_sbtRecordException;
  CUdeviceptr m_d_sbtRecordMiss;

  CUdeviceptr m_d_sbtRecordCallables;

  SbtRecordGeometryInstanceData m_sbtRecordHitRadiance;
  SbtRecordGeometryInstanceData m_sbtRecordHitShadow;
  SbtRecordGeometryInstanceData m_sbtRecordHitRadianceCutout;
  SbtRecordGeometryInstanceData m_sbtRecordHitShadowCutout;

  SbtRecordGeometryInstanceData* m_d_sbtRecordGeometryInstanceData;
};

#endif // APPLICATION_H

