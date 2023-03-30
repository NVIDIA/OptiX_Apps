/* 
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "shaders/app_config.h"

#include "inc/Application.h"
#include "inc/CheckMacros.h"

#include "dp/math/Quatt.h"


#ifdef _WIN32
// The cfgmgr32 header is necessary for interrogating driver information in the registry.
#include <cfgmgr32.h>
// For convenience the library is also linked in automatically using the #pragma command.
#pragma comment(lib, "Cfgmgr32.lib")
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string.h>
#include <time.h>
#include <vector>

#include "shaders/vector_math.h"
#include "shaders/system_parameter.h"
#include "shaders/material_parameter.h"
// Only needed for the FLAG_THINWALLED
#include "shaders/per_ray_data.h"

#include <inc/MyAssert.h>


#ifdef _WIN32
// Code based on helper function in optix_stubs.h
static void* optixLoadWindowsDll(void)
{
  const char* optixDllName = "nvoptix.dll";
  void* handle = NULL;

  // Get the size of the path first, then allocate
  unsigned int size = GetSystemDirectoryA(NULL, 0);
  if (size == 0)
  {
    // Couldn't get the system path size, so bail
    return NULL;
  }

  size_t pathSize = size + 1 + strlen(optixDllName);
  char*  systemPath = (char*) malloc(pathSize);

  if (GetSystemDirectoryA(systemPath, size) != size - 1)
  {
    // Something went wrong
    free(systemPath);
    return NULL;
  }

  strcat(systemPath, "\\");
  strcat(systemPath, optixDllName);

  handle = LoadLibraryA(systemPath);

  free(systemPath);

  if (handle)
  {
    return handle;
  }

  // If we didn't find it, go looking in the register store.  Since nvoptix.dll doesn't
  // have its own registry entry, we are going to look for the OpenGL driver which lives
  // next to nvoptix.dll. 0 (null) will be returned if any errors occured.

  static const char* deviceInstanceIdentifiersGUID = "{4d36e968-e325-11ce-bfc1-08002be10318}";
  const ULONG        flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;
  ULONG              deviceListSize = 0;

  if (CM_Get_Device_ID_List_SizeA(&deviceListSize, deviceInstanceIdentifiersGUID, flags) != CR_SUCCESS)
  {
    return NULL;
  }

  char* deviceNames = (char*) malloc(deviceListSize);

  if (CM_Get_Device_ID_ListA(deviceInstanceIdentifiersGUID, deviceNames, deviceListSize, flags))
  {
    free(deviceNames);
    return NULL;
  }

  DEVINST devID = 0;

  // Continue to the next device if errors are encountered.
  for (char* deviceName = deviceNames; *deviceName; deviceName += strlen(deviceName) + 1)
  {
    if (CM_Locate_DevNodeA(&devID, deviceName, CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS)
    {
      continue;
    }

    HKEY regKey = 0;
    if (CM_Open_DevNode_Key(devID, KEY_QUERY_VALUE, 0, RegDisposition_OpenExisting, &regKey, CM_REGISTRY_SOFTWARE) != CR_SUCCESS)
    {
      continue;
    }

    const char* valueName = "OpenGLDriverName";
    DWORD       valueSize = 0;

    LSTATUS     ret = RegQueryValueExA(regKey, valueName, NULL, NULL, NULL, &valueSize);
    if (ret != ERROR_SUCCESS)
    {
      RegCloseKey(regKey);
      continue;
    }

    char* regValue = (char*) malloc(valueSize);
    ret = RegQueryValueExA(regKey, valueName, NULL, NULL, (LPBYTE) regValue, &valueSize);
    if (ret != ERROR_SUCCESS)
    {
      free(regValue);
      RegCloseKey(regKey);
      continue;
    }

    // Strip the OpenGL driver dll name from the string then create a new string with
    // the path and the nvoptix.dll name
    for (int i = valueSize - 1; i >= 0 && regValue[i] != '\\'; --i)
    {
      regValue[i] = '\0';
    }

    size_t newPathSize = strlen(regValue) + strlen(optixDllName) + 1;
    char*  dllPath = (char*) malloc(newPathSize);
    strcpy(dllPath, regValue);
    strcat(dllPath, optixDllName);

    free(regValue);
    RegCloseKey(regKey);

    handle = LoadLibraryA((LPCSTR) dllPath);
    free(dllPath);

    if (handle)
    {
      break;
    }
  }

  free(deviceNames);

  return handle;
}
#endif


Application::Application(GLFWwindow* window,
                         Options const& options)
: m_window(window)
, m_logger(std::cerr)
{
  m_width   = std::max(1, options.getClientWidth());
  m_height  = std::max(1, options.getClientHeight());
  m_interop = options.getInterop();

  m_lightID = options.getLight();
  m_missID  = options.getMiss();
  m_environmentFilename = options.getEnvironment();

  m_isValid = false;

  m_sceneEpsilonFactor = 500;  // Factor on SCENE_EPSILOPN_SCALE (1.0e-7f) used to offset ray tmin interval along the path to reduce self-intersections.
  m_iterationIndex = 0; 
  
  m_pbo = 0;
  m_hdrTexture = 0;

  m_outputBuffer = new float4[m_width * m_height];

  m_present         = false;  // Update once per second. (The first half second shows all frames to get some initial accumulation).
  m_presentNext     = true;
  m_presentAtSecond = 1.0;

  m_frames = 0; // Samples per pixel. 0 == render forever.
    
  m_glslVS = 0;
  m_glslFS = 0;
  m_glslProgram = 0;

#if 1 // Tonemapper defaults
  m_gamma          = 2.2f;
  m_colorBalance   = make_float3(1.0f, 1.0f, 1.0f);
  m_whitePoint     = 1.0f;
  m_burnHighlights = 0.8f;
  m_crushBlacks    = 0.2f;
  m_saturation     = 1.2f;
  m_brightness     = 0.8f;
#else // Neutral tonemapper settings.
  m_gamma          = 1.0f;
  m_colorBalance   = make_float3(1.0f, 1.0f, 1.0f);
  m_whitePoint     = 1.0f;
  m_burnHighlights = 1.0f;
  m_crushBlacks    = 0.0f;
  m_saturation     = 1.0f;
  m_brightness     = 1.0f;
#endif

  m_guiState = GUI_STATE_NONE;

  m_isVisibleGUI = true;

  m_mouseSpeedRatio = 10.0f;

  m_pinholeCamera.setViewport(m_width, m_height);

  m_textureEnvironment = nullptr;
  m_textureAlbedo      = nullptr;
  m_textureCutout      = nullptr;
  
  m_vboAttributes = 0;
  m_vboIndices = 0;
    
  m_positionLocation = -1;
  m_texCoordLocation = -1;
  
  m_cudaGraphicsResource = nullptr;

  m_context = nullptr;

  m_iasRoot               = 0;
  m_d_iasRoot             = 0;
  m_instanceInputRoot     = {};
  m_accelBuildOptionsRoot = {};
  m_iasBufferSizesRoot    = {};
  m_d_instancesRoot       = 0;
  m_d_tmpRoot             = 0;

  m_pipeline = nullptr;
  
  m_d_systemParameter = nullptr;
  
  // The Shader Binding Table and data.
  m_d_sbtRecordRaygeneration = 0;
  m_d_sbtRecordException = 0;
  m_d_sbtRecordMiss = 0;

  m_d_sbtRecordCallables = 0;

  m_d_sbtRecordGeometryInstanceData = nullptr;

  m_cameraPosition  = make_float3(0.0f, 0.0f, 1.0f);
  m_hasCameraMotion = false; // No camera motion blur by default.

  // Initialize all renderer system parameters.
  m_systemParameter.topObject          = 0;
  m_systemParameter.outputBuffer       = nullptr;
  m_systemParameter.lightDefinitions   = nullptr;
  m_systemParameter.materialParameters = nullptr;
  m_systemParameter.envTexture         = 0;
  m_systemParameter.envCDF_U           = nullptr;
  m_systemParameter.envCDF_V           = nullptr;
  m_systemParameter.pathLengths        = make_int2(2, 5);
  m_systemParameter.envWidth           = 0;
  m_systemParameter.envHeight          = 0;
  m_systemParameter.envIntegral        = 1.0f;
  m_systemParameter.envRotation        = 0.0f;
  m_systemParameter.iterationIndex     = 0;
  m_systemParameter.sceneEpsilon       = m_sceneEpsilonFactor * SCENE_EPSILON_SCALE;
  m_systemParameter.numLights          = 0;
  m_systemParameter.cameraType         = 0;
  m_systemParameter.cameraPosition0    = m_cameraPosition;
  m_systemParameter.cameraPosition1    = m_cameraPosition;
  m_systemParameter.cameraU            = make_float3(1.0f, 0.0f, 0.0f);
  m_systemParameter.cameraV            = make_float3(0.0f, 1.0f, 0.0f);
  m_systemParameter.cameraW            = make_float3(0.0f, 0.0f, -1.0f);

  // Motion blur related fields.
  m_frameAnimation  = 0;
  m_framesPerSecond = 10;
  m_timePerFrame    = 1.0f / float(m_framesPerSecond);
  m_velocity        = 1.0f; // [m/s]
  m_angularVelocity = 1.0f; // [rad/s]

  // Setup ImGui binding.
  ImGui::CreateContext();
  ImGui_ImplGlfwGL3_Init(window, true);

  // This initializes the GLFW part including the font texture.
  ImGui_ImplGlfwGL3_NewFrame();
  ImGui::EndFrame();

#if 1
  // Style the GUI colors to a neutral greyscale with plenty of transparency to concentrate on the image.
  ImGuiStyle& style = ImGui::GetStyle();

  // Change these RGB values to get any other tint.
  const float r = 1.0f;
  const float g = 1.0f;
  const float b = 1.0f;
  
  style.Colors[ImGuiCol_Text]                  = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  style.Colors[ImGuiCol_TextDisabled]          = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
  style.Colors[ImGuiCol_WindowBg]              = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 0.6f);
  style.Colors[ImGuiCol_ChildWindowBg]         = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 1.0f);
  style.Colors[ImGuiCol_PopupBg]               = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 1.0f);
  style.Colors[ImGuiCol_Border]                = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
  style.Colors[ImGuiCol_BorderShadow]          = ImVec4(r * 0.0f, g * 0.0f, b * 0.0f, 0.4f);
  style.Colors[ImGuiCol_FrameBg]               = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
  style.Colors[ImGuiCol_FrameBgHovered]        = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
  style.Colors[ImGuiCol_FrameBgActive]         = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
  style.Colors[ImGuiCol_TitleBg]               = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
  style.Colors[ImGuiCol_TitleBgCollapsed]      = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 0.2f);
  style.Colors[ImGuiCol_TitleBgActive]         = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
  style.Colors[ImGuiCol_MenuBarBg]             = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 1.0f);
  style.Colors[ImGuiCol_ScrollbarBg]           = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 0.2f);
  style.Colors[ImGuiCol_ScrollbarGrab]         = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
  style.Colors[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
  style.Colors[ImGuiCol_ScrollbarGrabActive]   = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
  style.Colors[ImGuiCol_CheckMark]             = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
  style.Colors[ImGuiCol_SliderGrab]            = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
  style.Colors[ImGuiCol_SliderGrabActive]      = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
  style.Colors[ImGuiCol_Button]                = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
  style.Colors[ImGuiCol_ButtonHovered]         = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
  style.Colors[ImGuiCol_ButtonActive]          = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
  style.Colors[ImGuiCol_Header]                = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
  style.Colors[ImGuiCol_HeaderHovered]         = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
  style.Colors[ImGuiCol_HeaderActive]          = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
  style.Colors[ImGuiCol_Column]                = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
  style.Colors[ImGuiCol_ColumnHovered]         = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
  style.Colors[ImGuiCol_ColumnActive]          = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
  style.Colors[ImGuiCol_ResizeGrip]            = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
  style.Colors[ImGuiCol_ResizeGripHovered]     = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
  style.Colors[ImGuiCol_ResizeGripActive]      = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
  style.Colors[ImGuiCol_CloseButton]           = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
  style.Colors[ImGuiCol_CloseButtonHovered]    = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
  style.Colors[ImGuiCol_CloseButtonActive]     = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
  style.Colors[ImGuiCol_PlotLines]             = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 1.0f);
  style.Colors[ImGuiCol_PlotLinesHovered]      = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
  style.Colors[ImGuiCol_PlotHistogram]         = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 1.0f);
  style.Colors[ImGuiCol_PlotHistogramHovered]  = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
  style.Colors[ImGuiCol_TextSelectedBg]        = ImVec4(r * 0.5f, g * 0.5f, b * 0.5f, 1.0f);
  style.Colors[ImGuiCol_ModalWindowDarkening]  = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 0.2f);
  style.Colors[ImGuiCol_DragDropTarget]        = ImVec4(r * 1.0f, g * 1.0f, b * 0.0f, 1.0f); // Yellow
  style.Colors[ImGuiCol_NavHighlight]          = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
  style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
#endif

  initOpenGL();

  m_moduleFilenames.resize(NUM_MODULE_IDENTIFIERS);

  // Starting with OptiX SDK 7.5.0 and CUDA 11.7 either PTX or OptiX IR input can be used to create modules.
  // Just initialize the m_moduleFilenames depending on the definition of USE_OPTIX_IR.
  // That is added to the project definitions inside the CMake script when OptiX SDK 7.5.0 and CUDA 11.7 or newer are found.
#if defined(USE_OPTIX_IR)
  m_moduleFilenames[MODULE_ID_RAYGENERATION]                    = std::string("./intro_motion_blur_core/raygeneration.optixir");
  m_moduleFilenames[MODULE_ID_EXCEPTION]                        = std::string("./intro_motion_blur_core/exception.optixir");
  m_moduleFilenames[MODULE_ID_MISS]                             = std::string("./intro_motion_blur_core/miss.optixir");
  m_moduleFilenames[MODULE_ID_CLOSESTHIT]                       = std::string("./intro_motion_blur_core/closesthit.optixir");
  m_moduleFilenames[MODULE_ID_ANYHIT]                           = std::string("./intro_motion_blur_core/anyhit.optixir");
  m_moduleFilenames[MODULE_ID_LENS_SHADER]                      = std::string("./intro_motion_blur_core/lens_shader.optixir");
  m_moduleFilenames[MODULE_ID_LIGHT_SAMPLE]                     = std::string("./intro_motion_blur_core/light_sample.optixir");
  m_moduleFilenames[MODULE_ID_DIFFUSE_REFLECTION]               = std::string("./intro_motion_blur_core/bsdf_diffuse_reflection.optixir");
  m_moduleFilenames[MODULE_ID_SPECULAR_REFLECTION]              = std::string("./intro_motion_blur_core/bsdf_specular_reflection.optixir");
  m_moduleFilenames[MODULE_ID_SPECULAR_REFLECTION_TRANSMISSION] = std::string("./intro_motion_blur_core/bsdf_specular_reflection_transmission.optixir");
#else
  m_moduleFilenames[MODULE_ID_RAYGENERATION]                    = std::string("./intro_motion_blur_core/raygeneration.ptx");
  m_moduleFilenames[MODULE_ID_EXCEPTION]                        = std::string("./intro_motion_blur_core/exception.ptx");
  m_moduleFilenames[MODULE_ID_MISS]                             = std::string("./intro_motion_blur_core/miss.ptx");
  m_moduleFilenames[MODULE_ID_CLOSESTHIT]                       = std::string("./intro_motion_blur_core/closesthit.ptx");
  m_moduleFilenames[MODULE_ID_ANYHIT]                           = std::string("./intro_motion_blur_core/anyhit.ptx");
  m_moduleFilenames[MODULE_ID_LENS_SHADER]                      = std::string("./intro_motion_blur_core/lens_shader.ptx");
  m_moduleFilenames[MODULE_ID_LIGHT_SAMPLE]                     = std::string("./intro_motion_blur_core/light_sample.ptx");
  m_moduleFilenames[MODULE_ID_DIFFUSE_REFLECTION]               = std::string("./intro_motion_blur_core/bsdf_diffuse_reflection.ptx");
  m_moduleFilenames[MODULE_ID_SPECULAR_REFLECTION]              = std::string("./intro_motion_blur_core/bsdf_specular_reflection.ptx");
  m_moduleFilenames[MODULE_ID_SPECULAR_REFLECTION_TRANSMISSION] = std::string("./intro_motion_blur_core/bsdf_specular_reflection_transmission.ptx");
#endif

  m_isValid = initOptiX();
}


Application::~Application()
{
  if (m_isValid)
  {
    CU_CHECK( cuStreamSynchronize(m_cudaStream) );

    // Delete the textures while the context is still alive.
    delete m_textureEnvironment;
    delete m_textureAlbedo;
    delete m_textureCutout;

    if (m_interop)
    {
      CU_CHECK( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
      glDeleteBuffers(1, &m_pbo);
    }
    else
    {
      CU_CHECK( cuMemFree(reinterpret_cast<CUdeviceptr>(m_systemParameter.outputBuffer)) );
      delete[] m_outputBuffer;
    }

    CU_CHECK( cuMemFree(reinterpret_cast<CUdeviceptr>(m_systemParameter.lightDefinitions)) );
    CU_CHECK( cuMemFree(reinterpret_cast<CUdeviceptr>(m_systemParameter.materialParameters)) );
    CU_CHECK( cuMemFree(reinterpret_cast<CUdeviceptr>(m_d_systemParameter)) );

    for (size_t i = 0; i < m_geometries.size(); ++i)
    {
      CU_CHECK( cuMemFree(m_geometries[i].indices) );
      CU_CHECK( cuMemFree(m_geometries[i].attributes) );
      CU_CHECK( cuMemFree(m_geometries[i].gas) );
    }
  
    CU_CHECK( cuMemFree(m_d_srtMotionTransform) );
    CU_CHECK( cuMemFree(m_d_matrixMotionTransform) );

    CU_CHECK( cuMemFree(m_d_iasRoot) );
    CU_CHECK( cuMemFree(m_d_instancesRoot) );
    CU_CHECK( cuMemFree(m_d_tmpRoot) );

    CU_CHECK( cuMemFree(m_d_sbtRecordRaygeneration) );
    CU_CHECK( cuMemFree(m_d_sbtRecordException) );
    CU_CHECK( cuMemFree(m_d_sbtRecordMiss) );
    CU_CHECK( cuMemFree(m_d_sbtRecordCallables) );

    CU_CHECK( cuMemFree(reinterpret_cast<CUdeviceptr>(m_d_sbtRecordGeometryInstanceData)) );

    OPTIX_CHECK( m_api.optixPipelineDestroy(m_pipeline) );
    OPTIX_CHECK( m_api.optixDeviceContextDestroy(m_context) );

    CU_CHECK( cuStreamDestroy(m_cudaStream) );
    CU_CHECK( cuCtxDestroy(m_cudaContext) );
    
    glDeleteBuffers(1, &m_vboAttributes);
    glDeleteBuffers(1, &m_vboIndices);

    glDeleteProgram(m_glslProgram);
  }

  ImGui_ImplGlfwGL3_Shutdown();
  ImGui::DestroyContext();
}

bool Application::isValid() const
{
  return m_isValid;
}

void Application::reshape(int width, int height)
{
  if ((width != 0 && height != 0) && // Zero sized interop buffers are not allowed in OptiX.
      (m_width != width || m_height != height))
  {
    m_width  = width;
    m_height = height;

    glViewport(0, 0, m_width, m_height);

    if (m_interop)
    {
      CU_CHECK( cuGraphicsUnregisterResource(m_cudaGraphicsResource) ); // No flags for read-write access during accumulation.

      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
      glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

      CU_CHECK( cuGraphicsGLRegisterBuffer(&m_cudaGraphicsResource, m_pbo, CU_GRAPHICS_REGISTER_FLAGS_NONE) );

      size_t size;

      CU_CHECK( cuGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream) );
      CU_CHECK( cuGraphicsResourceGetMappedPointer(reinterpret_cast<CUdeviceptr*>(&m_systemParameter.outputBuffer), &size, m_cudaGraphicsResource) ); // Redundant. Must be done on each map anyway.
      CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) );
      
      MY_ASSERT(m_width * m_height * sizeof(float) * 4 <= size);
    }
    else
    {
      delete[] m_outputBuffer;
      m_outputBuffer = new float4[m_width * m_height];

      CU_CHECK( cuMemFree(reinterpret_cast<CUdeviceptr>(m_systemParameter.outputBuffer)) );
      CU_CHECK( cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&m_systemParameter.outputBuffer), sizeof(float4) * m_width * m_height) );
    }

    m_pinholeCamera.setViewport(m_width, m_height);

    restartAccumulation();
  }
}

void Application::guiNewFrame()
{
  ImGui_ImplGlfwGL3_NewFrame();
}

void Application::guiReferenceManual()
{
  ImGui::ShowTestWindow();
}

void Application::guiRender()
{
  ImGui::Render();
  ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());
}


void Application::getSystemInformation()
{
  int versionDriver = 0;
  CU_CHECK( cuDriverGetVersion(&versionDriver) ); 
  
  // The version is returned as (1000 * major + 10 * minor).
  int major =  versionDriver / 1000;
  int minor = (versionDriver - major * 1000) / 10;
  std::cout << "Driver Version  = " << major << "." << minor << '\n';
  
  int countDevices = 0;
  CU_CHECK( cuDeviceGetCount(&countDevices) );
  std::cout << "Device Count    = " << countDevices << '\n';

  char name[1024];
  name[1023] = 0;

  for (CUdevice device = 0; device < countDevices; ++device)
  {
    CU_CHECK( cuDeviceGetName(name, 1023, device) );
    std::cout << "Device " << device << ": " << name << '\n';

    DeviceAttribute attr = {};

    CU_CHECK( cuDeviceGetAttribute(&attr.maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxBlockDimX, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxBlockDimY, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxBlockDimZ, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxGridDimX, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxGridDimY, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxGridDimZ, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxSharedMemoryPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.sharedMemoryPerBlock, CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.totalConstantMemory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxPitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxRegistersPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.registersPerBlock, CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.textureAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.gpuOverlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.multiprocessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.kernelExecTimeout, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.canMapHostMemory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture1dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dDepth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLayeredHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dArrayWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dArrayHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dArrayNumslices, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.surfaceAlignment, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.concurrentKernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.eccEnabled, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.pciBusId, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.pciDeviceId, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.tccDriver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.memoryClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.globalMemoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.l2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxThreadsPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.asyncEngineCount, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.unifiedAddressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture1dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture1dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.canTex2dGather, CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dGatherWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dGatherHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dWidthAlternate, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dHeightAlternate, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dDepthAlternate, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.pciDomainId, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.texturePitchAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexturecubemapWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexturecubemapLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexturecubemapLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface1dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface2dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface2dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface3dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface3dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface3dDepth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface1dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface1dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface2dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface2dLayeredHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface2dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurfacecubemapWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurfacecubemapLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurfacecubemapLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture1dLinearWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLinearWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLinearHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLinearPitch, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dMipmappedWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dMipmappedHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture1dMipmappedWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.streamPrioritiesSupported, CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.globalL1CacheSupported, CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.localL1CacheSupported, CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxSharedMemoryPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxRegistersPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.managedMemory, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.multiGpuBoard, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.multiGpuBoardGroupId, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.hostNativeAtomicSupported, CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.singleToDoublePrecisionPerfRatio, CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.pageableMemoryAccess, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.concurrentManagedAccess, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.computePreemptionSupported, CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.canUseHostPointerForRegisteredMem, CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.canUse64BitStreamMemOps, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.canUseStreamWaitValueNor, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.cooperativeLaunch, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.cooperativeMultiDeviceLaunch, CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxSharedMemoryPerBlockOptin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.canFlushRemoteWrites, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.hostRegisterSupported, CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.pageableMemoryAccessUsesHostPageTables, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.directManagedMemAccessFromHost, CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, device) );

    m_deviceAttributes.push_back(attr);
  }
}

void Application::initOpenGL()
{
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

  glViewport(0, 0, m_width, m_height);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // glPixelStorei(GL_UNPACK_ALIGNMENT, 4); // default, works for BGRA8, RGBA16F, and RGBA32F.

  glDisable(GL_CULL_FACE);  // default
  glDisable(GL_DEPTH_TEST); // default

  if (m_interop)
  {
    // PBO for CUDA-OpenGL interop.
    glGenBuffers(1, &m_pbo);
    MY_ASSERT(m_pbo != 0); 

    // Buffer size must be > 0 or OptiX can't create a buffer from it.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * sizeof(float) * 4, (void*) 0, GL_DYNAMIC_DRAW); // RGBA32F from byte offset 0 in the pixel unpack buffer.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  }

  glGenTextures(1, &m_hdrTexture);
  MY_ASSERT(m_hdrTexture != 0);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glBindTexture(GL_TEXTURE_2D, 0);

  // DAR Local ImGui code has been changed to push the GL_TEXTURE_BIT so that this works. 
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  // GLSL shaders objects and program. 
  m_glslVS      = 0;
  m_glslFS      = 0;
  m_glslProgram = 0;

  m_positionLocation   = -1;
  m_texCoordLocation   = -1;

  initGLSL();

  // Two hardcoded triangles in the identity matrix pojection coordinate system with 2D texture coordinates.
  const float attributes[16] = 
  {
    // vertex2f,   texcoord2f
    -1.0f, -1.0f,  0.0f, 0.0f,
     1.0f, -1.0f,  1.0f, 0.0f,
     1.0f,  1.0f,  1.0f, 1.0f,
    -1.0f,  1.0f,  0.0f, 1.0f
  };

  unsigned int indices[6] = 
  {
    0, 1, 2, 
    2, 3, 0
  };

  glGenBuffers(1, &m_vboAttributes);
  MY_ASSERT(m_vboAttributes != 0);

  glGenBuffers(1, &m_vboIndices);
  MY_ASSERT(m_vboIndices != 0);

  // Setup the vertex arrays from the interleaved vertex attributes.
  glBindBuffer(GL_ARRAY_BUFFER, m_vboAttributes);
  glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) sizeof(float) * 16, (GLvoid const*) attributes, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vboIndices);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr) sizeof(unsigned int) * 6, (const GLvoid*) indices, GL_STATIC_DRAW);

  glVertexAttribPointer(m_positionLocation, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (GLvoid*) 0);
  //glEnableVertexAttribArray(m_positionLocation);

  glVertexAttribPointer(m_texCoordLocation, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (GLvoid*) (sizeof(float) * 2));
  //glEnableVertexAttribArray(m_texCoordLocation);
}


OptixResult Application::initOptiXFunctionTable()
{
#ifdef _WIN32
  void* handle = optixLoadWindowsDll();
  if (!handle)
  {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }

  void* symbol = reinterpret_cast<void*>(GetProcAddress((HMODULE) handle, "optixQueryFunctionTable"));
  if (!symbol)
  {
    return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
  }
#else
  void* handle = dlopen("libnvoptix.so.1", RTLD_NOW);
  if (!handle)
  {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }

  void* symbol = dlsym(handle, "optixQueryFunctionTable");
  if (!symbol)
  {
    return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
  }
#endif

  OptixQueryFunctionTable_t* optixQueryFunctionTable = reinterpret_cast<OptixQueryFunctionTable_t*>(symbol);

  return optixQueryFunctionTable(OPTIX_ABI_VERSION, 0, 0, 0, &m_api, sizeof(OptixFunctionTable));
}


bool Application::initOptiX()
{
  CUresult cuRes = cuInit(0); // Initialize CUDA driver API.
  if (cuRes != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() cuInit() failed: " << cuRes << '\n';
    return false;
  }

  getSystemInformation(); // Get device attributes of all found devices. Fills m_deviceAttributes.

  CUdevice device = 0;

  cuRes = cuCtxCreate(&m_cudaContext, CU_CTX_SCHED_SPIN, device); // DEBUG What is the best CU_CTX_SCHED_* setting here.
  if (cuRes != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() cuCtxCreate() failed: " << cuRes << '\n';
    return false;
  }

  // PERF Use CU_STREAM_NON_BLOCKING if there is any work running in parallel on multiple streams.
  cuRes = cuStreamCreate(&m_cudaStream, CU_STREAM_DEFAULT);
  if (cuRes != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() cuStreamCreate() failed: " << cuRes << '\n';
    return false;
  }

  OptixResult res = initOptiXFunctionTable();
  if (res != OPTIX_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() initOptiXFunctionTable() failed: " << res << '\n';
    return false;
  }

  OptixDeviceContextOptions options = {};

  options.logCallbackFunction = &Logger::callback;
  options.logCallbackData     = &m_logger;
  options.logCallbackLevel    = 3; // Keep at warning level to suppress the disk cache messages.

  res = m_api.optixDeviceContextCreate(m_cudaContext, &options, &m_context);
  if (res != OPTIX_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() optixDeviceContextCreate() failed: " << res << '\n';
    return false;
  }

  initRenderer(); // Initialize all the rest.

  return true;
}


void Application::restartAccumulation()
{
  m_iterationIndex  = 0;
  m_presentNext     = true;
  m_presentAtSecond = 1.0;

  CU_CHECK( cuStreamSynchronize(m_cudaStream) );
  CU_CHECK( cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(m_d_systemParameter), &m_systemParameter, sizeof(SystemParameter)) );

  m_timer.restart();
}


bool Application::render()
{
  bool repaint = false;

  bool cameraChanged = m_pinholeCamera.getFrustum(m_cameraPosition,
                                                  m_systemParameter.cameraU,
                                                  m_systemParameter.cameraV,
                                                  m_systemParameter.cameraW);
  if (cameraChanged)
  {
    updateCameraAnimation();
    restartAccumulation();
  }
  
  // Continue manual accumulation rendering if there is no limit (m_frames == 0) or the number of frames has not been reached.
  if (0 == m_frames || m_iterationIndex < m_frames)
  {
    // Update only the sysParameter.iterationIndex.
    m_systemParameter.iterationIndex = m_iterationIndex++;

    CU_CHECK( cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(&m_d_systemParameter->iterationIndex), &m_systemParameter.iterationIndex, sizeof(int)) );

    if (m_interop)
    {
      size_t size;

      CU_CHECK( cuGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream) );
      CU_CHECK( cuGraphicsResourceGetMappedPointer(reinterpret_cast<CUdeviceptr*>(&m_systemParameter.outputBuffer), &size, m_cudaGraphicsResource) ); // The pointer can change on every map!
      CU_CHECK( cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(&m_d_systemParameter->outputBuffer), &m_systemParameter.outputBuffer, sizeof(void*)) );

      OPTIX_CHECK( m_api.optixLaunch(m_pipeline, m_cudaStream, (CUdeviceptr) m_d_systemParameter, sizeof(SystemParameter), &m_sbt, m_width, m_height, /* depth */ 1) );
      
      CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) );
    }
    else
    {
      OPTIX_CHECK( m_api.optixLaunch(m_pipeline, m_cudaStream, (CUdeviceptr) m_d_systemParameter, sizeof(SystemParameter), &m_sbt, m_width, m_height, /* depth */ 1) );
    }
  }

  // Only update the texture when a restart happened or one second passed to reduce required bandwidth.
  if (m_presentNext)
  {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture); // Manual accumulation always renders into the m_hdrTexture.

    if (m_interop)
    {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_width, (GLsizei) m_height, 0, GL_RGBA, GL_FLOAT, (void*) 0); // RGBA32F from byte offset 0 in the pixel unpack buffer.
    }
    else
    {
      CU_CHECK( cuMemcpyDtoH((void*) m_outputBuffer, reinterpret_cast<CUdeviceptr>(m_systemParameter.outputBuffer), sizeof(float4) * m_width * m_height) );
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_width, (GLsizei) m_height, 0, GL_RGBA, GL_FLOAT, m_outputBuffer); // RGBA32F
    }

    repaint = true; // Indicate that there is a new image.

    m_presentNext = m_present;
  }

  double seconds = m_timer.getTime();
#if 1
  // Show the accumulation of the first half second to get some refinement after interaction.
  if (seconds < 0.5)
  {
    m_presentAtSecond = 1.0;
    m_presentNext     = true;
  }
  else 
#endif
  if (m_presentAtSecond < seconds)
  {
    m_presentAtSecond = ceil(seconds);
      
    const double fps = double(m_iterationIndex) / seconds;

    std::ostringstream stream; 
    stream.precision(3); // Precision is # digits in fraction part.
    // m_iterationIndex has already been incremented for the last rendered frame, so it is the actual framecount here.
    stream << std::fixed << m_iterationIndex << " / " << seconds << " = " << fps << " fps";
    std::cout << stream.str() << '\n';

    m_presentNext = true; // Present at least every second.
  }
  
  return repaint;
}

void Application::display()
{
  glBindBuffer(GL_ARRAY_BUFFER, m_vboAttributes);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vboIndices);

  glEnableVertexAttribArray(m_positionLocation);
  glEnableVertexAttribArray(m_texCoordLocation);

  glUseProgram(m_glslProgram);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_hdrTexture);

  glDrawElements(GL_TRIANGLES, (GLsizei) 6, GL_UNSIGNED_INT, (const GLvoid*) 0);

  glUseProgram(0);

  glDisableVertexAttribArray(m_positionLocation);
  glDisableVertexAttribArray(m_texCoordLocation);
}


void Application::checkInfoLog(const char *msg, GLuint object)
{
  GLint  maxLength;
  GLint  length;
  GLchar *infoLog;

  if (glIsProgram(object))
  {
    glGetProgramiv(object, GL_INFO_LOG_LENGTH, &maxLength);
  }
  else
  {
    glGetShaderiv(object, GL_INFO_LOG_LENGTH, &maxLength);
  }
  if (maxLength > 1) 
  {
    infoLog = (GLchar *) malloc(maxLength);
    if (infoLog != NULL)
    {
      if (glIsShader(object))
      {
        glGetShaderInfoLog(object, maxLength, &length, infoLog);
      }
      else
      {
        glGetProgramInfoLog(object, maxLength, &length, infoLog);
      }
      //fprintf(fileLog, "-- tried to compile (len=%d): %s\n", (unsigned int)strlen(msg), msg);
      //fprintf(fileLog, "--- info log contents (len=%d) ---\n", (int) maxLength);
      //fprintf(fileLog, "%s", infoLog);
      //fprintf(fileLog, "--- end ---\n");
      std::cout << infoLog << '\n';
      // Look at the info log string here...
      free(infoLog);
    }
  }
}

void Application::initGLSL()
{
  static const std::string vsSource =
    "#version 330\n"
    "layout(location = 0) in vec2 attrPosition;\n"
    "layout(location = 1) in vec2 attrTexCoord;\n"
    "out vec2 varTexCoord;\n"
    "void main()\n"
    "{\n"
    "  gl_Position = vec4(attrPosition, 0.0, 1.0);\n"
    "  varTexCoord = attrTexCoord;\n"
    "}\n";

  static const std::string fsSource =
    "#version 330\n"
    "uniform sampler2D samplerHDR;\n"
    "uniform vec3  colorBalance;\n"
    "uniform float invWhitePoint;\n"
    "uniform float burnHighlights;\n"
    "uniform float saturation;\n"
    "uniform float crushBlacks;\n"
    "uniform float invGamma;\n"
    "in vec2 varTexCoord;\n"
    "layout(location = 0, index = 0) out vec4 outColor;\n"
    "void main()\n"
    "{\n"
    "  vec3 hdrColor = texture(samplerHDR, varTexCoord).rgb;\n"
    "  vec3 ldrColor = invWhitePoint * colorBalance * hdrColor;\n"
    "  ldrColor *= (ldrColor * burnHighlights + 1.0) / (ldrColor + 1.0);\n"
    "  float luminance = dot(ldrColor, vec3(0.3, 0.59, 0.11));\n"
    "  ldrColor = max(mix(vec3(luminance), ldrColor, saturation), 0.0);\n"
    "  luminance = dot(ldrColor, vec3(0.3, 0.59, 0.11));\n"
    "  if (luminance < 1.0)\n"
    "  {\n"
    "    ldrColor = max(mix(pow(ldrColor, vec3(crushBlacks)), ldrColor, sqrt(luminance)), 0.0);\n"
    "  }\n"
    "  ldrColor = pow(ldrColor, vec3(invGamma));\n"
    "  outColor = vec4(ldrColor, 1.0);\n"
    "}\n";

  GLint vsCompiled = 0;
  GLint fsCompiled = 0;
    
  m_glslVS = glCreateShader(GL_VERTEX_SHADER);
  if (m_glslVS)
  {
    GLsizei len = (GLsizei) vsSource.size();
    const GLchar *vs = vsSource.c_str();
    glShaderSource(m_glslVS, 1, &vs, &len);
    glCompileShader(m_glslVS);
    checkInfoLog(vs, m_glslVS);

    glGetShaderiv(m_glslVS, GL_COMPILE_STATUS, &vsCompiled);
    MY_ASSERT(vsCompiled);
  }

  m_glslFS = glCreateShader(GL_FRAGMENT_SHADER);
  if (m_glslFS)
  {
    GLsizei len = (GLsizei) fsSource.size();
    const GLchar *fs = fsSource.c_str();
    glShaderSource(m_glslFS, 1, &fs, &len);
    glCompileShader(m_glslFS);
    checkInfoLog(fs, m_glslFS);

    glGetShaderiv(m_glslFS, GL_COMPILE_STATUS, &fsCompiled);
    MY_ASSERT(fsCompiled);
  }

  m_glslProgram = glCreateProgram();
  if (m_glslProgram)
  {
    GLint programLinked = 0;

    if (m_glslVS && vsCompiled)
    {
      glAttachShader(m_glslProgram, m_glslVS);
    }
    if (m_glslFS && fsCompiled)
    {
      glAttachShader(m_glslProgram, m_glslFS);
    }

    glLinkProgram(m_glslProgram);
    checkInfoLog("m_glslProgram", m_glslProgram);

    glGetProgramiv(m_glslProgram, GL_LINK_STATUS, &programLinked);
    MY_ASSERT(programLinked);

    if (programLinked)
    {
      glUseProgram(m_glslProgram);

      m_positionLocation = glGetAttribLocation(m_glslProgram, "attrPosition");
      MY_ASSERT(m_positionLocation != -1);

      m_texCoordLocation = glGetAttribLocation(m_glslProgram, "attrTexCoord");
      MY_ASSERT(m_texCoordLocation != -1);
      
      glUniform1i(glGetUniformLocation(m_glslProgram, "samplerHDR"), 0); // Always using texture image unit 0 for the display texture.
      glUniform1f(glGetUniformLocation(m_glslProgram, "invGamma"), 1.0f / m_gamma);
      glUniform3f(glGetUniformLocation(m_glslProgram, "colorBalance"), m_colorBalance.x, m_colorBalance.y, m_colorBalance.z);
      glUniform1f(glGetUniformLocation(m_glslProgram, "invWhitePoint"), m_brightness / m_whitePoint);
      glUniform1f(glGetUniformLocation(m_glslProgram, "burnHighlights"), m_burnHighlights);
      glUniform1f(glGetUniformLocation(m_glslProgram, "crushBlacks"), m_crushBlacks + m_crushBlacks + 1.0f);
      glUniform1f(glGetUniformLocation(m_glslProgram, "saturation"), m_saturation);

      glUseProgram(0);
    }
  }
}


void Application::guiWindow()
{
  if (!m_isVisibleGUI) // Use SPACE to toggle the display of the GUI window.
  {
    return;
  }

  ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);

  ImGuiWindowFlags window_flags = 0;
  if (!ImGui::Begin("intro_motion_blur", nullptr, window_flags)) // No bool flag to omit the close button.
  {
    // Early out if the window is collapsed, as an optimization.
    ImGui::End();
    return;
  }

  ImGui::PushItemWidth(-110); // Right-aligned, keep pixels for the labels.

  if (ImGui::CollapsingHeader("System"))
  {
    if (ImGui::Checkbox("Present", &m_present))
    {
      // No action needed, happens automatically on next frame.
    }
    if (ImGui::Combo("Camera", (int*) &m_systemParameter.cameraType, "Pinhole\0Fisheye\0Spherical\0\0"))
    {
      restartAccumulation();
    }
    if (ImGui::DragInt("Min Path Length", &m_systemParameter.pathLengths.x, 1.0f, 0, 100))
    {
      restartAccumulation();
    }
    if (ImGui::DragInt("Max Path Length", &m_systemParameter.pathLengths.y, 1.0f, 0, 100))
    {
      restartAccumulation();
    }
    if (ImGui::DragFloat("Scene Epsilon", &m_sceneEpsilonFactor, 1.0f, 0.0f, 10000.0f))
    {
      m_systemParameter.sceneEpsilon = m_sceneEpsilonFactor * SCENE_EPSILON_SCALE;
      restartAccumulation();
    }
    if (ImGui::DragFloat("Env Rotation", &m_systemParameter.envRotation, 0.001f, 0.0f, 1.0f))
    {
      restartAccumulation();
    }
    if (ImGui::DragInt("Frames", &m_frames, 1.0f, 0, 10000))
    {
      if (m_frames != 0 && m_frames < m_iterationIndex) // If we already rendered more frames, start again.
      {
        restartAccumulation();
      }
    }
    if (ImGui::DragFloat("Mouse Ratio", &m_mouseSpeedRatio, 0.1f, 0.1f, 1000.0f, "%.1f"))
    {
      m_pinholeCamera.setSpeedRatio(m_mouseSpeedRatio);
    }
  }
  if (ImGui::CollapsingHeader("Tonemapper"))
  {
    if (ImGui::ColorEdit3("Balance", (float*) &m_colorBalance))
    {
      glUseProgram(m_glslProgram);
      glUniform3f(glGetUniformLocation(m_glslProgram, "colorBalance"), m_colorBalance.x, m_colorBalance.y, m_colorBalance.z);
      glUseProgram(0);
    }
    if (ImGui::DragFloat("Gamma", &m_gamma, 0.01f, 0.01f, 10.0f)) // Must not get 0.0f
    {
      glUseProgram(m_glslProgram);
      glUniform1f(glGetUniformLocation(m_glslProgram, "invGamma"), 1.0f / m_gamma);
      glUseProgram(0);
    }
    if (ImGui::DragFloat("White Point", &m_whitePoint, 0.01f, 0.01f, 255.0f, "%.2f", 2.0f)) // Must not get 0.0f
    {
      glUseProgram(m_glslProgram);
      glUniform1f(glGetUniformLocation(m_glslProgram, "invWhitePoint"), m_brightness / m_whitePoint);
      glUseProgram(0);
    }
    if (ImGui::DragFloat("Burn Lights", &m_burnHighlights, 0.01f, 0.0f, 10.0f, "%.2f"))
    {
      glUseProgram(m_glslProgram);
      glUniform1f(glGetUniformLocation(m_glslProgram, "burnHighlights"), m_burnHighlights);
      glUseProgram(0);
    }
    if (ImGui::DragFloat("Crush Blacks", &m_crushBlacks, 0.01f, 0.0f, 1.0f, "%.2f"))
    {
      glUseProgram(m_glslProgram);
      glUniform1f(glGetUniformLocation(m_glslProgram, "crushBlacks"),  m_crushBlacks + m_crushBlacks + 1.0f);
      glUseProgram(0);
    }
    if (ImGui::DragFloat("Saturation", &m_saturation, 0.01f, 0.0f, 10.0f, "%.2f"))
    {
      glUseProgram(m_glslProgram);
      glUniform1f(glGetUniformLocation(m_glslProgram, "saturation"), m_saturation);
      glUseProgram(0);
    }
    if (ImGui::DragFloat("Brightness", &m_brightness, 0.01f, 0.0f, 100.0f, "%.2f", 2.0f))
    {
      glUseProgram(m_glslProgram);
      glUniform1f(glGetUniformLocation(m_glslProgram, "invWhitePoint"), m_brightness / m_whitePoint);
      glUseProgram(0);
    }
  }
  if (ImGui::CollapsingHeader("Animation"))
  {
    bool changed = false;
    if (ImGui::Checkbox("Camera Motion", &m_hasCameraMotion))
    {
      changed = true;
    }
    if (ImGui::DragInt("Frame", &m_frameAnimation, 1.0f, 0, 59)) // 60 frames long animation.
    {
      changed = true;
    }
    if (ImGui::DragInt("FPS", &m_framesPerSecond, 1.0f, 1, 20)) // 1 to 20 fps
    {
      m_timePerFrame = 1.0f / m_framesPerSecond;
      changed = true;
    }
    if (ImGui::DragFloat("Velocity", &m_velocity, 0.01f, 0.0f, 3.0f, "%.2f")) // Maximum of 3 m/s horizontal speed.
    {
      changed = true;
    }
    // Make sure the rotation cannot become bigger than 180 degrees (M_PIf) per frame interval.
    // Setting velocity and angular velocity to the same value will generate a nicely rolling object. 
    // That is, a sphere of radius 1.0f will exactly move a point on the great circle the same distance as it translates in horizontal direction.
    if (ImGui::DragFloat("Angular Vel.", &m_angularVelocity, 0.01f, 0.0f, 3.0, "%.2f"))
    {
      changed = true;
    }
    if (changed)
    {
      updateCameraAnimation();
      updateAnimation();
      restartAccumulation();
    }
  }
  if (ImGui::CollapsingHeader("Materials"))
  {
    bool changed = false;

    // HACK The last material is a black specular reflection for the area light and not editable
    // because this example does not support explicit light sampling of textured or cutout opacity geometry.
    for (int i = 0; i < int(m_guiMaterialParameters.size()) - 1; ++i)
    {
      if (ImGui::TreeNode((void*)(intptr_t) i, "Material %d", i))
      {
        MaterialParameterGUI& parameters = m_guiMaterialParameters[i];

        if (ImGui::Combo("BSDF Type", (int*) &parameters.indexBSDF,
                         "Diffuse Reflection\0Specular Reflection\0Specular Reflection Transmission\0\0"))
        {
          changed = true;
        }
        if (ImGui::ColorEdit3("Albedo", (float*) &parameters.albedo))
        {
          changed = true;
        }
        if (ImGui::Checkbox("Use Albedo Texture", &parameters.useAlbedoTexture))
        {
          changed = true;
        }
        if (ImGui::Checkbox("Use Cutout Texture", &parameters.useCutoutTexture))
        {
          // This chnages the hit group in the Shader Binding Table between opaque and cutout. (Opaque renders faster.)
          updateShaderBindingTable(i);
          changed = true; // This triggers the sysParameter.textureCutout object ID update.
        }
        if (ImGui::Checkbox("Thin-Walled", &parameters.thinwalled)) // Set this to true when using cutout opacity. Refracting materials won't look right with cutouts otherwise.
        {
          changed = true;
        }
        // Only show material parameters for the BSDFs which are affected.
        if (parameters.indexBSDF == INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION)
        {
          if (ImGui::ColorEdit3("Absorption", (float*) &parameters.absorptionColor))
          {
            changed = true;
          }
          if (ImGui::DragFloat("Volume Scale", &parameters.volumeDistanceScale, 0.01f, 0.0f, 100.0f, "%.2f"))
          {
            changed = true;
          }
          if (ImGui::DragFloat("IOR", &parameters.ior, 0.01f, 0.0f, 10.0f, "%.2f"))
          {
            changed = true;
          }
        }
        ImGui::TreePop();
      }
    }
    
    if (changed) // If any of the material parameters changed, simply upload them to the sysMaterialParameters again.
    {
      updateMaterialParameters();
      restartAccumulation();
    }
  }
  if (ImGui::CollapsingHeader("Lights"))
  {
    bool changed = false;
    
    for (int i = 0; i < int(m_lightDefinitions.size()); ++i)
    {
      LightDefinition& light = m_lightDefinitions[i];

      // Allow to change the emission (radiant exitance in Watt/m^2) of the rectangle lights in the scene.
      if (light.type == LIGHT_PARALLELOGRAM)
      {
        if (ImGui::TreeNode((void*)(intptr_t) i, "Light %d", i))
        {
          if (ImGui::DragFloat3("Emission", (float*) &light.emission, 1.0f, 0.0f, 10000.0f, "%.0f"))
          {
            changed = true;
          }
          ImGui::TreePop();
        }
      }
    }
    if (changed) // If any of the light parameters changed, simply upload them to the sysMaterialParameters again.
    {
      CU_CHECK( cuStreamSynchronize(m_cudaStream) );
      CU_CHECK( cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(m_systemParameter.lightDefinitions), m_lightDefinitions.data(), sizeof(LightDefinition) * m_lightDefinitions.size()) );

      restartAccumulation();
    }
  }

  ImGui::PopItemWidth();

  ImGui::End();
}

void Application::guiEventHandler()
{
  ImGuiIO const& io = ImGui::GetIO();

  if (ImGui::IsKeyPressed(' ', false)) // Toggle the GUI window display with SPACE key.
  {
    m_isVisibleGUI = !m_isVisibleGUI;
  }

  const ImVec2 mousePosition = ImGui::GetMousePos(); // Mouse coordinate window client rect.
  const int x = int(mousePosition.x);
  const int y = int(mousePosition.y);

  switch (m_guiState)
  {
    case GUI_STATE_NONE:
      if (!io.WantCaptureMouse) // Only allow camera interactions to begin when not interacting with the GUI.
      {
        if (ImGui::IsMouseDown(0)) // LMB down event?
        {
          m_pinholeCamera.setBaseCoordinates(x, y);
          m_guiState = GUI_STATE_ORBIT;
        }
        else if (ImGui::IsMouseDown(1)) // RMB down event?
        {
          m_pinholeCamera.setBaseCoordinates(x, y);
          m_guiState = GUI_STATE_DOLLY;
        }
        else if (ImGui::IsMouseDown(2)) // MMB down event?
        {
          m_pinholeCamera.setBaseCoordinates(x, y);
          m_guiState = GUI_STATE_PAN;
        }
        else if (io.MouseWheel != 0.0f) // Mouse wheel zoom.
        {
          m_pinholeCamera.zoom(io.MouseWheel);
        }
      }
      break;

    case GUI_STATE_ORBIT:
      if (ImGui::IsMouseReleased(0)) // LMB released? End of orbit mode.
      {
        m_guiState = GUI_STATE_NONE;
      }
      else
      {
        m_pinholeCamera.orbit(x, y);
      }
      break;

    case GUI_STATE_DOLLY:
      if (ImGui::IsMouseReleased(1)) // RMB released? End of dolly mode.
      {
        m_guiState = GUI_STATE_NONE;
      }
      else
      {
        m_pinholeCamera.dolly(x, y);
      }
      break;

    case GUI_STATE_PAN:
      if (ImGui::IsMouseReleased(2)) // MMB released? End of pan mode.
      {
        m_guiState = GUI_STATE_NONE;
      }
      else
      {
        m_pinholeCamera.pan(x, y);
      }
      break;
  }
}


// This part is always identical in the generated geometry creation routines.
OptixTraversableHandle Application::createGeometry(std::vector<VertexAttributes> const& attributes, std::vector<unsigned int> const& indices)
{
  CUdeviceptr d_attributes;
  CUdeviceptr d_indices;

  const size_t attributesSizeInBytes = sizeof(VertexAttributes) * attributes.size();

  CU_CHECK( cuMemAlloc(&d_attributes, attributesSizeInBytes) );
  CU_CHECK( cuMemcpyHtoD(d_attributes, attributes.data(), attributesSizeInBytes) );

  const size_t indicesSizeInBytes = sizeof(unsigned int) * indices.size();

  CU_CHECK( cuMemAlloc(&d_indices, indicesSizeInBytes) );
  CU_CHECK( cuMemcpyHtoD(d_indices, indices.data(), indicesSizeInBytes) );

  OptixBuildInput triangleInput = {};

  triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  triangleInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
  triangleInput.triangleArray.vertexStrideInBytes = sizeof(VertexAttributes);
  triangleInput.triangleArray.numVertices         = (unsigned int) attributes.size();
  triangleInput.triangleArray.vertexBuffers       = &d_attributes;

  triangleInput.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  triangleInput.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;

  triangleInput.triangleArray.numIndexTriplets   = (unsigned int) indices.size() / 3;
  triangleInput.triangleArray.indexBuffer        = d_indices;

  unsigned int triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

  triangleInput.triangleArray.flags         = triangleInputFlags;
  triangleInput.triangleArray.numSbtRecords = 1;

  OptixAccelBuildOptions accelBuildOptions = {};

  accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
  accelBuildOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes accelBufferSizes;
  
  OPTIX_CHECK( m_api.optixAccelComputeMemoryUsage(m_context, &accelBuildOptions, &triangleInput, 1, &accelBufferSizes) );

  CUdeviceptr d_gas; // This holds the geometry acceleration structure.

  CU_CHECK( cuMemAlloc(&d_gas, accelBufferSizes.outputSizeInBytes) );

  CUdeviceptr d_tmp;

  CU_CHECK( cuMemAlloc(&d_tmp, accelBufferSizes.tempSizeInBytes) ); // Allocate temporary buffers last to reduce fragmentation.

  OptixTraversableHandle traversableHandle = 0; // This is the handle which gets returned.

  OPTIX_CHECK( m_api.optixAccelBuild(m_context, m_cudaStream, 
                                     &accelBuildOptions, &triangleInput, 1,
                                     d_tmp, accelBufferSizes.tempSizeInBytes,
                                     d_gas, accelBufferSizes.outputSizeInBytes, 
                                     &traversableHandle, nullptr, 0) );

  CU_CHECK( cuStreamSynchronize(m_cudaStream) );

  CU_CHECK( cuMemFree(d_tmp) );
  
  // Track the GeometryData to be able to set them in the SBT record GeometryInstanceData and free them on exit.
  GeometryData geometry;

  geometry.indices       = d_indices;
  geometry.attributes    = d_attributes;
  geometry.numIndices    = indices.size();
  geometry.numAttributes = attributes.size();
  geometry.gas           = d_gas;

  m_geometries.push_back(geometry);

  return traversableHandle;
}


std::vector<char> Application::readData(std::string const& filename)
{
  std::ifstream inputData(filename, std::ios::binary);

  if (inputData.fail())
  {
    std::cerr << "ERROR: readData() Failed to open file " << filename << '\n';
    return std::vector<char>();
  }

  // Copy the input buffer to a char vector.
  std::vector<char> data(std::istreambuf_iterator<char>(inputData), {});

  if (inputData.fail())
  {
    std::cerr << "ERROR: readData() Failed to read file " << filename << '\n';
    return std::vector<char>();
  }

  return data;
}


// Convert the GUI material parameters to the device side structure and upload them into the m_systemParameter.materialParameters device pointer.
void Application::updateMaterialParameters()
{
  MY_ASSERT((sizeof(MaterialParameter) & 15) == 0); // Verify float4 alignment.

  std::vector<MaterialParameter> materialParameters(m_guiMaterialParameters.size());

  // PERF This could be made faster for GUI interactions on scenes with very many materials when really only copying the changed values.
  for (size_t i = 0; i < m_guiMaterialParameters.size(); ++i)
  {
    MaterialParameterGUI& src = m_guiMaterialParameters[i]; // GUI layout.
    MaterialParameter&    dst = materialParameters[i];      // Device layout.

    dst.indexBSDF     = src.indexBSDF;
    dst.albedo        = src.albedo;
    dst.textureAlbedo = (src.useAlbedoTexture) ? m_textureAlbedo->getTextureObject() : 0;
    dst.textureCutout = (src.useCutoutTexture) ? m_textureCutout->getTextureObject() : 0;
    dst.flags         = (src.thinwalled) ? FLAG_THINWALLED : 0;
    // Calculate the effective absorption coefficient from the GUI parameters. This is one reason why there are two structures.
    // Prevent logf(0.0f) which results in infinity.
    const float x = (0.0f < src.absorptionColor.x) ? -logf(src.absorptionColor.x) : RT_DEFAULT_MAX;
    const float y = (0.0f < src.absorptionColor.y) ? -logf(src.absorptionColor.y) : RT_DEFAULT_MAX;
    const float z = (0.0f < src.absorptionColor.z) ? -logf(src.absorptionColor.z) : RT_DEFAULT_MAX;
    dst.absorption    = make_float3(x, y, z) * src.volumeDistanceScale;
    dst.ior           = src.ior;
  }

  CU_CHECK( cuStreamSynchronize(m_cudaStream) );
  CU_CHECK( cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(m_systemParameter.materialParameters), materialParameters.data(), sizeof(MaterialParameter) * materialParameters.size()) );
}


void Application::initMaterials()
{
  Picture* picture = new Picture;

  unsigned int flags = IMAGE_FLAG_2D;

  const std::string filenameCutout = std::string("./slots_rgba.png");
  if (!picture->load(filenameCutout, flags))
  {
    picture->generateRGBA8(2, 2, 1, flags); // This will not have cutouts though.
  }
  m_textureCutout = new Texture();
  m_textureCutout->create(picture, flags);

  const std::string filenameDiffuse = std::string("./NVIDIA_Logo.jpg");
  if (!picture->load(filenameDiffuse, flags))
  {
    picture->generateRGBA8(2, 2, 1, flags); // 2x2 RGBA8 red-green-blue-yellow failure picture.
  }
  m_textureAlbedo = new Texture();
  m_textureAlbedo->create(picture, flags);

  delete picture;

  // Setup GUI material parameters, one for each of the implemented BSDFs.
  MaterialParameterGUI parameters;

  // The order in this array matches the instance ID in the root IAS!
  // Lambert material for the floor.
  parameters.indexBSDF           = INDEX_BSDF_DIFFUSE_REFLECTION; // Index for the direct callables.
  parameters.albedo              = make_float3(0.5f); // Grey. Modulates the albedo texture.
  parameters.useAlbedoTexture    = true;
  parameters.useCutoutTexture    = false;
  parameters.thinwalled          = false;
  parameters.absorptionColor     = make_float3(1.0f);
  parameters.volumeDistanceScale = 1.0f;
  parameters.ior                 = 1.5f;
  m_guiMaterialParameters.push_back(parameters); // 0

  // Lambert material for the reference box behind the moving object.
  parameters.indexBSDF           = INDEX_BSDF_DIFFUSE_REFLECTION;
  parameters.albedo              = make_float3(0.980392f, 0.729412f, 0.470588f);
  parameters.useAlbedoTexture    = false;
  parameters.useCutoutTexture    = false;
  parameters.thinwalled          = false;
  parameters.absorptionColor     = make_float3(1.0f);
  parameters.volumeDistanceScale = 1.0f;
  parameters.ior                 = 1.5f;
  m_guiMaterialParameters.push_back(parameters); // 1

  // The material for the linear motion object.
  parameters.indexBSDF           = INDEX_BSDF_DIFFUSE_REFLECTION;
  parameters.albedo              = make_float3(0.462745f, 0.72549f, 0.0f); // NVIDIA Green
  parameters.useAlbedoTexture    = false;
  parameters.useCutoutTexture    = false;
  parameters.thinwalled          = false;
  parameters.absorptionColor     = make_float3(0.9f, 0.8f, 0.8f); // Light red.
  parameters.volumeDistanceScale = 1.0f;
  parameters.ior                 = 1.33f; // Water
  m_guiMaterialParameters.push_back(parameters); // 2

  // The material for the SRT motion object.
  parameters.indexBSDF           = INDEX_BSDF_DIFFUSE_REFLECTION;
  parameters.albedo              = make_float3(1.0f); // Modulated by the texture
  parameters.useAlbedoTexture    = true; // Use a texture on it to show the SRT motion better.
  parameters.useCutoutTexture    = false;
  parameters.thinwalled          = false;
  parameters.absorptionColor     = make_float3(0.9f, 0.8f, 0.8f); // Light red.
  parameters.volumeDistanceScale = 1.0f;
  parameters.ior                 = 1.5f; // Glass
  m_guiMaterialParameters.push_back(parameters); // 3

  // Black BSDF for the light. This last material will not be shown inside the GUI!
  parameters.indexBSDF           = INDEX_BSDF_SPECULAR_REFLECTION;
  parameters.albedo              = make_float3(0.0f);
  parameters.useAlbedoTexture    = false;
  parameters.useCutoutTexture    = false;
  parameters.thinwalled          = false;
  parameters.absorptionColor     = make_float3(1.0f);
  parameters.volumeDistanceScale = 1.0f;
  parameters.ior                 = 1.0f;
  m_guiMaterialParameters.push_back(parameters); // 4
}


void Application::initPipeline()
{
  MY_ASSERT((sizeof(SbtRecordHeader)               % OPTIX_SBT_RECORD_ALIGNMENT) == 0);
  MY_ASSERT((sizeof(SbtRecordGeometryInstanceData) % OPTIX_SBT_RECORD_ALIGNMENT) == 0);

  // INSTANCES
  
  OptixInstance instance = {};

  // 1.) Static floor plane.
  OptixTraversableHandle geoPlane = createPlane(1, 1, 1);

  const float trafoPlane[12] =
  {
    8.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 8.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 8.0f, 0.0f
  };

  unsigned int id = static_cast<unsigned int>(m_instances.size());

  memcpy(instance.transform, trafoPlane, sizeof(float) * 12);
  instance.instanceId        = id;
  instance.visibilityMask    = 255;
  instance.sbtOffset         = id * NUM_RAYTYPES; // This controls the SBT instance offset!
  instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
  instance.traversableHandle = geoPlane;
    
  m_instances.push_back(instance); // Plane, no motion.

  // 2.) Static box object.
  OptixTraversableHandle geoBox = createBox();

  const float trafoBox[12] =
  {
    1.0f, 0.0f, 0.0f,  0.0f,
    0.0f, 1.0f, 0.0f,  1.5f, // Hovering over the ground, exactly behind the motion blur box.
    0.0f, 0.0f, 1.0f, -2.5f  // Push it back to not lie in the path of the moving object.
  };

  id = static_cast<unsigned int>(m_instances.size());

  memcpy(instance.transform, trafoBox, sizeof(float) * 12);
  instance.instanceId        = id;
  instance.visibilityMask    = 255;
  instance.sbtOffset         = id * NUM_RAYTYPES;
  instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
  instance.traversableHandle = geoBox;
    
  m_instances.push_back(instance); // Box, no motion.

  // 3.) Linear matrix motion object.
  OptixTraversableHandle geoMatrixMotion = createBox(); // Not instanced to allow exchange against other objects (sphere, torus).

  // Matrix motion blur instance with two keyframes.
  // (The OptiX API Reference shows how to change the transform array size for numKeys > 2.)
  m_matrixMotionTransform = {};

  m_matrixMotionTransform.child = geoMatrixMotion;
  m_matrixMotionTransform.motionOptions.numKeys   = 2;
  m_matrixMotionTransform.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;
  m_matrixMotionTransform.motionOptions.timeBegin = 0.0f;
  m_matrixMotionTransform.motionOptions.timeEnd   = 1.0f;

  getMotionMatrices(m_frameAnimation, &m_matrixMotionTransform.transform[0][0]);

  CU_CHECK( cuMemAlloc(&m_d_matrixMotionTransform, sizeof(OptixMatrixMotionTransform)) ); // Must be aligned to OPTIX_TRANSFORM_BYTE_ALIGNMENT.
  CU_CHECK( cuMemcpyHtoD(m_d_matrixMotionTransform, &m_matrixMotionTransform, sizeof(OptixMatrixMotionTransform)) );

  OPTIX_CHECK( m_api.optixConvertPointerToTraversableHandle(m_context, m_d_matrixMotionTransform, OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM, &m_matrixMotionTransformHandle) );

  // No transform on the instance. 
  // The object to world transformation is done by the optixMatrixMotionTransform.
  const float trafoIdentity[12] =
  {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f 
  };

  id = static_cast<unsigned int>(m_instances.size());

  memcpy(instance.transform, trafoIdentity, sizeof(float) * 12);
  instance.instanceId        = id;
  instance.visibilityMask    = 255;
  instance.sbtOffset         = id * NUM_RAYTYPES;
  instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
  instance.traversableHandle = m_matrixMotionTransformHandle;
    
  m_instances.push_back(instance); // Matrix motion blur object.

  // 4.) SRT motion transform object.
  OptixTraversableHandle geoSRTMotion = createBox(); // Not instanced to allow exchange against other objects (sphere, torus).

  // Motion blur SRT instance with two keyframes.
  // (The OptiX API Reference shows how to change the srtData array size for numKeys > 2.)
  m_srtMotionTransform = {};

  m_srtMotionTransform.child = geoSRTMotion;
  m_srtMotionTransform.motionOptions.numKeys   = 2;
  m_srtMotionTransform.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;
  m_srtMotionTransform.motionOptions.timeBegin = 0.0f;
  m_srtMotionTransform.motionOptions.timeEnd   = 1.0f;

  getMotionSRTs(m_frameAnimation, m_srtMotionTransform.srtData);

  CU_CHECK( cuMemAlloc(&m_d_srtMotionTransform, sizeof(OptixSRTMotionTransform)) ); // Must be aligned to OPTIX_TRANSFORM_BYTE_ALIGNMENT.
  CU_CHECK( cuMemcpyHtoD(m_d_srtMotionTransform, &m_srtMotionTransform, sizeof(OptixSRTMotionTransform)) );

  OPTIX_CHECK( m_api.optixConvertPointerToTraversableHandle(m_context, m_d_srtMotionTransform, OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM, &m_srtMotionTransformHandle) );

  // No transform on the instance. 
  // The object to world transformation is done by the optixSRTMotionTransform.
  //const float trafoIdentity[12] =
  //{
  //  1.0f, 0.0f, 0.0f, 0.0f,
  //  0.0f, 1.0f, 0.0f, 0.0f,
  //  0.0f, 0.0f, 1.0f, 0.0f 
  //};

  id = static_cast<unsigned int>(m_instances.size());

  memcpy(instance.transform, trafoIdentity, sizeof(float) * 12);
  instance.instanceId        = id;
  instance.visibilityMask    = 255;
  instance.sbtOffset         = id * NUM_RAYTYPES;
  instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
  instance.traversableHandle = m_srtMotionTransformHandle;
    
  m_instances.push_back(instance); // SRT motion blur object.

  // Note that the area light is also a static object.
  createLights();

  // Static root IAS
  // The motion blur objects in the scene don't move too much during the time of a single frame of the animation. 
  // Means the AABBs are reasonably tight and there isn't really a need for a motion IAS in that case.

  const size_t sizeInstancesInBytes = sizeof(OptixInstance) * m_instances.size();

  CU_CHECK( cuMemAlloc(&m_d_instancesRoot, sizeInstancesInBytes) );
  CU_CHECK( cuMemcpyHtoD(m_d_instancesRoot, m_instances.data(), sizeInstancesInBytes) );

  m_instanceInputRoot = {};

  m_instanceInputRoot.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  m_instanceInputRoot.instanceArray.instances    = m_d_instancesRoot;
  m_instanceInputRoot.instanceArray.numInstances = static_cast<unsigned int>(m_instances.size());
  // OptiX 7.2.0 doesn't have the aabbs and numAabbs on the OptixBuildInputInstanceArray anymore which simplifies the motion blur implementation.

  m_accelBuildOptionsRoot = {};

  // Root IAS AABBs need to be updated when the SRT motion transform or GAS morphing changes.
  m_accelBuildOptionsRoot.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  m_accelBuildOptionsRoot.operation  = OPTIX_BUILD_OPERATION_BUILD; // The initial build.
  // For reference: Adding these settings would build a motion IAS instead.
  //m_accelBuildOptionsRoot.motionOptions.numKeys   = 2;
  //m_accelBuildOptionsRoot.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;
  //m_accelBuildOptionsRoot.motionOptions.timeBegin = 0.0f;
  //m_accelBuildOptionsRoot.motionOptions.timeEnd   = 1.0f;
  
  m_iasBufferSizesRoot = {};

  OPTIX_CHECK( m_api.optixAccelComputeMemoryUsage(m_context, &m_accelBuildOptionsRoot, &m_instanceInputRoot, 1, &m_iasBufferSizesRoot) );

  CU_CHECK( cuMemAlloc(&m_d_iasRoot, m_iasBufferSizesRoot.outputSizeInBytes ) );

  CU_CHECK( cuMemAlloc(&m_d_tmpRoot, m_iasBufferSizesRoot.tempSizeInBytes) ); // Allocate temporary buffers last to reduce fragmentation.

  OPTIX_CHECK( m_api.optixAccelBuild(m_context, m_cudaStream,
                                     &m_accelBuildOptionsRoot, &m_instanceInputRoot, 1,
                                     m_d_tmpRoot, m_iasBufferSizesRoot.tempSizeInBytes,
                                     m_d_iasRoot, m_iasBufferSizesRoot.outputSizeInBytes,
                                     &m_iasRoot,  nullptr, 0));

  CU_CHECK( cuStreamSynchronize(m_cudaStream) );

  // MODULES

  OptixModuleCompileOptions moduleCompileOptions = {};

  moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT; // No explicit register limit.
#if USE_MAX_OPTIMIZATION
  moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3; // All optimizations, is the default.
  // Keep generated line info for Nsight Compute profiling. (NVCC_OPTIONS use --generate-line-info in CMakeLists.txt)
#if (OPTIX_VERSION >= 70400)
  moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL; 
#else
  moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif
#else // DEBUG
  moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
  moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

  OptixPipelineCompileOptions pipelineCompileOptions = {};

  pipelineCompileOptions.usesMotionBlur        = 1; // Using motion blur
  pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY; // Can't use single level instancing with motion transforms!
  pipelineCompileOptions.numPayloadValues      = 2; // I need two to encode a 64-bit pointer to the per ray payload structure.
  pipelineCompileOptions.numAttributeValues    = 2; // The minimum is two, for the barycentrics.
#if USE_MAX_OPTIMIZATION
  pipelineCompileOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
#else // DEBUG 
  pipelineCompileOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | 
                                                 OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                                 OPTIX_EXCEPTION_FLAG_USER |
                                                 OPTIX_EXCEPTION_FLAG_DEBUG;
#endif
  pipelineCompileOptions.pipelineLaunchParamsVariableName = "sysParameter";

  OptixProgramGroupOptions programGroupOptions = {}; // So far this is just a placeholder today.

  // Each source file results in one OptixModule.
  std::vector<OptixModule> modules(NUM_MODULE_IDENTIFIERS);
 
  // Create all modules:
  for (size_t i = 0; i < m_moduleFilenames.size(); ++i)
  {
    // Since OptiX 7.5.0 the program input can either be *.ptx source code or *.optixir binary code.
    // The module filenames are automatically switched between *.ptx or *.optixir extension based on the definition of USE_OPTIX_IR
    std::vector<char> programData = readData(m_moduleFilenames[i]); 

#if (OPTIX_VERSION >= 70700)
    OPTIX_CHECK( m_api.optixModuleCreate(m_context, &moduleCompileOptions, &pipelineCompileOptions, programData.data(), programData.size(), nullptr, nullptr, &modules[i]) );
#else
    OPTIX_CHECK( m_api.optixModuleCreateFromPTX(m_context, &moduleCompileOptions, &pipelineCompileOptions, programData.data(), programData.size(), nullptr, nullptr, &modules[i]) );
#endif
  }

  // Each program gets its own OptixProgramGroupDesc.
  std::vector<OptixProgramGroupDesc> programGroupDescriptions(NUM_PROGRAM_IDENTIFIERS);
  // Null out all entries inside the program group descriptions. 
  // This is important because the following code will only set the required fields.
  memset(programGroupDescriptions.data(), 0, sizeof(OptixProgramGroupDesc) * programGroupDescriptions.size());

  // Setup all program group descriptions.
  OptixProgramGroupDesc* pgd = &programGroupDescriptions[PROGRAM_ID_RAYGENERATION];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->raygen.module            = modules[MODULE_ID_RAYGENERATION];
  pgd->raygen.entryFunctionName = "__raygen__pathtracer";

  pgd = &programGroupDescriptions[PROGRAM_ID_EXCEPTION];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->exception.module            = modules[MODULE_ID_EXCEPTION];
  pgd->exception.entryFunctionName = "__exception__all";

  // MISS

  pgd = &programGroupDescriptions[PROGRAM_ID_MISS_RADIANCE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->miss.module = modules[MODULE_ID_MISS];
  switch (m_missID)
  {
    case 0: // Black, not a light.
      pgd->miss.entryFunctionName = "__miss__env_null";
      break;
    case 1: // Constant white environment.
    default:
      pgd->miss.entryFunctionName = "__miss__env_constant";
      break;
    case 2: // Spherical HDR environment light.
      pgd->miss.entryFunctionName = "__miss__env_sphere";
      break;
  }

  pgd = &programGroupDescriptions[PROGRAM_ID_MISS_SHADOW];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->miss.module            = nullptr; // Redundant after the memset() above, for code clarity.
  pgd->miss.entryFunctionName = nullptr; // No miss program for shadow rays. 

  // HIT

  pgd = &programGroupDescriptions[PROGRAM_ID_HIT_RADIANCE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleCH            = modules[MODULE_ID_CLOSESTHIT];
  pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance";

  pgd = &programGroupDescriptions[PROGRAM_ID_HIT_SHADOW];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleAH            = modules[MODULE_ID_ANYHIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__shadow";

  pgd = &programGroupDescriptions[PROGRAM_ID_HIT_RADIANCE_CUTOUT];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleCH            = modules[MODULE_ID_CLOSESTHIT];
  pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pgd->hitgroup.moduleAH            = modules[MODULE_ID_ANYHIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__radiance_cutout";

  pgd = &programGroupDescriptions[PROGRAM_ID_HIT_SHADOW_CUTOUT];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleAH            = modules[MODULE_ID_ANYHIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__shadow_cutout";

  // CALLABLES

  pgd = &programGroupDescriptions[PROGRAM_ID_LENS_PINHOLE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC = modules[MODULE_ID_LENS_SHADER];
  pgd->callables.entryFunctionNameDC = "__direct_callable__pinhole";

  pgd = &programGroupDescriptions[PROGRAM_ID_LENS_FISHEYE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC = modules[MODULE_ID_LENS_SHADER];
  pgd->callables.entryFunctionNameDC = "__direct_callable__fisheye";

  pgd = &programGroupDescriptions[PROGRAM_ID_LENS_SPHERE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC = modules[MODULE_ID_LENS_SHADER];
  pgd->callables.entryFunctionNameDC = "__direct_callable__sphere";

  // Two light sampling functions, one for the environment and one for the parallelogram.
  pgd = &programGroupDescriptions[PROGRAM_ID_LIGHT_ENV];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC = modules[MODULE_ID_LIGHT_SAMPLE];
  switch (m_missID)
  {
    case 0: // Black environment. 
      // This is not a light and doesn't appear in the sysParameter.lightDefinitions and therefore is never called.
      // Put a valid direct callable into this SBT record anyway to have the correct number of callables. Use the light_env_constant function.
      // Fall through.
    case 1: // White environment.
    default:
      pgd->callables.entryFunctionNameDC = "__direct_callable__light_env_constant";
      break;
    case 2:
      pgd->callables.entryFunctionNameDC = "__direct_callable__light_env_sphere";
      break;
  }

  pgd = &programGroupDescriptions[PROGRAM_ID_LIGHT_PARALLELOGRAM];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC = modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_parallelogram";

  pgd = &programGroupDescriptions[PROGRAM_ID_BRDF_DIFFUSE_SAMPLE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC = modules[MODULE_ID_DIFFUSE_REFLECTION];
  pgd->callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_diffuse_reflection";

  pgd = &programGroupDescriptions[PROGRAM_ID_BRDF_DIFFUSE_EVAL];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC = modules[MODULE_ID_DIFFUSE_REFLECTION];
  pgd->callables.entryFunctionNameDC = "__direct_callable__eval_bsdf_diffuse_reflection";

  pgd = &programGroupDescriptions[PROGRAM_ID_BRDF_SPECULAR_SAMPLE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC = modules[MODULE_ID_SPECULAR_REFLECTION];
  pgd->callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_specular_reflection";

  pgd = &programGroupDescriptions[PROGRAM_ID_BRDF_SPECULAR_EVAL];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC = modules[MODULE_ID_SPECULAR_REFLECTION];
  pgd->callables.entryFunctionNameDC = "__direct_callable__eval_bsdf_specular_reflection"; // black

  pgd = &programGroupDescriptions[PROGRAM_ID_BSDF_SPECULAR_SAMPLE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC = modules[MODULE_ID_SPECULAR_REFLECTION_TRANSMISSION];
  pgd->callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_specular_reflection_transmission";

  pgd = &programGroupDescriptions[PROGRAM_ID_BSDF_SPECULAR_EVAL];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  // Reuse the same black eval function from the specular BRDF.
  pgd->callables.moduleDC = modules[MODULE_ID_SPECULAR_REFLECTION];
  pgd->callables.entryFunctionNameDC = "__direct_callable__eval_bsdf_specular_reflection"; // black

  // Each OptixProgramGroupDesc results on one OptixProgramGroup.
  std::vector<OptixProgramGroup> programGroups(NUM_PROGRAM_IDENTIFIERS);

  // Construct all program groups at once.
  OPTIX_CHECK( m_api.optixProgramGroupCreate(m_context, programGroupDescriptions.data(), (unsigned int) programGroupDescriptions.size(), &programGroupOptions, nullptr, nullptr, programGroups.data()) );

  OptixPipelineLinkOptions pipelineLinkOptions = {};

  pipelineLinkOptions.maxTraceDepth = 2;

#if (OPTIX_VERSION < 70700)
  // OptixPipelineLinkOptions debugLevel is only present in OptiX SDK versions before 7.7.0.
  #if USE_MAX_OPTIMIZATION
    // Keep generated line info for Nsight Compute profiling. (NVCC_OPTIONS use --generate-line-info in CMakeLists.txt)
    #if (OPTIX_VERSION >= 70400)
      pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    #else
      pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
    #endif
  #else // DEBUG
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
  #endif
#endif // 70700
#if (OPTIX_VERSION == 70000)
  pipelineLinkOptions.overrideUsesMotionBlur = 0; // Does not exist in OptiX 7.1.0.
#endif

  OPTIX_CHECK( m_api.optixPipelineCreate(m_context, &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(), (unsigned int) programGroups.size(), nullptr, nullptr, &m_pipeline) );

  // STACK SIZES

  OptixStackSizes stackSizesPipeline = {};

  for (size_t i = 0; i < programGroups.size(); ++i)
  {
    OptixStackSizes stackSizes;

#if (OPTIX_VERSION >= 70700)
    OPTIX_CHECK( m_api.optixProgramGroupGetStackSize(programGroups[i], &stackSizes, m_pipeline) );
#else
    OPTIX_CHECK( m_api.optixProgramGroupGetStackSize(programGroups[i], &stackSizes) );
#endif

    stackSizesPipeline.cssRG = std::max(stackSizesPipeline.cssRG, stackSizes.cssRG);
    stackSizesPipeline.cssMS = std::max(stackSizesPipeline.cssMS, stackSizes.cssMS);
    stackSizesPipeline.cssCH = std::max(stackSizesPipeline.cssCH, stackSizes.cssCH);
    stackSizesPipeline.cssAH = std::max(stackSizesPipeline.cssAH, stackSizes.cssAH);
    stackSizesPipeline.cssIS = std::max(stackSizesPipeline.cssIS, stackSizes.cssIS);
    stackSizesPipeline.cssCC = std::max(stackSizesPipeline.cssCC, stackSizes.cssCC);
    stackSizesPipeline.dssDC = std::max(stackSizesPipeline.dssDC, stackSizes.dssDC);
  }
  
  // Temporaries
  const unsigned int cssCCTree           = stackSizesPipeline.cssCC; // Should be 0. No continuation callables in this pipeline. // maxCCDepth == 0
  const unsigned int cssCHOrMSPlusCCTree = std::max(stackSizesPipeline.cssCH, stackSizesPipeline.cssMS) + cssCCTree;

  // Arguments
  const unsigned int directCallableStackSizeFromTraversal = stackSizesPipeline.dssDC; // maxDCDepth == 1 // FromTraversal: DC is invoked from IS or AH.      // Possible stack size optimizations.
  const unsigned int directCallableStackSizeFromState     = stackSizesPipeline.dssDC; // maxDCDepth == 1 // FromState:     DC is invoked from RG, MS, or CH. // Possible stack size optimizations.
  const unsigned int continuationStackSize = stackSizesPipeline.cssRG + cssCCTree + cssCHOrMSPlusCCTree * (std::max(1u, pipelineLinkOptions.maxTraceDepth) - 1u) +
                                             std::min(1u, pipelineLinkOptions.maxTraceDepth) * std::max(cssCHOrMSPlusCCTree, stackSizesPipeline.cssAH + stackSizesPipeline.cssIS);
  // "The maxTraversableGraphDepth responds to the maximum number of traversables visited when calling optixTrace. 
  // Every acceleration structure and motion transform count as one level of traversal."
  // Render Graph is at maximum: static IAS_root -> optixSRTMotionTransform -> GAS_motion. 
  const unsigned int maxTraversableGraphDepth = 3; 

  OPTIX_CHECK( m_api.optixPipelineSetStackSize(m_pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState, continuationStackSize, maxTraversableGraphDepth) );

  // Set up Shader Binding Table (SBT)
  // The shader binding table is inherently connected to the scene graph geometry instances in this example.

  // Raygeneration group
  SbtRecordHeader sbtRecordRaygeneration;

  OPTIX_CHECK( m_api.optixSbtRecordPackHeader(programGroups[PROGRAM_ID_RAYGENERATION], &sbtRecordRaygeneration) );

  CU_CHECK( cuMemAlloc(&m_d_sbtRecordRaygeneration, sizeof(SbtRecordHeader)) );
  CU_CHECK( cuMemcpyHtoD(m_d_sbtRecordRaygeneration, &sbtRecordRaygeneration, sizeof(SbtRecordHeader)) );

  // Exception
  SbtRecordHeader sbtRecordException;

  OPTIX_CHECK( m_api.optixSbtRecordPackHeader(programGroups[PROGRAM_ID_EXCEPTION], &sbtRecordException) );

  CU_CHECK( cuMemAlloc(&m_d_sbtRecordException, sizeof(SbtRecordHeader)) );
  CU_CHECK( cuMemcpyHtoD(m_d_sbtRecordException, &sbtRecordException, sizeof(SbtRecordHeader)) );

  // Miss group
  std::vector<SbtRecordHeader> sbtRecordMiss(NUM_RAYTYPES);

  OPTIX_CHECK( m_api.optixSbtRecordPackHeader(programGroups[PROGRAM_ID_MISS_RADIANCE], &sbtRecordMiss[RAYTYPE_RADIANCE]) );
  OPTIX_CHECK( m_api.optixSbtRecordPackHeader(programGroups[PROGRAM_ID_MISS_SHADOW],   &sbtRecordMiss[RAYTYPE_SHADOW]) );

  CU_CHECK( cuMemAlloc(&m_d_sbtRecordMiss, sizeof(SbtRecordHeader) * NUM_RAYTYPES) );
  CU_CHECK( cuMemcpyHtoD(m_d_sbtRecordMiss, sbtRecordMiss.data(), sizeof(SbtRecordHeader) * NUM_RAYTYPES) );

  // Hit groups for radiance and shadow rays per instance.
  
  MY_ASSERT(NUM_RAYTYPES == 2); // The following code only works for two raytypes.

  // Note that the SBT record data field is uninitialized after these!
  OPTIX_CHECK( m_api.optixSbtRecordPackHeader(programGroups[PROGRAM_ID_HIT_RADIANCE],        &m_sbtRecordHitRadiance) );
  OPTIX_CHECK( m_api.optixSbtRecordPackHeader(programGroups[PROGRAM_ID_HIT_SHADOW],          &m_sbtRecordHitShadow) );
  OPTIX_CHECK( m_api.optixSbtRecordPackHeader(programGroups[PROGRAM_ID_HIT_RADIANCE_CUTOUT], &m_sbtRecordHitRadianceCutout) );
  OPTIX_CHECK( m_api.optixSbtRecordPackHeader(programGroups[PROGRAM_ID_HIT_SHADOW_CUTOUT],   &m_sbtRecordHitShadowCutout) );

  // The real content.
  const int numInstances = static_cast<int>(m_instances.size());

  // In this exmple, each instance has its own SBT hit record. 
  // The additional data in the SBT hit record defines the geometry attributes and topology, material and optional light indices.
  m_sbtRecordGeometryInstanceData.resize(NUM_RAYTYPES * numInstances);

  for (int i = 0; i < numInstances; ++i)
  {
    const int idx = i * NUM_RAYTYPES; // idx == radiance ray, idx + 1 == shadow ray

    if (!m_guiMaterialParameters[i].useCutoutTexture)
    {
      // Only update the header to switch the program hit group. The SBT record data field doesn't change. 
      memcpy(m_sbtRecordGeometryInstanceData[idx    ].header, m_sbtRecordHitRadiance.header, OPTIX_SBT_RECORD_HEADER_SIZE);
      memcpy(m_sbtRecordGeometryInstanceData[idx + 1].header, m_sbtRecordHitShadow.header,   OPTIX_SBT_RECORD_HEADER_SIZE);
    }
    else
    {
      memcpy(m_sbtRecordGeometryInstanceData[idx    ].header, m_sbtRecordHitRadianceCutout.header, OPTIX_SBT_RECORD_HEADER_SIZE);
      memcpy(m_sbtRecordGeometryInstanceData[idx + 1].header, m_sbtRecordHitShadowCutout.header,   OPTIX_SBT_RECORD_HEADER_SIZE);
    }

    m_sbtRecordGeometryInstanceData[idx    ].data.indices       = (int3*)             m_geometries[i].indices;
    m_sbtRecordGeometryInstanceData[idx    ].data.attributes    = (VertexAttributes*) m_geometries[i].attributes;
    m_sbtRecordGeometryInstanceData[idx    ].data.materialIndex = i;
    m_sbtRecordGeometryInstanceData[idx    ].data.lightIndex    = -1;

    m_sbtRecordGeometryInstanceData[idx + 1].data.indices       = (int3*)             m_geometries[i].indices;
    m_sbtRecordGeometryInstanceData[idx + 1].data.attributes    = (VertexAttributes*) m_geometries[i].attributes;
    m_sbtRecordGeometryInstanceData[idx + 1].data.materialIndex = i;
    m_sbtRecordGeometryInstanceData[idx + 1].data.lightIndex    = -1;
  }

  if (m_lightID)
  {
    const int idx = (numInstances - 1) * NUM_RAYTYPES; // HACK The last instance is the parallelogram light.
    const int lightIndex = (m_missID != 0) ? 1 : 0;    // HACK If there is any environment light that is in sysParameter.lightDefinitions[0] and the area light in index [1] then.
    m_sbtRecordGeometryInstanceData[idx    ].data.lightIndex = lightIndex;
    m_sbtRecordGeometryInstanceData[idx + 1].data.lightIndex = lightIndex;
  }

  CU_CHECK( cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&m_d_sbtRecordGeometryInstanceData), sizeof(SbtRecordGeometryInstanceData) * NUM_RAYTYPES * numInstances) );
  CU_CHECK( cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(m_d_sbtRecordGeometryInstanceData), m_sbtRecordGeometryInstanceData.data(), sizeof(SbtRecordGeometryInstanceData) * NUM_RAYTYPES * numInstances) );

  // CALLABLES

  // The callable programs are at the end of the ProgramIdentifier enums (from PROGRAM_ID_LENS_PINHOLE to PROGRAM_ID_BSDF_SPECULAR_EVAL)
  const int numCallables = static_cast<int>(NUM_PROGRAM_IDENTIFIERS) - static_cast<int>(PROGRAM_ID_LENS_PINHOLE);
  std::vector<SbtRecordHeader> sbtRecordCallables(numCallables);

  for (int i = 0; i < numCallables; ++i)
  {
    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(programGroups[static_cast<int>(PROGRAM_ID_LENS_PINHOLE) + i], &sbtRecordCallables[i]) );
  }

  CU_CHECK( cuMemAlloc(&m_d_sbtRecordCallables, sizeof(SbtRecordHeader) * sbtRecordCallables.size()) );
  CU_CHECK( cuMemcpyHtoD(m_d_sbtRecordCallables, sbtRecordCallables.data(), sizeof(SbtRecordHeader) * sbtRecordCallables.size()) );


  // Setup the OptixShaderBindingTable.
  m_sbt.raygenRecord = m_d_sbtRecordRaygeneration;

  m_sbt.exceptionRecord = m_d_sbtRecordException;

  m_sbt.missRecordBase          = m_d_sbtRecordMiss;
  m_sbt.missRecordStrideInBytes = (unsigned int) sizeof(SbtRecordHeader);
  m_sbt.missRecordCount         = NUM_RAYTYPES;

  m_sbt.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(m_d_sbtRecordGeometryInstanceData);
  m_sbt.hitgroupRecordStrideInBytes = (unsigned int) sizeof(SbtRecordGeometryInstanceData);
  m_sbt.hitgroupRecordCount         = NUM_RAYTYPES * numInstances;

  m_sbt.callablesRecordBase          = m_d_sbtRecordCallables;
  m_sbt.callablesRecordStrideInBytes = (unsigned int) sizeof(SbtRecordHeader);
  m_sbt.callablesRecordCount         = (unsigned int) sbtRecordCallables.size();

  // Setup "sysParameter" data.
  m_systemParameter.topObject = m_iasRoot;

  if (m_interop)
  {
    CU_CHECK( cuGraphicsGLRegisterBuffer(&m_cudaGraphicsResource, m_pbo, CU_GRAPHICS_REGISTER_FLAGS_NONE) ); // No flags for read-write access during accumulation.

    size_t size;

    CU_CHECK( cuGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream) );
    CU_CHECK( cuGraphicsResourceGetMappedPointer(reinterpret_cast<CUdeviceptr*>(&m_systemParameter.outputBuffer), &size, m_cudaGraphicsResource) ); // Redundant. Must be done on each map anyway.
    CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) );
    
    MY_ASSERT(m_width * m_height * sizeof(float4) <= size);
  }
  else
  {
    CU_CHECK( cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&m_systemParameter.outputBuffer), sizeof(float4) * m_width * m_height) ); // No initialization, that is done at iterationIndex == 0.
  }
  
  MY_ASSERT((sizeof(LightDefinition) & 15) == 0); // Check alignment to float4
  CU_CHECK( cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&m_systemParameter.lightDefinitions), sizeof(LightDefinition) * m_lightDefinitions.size()) );
  CU_CHECK( cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(m_systemParameter.lightDefinitions), m_lightDefinitions.data(), sizeof(LightDefinition) * m_lightDefinitions.size()) );
  
  CU_CHECK( cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&m_systemParameter.materialParameters), sizeof(MaterialParameter) * m_guiMaterialParameters.size()) );
  updateMaterialParameters();

  // Setup the environment texture values. These are all defaults when there is no environment texture filename given.
  m_systemParameter.envTexture  = m_textureEnvironment->getTextureObject();
  m_systemParameter.envCDF_U    = (float*) m_textureEnvironment->getCDF_U();
  m_systemParameter.envCDF_V    = (float*) m_textureEnvironment->getCDF_V();
  m_systemParameter.envWidth    = m_textureEnvironment->getWidth();
  m_systemParameter.envHeight   = m_textureEnvironment->getHeight();
  m_systemParameter.envIntegral = m_textureEnvironment->getIntegral();

  m_systemParameter.pathLengths    = make_int2(2, 10);  // Default max path length set to 10 for the nested materials and to match optixIntro_07 for performance comparison.
  m_systemParameter.sceneEpsilon   = m_sceneEpsilonFactor * SCENE_EPSILON_SCALE;
  m_systemParameter.numLights      = static_cast<unsigned int>(m_lightDefinitions.size());
  m_systemParameter.iterationIndex = 0;
  m_systemParameter.cameraType     = LENS_SHADER_PINHOLE;

  m_pinholeCamera.getFrustum(m_cameraPosition,
                             m_systemParameter.cameraU,
                             m_systemParameter.cameraV,
                             m_systemParameter.cameraW);
  updateCameraAnimation();

  CU_CHECK( cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&m_d_systemParameter), sizeof(SystemParameter)) );
  CU_CHECK( cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(m_d_systemParameter), &m_systemParameter, sizeof(SystemParameter)) );

  // After all required optixSbtRecordPackHeader, optixProgramGroupGetStackSize, and optixPipelineCreate
  // calls have been done, the OptixProgramGroup and OptixModule objects (opaque pointer to struct types)
  // can be destroyed.
  for (auto pg: programGroups)
  {
    OPTIX_CHECK( m_api.optixProgramGroupDestroy(pg) );
  }

  for (auto m: modules)
  {
    OPTIX_CHECK( m_api.optixModuleDestroy(m) );
  }
}

// In contrast to the original optixIntro_07, this example supports dynamic switching of the cutout opacity material parameter.
void Application::updateShaderBindingTable(const int instance)
{
  if (instance < m_instances.size()) // Make sure to only touch existing SBT records.
  {
    const int idx = instance * NUM_RAYTYPES; // idx == radiance ray, idx + 1 == shadow ray

    if (!m_guiMaterialParameters[instance].useCutoutTexture)
    {
      // Only update the header to switch the program hit group. The SBT record data field doesn't change. 
      memcpy(m_sbtRecordGeometryInstanceData[idx    ].header, m_sbtRecordHitRadiance.header, OPTIX_SBT_RECORD_HEADER_SIZE);
      memcpy(m_sbtRecordGeometryInstanceData[idx + 1].header, m_sbtRecordHitShadow.header,   OPTIX_SBT_RECORD_HEADER_SIZE);
    }
    else
    {
      memcpy(m_sbtRecordGeometryInstanceData[idx    ].header, m_sbtRecordHitRadianceCutout.header, OPTIX_SBT_RECORD_HEADER_SIZE);
      memcpy(m_sbtRecordGeometryInstanceData[idx + 1].header, m_sbtRecordHitShadowCutout.header,   OPTIX_SBT_RECORD_HEADER_SIZE);
    }

    // Make sure the SBT isn't changed while the renderer is active.
    CU_CHECK( cuStreamSynchronize(m_cudaStream) ); 
    // Only copy the two SBT entries which changed.
    CU_CHECK( cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(&m_d_sbtRecordGeometryInstanceData[idx]), &m_sbtRecordGeometryInstanceData[idx], sizeof(SbtRecordGeometryInstanceData) * NUM_RAYTYPES) );
  }
}


void Application::initRenderer()
{
  m_timer.restart();

  const double timeRenderer = m_timer.getTime();

  initMaterials();
  const double timeMaterials = m_timer.getTime();

  initPipeline();
  const double timePipeline = m_timer.getTime();

  std::cout << "initRenderer(): " << timePipeline - timeRenderer << " seconds overall\n";
  std::cout << "{\n";
  std::cout << "  materials  = " << timeMaterials - timeRenderer << " seconds\n";
  std::cout << "  pipeline   = " << timePipeline - timeMaterials << " seconds\n";
  std::cout << "}\n";
}


void Application::createLights()
{
  LightDefinition light;

  // Unused in environment lights. 
  light.position = make_float3(0.0f, 0.0f, 0.0f);
  light.vecU     = make_float3(1.0f, 0.0f, 0.0f);
  light.vecV     = make_float3(0.0f, 1.0f, 0.0f);
  light.normal   = make_float3(0.0f, 0.0f, 1.0f);
  light.area     = 1.0f;
  light.emission = make_float3(1.0f, 1.0f, 1.0f);
  
  m_textureEnvironment = new Texture(); // Allocate an empty environment texture to be able to initialize the sysParameters unconditionally.

  // The environment light is expected in sysParameter.lightDefinitions[0], but since there is only one, 
  // the sysParameter struct contains the data for the spherical HDR environment light when enabled.
  // All other lights are indexed by their position inside the array.
  switch (m_missID)
  {
  case 0: // No environment light at all. Faster than a zero emission constant environment!
  default:
    break;

  case 1: // Constant environment light.
    light.type = LIGHT_ENVIRONMENT;
    light.area = 4.0f * M_PIf; // Unused.

    m_lightDefinitions.push_back(light);
    break;

  case 2: // HDR Environment mapping with loaded texture.
    {
      Picture* picture = new Picture; // Separating image file handling from CUDA texture handling.

      const unsigned int flags = IMAGE_FLAG_2D | IMAGE_FLAG_ENV;
      if (!picture->load(m_environmentFilename, flags))
      {
        picture->generateEnvironment(8, 8); // Generate a white 8x8 RGBA32F dummy environment picture.
      }
      m_textureEnvironment->create(picture, flags);

      delete picture;
    }

    light.type = LIGHT_ENVIRONMENT;
    light.area = 4.0f * M_PIf; // Unused.

    m_lightDefinitions.push_back(light);
    break;
  }

  if (m_lightID)  // Add a square area light over the scene objects.
  {
    light.type     = LIGHT_PARALLELOGRAM;             // A geometric area light with diffuse emission distribution function.
    light.position = make_float3(-2.0f, 4.0f, -2.0f); // Corner position.
    light.vecU     = make_float3(4.0f, 0.0f, 0.0f);   // To the right.
    light.vecV     = make_float3(0.0f, 0.0f, 4.0f);   // To the front. 
    float3 n       = cross(light.vecU, light.vecV);   // Length of the cross product is the area.
    light.area     = length(n);                       // Calculate the world space area of that rectangle, unit is [m^2]
    light.normal   = n / light.area;                  // Normalized normal
    light.emission = make_float3(10.0f);              // Radiant exitance in Watt/m^2.

    m_lightDefinitions.push_back(light);
    
    OptixTraversableHandle geoLight = createParallelogram(light.position, light.vecU, light.vecV, light.normal);

    OptixInstance instance = {};

    // The geometric light is stored in world coordinates for now.
    const float trafoLight[12] =
    {
      1.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f
    };

    const unsigned int id = static_cast<unsigned int>(m_instances.size());

    memcpy(instance.transform, trafoLight, sizeof(float) * 12);
    instance.instanceId        = id;
    instance.visibilityMask    = 255;
    instance.sbtOffset         = id * NUM_RAYTYPES;
    instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
    instance.traversableHandle = geoLight;

    m_instances.push_back(instance); // Parallelogram light.
  }
}


void Application::getMotionMatrices(const unsigned int frame, float* transform)
{
  // Calculate the linear matrix motion transform data for the animation interval [frame, frame + 1].
  float time = frame * m_timePerFrame;
  // Translation, object moves above the ground along the x-axis in a straight line.
  float3 t   = make_float3(time * m_velocity, 1.5f, 0.0f);
  float* tr  = &transform[0];

  tr[ 0] = 1.0f; tr[ 1] = 0.0f; tr[ 2] = 0.0f; tr[ 3] = t.x;
  tr[ 4] = 0.0f; tr[ 5] = 1.0f; tr[ 6] = 0.0f; tr[ 7] = t.y;
  tr[ 8] = 0.0f; tr[ 9] = 0.0f; tr[10] = 1.0f; tr[11] = t.z;

  time += m_timePerFrame;
  t     = make_float3(time * m_velocity, 1.5f, 0.0f);
  tr    = &transform[12];

  tr[ 0] = 1.0f; tr[ 1] = 0.0f; tr[ 2] = 0.0f; tr[ 3] = t.x;
  tr[ 4] = 0.0f; tr[ 5] = 1.0f; tr[ 6] = 0.0f; tr[ 7] = t.y;
  tr[ 8] = 0.0f; tr[ 9] = 0.0f; tr[10] = 1.0f; tr[11] = t.z;
}


void Application::getMotionSRTs(const unsigned int frame, OptixSRTData* srt)
{
  // Calculate the SRT motion transform data for the animation interval [frame, frame + 1].
  for (unsigned int i = 0; i < 2; ++i)
  {
    const float time = (frame + i) * m_timePerFrame;

    // Rotation, object rotates around the z-axis. Rolling to the right means negative angle.
    const float angle = time * m_angularVelocity;
    dp::math::Quatf r(dp::math::Vec3f(0.0f, 0.0f, 1.0f), -angle);

    // Translation, object moves above the ground along the x-axis in a straight line.
    const float3 t = make_float3(time * m_velocity, 1.5f, 2.5f); // In front of the matrix motion object.
  
    // Pivot is object space origin: (pvx, pvy, pvz) = 0.0f
    // There is no scaling:          (sx, sy, sz)    = 1.0f
    // There is no shearing:         (a, b, c)       = 0.0f
    //        sx,   a,    b,    pvx,  sy,   c,    pvy,  sz,   pvz,  qx,   qy,   qz,   qw,   tx,  ty,  tz
    srt[i] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, r[0], r[1], r[2], r[3], t.x, t.y, t.z };
  }
}


void Application::updateAnimation()
{
  // Get the matrix motion transform for the current frame.
  getMotionMatrices(m_frameAnimation, &m_matrixMotionTransform.transform[0][0]);
  CU_CHECK( cuMemcpyHtoD(m_d_matrixMotionTransform, &m_matrixMotionTransform, sizeof(OptixMatrixMotionTransform)) );

  // Get the SRT motion transform for the current frame.
  getMotionSRTs(m_frameAnimation, m_srtMotionTransform.srtData);
  CU_CHECK( cuMemcpyHtoD(m_d_srtMotionTransform, &m_srtMotionTransform, sizeof(OptixSRTMotionTransform)) );

  // Only update the root IAS. Buffer sizes and handles all remain identical after the initial build.
  m_accelBuildOptionsRoot.operation = OPTIX_BUILD_OPERATION_UPDATE; 
  
  // None of the handles change! Trigger the required update build of the root IAS.
  OPTIX_CHECK( m_api.optixAccelBuild(m_context, m_cudaStream,
                                     &m_accelBuildOptionsRoot, &m_instanceInputRoot, 1,
                                     m_d_tmpRoot, m_iasBufferSizesRoot.tempSizeInBytes,
                                     m_d_iasRoot, m_iasBufferSizesRoot.outputSizeInBytes,
                                     &m_iasRoot, nullptr, 0));

  CU_CHECK( cuStreamSynchronize(m_cudaStream) ); // DEBUG This shouldn't be required if all data is kept allocated.
}


void Application::updateCameraAnimation()
{
  m_systemParameter.cameraPosition0 = m_cameraPosition;
  m_systemParameter.cameraPosition1 = m_cameraPosition;

  if (m_hasCameraMotion)
  {
    const float time = m_frameAnimation * m_timePerFrame;
    // The camera position moves with the same velocity and direction as the two motion blurred objects,
    // which lets either the static objects or the matrix motion object look sharp intentionally.
    m_systemParameter.cameraPosition0.x += m_velocity *  time;
    m_systemParameter.cameraPosition1.x += m_velocity * (time + m_timePerFrame);
  }
}
