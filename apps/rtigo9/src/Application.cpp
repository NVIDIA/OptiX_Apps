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

#include "inc/Application.h"
#include "inc/LoaderIES.h"
#include "inc/Parser.h"
#include "inc/Raytracer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stack>
#include <memory>

#include <dp/math/Matmnt.h>

#include "inc/CheckMacros.h"

#include "inc/MyAssert.h"

Application::Application(GLFWwindow* window, const Options& options)
: m_window(window)
, m_isValid(false)
, m_guiState(GUI_STATE_NONE)
, m_isVisibleGUI(true)
, m_width(512)
, m_height(512)
, m_mode(0)
, m_maskDevices(0x00FFFFFF) // A maximum of 24 devices is supported by default. Limited by the UUID arrays 
, m_sizeArena(64) // Default to 64 MiB Arenas when nothing is specified inside the system description.
, m_interop(0)
, m_peerToPeer(P2P_TEX | P2P_GAS) // Enable material texture and GAS sharing via NVLINK only by default.
, m_present(false)
, m_presentNext(true)
, m_presentAtSecond(1.0)
, m_previousComplete(false)
, m_typeLens(TYPE_LENS_PINHOLE)
, m_samplesSqrt(1)
, m_epsilonFactor(500.0f)
, m_clockFactor(1000.0f)
, m_useDirectLighting(true)
, m_mouseSpeedRatio(10.0f)
, m_idGroup(0)
, m_idInstance(0)
, m_idGeometry(0)
{
  try
  {
    m_timer.restart();

    // Initialize the top-level keywords of the scene description for faster search.
    
    // Camera parameters 
    // These can be set in either the system or scene description. Latter takes precedence.
    m_mapKeywordScene["lensShader"] = KS_LENS_SHADER;
    m_mapKeywordScene["center"]     = KS_CENTER; // Center of interest.
    m_mapKeywordScene["camera"]     = KS_CAMERA; // Camere location, polar orientation, field of view.
    // Tonemapper parameters
    // These can be set in either the system or scene description. Latter takes precedence.
    m_mapKeywordScene["gamma"]          = KS_GAMMA;
    m_mapKeywordScene["colorBalance"]   = KS_COLOR_BALANCE;
    m_mapKeywordScene["whitePoint"]     = KS_WHITE_POINT;
    m_mapKeywordScene["burnHighlights"] = KS_BURN_HIGHLIGHTS;
    m_mapKeywordScene["crushBlacks"]    = KS_CRUSH_BLACKS;
    m_mapKeywordScene["saturation"]     = KS_SATURATION;
    m_mapKeywordScene["brightness"]     = KS_BRIGHTNESS;
    // Material parameters
    m_mapKeywordScene["albedo"]          = KS_ALBEDO;
    m_mapKeywordScene["albedoTexture"]   = KS_ALBEDO_TEXTURE;
    m_mapKeywordScene["cutoutTexture"]   = KS_CUTOUT_TEXTURE;
    m_mapKeywordScene["roughness"]       = KS_ROUGHNESS;
    m_mapKeywordScene["absorption"]      = KS_ABSORPTION;
    m_mapKeywordScene["absorptionScale"] = KS_ABSORPTION_SCALE;
    m_mapKeywordScene["ior"]             = KS_IOR;
    m_mapKeywordScene["thinwalled"]      = KS_THINWALLED;
    // Emission parameters
    m_mapKeywordScene["emission"]           = KS_EMISSION;
    m_mapKeywordScene["emissionMultiplier"] = KS_EMISSION_MULTIPLIER;
    m_mapKeywordScene["emissionProfile"]    = KS_EMISSION_PROFILE;
    m_mapKeywordScene["emissionTexture"]    = KS_EMISSION_TEXTURE;
    m_mapKeywordScene["spotAngle"]          = KS_SPOT_ANGLE;
    m_mapKeywordScene["spotExponent"]       = KS_SPOT_EXPONENT;
    // Transformations
    m_mapKeywordScene["identity"]  = KS_IDENTITY;
    m_mapKeywordScene["push"]      = KS_PUSH;
    m_mapKeywordScene["pop"]       = KS_POP;
    m_mapKeywordScene["rotate"]    = KS_ROTATE;
    m_mapKeywordScene["scale"]     = KS_SCALE;
    m_mapKeywordScene["translate"] = KS_TRANSLATE;
    // BXDFs
    m_mapKeywordScene["bxdf"]           = KS_BXDF;
    m_mapKeywordScene["brdf_diffuse"]   = KS_BRDF_DIFFUSE;
    m_mapKeywordScene["brdf_specular"]  = KS_BRDF_SPECULAR;
    m_mapKeywordScene["bsdf_specular"]  = KS_BSDF_SPECULAR;
    m_mapKeywordScene["brdf_ggx_smith"] = KS_BRDF_GGX_SMITH;
    m_mapKeywordScene["bsdf_ggx_smith"] = KS_BSDF_GGX_SMITH;
    // EDFs
    m_mapKeywordScene["edf"]         = KS_EDF;
    m_mapKeywordScene["edf_diffuse"] = KS_EDF_DIFFUSE;
    m_mapKeywordScene["edf_spot"]    = KS_EDF_SPOT;
    m_mapKeywordScene["edf_ies"]     = KS_EDF_IES;
    // Models
    m_mapKeywordScene["plane"]  = KS_PLANE;
    m_mapKeywordScene["box"]    = KS_BOX;
    m_mapKeywordScene["sphere"] = KS_SPHERE;
    m_mapKeywordScene["torus"]  = KS_TORUS;
    m_mapKeywordScene["assimp"] = KS_ASSIMP;
    // Predefined light types
    m_mapKeywordScene["env"]  = KS_LIGHT_ENV;
    m_mapKeywordScene["rect"] = KS_LIGHT_RECT;
    // Scene elements
    m_mapKeywordScene["material"] = KS_MATERIAL;
    m_mapKeywordScene["light"]    = KS_LIGHT;
    m_mapKeywordScene["model"]    = KS_MODEL;

    // ===== IMGUI INTERFACE

    // Setup ImGui binding.
    ImGui::CreateContext();
    ImGui_ImplGlfwGL3_Init(window, true);

    // This initializes the GLFW part including the font texture.
    ImGui_ImplGlfwGL3_NewFrame();
    ImGui::EndFrame();

#if 0
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
    style.Colors[ImGuiCol_TitleBgCollapsed]      = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
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

    // ===== SYSTEM DESCRIPTION

    m_width  = std::max(1, options.getWidth());
    m_height = std::max(1, options.getHeight());
    m_mode   = std::max(0, options.getMode());
    m_optimize = options.getOptimize();

    // Initialize the system options to minimum defaults to work, but require useful settings inside the system options file.
    // The minimum path length values will generate useful direct lighting results, but transmissions will be mostly black.
    m_resolution  = make_int2(1, 1);
    m_tileSize    = make_int2(8, 8);
    m_pathLengths = make_int2(0, 2);

    m_prefixScreenshot = std::string("./img"); // Default to current working directory and prefix "img".

    // Tonmapper neutral defaults. The system description overrides these.
    m_tonemapperGUI.gamma           = 1.0f;
    m_tonemapperGUI.whitePoint      = 1.0f;
    m_tonemapperGUI.colorBalance[0] = 1.0f;
    m_tonemapperGUI.colorBalance[1] = 1.0f;
    m_tonemapperGUI.colorBalance[2] = 1.0f;
    m_tonemapperGUI.burnHighlights  = 1.0f;
    m_tonemapperGUI.crushBlacks     = 0.0f;
    m_tonemapperGUI.saturation      = 1.0f; 
    m_tonemapperGUI.brightness      = 1.0f;

    // System wide parameters are loaded from this file to keep the number of command line options small.
    const std::string filenameSystem = options.getSystem();
    if (!loadSystemDescription(filenameSystem))
    {
      std::cerr << "ERROR: Application() failed to load system description file " << filenameSystem << '\n';
      MY_ASSERT(!"Failed to load system description");
      return; // m_isValid == false.
    }

    std::cout << "Arena size = " << m_sizeArena << " MiB\n"; // DEBUG

    const double timeSystem = m_timer.getTime(); // Contains constructor and GUI initialization.

    // ===== SCENE DESCRIPTION

    // Host side scene graph information.
    m_scene = std::make_shared<sg::Group>(m_idGroup++); // Create the scene's root group first.
    
    // Load the scene description file and generate the host side materials, lights, and scene hierarchy with geometry.
    const std::string filenameScene = options.getScene();
    if (!loadSceneDescription(filenameScene))
    {
      std::cerr << "ERROR: Application() failed to load scene description file " << filenameScene << '\n';
      MY_ASSERT(!"Failed to load scene description");
      return;
    } 

    // Once the scene has been loaded into the host scene graph
    // traverse it once and generate light definitions for the meshes with emissive materials.
    createMeshLights();

    //createPictures(); // FIXME Implement deferred texture loading of only the referenced textures.
    createCameras();
    
    MY_ASSERT(m_idGeometry == m_geometries.size());

    const double timeScene = m_timer.getTime();

    // ===== RASTERIZER

    m_camera.setResolution(m_resolution.x, m_resolution.y);
    m_camera.setSpeedRatio(m_mouseSpeedRatio);

    // Initialize the OpenGL rasterizer.
    m_rasterizer = std::make_unique<Rasterizer>(m_width, m_height, m_interop);

    // Must set the resolution explicitly to be able to calculate 
    // the proper vertex attributes for display and the PBO size in case of interop.
    m_rasterizer->setResolution(m_resolution.x, m_resolution.y); 
    m_rasterizer->setTonemapper(m_tonemapperGUI);

    const unsigned int tex = m_rasterizer->getTextureObject();
    const unsigned int pbo = m_rasterizer->getPixelBufferObject();

    const double timeRasterizer = m_timer.getTime();

    // ===== RAYTRACER

    const TypeLight typeEnv = (!m_lightsGUI.empty()) ? m_lightsGUI[0].typeLight : NUM_LIGHT_TYPES; // NUM_LIGHT_TYPES means not an environment light either.

    m_raytracer = std::make_unique<Raytracer>(m_maskDevices, typeEnv, m_interop, tex, pbo, m_sizeArena, m_peerToPeer);

    // If the raytracer could not be initialized correctly, return and leave Application invalid.
    if (!m_raytracer->m_isValid)
    {
      std::cerr << "ERROR: Application() Could not initialize Raytracer\n";
      return; // Exit application.
    }

    // Determine which device is the one running the OpenGL implementation.
    // The first OpenGL-CUDA device match wins.
    int deviceMatch = -1;

#if 1
    // UUID works under Windows and Linux.
    const int numDevicesOGL = m_rasterizer->getNumDevices();

    for (int i = 0; i < numDevicesOGL && deviceMatch == -1; ++i)
    {
      deviceMatch = m_raytracer->matchUUID(reinterpret_cast<const char*>(m_rasterizer->getUUID(i)));
    }
#else
    // LUID only works under Windows because it requires the EXT_external_objects_win32 extension.
    // DEBUG With multicast enabled, both devices have the same LUID and the OpenGL node mask is the OR of the individual device node masks.
    // Means the result of the deviceMatch here is depending on the CUDA device order. 
    // Seems like multicast needs to handle CUDA - OpenGL interop differently.
    // With multicast enabled, uploading the PBO with glTexImage2D halves the framerate when presenting each image in both the single-GPU and multi-GPU P2P strategy.
    // Means there is an expensive PCI-E copy going on in that case.
    const unsigned char* luid = m_rasterizer->getLUID();
    const int nodeMask        = m_rasterizer->getNodeMask(); 

    // The cuDeviceGetLuid() takes char* and unsigned int though.
    deviceMatch = m_raytracer->matchLUID(reinterpret_cast<const char*>(luid), nodeMask);
#endif

    if (deviceMatch == -1)
    {
      if (m_interop == INTEROP_MODE_TEX)
      {
        std::cerr << "ERROR: Application() OpenGL texture image interop without OpenGL device in active devices will not display the image!\n";
        return; // Exit application.
      }
      if (m_interop == INTEROP_MODE_PBO)
      {
        std::cerr << "WARNING: Application() OpenGL pixel buffer interop without OpenGL device in active devices will result in reduced performance!\n";
      }
    }

    m_state.resolution    = m_resolution;
    m_state.tileSize      = m_tileSize;
    m_state.pathLengths   = m_pathLengths;
    m_state.samplesSqrt   = m_samplesSqrt;
    m_state.typeLens      = m_typeLens;
    m_state.epsilonFactor = m_epsilonFactor;
    m_state.clockFactor   = m_clockFactor;
    m_state.directLighting = (m_useDirectLighting) ? 1 : 0;

    // Sync the state with the default GUI data.
    m_raytracer->initState(m_state);

    // Device side scene information.
    m_raytracer->initTextures(m_mapPictures);
    m_raytracer->initCameras(m_cameras);                  // Currently there is only one but this supports arbitrary many which could be used to select viewpoints or do animation (and camera motion blur) in the future.
    m_raytracer->initMaterials(m_materialsGUI);           // This will handle the per material textures as well.
    m_raytracer->initScene(m_scene, m_idGeometry);        // m_idGeometry is the number of geometries in the scene.
    m_raytracer->initLights(m_lightsGUI, m_materialsGUI); // With arbitrary mesh lights, the geometry attributes and indices can only be filled after initScene().
    
    const double timeRaytracer = m_timer.getTime();

    // Print out hiow long the initialization of each module took.
    std::cout << "Application() " << timeRaytracer << " seconds overall\n";
    std::cout << "{\n";
    std::cout << "  System     = " << timeSystem                      << " seconds\n";
    std::cout << "  Scene      = " << timeScene      - timeSystem     << " seconds\n";
    std::cout << "  Rasterizer = " << timeRasterizer - timeScene      << " seconds\n";
    std::cout << "  Raytracer  = " << timeRaytracer  - timeRasterizer << " seconds\n";
    std::cout << "}\n";

    restartRendering(); // Trigger a new rendering.

    m_isValid = true;
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
  }
}

Application::~Application()
{
  for (std::map<std::string, Picture*>::const_iterator it =  m_mapPictures.begin(); it != m_mapPictures.end(); ++it)
  {
    delete it->second;
  }

  ImGui_ImplGlfwGL3_Shutdown();
  ImGui::DestroyContext();
}

bool Application::isValid() const
{
  return m_isValid;
}

void Application::reshape(const int w, const int h)
{
   // Do not allow zero size client windows! All other reshape() routines depend on this and do not check this again.
  if ( (m_width != w || m_height != h) && w != 0 && h != 0)
  {
    m_width  = w;
    m_height = h;
    
    m_rasterizer->reshape(m_width, m_height);
  }
}

void Application::restartRendering()
{
  guiRenderingIndicator(true);

  m_presentNext      = true;
  m_presentAtSecond  = 1.0;
  
  m_previousComplete = false;
  
  m_timer.restart();
}

bool Application::render()
{
  bool finish = false;
  bool flush  = false;

  try
  {
    CameraDefinition camera;

    const bool cameraChanged = m_camera.getFrustum(camera.P, camera.U, camera.V, camera.W);
    if (cameraChanged)
    {
      m_cameras[0] = camera;
      m_raytracer->updateCamera(0, camera);

      restartRendering();
    }

    const unsigned int iterationIndex = m_raytracer->render();
    
    // When the renderer has completed all iterations, change the GUI title bar to green.
    const bool complete = ((unsigned int)(m_samplesSqrt * m_samplesSqrt) <= iterationIndex);

    if (complete)
    {
      guiRenderingIndicator(false); // Not rendering anymore.

      flush = !m_previousComplete && complete; // Completion status changed to true.
    }
    
    m_previousComplete = complete;
    
    // When benchmark is enabled, exit the application when the requested samples per pixel have been rendered.
    // Actually this render() function is not called when m_mode == 1 but keep the finish here to exit on exceptions.
    finish = ((m_mode == 1) && complete);
    
    // Only update the texture when a restart happened, one second passed to reduce required bandwidth, or the rendering is newly complete.
    if (m_presentNext || flush)
    {
      m_raytracer->updateDisplayTexture(); // This directly updates the display HDR texture for all rendering strategies.

      m_presentNext = m_present;
    }

    double seconds = m_timer.getTime();
#if 1
    // When in interactive mode, show the all rendered frames during the first half second to get some initial refinement.
    if (m_mode == 0 && seconds < 0.5)
    {
      m_presentAtSecond = 1.0;
      m_presentNext     = true;
    }
#endif

    if (m_presentAtSecond < seconds || flush || finish) // Print performance every second or when the rendering is complete or the benchmark finished.
    {
      m_presentAtSecond = ceil(seconds);
      m_presentNext     = true; // Present at least every second.
      
      if (flush || finish) // Only print the performance when the samples per pixels are reached.
      {
        const double fps = double(iterationIndex) / seconds;

        std::ostringstream stream; 
        stream.precision(3); // Precision is # digits in fraction part.
        stream << std::fixed << iterationIndex << " / " << seconds << " = " << fps << " fps";
        std::cout << stream.str() << '\n';

#if 0   // Automated benchmark in interactive mode. Change m_isVisibleGUI default to false!
        std::ostringstream filename;
        filename << "result_interactive_" << m_interop << "_" << m_tileSize.x << "_" << m_tileSize.y << ".log";
        const bool success = saveString(filename.str(), stream.str());
        if (success)
        {
          std::cout << filename.str() << '\n'; // Print out the filename to indicate success.
        }
        finish = true; // Exit application after interactive rendering finished.
#endif
      }
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    finish = true;
  }
  return finish;
}

void Application::benchmark()
{
  try
  {
    CameraDefinition camera;

    const bool cameraChanged = m_camera.getFrustum(camera.P, camera.U, camera.V, camera.W);
    if (cameraChanged)
    {
      m_cameras[0] = camera;
      m_raytracer->updateCamera(0, camera);

      // restartRendering();
    }

    const unsigned int spp = (unsigned int)(m_samplesSqrt * m_samplesSqrt);
    unsigned int iterationIndex = 0; 

    m_timer.restart();

    while (iterationIndex < spp)
    {
      iterationIndex = m_raytracer->render(m_mode);
    }
    
    m_raytracer->synchronize(); // Wait until any asynchronous operations have finished.

    const double seconds = m_timer.getTime();
    const double fps = double(iterationIndex) / seconds;

    std::ostringstream stream;
    stream.precision(3); // Precision is # digits in fraction part.
    stream << std::fixed << iterationIndex << " / " << seconds << " = " << fps << " fps";
    std::cout << stream.str() << '\n';

#if 0 // Automated benchmark in batch mode.
    std::ostringstream filename;
    filename << "result_batch_" << m_interop << "_" << m_tileSize.x << "_" << m_tileSize.y << ".log";
    const bool success = saveString(filename.str(), stream.str());
    if (success)
    {
      std::cout << filename.str() << '\n'; // Print out the filename to indicate success.
    }
#endif

    screenshot(true);
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
  }
}

void Application::display()
{
  m_rasterizer->display();
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

void Application::createCameras()
{
  CameraDefinition camera;

  m_camera.getFrustum(camera.P, camera.U, camera.V, camera.W, true);

  m_cameras.push_back(camera);
}

void Application::guiEventHandler()
{
  const ImGuiIO& io = ImGui::GetIO();

  if (ImGui::IsKeyPressed(' ', false)) // Key Space: Toggle the GUI window display.
  {
    m_isVisibleGUI = !m_isVisibleGUI;
  }
  if (ImGui::IsKeyPressed('S', false)) // Key S: Save the current system options to a file "system_rtigo9_<year><month><day>_<hour><minute><second>_<millisecond>.txt"
  {
    MY_VERIFY( saveSystemDescription() );
  }
  if (ImGui::IsKeyPressed('P', false)) // Key P: Save the current output buffer with tonemapping into a *.png file.
  {
    MY_VERIFY( screenshot(true) );
  }
  if (ImGui::IsKeyPressed('H', false)) // Key H: Save the current linear output buffer into a *.hdr file.
  {
    MY_VERIFY( screenshot(false) );
  }

  const ImVec2 mousePosition = ImGui::GetMousePos(); // Mouse coordinate window client rect.
  const int x = int(mousePosition.x);
  const int y = int(mousePosition.y);

  switch (m_guiState)
  {
    case GUI_STATE_NONE:
      if (!io.WantCaptureMouse) // Only allow camera interactions to begin when interacting with the GUI.
      {
        if (ImGui::IsMouseDown(0)) // LMB down event?
        {
          m_camera.setBaseCoordinates(x, y);
          m_guiState = GUI_STATE_ORBIT;
        }
        else if (ImGui::IsMouseDown(1)) // RMB down event?
        {
          m_camera.setBaseCoordinates(x, y);
          m_guiState = GUI_STATE_DOLLY;
        }
        else if (ImGui::IsMouseDown(2)) // MMB down event?
        {
          m_camera.setBaseCoordinates(x, y);
          m_guiState = GUI_STATE_PAN;
        }
        else if (io.MouseWheel != 0.0f) // Mouse wheel zoom.
        {
          m_camera.zoom(io.MouseWheel);
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
        m_camera.orbit(x, y);
      }
      break;

    case GUI_STATE_DOLLY:
      if (ImGui::IsMouseReleased(1)) // RMB released? End of dolly mode.
      {
        m_guiState = GUI_STATE_NONE;
      }
      else
      {
        m_camera.dolly(x, y);
      }
      break;

    case GUI_STATE_PAN:
      if (ImGui::IsMouseReleased(2)) // MMB released? End of pan mode.
      {
        m_guiState = GUI_STATE_NONE;
      }
      else
      {
        m_camera.pan(x, y);
      }
      break;
  }
}

void Application::guiWindow()
{
  if (!m_isVisibleGUI || m_mode == 1) // Use SPACE to toggle the display of the GUI window.
  {
    return;
  }

  bool refresh = false;

  ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);

  ImGuiWindowFlags window_flags = 0;
  if (!ImGui::Begin("rtigo9", nullptr, window_flags)) // No bool flag to omit the close button.
  {
    // Early out if the window is collapsed, as an optimization.
    ImGui::End();
    return;
  }

  ImGui::PushItemWidth(-120); // Right-aligned, keep pixels for the labels.

  if (ImGui::CollapsingHeader("System"))
  {
    if (ImGui::DragFloat("Mouse Ratio", &m_mouseSpeedRatio, 0.1f, 0.1f, 1000.0f, "%.1f"))
    {
      m_camera.setSpeedRatio(m_mouseSpeedRatio);
    }
    if (ImGui::Checkbox("Present", &m_present))
    {
      // No action needed, happens automatically.
    }
    if (ImGui::Checkbox("Direct Lighting", &m_useDirectLighting))
    {
      m_state.directLighting = (m_useDirectLighting) ? 1 : 0;
      m_raytracer->updateState(m_state);
      refresh = true;
    }
    if (ImGui::Combo("Camera", (int*) &m_typeLens, "Pinhole\0Fisheye\0Spherical\0\0"))
    {
      m_state.typeLens = m_typeLens;
      m_raytracer->updateState(m_state);
      refresh = true;
    }
    if (ImGui::InputInt2("Resolution", &m_resolution.x, ImGuiInputTextFlags_EnterReturnsTrue)) // This requires RETURN to apply a new value.
    {
      m_resolution.x = std::max(1, m_resolution.x);
      m_resolution.y = std::max(1, m_resolution.y);

      m_camera.setResolution(m_resolution.x, m_resolution.y);
      m_rasterizer->setResolution(m_resolution.x, m_resolution.y);
      m_state.resolution = m_resolution;
      m_raytracer->updateState(m_state);
      refresh = true;
    }
    if (ImGui::InputInt("SamplesSqrt", &m_samplesSqrt, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue))
    {
      m_samplesSqrt = clamp(m_samplesSqrt, 1, 256); // Samples per pixel are squares in the range [1, 65536].
      m_state.samplesSqrt = m_samplesSqrt;
      m_raytracer->updateState(m_state);
      refresh = true;
    }
    if (ImGui::DragInt2("Path Lengths", &m_pathLengths.x, 1.0f, 0, 100))
    {
      m_state.pathLengths = m_pathLengths;
      m_raytracer->updateState(m_state);
      refresh = true;
    }
    if (ImGui::DragFloat("Scene Epsilon", &m_epsilonFactor, 1.0f, 0.0f, 10000.0f))
    {
      m_state.epsilonFactor = m_epsilonFactor;
      m_raytracer->updateState(m_state);
      refresh = true;
    }
#if USE_TIME_VIEW
    if (ImGui::DragFloat("Clock Factor", &m_clockFactor, 1.0f, 0.0f, 1000000.0f, "%.0f"))
    {
      m_state.clockFactor = m_clockFactor;
      m_raytracer->updateState(m_state);
      refresh = true;
    }
#endif
  }

#if !USE_TIME_VIEW
  if (ImGui::CollapsingHeader("Tonemapper"))
  {
    bool changed = false;
    if (ImGui::ColorEdit3("Balance", (float*) &m_tonemapperGUI.colorBalance))
    {
      changed = true;
    }
    if (ImGui::DragFloat("Gamma", &m_tonemapperGUI.gamma, 0.01f, 0.01f, 10.0f)) // Must not get 0.0f
    {
      changed = true;
    }
    if (ImGui::DragFloat("White Point", &m_tonemapperGUI.whitePoint, 0.01f, 0.01f, 255.0f, "%.2f", 2.0f)) // Must not get 0.0f
    {
      changed = true;
    }
    if (ImGui::DragFloat("Burn Lights", &m_tonemapperGUI.burnHighlights, 0.01f, 0.0f, 10.0f, "%.2f"))
    {
      changed = true;
    }
    if (ImGui::DragFloat("Crush Blacks", &m_tonemapperGUI.crushBlacks, 0.01f, 0.0f, 1.0f, "%.2f"))
    {
      changed = true;
    }
    if (ImGui::DragFloat("Saturation", &m_tonemapperGUI.saturation, 0.01f, 0.0f, 10.0f, "%.2f"))
    {
      changed = true;
    }
    if (ImGui::DragFloat("Brightness", &m_tonemapperGUI.brightness, 0.01f, 0.0f, 100.0f, "%.2f", 2.0f))
    {
      changed = true;
    }
    if (changed)
    {
      m_rasterizer->setTonemapper(m_tonemapperGUI); // This doesn't need a refresh.
    }
  }
#endif // !USE_TIME_VIEW
  if (ImGui::CollapsingHeader("Materials"))
  {
    // Note that this simple material parameter GUI doesn't allow changing assigned textures.
    for (int iMaterial = 0; iMaterial < static_cast<int>(m_materialsGUI.size()); ++iMaterial)
    {
      bool changed = false;

      MaterialGUI& materialGUI = m_materialsGUI[iMaterial];

      if (ImGui::TreeNode((void*)(intptr_t) iMaterial, "%s", m_materialsGUI[iMaterial].name.c_str()))
      {
        if (ImGui::Combo("BxDF Type", (int*) &materialGUI.typeBXDF, "BXDF\0BRDF Diffuse\0BRDF Specular\0BSDF Specular\0BRDF GGX Smith\0BSDF GGX Smith\0\0"))
        {
          changed = true;
        }
        if (materialGUI.typeBXDF != TYPE_BXDF) // The default BXDF has no parameters.
        {
          if (ImGui::ColorEdit3("Albedo", (float*) &materialGUI.colorAlbedo))
          {
            changed = true;
          }
        }
        if (ImGui::Checkbox("Thin-Walled", &materialGUI.thinwalled)) // Set this to true when using cutout opacity!
        {
          changed = true;
        }	
        // Only show material parameters for the BxDFs which are affected by IOR and volume absorption.
        if (materialGUI.typeBXDF == TYPE_BSDF_SPECULAR ||
            materialGUI.typeBXDF == TYPE_BSDF_GGX_SMITH)
        {
          if (ImGui::ColorEdit3("Absorption", (float*) &materialGUI.colorAbsorption))
          {
            changed = true;
          }
          if (ImGui::DragFloat("Absorption Scale", &materialGUI.scaleAbsorption, 0.01f, 0.0f, 1000.0f, "%.2f"))
          {
            changed = true;
          }
          if (ImGui::DragFloat("IOR", &materialGUI.ior, 0.01f, 0.0f, 10.0f, "%.2f"))
          {
            changed = true;
          }
        }
        // Only show material parameters for the BxDFs which are affected by roughness.
        if (materialGUI.typeBXDF == TYPE_BRDF_GGX_SMITH ||
            materialGUI.typeBXDF == TYPE_BSDF_GGX_SMITH)
        {
          if (ImGui::DragFloat2("Roughness", reinterpret_cast<float*>(&materialGUI.roughness), 0.001f, 0.0f, 1.0f, "%.3f"))
          {
            // Clamp the microfacet roughness to working values minimum values.
            // FIXME When both roughness values fall below that threshold, use TYPE_BRDF_SPECULAR instead.
            if (materialGUI.roughness.x < MICROFACET_MIN_ROUGHNESS)
            {
              materialGUI.roughness.x = MICROFACET_MIN_ROUGHNESS;
            }
            if (materialGUI.roughness.y < MICROFACET_MIN_ROUGHNESS)
            {
              materialGUI.roughness.y = MICROFACET_MIN_ROUGHNESS;
            }
            changed = true;
          }
        }

        // Some material parameters also affect lights.
        if (materialGUI.typeEDF != TYPE_EDF) // The default EDF has no parameters.
        {
          if (ImGui::ColorEdit3("Emission", (float*) &materialGUI.colorEmission))
          {
            changed = true;
          }
          if (ImGui::DragFloat("Emission Multiplier", &materialGUI.multiplierEmission, 0.01f, 0.0f, 1000000.0f, "%.2f", 2.0f))
          {
            materialGUI.multiplierEmission = dp::math::clamp(materialGUI.multiplierEmission, 0.0f, 1000000.0f);
            changed = true;
          }
          // These two values only affect spot lights.
          if (ImGui::DragFloat("Spot Angle", &materialGUI.spotAngle, 0.01f, 0.0f, 180.0f, "%.2f"))
          {
            // It's possible to double click and edit DragFloat fields with values outside the drag range.
            // Make sure the spotAngle is in the expected range. This is not as critical for the multiplier and the exponent.
            materialGUI.spotAngle = dp::math::clamp(materialGUI.spotAngle, 0.0f, 180.0f);
            changed = true;
          }
          if (ImGui::DragFloat("Spot Exponent", &materialGUI.spotExponent, 0.01f, 0.0f, 100.0f, "%.2f"))
          {
            materialGUI.spotExponent = dp::math::clamp(materialGUI.spotExponent, 0.0f, 100.0f);
            changed = true;
          }
        }

        if (changed)
        {
          m_raytracer->updateMaterial(iMaterial, materialGUI);

          // Some material parameters affect the lights using this material.
          // Right now only the emission and spot parameters can be changed here.
          // These are only stored in the device-side structure LightDefinition.
          // Means we only need to find the lights which use the changed material index and update the parameters of that.
          // FIXME Allow changing the orientation of the environment light inside the GUI as well.
          for (size_t iLight = 0; iLight < m_lightsGUI.size(); ++iLight)
          {
            const LightGUI& lightGUI = m_lightsGUI[iLight]; // LightGUI data on the host.

            // If this material index is used by this light index, update the device side LightDefinition values.
            if (iMaterial == lightGUI.idMaterial)
            {
              m_raytracer->updateLight(iLight, materialGUI);
            }
          }
          refresh = true;
        }
        ImGui::TreePop();
      }
    }
  }

  ImGui::PopItemWidth();

  ImGui::End();

  if (refresh)
  {
    restartRendering();
  }
}

void Application::guiRenderingIndicator(const bool isRendering)
{
  // NVIDIA Green when rendering is complete.
  float r = 0.462745f;
  float g = 0.72549f;
  float b = 0.0f;
  
  if (isRendering)
  {
    // Neutral grey while rendering.
    r = 1.0f;
    g = 1.0f;
    b = 1.0f;
  }

  ImGuiStyle& style = ImGui::GetStyle();

  // Use the GUI window title bar color as rendering indicator. Green when rendering is completed.
  style.Colors[ImGuiCol_TitleBg]          = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
  style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
  style.Colors[ImGuiCol_TitleBgActive]    = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
}


bool Application::loadSystemDescription(const std::string& filename)
{
  Parser parser;

  if (!parser.load(filename))
  {
    std::cerr << "ERROR: loadSystemDescription() failed in loadString(" << filename << ")\n";
    return false;
  }

  ParserTokenType tokenType;
  std::string token;

  while ((tokenType = parser.getNextToken(token)) != PTT_EOF)
  {
    if (tokenType == PTT_UNKNOWN)
    {
      std::cerr << "ERROR: loadSystemDescription() " << filename << " (" << parser.getLine() << "): Unknown token type.\n";
      MY_ASSERT(!"Unknown token type.");
      return false;
    }

    if (tokenType == PTT_ID) 
    {
      //if (token == "strategy")
      //{
      //  // Ignored in this renderer. Behaves like RS_INTERACTIVE_MULTI_GPU_LOCAL_COPY, but single-GPU is optimized to not invoke the compositor.
      //  tokenType = parser.getNextToken(token);
      //  MY_ASSERT(tokenType == PTT_VAL);
      //  const int strategy = atoi(token.c_str());

      //  std::cerr << "WARNING: loadSystemDescription() renderer strategy " << strategy << " ignored.\n";
      //}
      //else 
      if (token == "devicesMask") // FIXME Kept the old name to be able to mix and match apps.
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_maskDevices = atoi(token.c_str());
      }
      else if (token == "arenaSize") // In mebi-bytes.
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_sizeArena = std::max(1, atoi(token.c_str()));
      }
      else if (token == "interop")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_interop = atoi(token.c_str());
        if (m_interop < 0 || 2 < m_interop)
        {
          std::cerr << "WARNING: loadSystemDescription() Invalid interop value " << m_interop << ", using interop 0 (host).\n";
          m_interop = 0;
        }
      }
      else if (token == "peerToPeer")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_peerToPeer = atoi(token.c_str());
      }
      else if (token == "present")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_present = (atoi(token.c_str()) != 0);
      }
      else if (token == "resolution")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_resolution.x = std::max(1, atoi(token.c_str()));
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_resolution.y = std::max(1, atoi(token.c_str()));
      }
      else if (token == "tileSize")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tileSize.x = std::max(1, atoi(token.c_str()));
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tileSize.y = std::max(1, atoi(token.c_str()));
       
        // Make sure the values are power-of-two.
        if (m_tileSize.x & (m_tileSize.x - 1))
        {
          std::cerr << "ERROR: loadSystemDescription() tileSize.x = " << m_tileSize.x << " is not power-of-two, using 8.\n";
          m_tileSize.x = 8;
        }
        if (m_tileSize.y & (m_tileSize.y - 1))
        {
          std::cerr << "ERROR: loadSystemDescription() tileSize.y = " << m_tileSize.y << " is not power-of-two, using 8.\n";
          m_tileSize.y = 8;
        }
      }
      else if (token == "samplesSqrt")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_samplesSqrt = std::max(1, atoi(token.c_str())); // spp = m_samplesSqrt * m_samplesSqrt
      }
      else  if (token == "clockFactor")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_clockFactor = (float) atof(token.c_str());
      }
      else if (token == "pathLengths")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_pathLengths.x = atoi(token.c_str()); // min path length before Russian Roulette kicks in
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_pathLengths.y = atoi(token.c_str()); // max path length
      }
      else if (token == "epsilonFactor")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_epsilonFactor = (float) atof(token.c_str());
      }
      // The lens shader, center of interest, and camera values can be set in either the system or scene definition file.
      // The scene defintion will take precedence because it's loaded afterwards.
      else if (token == "lensShader")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_typeLens = static_cast<TypeLens>(atoi(token.c_str()));
        if (m_typeLens < TYPE_LENS_PINHOLE || TYPE_LENS_SPHERE < m_typeLens)
        {
          m_typeLens = TYPE_LENS_PINHOLE;
        }
      }
      else if (token == "center")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        const float x = (float) atof(token.c_str());
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        const float y = (float) atof(token.c_str());
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        const float z = (float) atof(token.c_str());
        m_camera.m_center = make_float3(x, y, z);
        m_camera.markDirty();
      }
      else if (token == "camera")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_camera.m_phi = (float) atof(token.c_str());
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_camera.m_theta = (float) atof(token.c_str());
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_camera.m_fov = (float) atof(token.c_str());
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_camera.m_distance = (float) atof(token.c_str());
        m_camera.markDirty();
      }
      else if (token == "prefixScreenshot")
      {
        tokenType = parser.getNextToken(token); // Needs to be a path in quotation marks.
        MY_ASSERT(tokenType == PTT_STRING);
        convertPath(token);
        m_prefixScreenshot = token;
      }
      // All Tonemapper values can be set in either the system or scene definition file.
      // The scene defintion will take precedence because it's loaded afterwards.
      else if (token == "gamma")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.gamma = (float) atof(token.c_str());
      }
      else if (token == "colorBalance")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.colorBalance[0] = (float) atof(token.c_str());
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.colorBalance[1] = (float) atof(token.c_str());
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.colorBalance[2] = (float) atof(token.c_str());
      }
      else if (token == "whitePoint")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.whitePoint = (float) atof(token.c_str());
      }
      else if (token == "burnHighlights")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.burnHighlights = (float) atof(token.c_str());
      }
      else if (token == "crushBlacks")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.crushBlacks = (float) atof(token.c_str());
      }
      else if (token == "saturation")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.saturation = (float) atof(token.c_str());
      }
      else if (token == "brightness")
      {
        tokenType = parser.getNextToken(token);
        MY_ASSERT(tokenType == PTT_VAL);
        m_tonemapperGUI.brightness = (float) atof(token.c_str());
      }
      else
      {
        std::cerr << "WARNING: loadSystemDescription() Unknown system option name: " << token << '\n';
      }
    }
  }
  return true;
}


bool Application::saveSystemDescription()
{
  std::ostringstream description;

  description << "strategy " << m_strategy << '\n'; // Ignored in this renderer.
  description << "devicesMask " << m_maskDevices << '\n';
  description << "arenaSize " << m_sizeArena << '\n';
  description << "interop " << m_interop << '\n';
  description << "present " << ((m_present) ? "1" : "0") << '\n';
  description << "resolution " << m_resolution.x << " " << m_resolution.y << '\n';
  description << "tileSize " << m_tileSize.x << " " << m_tileSize.y << '\n';
  description << "samplesSqrt " << m_samplesSqrt << '\n';
  description << "clockFactor " << m_clockFactor << '\n';
  description << "pathLengths " << m_pathLengths.x << " " << m_pathLengths.y << '\n';
  description << "epsilonFactor " << m_epsilonFactor << '\n';
  description << "lensShader " << m_typeLens << '\n';
  description << "center " << m_camera.m_center.x << " " << m_camera.m_center.y << " " << m_camera.m_center.z << '\n';
  description << "camera " << m_camera.m_phi << " " << m_camera.m_theta << " " << m_camera.m_fov << " " << m_camera.m_distance << '\n';
  if (!m_prefixScreenshot.empty())
  {
    description << "prefixScreenshot \"" << m_prefixScreenshot << "\"\n";
  }
  description << "gamma " << m_tonemapperGUI.gamma << '\n';
  description << "colorBalance " << m_tonemapperGUI.colorBalance[0] << " " << m_tonemapperGUI.colorBalance[1] << " " << m_tonemapperGUI.colorBalance[2] << '\n';
  description << "whitePoint " << m_tonemapperGUI.whitePoint << '\n';
  description << "burnHighlights " << m_tonemapperGUI.burnHighlights << '\n';
  description << "crushBlacks " << m_tonemapperGUI.crushBlacks << '\n';
  description << "saturation " << m_tonemapperGUI.saturation << '\n';
  description << "brightness " << m_tonemapperGUI.brightness << '\n';

  const std::string filename = std::string("system_rtigo9_") + getDateTime() + std::string(".txt");
  const bool success = saveString(filename, description.str());
  if (success)
  {
    std::cout << filename  << '\n'; // Print out the filename to indicate success.
  }
  return success;
}

int Application::findMaterial(const std::string& reference) const
{
  int indexMaterial = -1; // -1 means not found. This is a critical error.

  std::map<std::string, int>::const_iterator itm = m_mapMaterialReferences.find(reference);
  if (itm != m_mapMaterialReferences.end())
  {
    indexMaterial = itm->second;
  }
  else
  {
    std::cerr << "WARNING: appendInstance() No material found for " << reference << ". Trying default.\n";

    std::map<std::string, int>::const_iterator itmd = m_mapMaterialReferences.find(std::string("default"));
    if (itmd != m_mapMaterialReferences.end())
    {
      indexMaterial = itmd->second;
    }
    else 
    {
      std::cerr << "ERROR: appendInstance() No default material found.\n";
    }
  }
  return indexMaterial;
}

void Application::appendInstance(std::shared_ptr<sg::Group>& group,
                                 std::shared_ptr<sg::Node> geometry,
                                 const dp::math::Mat44f& matrix, 
                                 const int indexMaterial, 
                                 const int indexLight)
{
  // nvpro-pipeline matrices are row-major multiplied from the right, means the translation is in the last row. Transpose!
  const float trafo[12] =
  {
    matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0], 
    matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1], 
    matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2]
  };

  MY_ASSERT(matrix[0][3] == 0.0f && 
            matrix[1][3] == 0.0f && 
            matrix[2][3] == 0.0f && 
            matrix[3][3] == 1.0f);

  std::shared_ptr<sg::Instance> instance(new sg::Instance(m_idInstance++));

  instance->setTransform(trafo);
  instance->setChild(geometry);
  instance->setMaterial(indexMaterial);
  instance->setLight(indexLight);

  group->addChild(instance);
}


TypeBXDF Application::determineTypeBXDF(const std::string& token) const
{
  std::map<std::string, KeywordScene>::const_iterator it = m_mapKeywordScene.find(token);
  if (it == m_mapKeywordScene.end())
  {
    std::cerr << "ERROR: determineTypeBXDF() Unknown token " << token << '\n';
    return TYPE_BXDF;
  }

  switch (it->second)
  {
    case KS_BXDF:
      return TYPE_BXDF;
    case KS_BRDF_DIFFUSE:
      return TYPE_BRDF_DIFFUSE;
    case KS_BRDF_SPECULAR:
      return TYPE_BRDF_SPECULAR;
    case KS_BSDF_SPECULAR:
      return TYPE_BSDF_SPECULAR;
    case KS_BRDF_GGX_SMITH:
      return TYPE_BRDF_GGX_SMITH;
    case KS_BSDF_GGX_SMITH:
      return TYPE_BSDF_GGX_SMITH;
    default:
      std::cerr << "ERROR: determineTypeBXDF() Unexpected BXDF keyword " << token << '\n';
      return TYPE_BXDF;
  }
}

TypeEDF Application::determineTypeEDF(const std::string& token) const
{
  std::map<std::string, KeywordScene>::const_iterator it = m_mapKeywordScene.find(token);
  if (it == m_mapKeywordScene.end())
  {
    std::cerr << "ERROR: determineTypeEDF() Unknown token " << token << '\n';
    return TYPE_EDF;
  }

  switch (it->second)
  {
    case KS_EDF:
      return TYPE_EDF;
    case KS_EDF_DIFFUSE:
      return TYPE_EDF_DIFFUSE;
    case KS_EDF_SPOT:
      return TYPE_EDF_SPOT;
    case KS_EDF_IES:
      return TYPE_EDF_IES;
    default:
      std::cerr << "ERROR: determineTypeEDF() Unexpected BXDF keyword " << token << '\n';
      return TYPE_EDF;
  }
}


bool Application::loadSceneDescription(const std::string& filename)
{
  Parser parser;

  if (!parser.load(filename))
  {
    std::cerr << "ERROR: loadSceneDescription() failed in loadString(" << filename << ")\n";
    return false;
  }

  ParserTokenType tokenType;
  std::string token;

  // Reusing some math routines from the NVIDIA nvpro-pipeline https://github.com/nvpro-pipeline/pipeline
  // Note that matrices in the nvpro-pipeline are defined row-major but are multiplied from the right,
  // which means the order of transformations is simply from left to right matrix, means first matrix is applied first,
  // but puts the translation into the last row elements (12 to 14).

  std::stack<SceneState> stackSceneState;

  SceneState cur;

  cur.reset(); // Reset state to identity transforms and default material parameters.

  while ((tokenType = parser.getNextToken(token)) != PTT_EOF)
  {
    if (tokenType == PTT_UNKNOWN)
    {
      std::cerr << "ERROR: loadSceneDescription() " << filename << " (" << parser.getLine() << "): Unknown token type.\n";
      MY_ASSERT(!"Unknown token type.");
      return false;
    }

    if (tokenType == PTT_ID) 
    {
      std::map<std::string, KeywordScene>::const_iterator itKeyword = m_mapKeywordScene.find(token);
      if (itKeyword == m_mapKeywordScene.end())
      {
        std::cerr << "WARNING: loadSceneDescription() Unknown token " << token << " ignored.\n";
        // MY_ASSERT(!"loadSceneDescription() Unknown token ignored.");
        continue; // Just keep getting the next token until a known keyword is found.
      }

      const KeywordScene keyword = itKeyword->second;

      switch (keyword)
      {
        // Lens shader, center, camera and tonemapper values 
        // can be set in system and scene description. Latter takes precendence.
        // These are global states not affected by push/pop. Last one wins.
        case KS_LENS_SHADER:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          m_typeLens = static_cast<TypeLens>(atoi(token.c_str()));
          if (m_typeLens < TYPE_LENS_PINHOLE || TYPE_LENS_SPHERE < m_typeLens)
          {
            m_typeLens = TYPE_LENS_PINHOLE;
          }
          break;

        case KS_CENTER:
        {
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          const float x = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          const float y = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          const float z = (float) atof(token.c_str());
          m_camera.m_center = make_float3(x, y, z);
          m_camera.markDirty();
          break;
        }

        case KS_CAMERA:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          m_camera.m_phi = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          m_camera.m_theta = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          m_camera.m_fov = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          m_camera.m_distance = (float) atof(token.c_str());
          m_camera.markDirty();
          break;

        case KS_GAMMA:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          m_tonemapperGUI.gamma = (float) atof(token.c_str());
          break;

        case KS_COLOR_BALANCE:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          m_tonemapperGUI.colorBalance[0] = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          m_tonemapperGUI.colorBalance[1] = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          m_tonemapperGUI.colorBalance[2] = (float) atof(token.c_str());
          break;

        case KS_WHITE_POINT:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          m_tonemapperGUI.whitePoint = (float) atof(token.c_str());
          break;

        case KS_BURN_HIGHLIGHTS:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          m_tonemapperGUI.burnHighlights = (float) atof(token.c_str());
          break;

        case KS_CRUSH_BLACKS:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          m_tonemapperGUI.crushBlacks = (float) atof(token.c_str());
          break;

        case KS_SATURATION:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          m_tonemapperGUI.saturation = (float) atof(token.c_str());
          break;

        case KS_BRIGHTNESS:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          m_tonemapperGUI.brightness = (float) atof(token.c_str());
          break;

        case KS_ALBEDO:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.colorAlbedo.x = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.colorAlbedo.y = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.colorAlbedo.z = (float) atof(token.c_str());
          break;

        case KS_ALBEDO_TEXTURE:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_STRING);
          convertPath(token);
          cur.material.nameAlbedo = token;
          break;

        case KS_CUTOUT_TEXTURE:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_STRING);
          convertPath(token);
          cur.material.nameCutout = token;
          break;

        case KS_ROUGHNESS:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.roughness.x = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.roughness.y = (float) atof(token.c_str());
          break;

        case KS_ABSORPTION: // For convenience this is an 'absorption color' used to calculate the absorption coefficient.
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.colorAbsorption.x = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.colorAbsorption.y = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.colorAbsorption.z = (float) atof(token.c_str());
          break;

        case KS_ABSORPTION_SCALE:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.scaleAbsorption = (float) atof(token.c_str());
          break;

        case KS_IOR:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.ior = (float) atof(token.c_str());
          break;

        case KS_THINWALLED:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.thinwalled = (atoi(token.c_str()) != 0);
          break;

        case KS_EMISSION:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.colorEmission.x = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.colorEmission.y = (float) atof(token.c_str());
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.colorEmission.z = (float) atof(token.c_str());
          break;

        case KS_EMISSION_MULTIPLIER:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.multiplierEmission = (float) atof(token.c_str());
          break;

        case KS_EMISSION_PROFILE:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_STRING);
          convertPath(token);
          cur.material.nameProfile = token;
          break;

        case KS_EMISSION_TEXTURE:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_STRING);
          convertPath(token);
          cur.material.nameEmission = token;
          break;

        case KS_SPOT_ANGLE:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.spotAngle = (float) atof(token.c_str());
          cur.material.spotAngle = clamp(cur.material.spotAngle, 0.0f, 180.0f);
          break;

        case KS_SPOT_EXPONENT:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_VAL);
          cur.material.spotExponent = (float) atof(token.c_str());
          break;

        case KS_MATERIAL:
          {
            tokenType = parser.getNextToken(cur.material.name); // Internal material reference name. If there are duplicates the last name wins.
            //MY_ASSERT(tokenType == PTT_ID); // Allow any type of identifier, including strings and numbers.

            tokenType = parser.getNextToken(token); // BXDF identifier.
            MY_ASSERT(tokenType == PTT_ID);
            cur.material.typeBXDF = determineTypeBXDF(token);

            tokenType = parser.getNextToken(token); // EDF identifier
            MY_ASSERT(tokenType == PTT_ID);
            cur.material.typeEDF = determineTypeEDF(token);

            const int indexMaterial = static_cast<int>(m_materialsGUI.size());

            m_materialsGUI.push_back(cur.material); // Store all current host material parameters.

            m_mapMaterialReferences[cur.material.name] = indexMaterial;

            // Cache the referenced pictures to load them only once.
            // FIXME Put this into a helper function.
            // FIXME Only load textures actually referenced inside the scene.
            if (!cur.material.nameAlbedo.empty())
            {
              std::map<std::string, Picture*>::const_iterator it = m_mapPictures.find(cur.material.nameAlbedo);
              if (it == m_mapPictures.end())
              {
                Picture* picture = new Picture();
                picture->load(cur.material.nameAlbedo, IMAGE_FLAG_2D);

                m_mapPictures[cur.material.nameAlbedo] = picture;
              }
            }
            
            if (!cur.material.nameCutout.empty())
            {
              std::map<std::string, Picture*>::const_iterator it = m_mapPictures.find(cur.material.nameCutout);
              if (it == m_mapPictures.end())
              {
                Picture* picture = new Picture();
                picture->load(cur.material.nameCutout, IMAGE_FLAG_2D);

                m_mapPictures[cur.material.nameCutout] = picture;
              }
            }

            if (!cur.material.nameEmission.empty())
            {
              std::map<std::string, Picture*>::const_iterator it = m_mapPictures.find(cur.material.nameEmission);
              if (it == m_mapPictures.end())
              {
                Picture* picture = new Picture();
                picture->load(cur.material.nameEmission, IMAGE_FLAG_2D); // Load as 2D by default. env or rect will be ORed into the flags later.

                m_mapPictures[cur.material.nameEmission] = picture;
              }
            }

            // FIXME Maybe use the emissionTexture with *.ies files as special case?
            // Then IES profiles could be used as environment light and projection for singular lights is implemented already.
            // This doesn't work for mesh lights. Also the environment lights are expected to be colored. 
            // This calls even more for a separation of Pictures from Samplers (texture objects).
            if (!cur.material.nameProfile.empty())
            {
              std::map<std::string, Picture*>::const_iterator it = m_mapPictures.find(cur.material.nameProfile);
              if (it == m_mapPictures.end())
              {
                // Load the IES profile and convert it to an omnidirectional projection texture for a point light. 
                LoaderIES loaderIES;

                if (loaderIES.load(cur.material.nameProfile))
                {
                  if (loaderIES.parse())
                  {
                    Picture* picture = new Picture();
                    picture->createIES(loaderIES.getData()); // This generates the 2D 1-component luminance float image.

                    m_mapPictures[cur.material.nameProfile] = picture;
                  }
                }
              }
            }
          }
          break;

        case KS_IDENTITY:
          cur.matrix         = dp::math::cIdentity44f;
          cur.matrixInv      = dp::math::cIdentity44f;
          cur.orientation    = dp::math::Quatf(0.0f, 0.0f, 0.0f, 1.0f);
          cur.orientationInv = dp::math::Quatf(0.0f, 0.0f, 0.0f, 1.0f);
          break;

        case KS_PUSH:
          stackSceneState.push(cur);
          break;

        case KS_POP:
          if (!stackSceneState.empty())
          {
            cur = stackSceneState.top();
            stackSceneState.pop();
          }
          else
          {
            std::cerr << "ERROR: loadSceneDescription() pop on empty stack. Resetting SceneState to defaults.\n";
            cur.reset();
          }
          break;

        case KS_ROTATE:
          {
            dp::math::Vec3f axis;

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            axis[0] = (float) atof(token.c_str());
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            axis[1] = (float) atof(token.c_str());
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            axis[2] = (float) atof(token.c_str());
            axis.normalize();

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const float angle = dp::math::degToRad((float) atof(token.c_str()));

            const dp::math::Quatf rotation(axis, angle);
            cur.orientation *= rotation;
           
            // Inverse. Opposite order of multiplications.
            const dp::math::Quatf rotationInv(axis, -angle);
            cur.orientationInv = rotationInv * cur.orientationInv;
        
            const dp::math::Mat44f matrix(rotation, dp::math::Vec3f(0.0f, 0.0f, 0.0f)); // Zero translation to get a Mat44f back. 
            cur.matrix *= matrix; // DEBUG No need for the local matrix variable.
        
            // Inverse. Opposite order of matrix multiplications to make M * M^-1 = I.
            const dp::math::Mat44f matrixInv(rotationInv, dp::math::Vec3f(0.0f, 0.0f, 0.0f)); // Zero translation to get a Mat44f back. 
            cur.matrixInv = matrixInv * cur.matrixInv; // DEBUG No need for the local matrixInv variable.
          }
          break;

        case KS_SCALE:
          {
            dp::math::Mat44f scaling(dp::math::cIdentity44f);

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            scaling[0][0] = (float) atof(token.c_str());
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            scaling[1][1] = (float) atof(token.c_str());
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            scaling[2][2] = (float) atof(token.c_str());

            cur.matrix *= scaling;

            // Inverse. // DEBUG Requires scalings to not contain zeros.
            scaling[0][0] = 1.0f / scaling[0][0];
            scaling[1][1] = 1.0f / scaling[1][1];
            scaling[2][2] = 1.0f / scaling[2][2];

            cur.matrixInv = scaling * cur.matrixInv;
          }
          break;

        case KS_TRANSLATE:
          {
            dp::math::Mat44f translation(dp::math::cIdentity44f);
        
            // Translation is in the third row in dp::math::Mat44f.
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            translation[3][0] = (float) atof(token.c_str());
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            translation[3][1] = (float) atof(token.c_str());
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            translation[3][2] = (float) atof(token.c_str());

            cur.matrix *= translation;

            translation[3][0] = -translation[3][0];
            translation[3][1] = -translation[3][1];
            translation[3][2] = -translation[3][2];

            cur.matrixInv = translation * cur.matrixInv;
          }
          break;

        case KS_MODEL:
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_ID);

          if (token == "plane")
          {
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const unsigned int tessU = atoi(token.c_str());

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const unsigned int tessV = atoi(token.c_str());

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const unsigned int upAxis = atoi(token.c_str());

            std::string nameMaterialReference;
            tokenType = parser.getNextToken(nameMaterialReference);

            std::ostringstream keyGeometry;

            keyGeometry << "plane_" << tessU << "_" << tessV << "_" << upAxis;

            std::shared_ptr<sg::Triangles> geometry;

            std::map<std::string, unsigned int>::const_iterator itg = m_mapGeometries.find(keyGeometry.str());
            if (itg == m_mapGeometries.end())
            {
              m_mapGeometries[keyGeometry.str()] = m_idGeometry; // PERF Equal to static_cast<unsigned int>(m_geometries.size());

              geometry = std::make_shared<sg::Triangles>(m_idGeometry++);
              geometry->createPlane(tessU, tessV, upAxis);

              m_geometries.push_back(geometry);
            }
            else
            {
              geometry = std::dynamic_pointer_cast<sg::Triangles>(m_geometries[itg->second]);
            }

            const int indexMaterial = findMaterial(nameMaterialReference);

            appendInstance(m_scene, geometry, cur.matrix, indexMaterial, -1);
          }
          else if (token == "box")
          {
            std::string nameMaterialReference;
            tokenType = parser.getNextToken(nameMaterialReference);
          
            // FIXME Implement tessellation. Must be a single value to get even distributions across edges.
            std::string keyGeometry("box_1_1");

            std::shared_ptr<sg::Triangles> geometry;

            std::map<std::string, unsigned int>::const_iterator itg = m_mapGeometries.find(keyGeometry);
            if (itg == m_mapGeometries.end())
            {
              m_mapGeometries[keyGeometry] = m_idGeometry;

              geometry = std::make_shared<sg::Triangles>(m_idGeometry++);
              geometry->createBox();

              m_geometries.push_back(geometry);
            }
            else
            {
              geometry = std::dynamic_pointer_cast<sg::Triangles>(m_geometries[itg->second]);
            }

            const int indexMaterial = findMaterial(nameMaterialReference);

            appendInstance(m_scene, geometry, cur.matrix, indexMaterial, -1);
          }
          else if (token == "sphere")
          {
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const unsigned int tessU = atoi(token.c_str());

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const unsigned int tessV = atoi(token.c_str());

            // Theta is in the range [0.0f, 1.0f] and 1.0f means closed sphere, smaller values open the noth pole.
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const float theta = float(atof(token.c_str()));

            std::string nameMaterialReference;
            tokenType = parser.getNextToken(nameMaterialReference);
                    
            std::ostringstream keyGeometry;
            keyGeometry << "sphere_" << tessU << "_" << tessV << "_" << theta;

            std::shared_ptr<sg::Triangles> geometry;

            std::map<std::string, unsigned int>::const_iterator itg = m_mapGeometries.find(keyGeometry.str());
            if (itg == m_mapGeometries.end())
            {
              m_mapGeometries[keyGeometry.str()] = m_idGeometry;

              geometry = std::make_shared<sg::Triangles>(m_idGeometry++);
              geometry->createSphere(tessU, tessV, 1.0f, theta * M_PIf);

              m_geometries.push_back(geometry);
            }
            else
            {
              geometry = std::dynamic_pointer_cast<sg::Triangles>(m_geometries[itg->second]);
            }

            const int indexMaterial = findMaterial(nameMaterialReference);

            appendInstance(m_scene, geometry, cur.matrix, indexMaterial, -1);
          }
          else if (token == "torus")
          {
            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const unsigned int tessU = atoi(token.c_str());

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const unsigned int tessV = atoi(token.c_str());

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const float innerRadius = float(atof(token.c_str()));

            tokenType = parser.getNextToken(token);
            MY_ASSERT(tokenType == PTT_VAL);
            const float outerRadius = float(atof(token.c_str()));

            std::string nameMaterialReference;
            tokenType = parser.getNextToken(nameMaterialReference);
                    
            std::ostringstream keyGeometry;
            keyGeometry << "torus_" << tessU << "_" << tessV << "_" << innerRadius << "_" << outerRadius;

            std::shared_ptr<sg::Triangles> geometry;

            std::map<std::string, unsigned int>::const_iterator itg = m_mapGeometries.find(keyGeometry.str());
            if (itg == m_mapGeometries.end())
            {
              m_mapGeometries[keyGeometry.str()] = m_idGeometry;

              geometry = std::make_shared<sg::Triangles>(m_idGeometry++);
              geometry->createTorus(tessU, tessV, innerRadius, outerRadius);

              m_geometries.push_back(geometry);
            }
            else
            {
              geometry = std::dynamic_pointer_cast<sg::Triangles>(m_geometries[itg->second]);
            }

            const int indexMaterial = findMaterial(nameMaterialReference);

            appendInstance(m_scene, geometry, cur.matrix, indexMaterial, -1);
          }
          else if (token == "assimp")
          {
            std::string filenameModel;
            tokenType = parser.getNextToken(filenameModel); // Needs to be a path in quotation marks.
            MY_ASSERT(tokenType == PTT_STRING);
            convertPath(filenameModel);

            std::shared_ptr<sg::Group> model = createASSIMP(filenameModel);

            // nvpro-pipeline matrices are row-major multiplied from the right, means the translation is in the last row. Transpose!
            const float trafo[12] =
            {
              cur.matrix[0][0], cur.matrix[1][0], cur.matrix[2][0], cur.matrix[3][0], 
              cur.matrix[0][1], cur.matrix[1][1], cur.matrix[2][1], cur.matrix[3][1], 
              cur.matrix[0][2], cur.matrix[1][2], cur.matrix[2][2], cur.matrix[3][2]
            };

            MY_ASSERT(cur.matrix[0][3] == 0.0f &&
                      cur.matrix[1][3] == 0.0f &&
                      cur.matrix[2][3] == 0.0f &&
                      cur.matrix[3][3] == 1.0f);

            std::shared_ptr<sg::Instance> instance(new sg::Instance(m_idInstance++));
            instance->setTransform(trafo);
            instance->setChild(model);

            m_scene->addChild(instance);
          }
          else
          {
            std::cerr << "WARNING: loadSceneDescription() unknown model type "<< token << '\n';
          }
          break;

        case KS_LIGHT:
        {
          // One of the predefined light types: env, rect, point, spot, ies
          tokenType = parser.getNextToken(token);
          MY_ASSERT(tokenType == PTT_ID);

          std::string nameMaterialReference;
          tokenType = parser.getNextToken(nameMaterialReference);
          //MY_ASSERT(tokenType == PTT_ID); // Allow any type of identifier, including strings and numbers.

          const int indexMaterial = findMaterial(nameMaterialReference);
          
          if (indexMaterial < 0) // Lights require a material reference. 
          {
            std::cerr << "ERROR: loadSceneDescription() Light referenced unknown material " << nameMaterialReference << '\n';
            break;
          }

          const MaterialGUI& materialGUI = m_materialsGUI[indexMaterial];
          
          // FIXME Use the m_mapKeywords and a switch here.
          if (token == "env") // Constant or spherical texture map environment light.
          {
            // There can be only one environment light and it must be the first light definition.
            if (m_lightsGUI.size() == 0)
            {
              LightGUI lightGUI;
            
              // FIXME This type selection would need to be handled via different env keywords for more than two environment lights (e.g. hemisphere).
              lightGUI.typeLight = (materialGUI.nameEmission.empty()) ? TYPE_LIGHT_ENV_CONST : TYPE_LIGHT_ENV_SPHERE;
              // Transformation is taken from the current scene state. Only orientation is used for spherical environment lights.
              lightGUI.matrix         = cur.matrix;
              lightGUI.matrixInv      = cur.matrixInv;
              lightGUI.orientation    = cur.orientation;
              lightGUI.orientationInv = cur.orientationInv;
              // All other light parameters are taken from the referenced material.
              lightGUI.idGeometry         = ~0u; // Unused. Not a mesh light.
              lightGUI.idMaterial         = indexMaterial;
              lightGUI.area               = 4.0f * M_PIf; // Unused.

              m_lightsGUI.push_back(lightGUI);

              if (!materialGUI.nameEmission.empty())
              {
                std::map<std::string, Picture*>::iterator it = m_mapPictures.find(materialGUI.nameEmission);
                MY_ASSERT(it != m_mapPictures.end()); // Must have been loaded when the material was parsed.
                if (it != m_mapPictures.end())
                {
                  MY_ASSERT((it->second->getFlags() & IMAGE_FLAG_RECT) == 0); // Cannot have the same texture for env and rect. There is only one CDF.
                  it->second->addFlags(IMAGE_FLAG_ENV);
                }
              }
            }
            else
            {
              std::cerr << "ERROR: loadSceneDescription() Environment lights must be specified first.\n";
            }
          }
          else if (token == "rect") // Rectangle area light, supports importance sampled textures. (Mesh lights don't.)
          {
            // Rectangular area light source, emission is singled sided on the front face.
            // Represented as two triangles with extents from -0.5 to 0.5 (== 1.0 m^2) on the xy-plane with the positive z-axis as normal.
            // Use the transformations to scale, rotate and translate it in world space.
            LightGUI lightGUI;
            
            lightGUI.typeLight = TYPE_LIGHT_RECT;
            // Transformation is taken from the current scene state. Only orientation is used for spherical environment lights.
            lightGUI.matrix         = cur.matrix;
            lightGUI.matrixInv      = cur.matrixInv;
            lightGUI.orientation    = cur.orientation;
            lightGUI.orientationInv = cur.orientationInv;
            // All other light parameters are taken from the referenced material.
            lightGUI.idGeometry = ~0u;
            lightGUI.idMaterial = indexMaterial;
            
            // Calculate the world space rectangle area.
            //const dp::math::Vec3f vecU(dp::math::Vec4f(1.0f, 0.0f, 0.0f, 0.0f) * lightGUI.matrix);
            //const dp::math::Vec3f vecV(dp::math::Vec4f(0.0f, 1.0f, 0.0f, 0.0f) * lightGUI.matrix);
            // Optimized version: 
            const dp::math::Vec3f vecU(lightGUI.matrix[0]);
            const dp::math::Vec3f vecV(lightGUI.matrix[1]);

            lightGUI.area = dp::math::length(vecU ^ vecV); // Rectangle area is the length of the cross product of the edges.

            const int indexLight = static_cast<int>(m_lightsGUI.size());

            m_lightsGUI.push_back(lightGUI);

            if (!materialGUI.nameEmission.empty())
            {
              std::map<std::string, Picture*>::iterator it = m_mapPictures.find(materialGUI.nameEmission);
              MY_ASSERT(it != m_mapPictures.end()); // Must have been loaded when the material was parsed.
              if (it != m_mapPictures.end())
              {
                MY_ASSERT((it->second->getFlags() & IMAGE_FLAG_ENV) == 0); // Cannot have the same texture for env and rect. There is only one CDF.
                it->second->addFlags(IMAGE_FLAG_RECT);
              }
            }

            // Create the rectangle geometry.
            std::ostringstream keyGeometry;
            keyGeometry << "rect";

            std::shared_ptr<sg::Triangles> geometry;

            std::map<std::string, unsigned int>::const_iterator itg = m_mapGeometries.find(keyGeometry.str());
            if (itg == m_mapGeometries.end())
            {
              m_mapGeometries[keyGeometry.str()] = m_idGeometry;

              geometry = std::make_shared<sg::Triangles>(m_idGeometry++);
              geometry->createRect();

              m_geometries.push_back(geometry);
            }
            else
            {
              geometry = std::dynamic_pointer_cast<sg::Triangles>(m_geometries[itg->second]);
            }
            // The rect light is adding a light definition, but also add geometry with an emissive material to the scene graph.
            // That the indexLight is not -1 here will let the createMeshLights() routine, or more precisely the traverseGraph()
            // function in there not append another light definition.
            appendInstance(m_scene, geometry, cur.matrix, indexMaterial, indexLight);
          }
          else if (token == "point")
          {
            LightGUI lightGUI;
            
            lightGUI.typeLight = TYPE_LIGHT_POINT;
            // Transformation is taken from the current scene state.
            lightGUI.matrix         = cur.matrix;
            lightGUI.matrixInv      = cur.matrixInv;
            lightGUI.orientation    = cur.orientation;
            lightGUI.orientationInv = cur.orientationInv;
            // All other light parameters are taken from the referenced material.
            lightGUI.idGeometry = ~0u;
            lightGUI.idMaterial = indexMaterial;
            
            lightGUI.area = 1.0f; // Unused for singular lights. 

            const int indexLight = static_cast<int>(m_lightsGUI.size());

            m_lightsGUI.push_back(lightGUI);
            
            if (!materialGUI.nameEmission.empty())
            {
              std::map<std::string, Picture*>::iterator it = m_mapPictures.find(materialGUI.nameEmission);
              MY_ASSERT(it != m_mapPictures.end()); // Must have been loaded when the material was parsed.
              if (it != m_mapPictures.end())
              {
                it->second->addFlags(IMAGE_FLAG_POINT);
              }
            }
            // No geometry. A singular light cannot be hit implicitly.
          }
          else if (token == "spot") // Spot light singular light type. Emission texture is projected over the open spread angle.
          {
            LightGUI lightGUI;
            
            lightGUI.typeLight = TYPE_LIGHT_SPOT; 
            // Transformation is taken from the current scene state.
            lightGUI.matrix         = cur.matrix;
            lightGUI.matrixInv      = cur.matrixInv;
            lightGUI.orientation    = cur.orientation;
            lightGUI.orientationInv = cur.orientationInv;
            // All other light parameters are taken from the referenced material.
            lightGUI.idGeometry = ~0u;
            lightGUI.idMaterial = indexMaterial;
            
            lightGUI.area = 1.0f; // Unused for singular lights. 

            const int indexLight = static_cast<int>(m_lightsGUI.size());

            m_lightsGUI.push_back(lightGUI);
            
            if (!materialGUI.nameEmission.empty())
            {
              std::map<std::string, Picture*>::iterator it = m_mapPictures.find(materialGUI.nameEmission);
              MY_ASSERT(it != m_mapPictures.end()); // Must have been loaded when the material was parsed.
              if (it != m_mapPictures.end())
              {
                it->second->addFlags(IMAGE_FLAG_SPOT);
              }
            }
            // No geometry. A singular light cannot be hit implicitly.
          }
          else if (token == "ies")
          {
            LightGUI lightGUI;
            
            lightGUI.typeLight = TYPE_LIGHT_IES;
            // Transformation is taken from the current scene state.
            lightGUI.matrix         = cur.matrix;
            lightGUI.matrixInv      = cur.matrixInv;
            lightGUI.orientation    = cur.orientation;
            lightGUI.orientationInv = cur.orientationInv;
            // All other light parameters are taken from the referenced material.
            lightGUI.idGeometry = ~0u;
            lightGUI.idMaterial = indexMaterial;
            
            lightGUI.area = 1.0f; // Unused for singular lights. 

            const int indexLight = static_cast<int>(m_lightsGUI.size());

            m_lightsGUI.push_back(lightGUI);

            if (!materialGUI.nameProfile.empty())
            {
              std::map<std::string, Picture*>::iterator it = m_mapPictures.find(materialGUI.nameProfile);
              MY_ASSERT(it != m_mapPictures.end()); // Must have been loaded when the material was parsed.
              if (it != m_mapPictures.end())
              {
                it->second->addFlags(IMAGE_FLAG_IES); // FIXME This is redundant because generateIES() sets that already.
              }
            }
            // No geometry. A singular light cannot be hit implicitly.
          }
          else
          {
            std::cerr << "WARNING: loadSceneDescription() unknown light type "<< token << '\n';
          }
        }
        break;

        default:
          std::cerr << "ERROR: loadSceneDescription() Unexpected KeywordScene value " << keyword << " ignored.\n";
          MY_ASSERT(!"ERROR: loadSceneDescription() Unexpected KeywordScene value");
          break;
      }
    }
  }

  std::cout << "loadSceneDescription() m_idGroup = " << m_idGroup << ", m_idInstance = " << m_idInstance << ", m_idGeometry = " << m_idGeometry << '\n';

  return true;
}


void Application::createMeshLights()
{
  // Traverse the host scene graph and append light definitions for all meshes with emissive materials.
  InstanceData instanceData(~0u, -1, -1);

  float matrix[12];

  // Set the affine matrix to identity by default.
  memset(matrix, 0, sizeof(float) * 12);
  matrix[ 0] = 1.0f;
  matrix[ 5] = 1.0f;
  matrix[10] = 1.0f;

  traverseGraph(m_scene, instanceData, matrix);
}


// m = a * b;
static void multiplyMatrix(float* m, const float* a, const float* b)
{
  m[0] = a[0] * b[0] + a[1] * b[4] + a[2] * b[ 8]; // + a[3] * 0
  m[1] = a[0] * b[1] + a[1] * b[5] + a[2] * b[ 9]; // + a[3] * 0
  m[2] = a[0] * b[2] + a[1] * b[6] + a[2] * b[10]; // + a[3] * 0
  m[3] = a[0] * b[3] + a[1] * b[7] + a[2] * b[11] + a[3]; // * 1

  m[4] = a[4] * b[0] + a[5] * b[4] + a[6] * b[ 8]; // + a[7] * 0
  m[5] = a[4] * b[1] + a[5] * b[5] + a[6] * b[ 9]; // + a[7] * 0
  m[6] = a[4] * b[2] + a[5] * b[6] + a[6] * b[10]; // + a[7] * 0
  m[7] = a[4] * b[3] + a[5] * b[7] + a[6] * b[11] + a[7]; // * 1

  m[ 8] = a[8] * b[0] + a[9] * b[4] + a[10] * b[ 8]; // + a[11] * 0
  m[ 9] = a[8] * b[1] + a[9] * b[5] + a[10] * b[ 9]; // + a[11] * 0
  m[10] = a[8] * b[2] + a[9] * b[6] + a[10] * b[10]; // + a[11] * 0
  m[11] = a[8] * b[3] + a[9] * b[7] + a[10] * b[11] + a[11]; // * 1
}


// Depth-first traversal of the scene graph to flatten all unique paths to a geometry node to one-level instancing inside the OptiX render graph.
void Application::traverseGraph(std::shared_ptr<sg::Node> node, InstanceData& instanceData, float matrix[12])
{
  switch (node->getType())
  {
    case sg::NodeType::NT_GROUP:
    {
      std::shared_ptr<sg::Group> group = std::dynamic_pointer_cast<sg::Group>(node);
      //std::cout << "Group " << group->getId() << ", " << group->getNumChildren() << '\n';

      for (size_t i = 0; i < group->getNumChildren(); ++i)
      {
        traverseGraph(group->getChild(i), instanceData, matrix);
      }
    }
    break;

    case sg::NodeType::NT_INSTANCE:
    {
      std::shared_ptr<sg::Instance> instance = std::dynamic_pointer_cast<sg::Instance>(node);
      //std::cout << "Instance " << instance->getId() << ", " << instance->getMaterial() << ", " << instance->getLight() << '\n';

      // Track the assigned material and light indices. Only the bottom-most instance node matters.
      instanceData.idMaterial = instance->getMaterial();
      instanceData.idLight    = instance->getLight();

      // Concatenate the transformations along the path.
      float trafo[12];

      multiplyMatrix(trafo, matrix, instance->getTransform());

      traverseGraph(instance->getChild(), instanceData, trafo);
      
      // If there was no light ID on the instance, but the geometry is a mesh light,
      // assign the new light ID to the bottom-most instance.
      if (instance->getLight() == -1 && instanceData.idLight != -1)
      {
        instance->setLight(instanceData.idLight);
      }
      instanceData.idLight = -1; // Clear the returned value to not populate it upwards.
    }
    break;

    case sg::NodeType::NT_TRIANGLES:
    {
      // Only create a new light definition if the instance is not already holding a valid light index!
      // This is the case for rectangle lights which add a light definition and add geometry to the scene.
      if (instanceData.idLight == -1)
      {
        std::shared_ptr<sg::Triangles> geometry = std::dynamic_pointer_cast<sg::Triangles>(node);
        //std::cout << "Triangles " << geometry->getId() << '\n';

        instanceData.idGeometry = geometry->getId();

        // Check if the material assigned to this mesh is emissive and 
        // return the new idLight to the caller to set it inside the instance.
        instanceData.idLight = createMeshLight(geometry, instanceData.idMaterial, matrix);
      }
    }
    break;
  }
}

int Application::createMeshLight(const std::shared_ptr<sg::Triangles> geometry, const int indexMaterial, const float matrix[12])
{
  int indexLight = -1;

  // If an emissive material (edf != TYPE_EDF) is assigned to an arbitrary triangle mesh geometry inside the scene,
  // create a LightDefinition with the given geometry and append it to m_lightsGUI.
  if (0 <= indexMaterial)
  {
    const MaterialGUI& materialGUI = m_materialsGUI[indexMaterial];
    
    if (materialGUI.typeEDF != TYPE_EDF)
    {
      LightGUI lightGUI;

      lightGUI.typeLight = TYPE_LIGHT_MESH;

      // nvpro-pipeline matrices are row-major multiplied from the right, means the translation is in the last row. Transpose!
      dp::math::Mat44f mat44f( { matrix[0], matrix[4], matrix[ 8], 0.0f,
                                 matrix[1], matrix[5], matrix[ 9], 0.0f,
                                 matrix[2], matrix[6], matrix[10], 0.0f,
                                 matrix[3], matrix[7], matrix[11], 1.0f } );

      lightGUI.matrix    = mat44f;
      lightGUI.matrixInv = mat44f;
      if (!lightGUI.matrixInv.invert())
      {
        std::cerr << "ERROR: Application::createMeshLight() matrix inversion failed.";
        MY_ASSERT(!"ERROR: Application::createMeshLight() matrix inversion failed.");
      }

      // HACK PERF It would be involved to decompose the pure rotational part from a generic 4x4 matrix. (dp::math decompose() can do that.)
      // But rectangle and arbitrary mesh lights use only the 4x4 matrices above, so no need to set the orientation in this mesh light case at all.
      // Just set the orientation quaternions to identity.
      lightGUI.orientation    = dp::math::Quatf(0.0f, 0.0f, 0.0f, 1.0f);
      lightGUI.orientationInv = dp::math::Quatf(0.0f, 0.0f, 0.0f, 1.0f);

      lightGUI.idGeometry = geometry->getId();
      lightGUI.idMaterial = indexMaterial;

      // Fill in the areaTriangles and cdfAreas and area.
      geometry->calculateLightArea(lightGUI);

      indexLight = static_cast<int>(m_lightsGUI.size()); // Success, set the proper light index.

      m_lightsGUI.push_back(lightGUI);
    }
  }
  return indexLight;
}


bool Application::loadString(const std::string& filename, std::string& text)
{
  std::ifstream inputStream(filename);

  if (!inputStream)
  {
    std::cerr << "ERROR: loadString() Failed to open file " << filename << '\n';
    return false;
  }

  std::stringstream data;

  data << inputStream.rdbuf();

  if (inputStream.fail())
  {
    std::cerr << "ERROR: loadString() Failed to read file " << filename << '\n';
    return false;
  }

  text = data.str();
  return true;
}

bool Application::saveString(const std::string& filename, const std::string& text)
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

std::string Application::getDateTime()
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

static void updateAABB(float3& minimum, float3& maximum, const float3& v)
{
  if (v.x < minimum.x)
  {
    minimum.x = v.x;
  }
  else if (maximum.x < v.x)
  {
    maximum.x = v.x;
  }

  if (v.y < minimum.y)
  {
    minimum.y = v.y;
  }
  else if (maximum.y < v.y)
  {
    maximum.y = v.y;
  }

  if (v.z < minimum.z)
  {
    minimum.z = v.z;
  }
  else if (maximum.z < v.z)
  {
    maximum.z = v.z;
  }
}

//static void calculateTexcoordsSpherical(std::vector<InterleavedHost>& attributes, const std::vector<unsigned int>& indices)
//{
//  dp::math::Vec3f center(0.0f, 0.0f, 0.0f);
//  for (size_t i = 0; i < attributes.size(); ++i)
//  {
//    center += attributes[i].vertex;
//  }
//  center /= (float) attributes.size();
//
//  float u;
//  float v;
//
//  for (size_t i = 0; i < attributes.size(); ++i)
//  {
//    dp::math::Vec3f p = attributes[i].vertex - center;
//    if (FLT_EPSILON < fabsf(p[1]))
//    {
//      u = 0.5f * atan2f(p[0], -p[1]) / dp::math::PI + 0.5f;
//    }
//    else
//    {
//      u = (0.0f <= p[0]) ? 0.75f : 0.25f;
//    }
//    float d = sqrtf(dp::math::square(p[0]) + dp::math::square(p[1]));
//    if (FLT_EPSILON < d)
//    {
//      v = atan2f(p[2], d) / dp::math::PI + 0.5f;
//    }
//    else
//    {
//      v = (0.0f <= p[2]) ? 1.0f : 0.0f;
//    }
//    attributes[i].texcoord0 = dp::math::Vec3f(u, v, 0.0f);
//  }
//
//  //// The code from the environment texture lookup.
//  //for (size_t i = 0; i < attributes.size(); ++i)
//  //{
//  //  dp::math::Vec3f R = attributes[i].vertex - center;
//  //  dp::math::normalize(R);
//
//  //  // The seam u == 0.0 == 1.0 is in positive z-axis direction.
//  //  // Compensate for the environment rotation done inside the direct lighting.
//  //  const float u = (atan2f(R[0], -R[2]) + dp::math::PI) * 0.5f / dp::math::PI;
//  //  const float theta = acosf(-R[1]); // theta == 0.0f is south pole, theta == M_PIf is north pole.
//  //  const float v = theta / dp::math::PI; // Texture is with origin at lower left, v == 0.0f is south pole.
//
//  //  attributes[i].texcoord0 = dp::math::Vecf(u, v, 0.0f);
//  //}
//}


// Calculate texture tangents based on the texture coordinate gradients.
// Doesn't work when all texture coordinates are identical! Thats the reason for the other routine below.
//static void calculateTangents(std::vector<InterleavedHost>& attributes, const std::vector<unsigned int>& indices)
//{
//  for (size_t i = 0; i < indices.size(); i += 4)
//  {
//    unsigned int i0 = indices[i    ];
//    unsigned int i1 = indices[i + 1];
//    unsigned int i2 = indices[i + 2];
//
//    dp::math::Vec3f e0 = attributes[i1].vertex - attributes[i0].vertex;
//    dp::math::Vec3f e1 = attributes[i2].vertex - attributes[i0].vertex;
//    dp::math::Vec2f d0 = dp::math::Vec2f(attributes[i1].texcoord0) - dp::math::Vec2f(attributes[i0].texcoord0);
//    dp::math::Vec2f d1 = dp::math::Vec2f(attributes[i2].texcoord0) - dp::math::Vec2f(attributes[i0].texcoord0);
//    attributes[i0].tangent += d1[1] * e0 - d0[1] * e1;
//
//    e0 = attributes[i2].vertex - attributes[i1].vertex;
//    e1 = attributes[i0].vertex - attributes[i1].vertex;
//    d0 = dp::math::Vec2f(attributes[i2].texcoord0) - dp::math::Vec2f(attributes[i1].texcoord0);
//    d1 = dp::math::Vec2f(attributes[i0].texcoord0) - dp::math::Vec2f(attributes[i1].texcoord0);
//    attributes[i1].tangent += d1[1] * e0 - d0[1] * e1;
//
//    e0 = attributes[i0].vertex - attributes[i2].vertex;
//    e1 = attributes[i1].vertex - attributes[i2].vertex;
//    d0 = dp::math::Vec2f(attributes[i0].texcoord0) - dp::math::Vec2f(attributes[i2].texcoord0);
//    d1 = dp::math::Vec2f(attributes[i1].texcoord0) - dp::math::Vec2f(attributes[i2].texcoord0);
//    attributes[i2].tangent += d1[1] * e0 - d0[1] * e1;
//  }
//
//  for (size_t i = 0; i < attributes.size(); ++i)
//  {
//    dp::math::Vec3f tangent(attributes[i].tangent);
//    dp::math::normalize(tangent); // This normalizes the sums from above!
//
//    dp::math::Vec3f normal(attributes[i].normal);
//
//    dp::math::Vec3f bitangent = normal ^ tangent;
//    dp::math::normalize(bitangent);
//
//    tangent = bitangent ^ normal;
//    dp::math::normalize(tangent);
//    
//    attributes[i].tangent = tangent;
//
//#if USE_BITANGENT
//    attributes[i].bitangent = bitantent;
//#endif
//  }
//}

// Calculate (geometry) tangents with the global tangent direction aligned to the biggest AABB extend of this part.
void Application::calculateTangents(std::vector<TriangleAttributes>& attributes, const std::vector<unsigned int>& indices)
{
  MY_ASSERT(3 <= indices.size());

  // Initialize with the first vertex to be able to use else-if comparisons in updateAABB().
  float3 aabbLo = attributes[indices[0]].vertex;
  float3 aabbHi = attributes[indices[0]].vertex;

  // Build an axis aligned bounding box.
  for (size_t i = 0; i < indices.size(); i += 3)
  {
    unsigned int i0 = indices[i    ];
    unsigned int i1 = indices[i + 1];
    unsigned int i2 = indices[i + 2];

    updateAABB(aabbLo, aabbHi, attributes[i0].vertex);
    updateAABB(aabbLo, aabbHi, attributes[i1].vertex);
    updateAABB(aabbLo, aabbHi, attributes[i2].vertex);
  }

  // Get the longest extend and use that as general tangent direction.
  const float3 extents = aabbHi - aabbLo;
  
  float f = extents.x;
  int maxComponent = 0;

  if (f < extents.y)
  {
    f = extents.y;
    maxComponent = 1;
  }
  if (f < extents.z)
  {
    maxComponent = 2;
  }

  float3 direction;
  float3 bidirection;

  switch (maxComponent)
  {
  case 0: // x-axis
  default:
    direction   = make_float3(1.0f, 0.0f, 0.0f);
    bidirection = make_float3(0.0f, 1.0f, 0.0f);
    break;
  case 1: // y-axis // DEBUG It might make sense to keep these directions aligned to the global coordinate system. Use the same coordinates as for z-axis then.
    direction   = make_float3(0.0f, 1.0f, 0.0f); 
    bidirection = make_float3(0.0f, 0.0f, -1.0f);
    break;
  case 2: // z-axis
    direction   = make_float3(0.0f, 0.0f, -1.0f);
    bidirection = make_float3(0.0f, 1.0f,  0.0f);
    break;
  }

  // Build an ortho-normal basis with the existing normal.
  for (size_t i = 0; i < attributes.size(); ++i)
  {
    float3 tangent   = direction;
    float3 bitangent = bidirection;
    // float3 normal    = attributes[i].normal;
    float3 normal;
    normal.x = attributes[i].normal.x;
    normal.y = attributes[i].normal.y;
    normal.z = attributes[i].normal.z;

    if (0.001f < 1.0f - fabsf(dot(normal, tangent)))
    {
      bitangent = normalize(cross(normal, tangent));
      tangent   = normalize(cross(bitangent, normal));
    }
    else // Normal and tangent direction too collinear.
    {
      MY_ASSERT(0.001f < 1.0f - fabsf(dot(bitangent, normal)));
      tangent   = normalize(cross(bitangent, normal));
      //bitangent = normalize(cross(normal, tangent));
    }
    attributes[i].tangent = tangent;
  }
}

bool Application::screenshot(const bool tonemap)
{
  ILboolean hasImage = false;
  
  const int spp = m_samplesSqrt * m_samplesSqrt; // Add the samples per pixel to the filename for quality comparisons.

  std::ostringstream path;
   
  path << m_prefixScreenshot << "_" << spp << "spp_" << getDateTime();
  
  unsigned int imageID;

  ilGenImages(1, (ILuint *) &imageID);

  ilBindImage(imageID);
  ilActiveImage(0);
  ilActiveFace(0);

  ilDisable(IL_ORIGIN_SET);

  const float4* bufferHost = reinterpret_cast<const float4*>(m_raytracer->getOutputBufferHost());
  
  if (tonemap)
  {
    // Store a tonemapped RGB8 *.png image
    path << ".png";

    if (ilTexImage(m_resolution.x, m_resolution.y, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, nullptr))
    {
      uchar3* dst = reinterpret_cast<uchar3*>(ilGetData());

      const float  invGamma       = 1.0f / m_tonemapperGUI.gamma;
      const float3 colorBalance   = make_float3(m_tonemapperGUI.colorBalance[0], m_tonemapperGUI.colorBalance[1], m_tonemapperGUI.colorBalance[2]);
      const float  invWhitePoint  = m_tonemapperGUI.brightness / m_tonemapperGUI.whitePoint;
      const float  burnHighlights = m_tonemapperGUI.burnHighlights;
      const float  crushBlacks    = m_tonemapperGUI.crushBlacks + m_tonemapperGUI.crushBlacks + 1.0f;
      const float  saturation     = m_tonemapperGUI.saturation;

      for (int y = 0; y < m_resolution.y; ++y)
      {
        for (int x = 0; x < m_resolution.x; ++x)
        {
          const int idx = y * m_resolution.x + x;

          // Tonemapper. // PERF Add a native CUDA kernel doing this.
          float3 hdrColor = make_float3(bufferHost[idx]);
          float3 ldrColor = invWhitePoint * colorBalance * hdrColor;
          ldrColor       *= ((ldrColor * burnHighlights) + 1.0f) / (ldrColor + 1.0f);
          
          float luminance = dot(ldrColor, make_float3(0.3f, 0.59f, 0.11f));
          ldrColor = lerp(make_float3(luminance), ldrColor, saturation); // This can generate negative values for saturation > 1.0f!
          ldrColor = fmaxf(make_float3(0.0f), ldrColor); // Prevent negative values.

          luminance = dot(ldrColor, make_float3(0.3f, 0.59f, 0.11f));
          if (luminance < 1.0f)
          {
            const float3 crushed = powf(ldrColor, crushBlacks);
            ldrColor = lerp(crushed, ldrColor, sqrtf(luminance));
            ldrColor = fmaxf(make_float3(0.0f), ldrColor); // Prevent negative values.
          }
          ldrColor = clamp(powf(ldrColor, invGamma), 0.0f, 1.0f); // Saturate, clamp to range [0.0f, 1.0f].

          dst[idx] = make_uchar3((unsigned char) (ldrColor.x * 255.0f),
                                 (unsigned char) (ldrColor.y * 255.0f),
                                 (unsigned char) (ldrColor.z * 255.0f));
        }
      }
      hasImage = true;
    }
  }
  else
  {
    // Store the float4 linear output buffer as *.hdr image.
    // FIXME Add a half float conversion and store as *.exr. (Pre-built DevIL 1.7.8 supports EXR, DevIL 1.8.0 doesn't!)
    path << ".hdr";

    hasImage = ilTexImage(m_resolution.x, m_resolution.y, 1, 4, IL_RGBA, IL_FLOAT, (void*) bufferHost);
  }

  if (hasImage)
  {
    ilEnable(IL_FILE_OVERWRITE); // By default, always overwrite
    
    std::string filename = path.str();
    convertPath(filename);
	
    if (ilSaveImage((const ILstring) filename.c_str()))
    {
      ilDeleteImages(1, &imageID);

      std::cout << filename << '\n'; // Print out filename to indicate that a screenshot has been taken.
      return true;
    }
  }

  // There was an error when reaching this code.
  ILenum error = ilGetError(); // DEBUG 
  std::cerr << "ERROR: screenshot() failed with IL error " << error << '\n';

  while (ilGetError() != IL_NO_ERROR) // Clean up errors.
  {
  }

  // Free all resources associated with the DevIL image
  ilDeleteImages(1, &imageID);

  return false;
}

// Convert between slashes and backslashes in paths depending on the operating system
void Application::convertPath(std::string& path)
{
#if defined(_WIN32)
  std::string::size_type pos = path.find("/", 0);
  while (pos != std::string::npos)
  {
    path[pos] = '\\';
    pos = path.find("/", pos);
  }
#elif defined(__linux__)
  std::string::size_type pos = path.find("\\", 0);
  while (pos != std::string::npos)
  {
    path[pos] = '/';
    pos = path.find("\\", pos);
  }
#endif
}

void Application::convertPath(char* path)
{
#if defined(_WIN32)
  for (size_t i = 0; i < strlen(path); ++i)
  {
    if (path[i] == '/')
    {
      path[i] = '\\';
    }
  }
#elif defined(__linux__)
  for (size_t i = 0; i < strlen(path); ++i)
  {
    if (path[i] == '\\')
    {
      path[i] = '/';
    }
  }
#endif
}
