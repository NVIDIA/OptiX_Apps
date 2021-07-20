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

#pragma once

#ifndef APPLICATION_H
#define APPLICATION_H

// DAR This renderer only uses the CUDA Driver API!
// (CMake uses the CUDA_CUDA_LIBRARY which is nvcuda.lib. At runtime that loads nvcuda.dll from the driver.)
//#include <cuda.h>

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

#include <GL/glew.h>
#if defined( _WIN32 )
#include <GL/wglew.h>
#endif

#include "inc/Camera.h"
#include "inc/Options.h"
#include "inc/Rasterizer.h"
#include "inc/Raytracer.h"
#include "inc/SceneGraph.h"
#include "inc/Texture.h"
#include "inc/Timer.h"

#include <dp/math/Matmnt.h>

#include "shaders/material_definition.h"

// This include gl.h and needs to be done after glew.h
#include <GLFW/glfw3.h>

// assimp include files.
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/LogStream.hpp>

#include <map>
#include <memory>


constexpr int APP_EXIT_SUCCESS        =  0;
constexpr int APP_ERROR_UNKNOWN       = -1;
constexpr int APP_ERROR_CREATE_WINDOW = -2;
constexpr int APP_ERROR_GLFW_INIT     = -3;
constexpr int APP_ERROR_GLEW_INIT     = -4;
constexpr int APP_ERROR_APP_INIT      = -5;


enum GuiState
{
  GUI_STATE_NONE,
  GUI_STATE_ORBIT,
  GUI_STATE_PAN,
  GUI_STATE_DOLLY,
  GUI_STATE_FOCUS
};


enum KeywordScene
{
  KS_ALBEDO,
  KS_ROUGHNESS,
  KS_ABSORPTION,
  KS_ABSORPTION_SCALE,
  KS_IOR,
  KS_THINWALLED,
  KS_MATERIAL,
  KS_IDENTITY,
  KS_PUSH,
  KS_POP,
  KS_ROTATE,
  KS_SCALE,
  KS_TRANSLATE,
  KS_MODEL
};


class Application
{
public:

  Application(GLFWwindow* window, Options const& options);
  ~Application();

  bool isValid() const;

  void reshape(const int w, const int h);
  bool render();
  void benchmark();

  void display();

  void guiNewFrame();
  void guiWindow();
  void guiEventHandler();
  void guiReferenceManual(); // The ImGui "programming manual" in form of a live window.
  void guiRender();

private:
  bool loadSystemDescription(std::string const& filename);
  bool saveSystemDescription();
  bool loadSceneDescription(std::string const& filename);

  void restartRendering();

  bool screenshot(const bool tonemap);

  void createCameras();
  void createLights();
  void createPictures();

  void appendInstance(std::shared_ptr<sg::Group>& group,
                      std::shared_ptr<sg::Triangles> geometry, 
                      dp::math::Mat44f const& matrix, 
                      std::string const& reference, 
                      unsigned int& idInstance);

  std::shared_ptr<sg::Group> createASSIMP(std::string const& filename);
  std::shared_ptr<sg::Group> traverseScene(const struct aiScene *scene, const unsigned int indexSceneBase, const struct aiNode* node);

  void calculateTangents(std::vector<TriangleAttributes>& attributes, std::vector<unsigned int> const& indices);

  void guiRenderingIndicator(const bool isRendering);

  bool loadString(std::string const& filename, std::string& text);
  bool saveString(std::string const& filename, std::string const& text);
  std::string getDateTime();
  void convertPath(std::string& path);
  void convertPath(char* path);

private:
  GLFWwindow* m_window;
  bool        m_isValid;

  GuiState m_guiState;
  bool     m_isVisibleGUI;

  // Command line options:
  int         m_width;   // Client window size.
  int         m_height;
  int         m_mode;   // Application mode 0 = interactive, 1 = batched benchmark (single shot).
  bool        m_optimize; // Command line option to let the assimp importer optimize the graph (sorts by material).

  // System options:
  int         m_strategy;    // "strategy"
  int         m_devicesMask; // "devicesMask" // Bitmask with enabled devices, default 0xFF for 8 devices. Only the visible ones will be used.
  int         m_light;       // "light"
  int         m_miss;        // "miss"
  std::string m_environment; // "envMap"
  int         m_interop;     // "interop"´// 0 = none all through host, 1 = register texture image, 2 = register pixel buffer
  bool        m_present;     // "present"
  
  bool        m_presentNext;      // (derived)
  double      m_presentAtSecond;  // (derived)
  bool        m_previousComplete; // (derived) // Prevents spurious benchmark prints and image updates.

  // GUI Data representing raytracer settings.
  LensShader m_lensShader;          // "lensShader"
  int2       m_pathLengths;         // "pathLengths"   // min, max
  int2       m_resolution;          // "resolution"    // The actual size of the rendering, independent of the window's client size. (Preparation for final frame rendering.)
  int2       m_tileSize;            // "tileSize"      // Multi-GPU distribution tile size. Must be power-of-two values.
  int        m_samplesSqrt;         // "sampleSqrt"
  float      m_epsilonFactor;       // "epsilonFactor"
  float      m_environmentRotation; // "envRotation"
  float      m_clockFactor;         // "clockFactor"

  std::string m_prefixScreenshot;   // "prefixScreenshot", allows to set a path and the prefix for the screenshot filename. spp, data, time and extension will be appended.
  
  TonemapperGUI m_tonemapperGUI;    // "gamma", "whitePoint", "burnHighlights", "crushBlacks", "saturation", "brightness"
  
  Camera m_camera;                  // "center", "camera"

  float m_mouseSpeedRatio;
  
  Timer m_timer;

  std::map<std::string, KeywordScene> m_mapKeywordScene;

  std::unique_ptr<Rasterizer> m_rasterizer;
  
  std::unique_ptr<Raytracer> m_raytracer; // This is a base class.
  DeviceState                m_state;

  // The scene description:
  // Unique identifiers per host scene node.
  unsigned int m_idGroup;
  unsigned int m_idInstance;
  unsigned int m_idGeometry;

  std::shared_ptr<sg::Group> m_scene; // Root group node of the scene.
  
  std::vector< std::shared_ptr<sg::Triangles> > m_geometries; // All geometries in the scene.

  // For the runtime generated objects, this allows to find geometries with the same type and construction parameters.
  std::map<std::string, unsigned int> m_mapGeometries;

  // For all model file format loaders. Allows instancing of full models in the host side scene graph.
  std::map< std::string, std::shared_ptr<sg::Group> > m_mapGroups;

  std::vector<CameraDefinition> m_cameras;
  std::vector<LightDefinition>  m_lights;
  std::vector<MaterialGUI>      m_materialsGUI;

  // Map of local material names to indices in the m_materialsGUI vector.
  std::map<std::string, int> m_mapMaterialReferences; 

  std::map<std::string, Picture*> m_mapPictures;

  std::vector<unsigned int> m_remappedMeshIndices; 
};

#endif // APPLICATION_H

