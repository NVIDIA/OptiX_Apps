/* 
 * Copyright (c) 2013-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "inc/MaterialMDL.h"

#include <dp/math/Matmnt.h>

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
#include <string>
#include <vector>


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
  KS_LENS_SHADER,
  KS_CENTER,
  KS_CAMERA,
  KS_GAMMA,
  KS_COLOR_BALANCE,
  KS_WHITE_POINT,
  KS_BURN_HIGHLIGHTS,
  KS_CRUSH_BLACKS,
  KS_SATURATION,
  KS_BRIGHTNESS,
  KS_EMISSION,
  KS_EMISSION_MULTIPLIER,
  KS_EMISSION_PROFILE,
  KS_EMISSION_TEXTURE,
  KS_SPOT_ANGLE,
  KS_SPOT_EXPONENT,
  KS_IDENTITY,
  KS_PUSH,
  KS_POP,
  KS_ROTATE,
  KS_SCALE,
  KS_TRANSLATE,
  //KS_PLANE,
  //KS_BOX,
  //KS_SPHERE,
  //KS_TORUS,
  //KS_ASSIMP,
  KS_MDL,
  KS_LIGHT,
  KS_MODEL
};


struct SceneState
{
  void reset()
  {
    matrix    = dp::math::cIdentity44f;
    matrixInv = dp::math::cIdentity44f;

    orientation    = dp::math::Quatf(0.0f, 0.0f, 0.0f, 1.0f);
    orientationInv = dp::math::Quatf(0.0f, 0.0f, 0.0f, 1.0f);

    nameEmission.clear();
    nameProfile.clear();

    colorEmission      = make_float3(0.0f); // Default black
    multiplierEmission = 1.0f;

    spotAngle    = 45.0f;
    spotExponent = 0.0f; // No cosine falloff.
  }

  // Transformation state
  dp::math::Mat44f matrix;
  dp::math::Mat44f matrixInv;
  // The orientation (the pure rotational part of the above matrices).
  dp::math::Quatf  orientation;
  dp::math::Quatf  orientationInv;

  std::string nameEmission; // The filename of the emission texture. Empty when none.
  std::string nameProfile;  // The filename of the IES light profile. Empty when none.

  float3 colorEmission;      // The emission base color.
  float  multiplierEmission; // A multiplier on top of colorEmission to get HDR lights.

  float spotAngle;    // Full cone angle in degrees, means max. 180 degrees is a hemispherical distribution.
  float spotExponent; // Exponent on the cosine of the sotAngle, used to generate intensity falloff from spot cone center to outer angle. Set to 0.0 for no falloff.
};


class Application
{
public:

  Application(GLFWwindow* window, const Options& options);
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
  bool loadSystemDescription(const std::string& filename);
  bool saveSystemDescription();

  //TypeBXDF determineTypeBXDF(const std::string& token) const;
  //TypeEDF  determineTypeEDF(const std::string& token) const;
  bool loadSceneDescription(const std::string& filename);

  void restartRendering();

  bool screenshot(const bool tonemap);

  void createCameras();
  
  int findMaterial(const std::string& reference);

  void appendInstance(std::shared_ptr<sg::Group>& group,
                      std::shared_ptr<sg::Node> geometry, // Baseclass Node to be prepared for different geometric primitives.
                      const dp::math::Mat44f& matrix, 
                      const int indexMaterial,
                      const int indexLight);

  std::shared_ptr<sg::Group> createASSIMP(const std::string& filename);
  std::shared_ptr<sg::Group> traverseScene(const struct aiScene *scene, const unsigned int indexSceneBase, const struct aiNode* node);

  void calculateTangents(std::vector<TriangleAttributes>& attributes, const std::vector<unsigned int>& indices);

  void guiRenderingIndicator(const bool isRendering);

  void createMeshLights();
  void traverseGraph(std::shared_ptr<sg::Node> node, InstanceData& instanceData, float matrix[12]);
  int createMeshLight(const std::shared_ptr<sg::Triangles> geometry, const InstanceData& instanceData, const float matrix[12]);

  bool loadString(const std::string& filename, std::string& text);
  bool saveString(const std::string& filename, const std::string& text);
  std::string getDateTime();
  void convertPath(std::string& path);
  void convertPath(char* path);

  void addSearchPath(const std::string& path);

  bool isEmissiveMaterial(const int indexMaterial) const;

private:
  GLFWwindow* m_window;
  bool        m_isValid;

  GuiState m_guiState;
  bool     m_isVisibleGUI;

  // Command line options:
  int         m_width;    // Client window size.
  int         m_height;
  int         m_mode;     // Application mode 0 = interactive, 1 = batched benchmark (single shot).
  bool        m_optimize; // Command line option to let the assimp importer optimize the graph (sorts by material).

  // System options:
  int         m_strategy;    // "strategy"    // Ignored in this renderer. Always behaves like RS_INTERACTIVE_MULTI_GPU_LOCAL_COPY.
  int         m_maskDevices; // "devicesMask" // Bitmask with enabled devices, default 0x00FFFFFF for max 24 devices. Only the visible ones will be used.
  size_t      m_sizeArena;   // "arenaSize"   // Default size for Arena allocations in mega-bytes.
  int         m_interop;     // "interop"     // 0 = none all through host, 1 = register texture image, 2 = register pixel buffer
  int         m_peerToPeer;  // "peerToPeer   // Bitfield controlling P2P resource sharing:
                                              // Bit 0 = Allow sharing via PCI-E bus. Only share across NVLINK bridges when off (default off)
                                              // Bit 1 = Allow sharing of texture CUarray or CUmipmappedArray data (legacy and MDL) (fast) (default on)
                                              // Bit 2 = Allow sharing of geometry acceleration structures and vertex attributes (slowest) (default off)
                                              // Bit 3 = Allow sharing of spherical environment light texture and CDFs (slow) (default off)
                                              // Bit 4 = Allow sharing of MDL Measured BSDF and their CDFs (slow) (default off)
                                              // Bit 5 = Allow sharing of MDL Lightprofiles and their CDFs (slow) (default off)
  bool        m_present;     // "present"
  bool        m_presentNext;      // (derived)
  double      m_presentAtSecond;  // (derived)
  bool        m_previousComplete; // (derived) // Prevents spurious benchmark prints and image updates.

  // GUI Data representing raytracer settings.
  TypeLens   m_typeLens;            // "lensShader"
  int2       m_pathLengths;         // "pathLengths"   // min, max
  int        m_walkLength;          // "walkLength"    // Number of volume scattering random walk steps until the maximum distance is to try gettting out of the volumes. Minimum 1 for single scattering.
  int2       m_resolution;          // "resolution"    // The actual size of the rendering, independent of the window's client size. (Preparation for final frame rendering.)
  int2       m_tileSize;            // "tileSize"      // Multi-GPU distribution tile size. Must be power-of-two values.
  int        m_samplesSqrt;         // "sampleSqrt"
  float      m_epsilonFactor;       // "epsilonFactor"
  float      m_clockFactor;         // "clockFactor"
  bool       m_useDirectLighting; 

  std::string m_prefixScreenshot;   // "prefixScreenshot", allows to set a path and the prefix for the screenshot filename. spp, data, time and extension will be appended.
  
  TonemapperGUI m_tonemapperGUI;    // "gamma", "whitePoint", "burnHighlights", "crushBlacks", "saturation", "brightness"
  
  Camera m_camera;                  // "center", "camera"

  float m_mouseSpeedRatio;
  
  Timer m_timer;

  std::map<std::string, KeywordScene> m_mapKeywordScene;

  std::unique_ptr<Rasterizer> m_rasterizer;
  
  std::unique_ptr<Raytracer> m_raytracer;

  DeviceState                m_state;

  // The scene description:
  // Unique identifiers per host scene node.
  unsigned int m_idGroup;
  unsigned int m_idInstance;
  unsigned int m_idGeometry;

  std::shared_ptr<sg::Group> m_scene; // Root group node of the scene.
  
  std::vector< std::shared_ptr<sg::Node> > m_geometries; // All geometries in the scene. Baseclass Node to be prepared for different geometric primitives.

  // For the runtime generated objects, this allows to find geometries with the same type and construction parameters.
  std::map<std::string, unsigned int> m_mapGeometries;

  // For all model file format loaders. Allows instancing of full models in the host side scene graph.
  std::map< std::string, std::shared_ptr<sg::Group> > m_mapGroups;

  std::vector<CameraDefinition> m_cameras;
  std::vector<LightGUI>         m_lightsGUI;

  // Map picture file paths to their loaded data on the host. 
  // (This is not including the texture images loaded by MDL, only the ones usded for hardcoded lights.)
  std::map<std::string, Picture*> m_mapPictures;

  // A temporary vector of mesh indices used inside the ASSIMP loader to handle the triangle meshes inside a model.
  std::vector<unsigned int> m_remappedMeshIndices; 

  // MDL material handling.

  // This vector of search paths defines where MDL is searching for *.mdl and resource files.
  // This is set via the "searchPath" option in the system description file.
  // Multiple searchPaths entries are possible and define the search order.
  std::vector<std::string> m_searchPaths;

  // This map stores all declared materials under their reference name. 
  // Only the ones which are referenced inside the scene will be loaded 
  // into the m_materialsMDL vector and tracked inside m_mapReferenceToMaterialIndex.
  std::map<std::string, MaterialDeclaration> m_mapReferenceToDeclaration;

  // This maps a material reference name to an index into the m_materialsMDL vector.
  std::map<std::string, int> m_mapReferenceToMaterialIndex;
  
  // Vector of MaterialMDL pointers, one per unique material reference name inside the scene description.
  std::vector<MaterialMDL*> m_materialsMDL;

  // This tracks which material is selected inside the GUI combo box.
  int m_indexMaterialGUI;
};


#endif // APPLICATION_H

