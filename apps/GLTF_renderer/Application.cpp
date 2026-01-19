/*
 * Copyright (c) 2013-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda/config.h>

#include "Application.h"
#include "CheckMacros.h"
#include "HostKernels.h"
#include "Utils.h"
#include "ConversionArguments.h"

// STB
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <map>

#ifdef __linux__
  #include <dlfcn.h> // RTLD_NOW
#endif

#include "cuda/hit_group_data.h"
#include "cuda/light_definition.h"
#include "cuda/vector_math.h"

#include "Record.h"
#include "Mesh.h"

#include <glm/gtc/matrix_access.hpp>

// CUDA Driver API version of the OpenGL interop header. 
#include <cudaGL.h>

#include <MyAssert.h>

static const char* CUDA_PROGRAMS_PATH = "./GLTF_renderer_core/";

// gltf node's transform
using GltfTransform = std::variant<fastgltf::TRS, fastgltf::math::fmat4x4>;

namespace detail {

  // Fill a MaterialData::Texture.
  template<typename T>
  void parseTextureInfo(const std::vector<cudaTextureObject_t>& samplers,
                        const T&                                textureInfo,
                        MaterialData::Texture&                  texture)
  {
    size_t texCoordIndex = textureInfo.texCoordIndex;

    // KHR_texture_transform extension data.
    float2 scale = make_float2(1.0f);
    float  rotation = 0.0f;
    float2 translation = make_float2(0.0f);

    // Optional KHR_texture_transform extension data.
    if (textureInfo.transform != nullptr)
    {
      scale.x = textureInfo.transform->uvScale[0];
      scale.y = textureInfo.transform->uvScale[1];

      rotation = textureInfo.transform->rotation;

      translation.x = textureInfo.transform->uvOffset[0];
      translation.y = textureInfo.transform->uvOffset[1];

      // KHR_texture_transform can override the texture coordinate index.
      if (textureInfo.transform->texCoordIndex.has_value())
      {
        texCoordIndex = textureInfo.transform->texCoordIndex.value();
      }
    }

    if (NUM_ATTR_TEXCOORDS <= texCoordIndex)
    {
      std::cerr << "ERROR: detail::parseTextureInfo() Maximum supported texture coordinate index exceeded, using 0.\n";
      texCoordIndex = 0; // PERF This means the device code doesn't need to check if the texcoord index is in the valid range!
    }

    MY_ASSERT(0 <= textureInfo.textureIndex && textureInfo.textureIndex < samplers.size());

    texture.index = static_cast<int>(texCoordIndex);
    //texture.angle       = rotation; // For optional GUI only, needed to recalculate sin and cos below.
    texture.object = samplers[textureInfo.textureIndex];
    texture.scale = scale;
    texture.rotation = make_float2(sinf(rotation), cosf(rotation));
    texture.translation = translation;
  }


  /// Convert glTF types to ours.
  dev::PrimitiveType toDevPrimitiveType(fastgltf::PrimitiveType t)
  {
    // TODO rename namespace dev to app (dev sounds like device, confusing)
    switch (t)
    {
      case fastgltf::PrimitiveType::Points:
      return dev::PrimitiveType::Points;
      case fastgltf::PrimitiveType::Triangles:
      return dev::PrimitiveType::Triangles;
      default:
      return dev::PrimitiveType::Undefined;
    }
  }

  const std::string& getDevPrimitiveTypeName(fastgltf::PrimitiveType t)
  {
    static const std::string names[]
    {
      "Points",
      "Lines",
      "LineLoop",
      "LineStrip",
      "Triangles",
      "TriangleStrip",
      "TriangleFan",
    };
    return names[static_cast<int>(t)];
  }

  // Build a glm matrix from translation, rotation, scale. PERF: this is slow.
  auto makeMatrix = [](const glm::vec3& tr, const glm::quat& rot, const glm::vec3& scale)
  {
    glm::mat4 mTranslation = glm::translate(glm::mat4(1.0f), tr);
    glm::mat4 mRotation = glm::toMat4(rot);
    glm::mat4 mScale = glm::scale(glm::mat4(1.0f), scale);
    return mTranslation * mRotation * mScale;
  };

  // Convert gltf transform to GLM.
  glm::mat4x4 toGLMTransform(const GltfTransform& transform)
  {
    glm::mat4x4 mtx;

    // Matrix and TRS values are mutually exclusive according to the spec.
    if (const fastgltf::math::fmat4x4* matrix = std::get_if<fastgltf::math::fmat4x4>(&transform))
    {
      mtx = glm::make_mat4x4(matrix->data());
    }
    else if (const fastgltf::TRS* trs = std::get_if<fastgltf::TRS>(&transform))
    {
      // Warning: The quaternion to mat4x4 conversion here is not correct with all versions of GLM.
      // glTF provides the quaternion as (x, y, z, w), which is the same layout GLM used up to version 0.9.9.8.
      // However, with commit 59ddeb7 (May 2021) the default order was changed to (w, x, y, z).
      // You could either define GLM_FORCE_QUAT_DATA_XYZW to return to the old layout,
      // or you could use the recently added static factory constructor glm::quat::wxyz(w, x, y, z),
      // which guarantees the parameter order.
      // => 
      // Using GLM version 0.9.9.9 (or newer) and glm::quat::wxyz(w, x, y, z).
      // If this is not compiling your glm version is too old!
      const auto translation = glm::make_vec3(trs->translation.data());
      const auto rotation = glm::quat::wxyz(trs->rotation[3], trs->rotation[0], trs->rotation[1], trs->rotation[2]);
      const auto scale = glm::make_vec3(trs->scale.data());
      mtx = makeMatrix(translation, rotation, scale);
    }
    else
    {
      std::cerr << "Missing transform " << __FUNCTION__ << std::endl;
      MY_ASSERT(false);
    }
    return mtx;
  }
}//detail namespace



void Application::initSheenLUT()
{
  // Create the sheen lookup table which is required to weight the sheen sampling.
  m_picSheenLUT = new Picture();

  if (!m_picSheenLUT->load("sheen_lut.hdr", IMAGE_FLAG_2D)) // This frees all images inside an existing Picture.
  {
    delete m_picSheenLUT;
    m_picSheenLUT = nullptr;

    throw std::runtime_error("ERROR: initSheenLUT() Picture::load() failed.");
  }

  // Create a new texture to keep the old texture intact in case anything goes wrong.
  m_texSheenLUT = new Texture(m_allocator);

  m_texSheenLUT->setAddressMode(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP);

  if (!m_texSheenLUT->create(m_picSheenLUT, IMAGE_FLAG_2D | IMAGE_FLAG_SHEEN))
  {
    delete m_texSheenLUT;
    m_texSheenLUT = nullptr;

    throw std::runtime_error("ERROR: initSheenLUT Texture::create() failed.");
  }
}


Application::Application(GLFWwindow* window,
                         const Options& options)
  : m_window(window)
  , m_logger(std::cerr)
{
  m_pathAsset    = options.getFilename();
  m_width        = std::max(1, options.getWidthClient());
  m_height       = std::max(1, options.getHeightClient());
  m_resolution.x = std::max(1, options.getWidthResolution());
  m_resolution.y = std::max(1, options.getHeightResolution());
  m_isDirtyResolution = true;
  m_launches  = options.getLaunches();
  m_interop   = options.getInterop();
  m_punctual  = options.getPunctual();
  m_missID    = options.getMiss();
  m_pathEnv   = options.getEnvironment();

  m_iterations.resize(MAX_LAUNCHES); // The size of this vector must always be greater or equal to m_launches. Just size it once to the maximum.
  
  m_benchmarkValues.resize(SIZE_BENCHMARK_VALUES); // This is a vector of a running average of the the last m_benchmarkCapacity results.

  m_colorEnv[0] = 1.0f;
  m_colorEnv[1] = 1.0f;
  m_colorEnv[2] = 1.0f;
  m_intensityEnv = 1.0f;
  m_rotationEnv[0] = 0.0f;
  m_rotationEnv[1] = 0.0f;
  m_rotationEnv[2] = 0.0f;

  m_bufferHost = nullptr; // Allocated inside updateBuffers() when needed.

#if 1 // Tonemapper defaults
    m_gamma          = 2.2f;
    m_colorBalance   = make_float3(1.0f, 1.0f, 1.0f);
    m_whitePoint     = 1.0f;
    m_burnHighlights = 0.8f;
    m_crushBlacks    = 0.2f;
    m_saturation     = 1.2f;
    m_brightness     = 1.0f;
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

  m_mouseSpeedRatio = 100.0f;
  m_trackball.setSpeedRatio(m_mouseSpeedRatio);

  m_cudaGraphicsResource = nullptr;
  m_sphereRadiusFraction = options.getSphereRadiusFraction();
  
  // Setup ImGui binding.
  ImGui::CreateContext();

  ImGuiIO& io = ImGui::GetIO(); 
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Use Tab and arrow keys to navigate through widgets.
  //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  updateFonts();

#ifdef _WIN32
  // HACK Only enable Multi-Viewport under Windows because of
  // https://github.com/ocornut/imgui/wiki/Multi-Viewports#issues
  // "The feature tends to be broken on Linux/X11 with many window managers.
  //  The feature doesn't work in Wayland."
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport/Platform Windows"
#endif
  io.ConfigWindowsResizeFromEdges      = true; // More consistent window resize behavior, esp. when using multi-viewports.
  io.ConfigWindowsMoveFromTitleBarOnly = true; // Prevent moving the GUI window when inadvertently clicking on an empty space.

  //ImGui::StyleColorsDark(); // default
  //ImGui::StyleColorsLight();
  //ImGui::StyleColorsClassic();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init();

  // This initializes ImGui resources like the font texture.
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  // Do nothing.
  ImGui::EndFrame();

  // This must always be called after each ImGui::EndFrame() when ImGuiConfigFlags_ViewportsEnable is set.
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
  {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    // Platform windows can change the OpenGL context.
    glfwMakeContextCurrent(m_window);
  }

  initCUDA();   // CUDA must be initialized before OpenGL to have the CUDA device UUID/LUID for the OpenGL-CUDA interop check.
  initOpenGL(); // OpenGL must be initialized before OptiX because that determines the OpenGL-CUDA interop mode and generates resources.
  initOptiX();
  initSheenLUT();

  try
  {
    // This uses fastgltf to load the glTF into Application::m_asset.
    // If anything goes wrong, like calling it with an empty asset filename,
    // application running with an empty scene.
    loadGLTF(m_pathAsset);
  }
  catch(std::exception& e)
  {
    std::cerr << "WARNING: Caught exception in the app constructor: " << e.what() << "\n";
  }
  
  initRenderer(true); // First time initialization.
}


void Application::initRenderer(const bool first)
{
  // Print which extensions the asset uses.
  // This is helpful when adding support for new extensions in loadGLTF().
  // Which material extensions are used is determined inside initMaterials() per individual material.
  auto print = [](const char* info, const auto& extensions) {
    std::cout << info;
    for (const auto& x : extensions)
    {
      std::cout << "  " << x << '\n';
    }
    std::cout << "}\n";
  };

  print("extensionsUsed = {\n", m_asset.extensionsUsed);
  // If this would list any extensions which aren't supported, the loadGLTF() above already threw an error.
  print("extensionsRequired = {\n", m_asset.extensionsRequired);

  // Initialize the GLTF host and device resource vectors (sizes) upfront 
  // to match the GLTF indices used in various asset objects.
  initImages();
  initTextures();
  initMaterials();
  initMeshes();
  initLights(first); // Copy the data from the m_asset.lights. This is not the device side representation. 
  initCameras();     // This also creates a default camera when there isn't one inside the asset.
  initNodes();       // This builds a vector of dev::Node which are used to track animations.
  initSkins();       // This builds a vector of dev::Skin.
  initAnimations();

  // First time scene initialization, creating or using the default scene.
  initScene(-1);
  initSceneExtent();  // find scene extents, for the spehres' radius and other scene params.

  // Initialize all acceleration structures, pipeline, shader binding table.
  updateScene(true       /*rebuild*/);
  buildInstanceAccel(true/*rebuild*/);

  if (first)
  {
    initPipeline();
    initSBT();
    initLaunchParameters();
  }
  else // Drag-and-drop of another asset.
  {
    updateSBT();
    updateLaunchParameters();
  }

  m_isDirtyScene = false;

  // In case there was no camera inside the asset, this places and centers 
  // the added default camera according to the selected scene.
  // This requires that the top-level IAS is already built to have the scene extent.
  initTrackball();
}


Application::~Application()
{
  try
  {
    cleanup();

    delete m_allocator; // This frees all CUDA allocations done with the arena allocator!

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
  }
  catch (const std::exception& e)
  {
    std::cerr << "ERROR: Caught exception (in the dtor!): " << e.what() << "\n";
  }
}


// Arena version of cudaMalloc(), but asynchronous!
CUdeviceptr Application::memAlloc(const size_t size, const size_t alignment, const cuda::Usage usage)
{
  return m_allocator->alloc(size, alignment, usage);
}

// Arena version of cudaFree(), but asynchronous!
void Application::memFree(const CUdeviceptr ptr)
{
  m_allocator->free(ptr);
}


void Application::updateProjectionMatrix()
{
  // No need to set this when using shaders only.
  //glMatrixMode(GL_PROJECTION);
  //glLoadIdentity();
  //glOrtho(0.0, GLdouble(m_width), 0.0, GLdouble(m_height), -1.0, 1.0);

  //glMatrixMode(GL_MODELVIEW);

  // Full projection matrix calculation:
  //const float l = 0.0f;
  const float r = float(m_width);
  //const float b = 0.0f;
  const float t = float(m_height);
  //const float n = -1.0f;
  //const float f =  1.0;

  //const float m00 =  2.0f / (r - l);   // == 2.0f / r with l == 0.0f
  //const float m11 =  2.0f / (t - b);   // == 2.0f / t with b == 0.0f
  //const float m22 = -2.0f / (f - n);   // Always -1.0f with f == 1.0f and n == -1.0f
  //const float tx = -(r + l) / (r - l); // Always -1.0f with l == 0.0f
  //const float ty = -(t + b) / (t - b); // Always -1.0f with b == 0.0f 
  //const float tz = -(f + n) / (f - n); // Always  0.0f with f = -n

  // Row-major layout, needs transpose in glUniformMatrix4fv.
  //const float projection[16] =
  //{
  //  m00,  0.0f, 0.0f, tx,
  //  0.0f, m11,  0.0f, ty,
  //  0.0f, 0.0f, m22,  tz,
  //  0.0f, 0.0f, 0.0f, 1.0f
  //};

  // Optimized version and colum-major layout:
  const float projection[16] =
  {
     2.0f / r, 0.0f,     0.0f, 0.0f,
     0.0f,     2.0f / t, 0.0f, 0.0f,
     0.0f,     0.0f,    -1.0f, 0.0f,
    -1.0f,    -1.0f,     0.0f, 1.0f
  };
  
  glUseProgram(m_glslProgram);
  glUniformMatrix4fv(m_locProjection, 1, GL_FALSE, projection); // Column-major memory layout, no transpose.
  glUseProgram(0);
}


void Application::updateVertexAttributes()
{
  // This routine calculates the vertex attributes for the diplay routine.
  // It calculates screen space vertex coordinates to display the full rendered image 
  // in the correct aspect ratio independently of the window client size. 
  // The image gets scaled down when it's bigger than the client window.

  // The final screen space vertex coordinates for the texture blit.
  float x0;
  float y0;
  float x1;
  float y1;

  // This routine picks the required filtering mode for this texture.
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
  
  if (m_resolution.x <= m_width && m_resolution.y <= m_height)
  {
    // Texture fits into viewport without scaling.
    // Calculate the amount of cleared border pixels.
    int w1 = m_width  - m_resolution.x;
    int h1 = m_height - m_resolution.y;
    // Halve the border size to get the lower left offset 
    int w0 = w1 >> 1;
    int h0 = h1 >> 1;
    // Subtract from the full border to get the right top offset.
    w1 -= w0;
    h1 -= h0;
    // Calculate the texture blit screen space coordinates.
    x0 = float(w0);
    y0 = float(h0);
    x1 = float(m_width  - w1);
    y1 = float(m_height - h1);

    // Fill the background with black to indicate that all pixels are visible without scaling.
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    // Use nearest filtering to display the pixels exactly.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  }
  else
  {
    // Texture needs to be scaled down to fit into client window.
    // Check which extent defines the necessary scaling factor.
    const float wC = float(m_width);
    const float hC = float(m_height);
    const float wR = float(m_resolution.x);
    const float hR = float(m_resolution.y);

    const float scale = std::min(wC / wR, hC / hR);

    const float swR = scale * wR;
    const float shR = scale * hR;

    x0 = 0.5f * (wC - swR);
    y0 = 0.5f * (hC - shR);
    x1 = x0 + swR;
    y1 = y0 + shR;

    // Render surrounding pixels in dark red to indicate that the image is scaled down.
    glClearColor(0.2f, 0.0f, 0.0f, 0.0f); 

    // Use linear filtering to smooth the downscaling.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  }

  // Update the vertex attributes with the new texture blit screen space coordinates.
  const float attributes[16] = 
  {
    // vertex2f
    x0, y0,
    x1, y0,      
    x1, y1,      
    x0, y1,
    // texcoord2f
    0.0f, 0.0f,
    1.0f, 0.0f,
    1.0f, 1.0f,
    0.0f, 1.0f
  };

  glBindBuffer(GL_ARRAY_BUFFER, m_vboAttributes);
  glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) sizeof(float) * 16, (GLvoid const*) attributes, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0); // PERF It should be faster to keep them bound.
}


// Input is client window relative mouse coordinate with origin at top-left.
float2 Application::getPickingCoordinate(const int x, const int y)
{
  // The final screen space display rectangle coordinates.
  float x0;
  float y0;
  float x1;
  float y1;

  if (m_resolution.x <= m_width && m_resolution.y <= m_height)
  {
    // Texture fits into viewport without scaling.
    // Calculate the amount of cleared border pixels.
    int w1 = m_width  - m_resolution.x;
    int h1 = m_height - m_resolution.y;
    // Halve the border size to get the lower left offset 
    int w0 = w1 >> 1;
    int h0 = h1 >> 1;
    // Subtract from the full border to get the right top offset.
    w1 -= w0;
    h1 -= h0;
    // Calculate the texture blit screen space coordinates.
    x0 = float(w0);
    y0 = float(h0);
    x1 = float(m_width  - w1);
    y1 = float(m_height - h1);
  }
  else // Resolution bigger than client area, image needs to be scaled down.
  {
    // Check which extent defines the necessary scaling factor.
    const float wC = float(m_width);
    const float hC = float(m_height);
    const float wR = float(m_resolution.x);
    const float hR = float(m_resolution.y);

    const float scale = std::min(wC / wR, hC / hR);

    const float swR = scale * wR;
    const float shR = scale * hR;

    x0 = 0.5f * (wC - swR);
    y0 = 0.5f * (hC - shR);
    x1 = x0 + swR;
    y1 = y0 + shR;
  }

  // Pick in the center of the screen pixel.
  float xp = float(x) + 0.5f;
  float yp = float(y) + 0.5f;

  // If the mouse coordinate is inside the display rectangle
  // return a picking coordinate normalized to the rendering resolution.
  if (x0 <= xp && xp <= x1 && y0 <= yp && yp <= y1)
  {
    xp = float(m_resolution.x) *         ((xp - x0) / (x1 - x0));
    yp = float(m_resolution.y) * (1.0f - ((yp - y0) / (y1 - y0)));
    
    return make_float2(xp, yp); // Picking coordinate in resolution (launch dimension) rectangle, bottom-left origin.
  }

  return make_float2(-1.0f, -1.0f); // No picking.
}


int Application::getBenchmarkMode() const
{
  return m_benchmarkMode;
}


void Application::setBenchmarkValue(const float value)
{
  if (m_benchmarkMode != OFF)
  {
    m_benchmarkValues[m_benchmarkCell++] = value;                       // Set value and increment cell index.
    m_benchmarkEntries = std::max(m_benchmarkEntries, m_benchmarkCell); // Number of valid entries insde m_benchmarkValues.
    m_benchmarkCell %= SIZE_BENCHMARK_VALUES;                           // Next value index modulo benchmark values capacity.
  }
}


void Application::reshape(int width, int height)
{
  // Zero sized interop buffers are not allowed in OptiX.
  if ((width != 0 && height != 0) && (m_width != width || m_height != height))
  {
    m_width  = width;
    m_height = height;

    glViewport(0, 0, m_width, m_height);

    updateProjectionMatrix();
    updateVertexAttributes();
  }
}


void Application::drop(const int countPaths, const char* paths[])
{
  // DEBUG
  //std::cout << "drop(): count = " << countPaths << '\n'; 
  //for (int i = 0; i < countPaths; ++i)
  //{
  //  std::cout << paths[i] << '\n';
  //}
  //std::cout << std::endl;

  // Check if there is any *.hdr file inside the dropped paths.
  for (int i = 0; i < countPaths; ++i)
  {
    std::string strPath(paths[i]);
    utils::convertPath(strPath);

    std::filesystem::path path(strPath);
    std::filesystem::path ext = path.extension();

    if (ext.string() == std::string(".hdr"))
    {
      // The first found *.hdr file is the current environment light. Nothing changes.
      if (m_pathEnv == strPath)
      {
        std::cerr << "WARNING: Environment light " << strPath << " already used.\n";
        return;
      }
      // Exchanging the light type in m_lightDefinitions[0] only works when it's the same type,
      // because the pipeline miss program is setup for that case.
      if (m_missID != 2)
      {
        std::cerr << "WARNING: Environment texture exchange requires spherical environment light. Use command line without --miss (-m) option or set it to 2.\n";
        return;
      }

      std::cout << "drop() Replacing environment light with image "<< strPath << '\n';

      CUDA_CHECK( cudaDeviceSynchronize() ); // Wait until all rendering is finished before deleting and existing environment light.
      
      // Create the new environment light.
      m_pathEnv = strPath;

      if (m_picEnv == nullptr)
      {
        m_picEnv = new Picture();
      }

      if (!m_picEnv->load(m_pathEnv, IMAGE_FLAG_2D)) // This frees all images inside an existing Picture.
      {
        return;
      }

      // Create a new texture to keep the old texture intact in case anything goes wrong.
      Texture *texture = new Texture(m_allocator);

      if (!texture->create(m_picEnv, IMAGE_FLAG_2D | IMAGE_FLAG_ENV))
      {
        delete texture;
        return;
      }

      if (m_texEnv != nullptr)
      {
        delete m_texEnv;
        //m_texEnv = nullptr;
      }

      m_texEnv = texture;

      // Replace the spherical environment light in entry 0.
      LightDefinition& light = m_lightDefinitions[0];
      MY_ASSERT(light.typeLight == TYPE_LIGHT_ENV_SPHERE);

      // Only change the values affected by the new texture.
      // Not reseting the orientation matrices means the 
      // m_environmentRotation values from the GUI stay intact.
      light.cdfU            = m_texEnv->getCDF_U(); 
      light.cdfV            = m_texEnv->getCDF_V();
      light.textureEmission = m_texEnv->getTextureObject();
      light.emission        = make_float3(1.0f);
      light.invIntegral     = 1.0f / m_texEnv->getIntegral();
      light.width           = m_texEnv->getWidth(); 
      light.height          = m_texEnv->getHeight();

      // Only update the first light definition inside the device buffer.
      CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_d_lightDefinitions), m_lightDefinitions.data(), sizeof(LightDefinition), cudaMemcpyHostToDevice) );
      
      m_launchParameters.iteration = 0; // Restart accumulation when any launch parameter changes.

      return; // Only load the first *.hdr file inside the dropped list.
    }

    if (ext.string() == std::string(".gltf") ||
        ext.string() == std::string(".glb"))
    {
      // The first found *.gltf or *.glb.
      std::filesystem::path pathDrop(strPath);

      if (m_pathAsset == pathDrop)
      {
        std::cerr << "WARNING: Asset " << strPath << " already used.\n";
        return;
      }
     
      try
      {
        // Load the dropped asset. 
        // If this fails, it throws an exception and the previous asset stays intact.
        loadGLTF(pathDrop); 
        // Indicate that the loadGLTF() succeeded.
        m_isDirtyScene = true; 
      }
      catch(std::exception& e)
      {
        std::cerr << "WARNING: Caught exception: (Application::drop()) " << e.what() << "\n";
      }
      if (m_isDirtyScene)
      {
        // Initialize all resources from the new asset, but only update the pipeline, SBT, and launch parameters.
        // Note that the CUDA allocations for the IAS aren't touched when switching scenes. They are kept since they only grow.
        // If any of this fails, that also throws exceptions, but these are fatal and exit the application.
        initRenderer(false);

        m_pathAsset = pathDrop; // Remember the current asset path.

        return; // Only load the first *.gltf or *.glb file inside the dropped list.
      }
    }
  }
}


void Application::guiNewFrame()
{
  if (utils::getFontScale() != m_fontScale)
  {
    updateFonts();
  }
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
}


void Application::guiReferenceManual()
{
  ImGui::ShowDemoWindow();
}


void Application::guiRender()
{
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  ImGuiIO& io = ImGui::GetIO();
  // This must always be called after each ImGui::EndFrame() when ImGuiConfigFlags_ViewportsEnable is set.
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
  {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    // Platform windows can change the OpenGL context.
    glfwMakeContextCurrent(m_window);
  }
}




void Application::initOpenGL()
{
  // Find out which device is running the OpenGL implementation to be able to allocate the PBO peer-to-peer staging buffer on the same device.
  // Needs these OpenGL extensions: 
  // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_external_objects.txt
  // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_external_objects_win32.txt
  // and on CUDA side the CUDA 10.0 Driver API function cuDeviceGetLuid().
  // While the extensions are named EXT_external_objects, the enums and functions are found under name string EXT_memory_object!
  if (GLEW_EXT_memory_object)
  {
    // LUID 
    // "The devices in use by the current context may also be identified by an (LUID, node) pair.
    //  To determine the LUID of the current context, call GetUnsignedBytev with <pname> set to DEVICE_LUID_EXT and <data> set to point to an array of LUID_SIZE_EXT unsigned bytes.
    //  Following the call, <data> can be cast to a pointer to an LUID object that will be equal to the locally unique identifier 
    //  of an IDXGIAdapter1 object corresponding to the adapter used by the current context.
    //  To identify which individual devices within an adapter are used by the current context, call GetIntegerv with <pname> set to DEVICE_NODE_MASK_EXT.
    //  A bitfield is returned with one bit set for each device node used by the current context.
    //  The bits set will be subset of those available on a Direct3D 12 device created on an adapter with the same LUID as the current context."
    if (GLEW_EXT_memory_object_win32) // LUID
    {
      // LUID only works under Windows and only in WDDM mode, not in TCC mode!
      // Get the LUID and node mask from the CUDA device.
      char cudaDeviceLUID[8];
      unsigned int cudaNodeMask = 0;
      
      memset(cudaDeviceLUID, 0, 8);
      CU_CHECK( cuDeviceGetLuid(cudaDeviceLUID, &cudaNodeMask, m_cudaDevice) ); // This means initCUDA() must run before initOpenGL().

      // Now compare that with the OpenGL device.
      GLubyte glDeviceLUID[GL_LUID_SIZE_EXT]; // 8 bytes identifier.
      GLint   glNodeMask = 0;                 // Node mask used together with the LUID to identify OpenGL device uniquely.

      // It is not expected that a single context will be associated with multiple DXGI adapters, so only one LUID is returned.
      memset(glDeviceLUID, 0, GL_LUID_SIZE_EXT);
      glGetUnsignedBytevEXT(GL_DEVICE_LUID_EXT, glDeviceLUID);
      glGetIntegerv(GL_DEVICE_NODE_MASK_EXT, &glNodeMask);

      if (!utils::matchLUID(cudaDeviceLUID,
                            cudaNodeMask,
                            reinterpret_cast<const char*>(glDeviceLUID),
                            glNodeMask))
      {
        // The CUDA and OpenGL devices do not match, there is no interop possible!
        std::cerr << "WARNING: OpenGL-CUDA interop disabled, LUID mismatch.\n";
        m_interop = INTEROP_OFF;
      }
    }
    else // UUID
    {
      // UUID works under Windows and Linux.
      CUuuid cudaDeviceUUID;

      memset(&cudaDeviceUUID, 0, 16);
      CU_CHECK( cuDeviceGetUuid(&cudaDeviceUUID, m_cudaDevice) ); // This means initCUDA() must run before initOpenGL().

      GLint numDevices = 0; // Number of OpenGL devices. Normally 1, unless multicast is enabled.

      // To determine which devices are used by the current context, first call GetIntegerv with <pname> set to NUM_DEVICE_UUIDS_EXT, 
      // then call GetUnsignedBytei_vEXT with <target> set to DEVICE_UUID_EXT, <index> set to a value in the range [0, <number of device UUIDs>),
      // and <data> set to point to an array of UUID_SIZE_EXT unsigned bytes. 
      glGetIntegerv(GL_NUM_DEVICE_UUIDS_EXT, &numDevices);
    
      int deviceMatch = -1;
      for (GLint i = 0; i < numDevices; ++i)
      {
        GLubyte glDeviceUUID[GL_UUID_SIZE_EXT];  // 16 bytes identifier. This example only supports one device but check up to 8 device in a machine.

        memset(glDeviceUUID, 0, GL_UUID_SIZE_EXT);
        glGetUnsignedBytei_vEXT(GL_DEVICE_UUID_EXT, i, glDeviceUUID);

        if (utils::matchUUID(cudaDeviceUUID, reinterpret_cast<const char*>(glDeviceUUID)))
        {
          deviceMatch = i;
          break;
        }
      }
      if (deviceMatch == -1)
      {
        // The CUDA and OpenGL devices do not match, there is no interop possible!
        std::cerr << "WARNING: OpenGL-CUDA interop disabled, UUID mismatch.\n";
        m_interop = INTEROP_OFF;
      }
    }
  }

  // Report which OpenGL-CUDA interop mode is used.
  switch (m_interop) 
  {
    case INTEROP_OFF:
    default:
      std::cout << "OpenGL-CUDA interop OFF\n";
      break;
    case INTEROP_PBO:
      std::cout << "OpenGL-CUDA interop PBO\n";
      break;
    case INTEROP_TEX:
      std::cout << "OpenGL-CUDA interop TEX\n";
      break;
    case INTEROP_IMG:
      std::cout << "OpenGL-CUDA interop IMG\n";
      break;
  }

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

  glViewport(0, 0, m_width, m_height);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // glPixelStorei(GL_UNPACK_ALIGNMENT, 4); // default, works for BGRA8, RGBA16F, and RGBA32F.

  glDisable(GL_CULL_FACE);  // default
  glDisable(GL_DEPTH_TEST); // default

  glGenTextures(1, &m_hdrTexture);
  MY_ASSERT(m_hdrTexture != 0);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glBindTexture(GL_TEXTURE_2D, 0);

  // For all interop modes, updateBuffers() resizes m_hdrTexture before the first render() call and registers the resource as needed.
  switch (m_interop) 
  {
    // The "enum InteropMode" declaration documents what these OpenGL-CUDA interop modes do.
    case INTEROP_OFF:
    case INTEROP_TEX:
    case INTEROP_IMG:
    default:
      // Nothing else to initialize on OpenGL side when interop is OFF, TEX, or IMG.
      break;

    case INTEROP_PBO:
      glGenBuffers(1, &m_pbo); // PBO for OpenGL-CUDA interop.
      MY_ASSERT(m_pbo != 0); 
      // First time initialization of the PBO size happens in updateBuffers().
      break;
  }
      
  initGLSL();

  // This initialization is just to generate the vertex buffer objects and bind the VertexAttribPointers.
  // Two hardcoded triangles in the viewport size projection coordinate system with 2D texture coordinates.
  const float attributes[16] = 
  {
    // vertex2f,   
    0.0f, 0.0f,
    1.0,  0.0f,
    1.0,  1.0,
    0.0f, 1.0,
    //texcoord2f
    0.0f, 0.0f,
    1.0f, 0.0f,
    1.0f, 1.0f,
    0.0f, 1.0f
  };

  const unsigned int indices[6] = 
  {
    0, 1, 2, 
    2, 3, 0
  };

  glGenBuffers(1, &m_vboAttributes);
  MY_ASSERT(m_vboAttributes != 0);

  glGenBuffers(1, &m_vboIndices);
  MY_ASSERT(m_vboIndices != 0);

  // Setup the vertex arrays from the vertex attributes.
  glBindBuffer(GL_ARRAY_BUFFER, m_vboAttributes);
  glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) sizeof(float) * 16, (GLvoid const*) attributes, GL_DYNAMIC_DRAW);
  // This requires a bound array buffer!
  glVertexAttribPointer(m_locAttrPosition, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (GLvoid*) 0);
  glVertexAttribPointer(m_locAttrTexCoord, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (GLvoid*) (sizeof(float) * 8));
  glBindBuffer(GL_ARRAY_BUFFER, 0); // PERF It should be faster to keep these buffers bound.

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vboIndices);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr) sizeof(unsigned int) * 6, (const GLvoid*) indices, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); // PERF It should be faster to keep these buffers bound.

  // Synchronize data with the current values.
  updateProjectionMatrix();
  updateVertexAttributes();
}


OptixResult Application::initOptiXFunctionTable()
{
#ifdef _WIN32
  void* handle = utils::optixLoadWindowsDll();
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


void Application::initCUDA()
{
  utils::getSystemInformation();

  cudaError_t cudaErr = cudaFree(0); // Creates a CUDA context.
  if (cudaErr != cudaSuccess)
  {
    std::cerr << "ERROR: initCUDA() cudaFree(0) failed: " << cudaErr << '\n';
    throw std::runtime_error("initCUDA() cudaFree(0) failed");
  }

  // Get the CUdevice handle from the CUDA device ordinal.
  // This single-GPU example uses the first visible CUDA device ordinal.
  // Use the environment variable CUDA_VISIBLE_DEVICES to control which installed device is the first visible one.
  // Note that OpenGL interop is only possible of that CUDA device also runs the NVIDIA OpenGL implementation.
  // That is checked in initOpenGL() with this m_cudaDevice when m_interop != INTEROP_OFF.
  CU_CHECK( cuDeviceGet(&m_cudaDevice, 0) );

  CUresult cuRes = cuCtxGetCurrent(&m_cudaContext);
  if (cuRes != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initCUDA() cuCtxGetCurrent() failed: " << cuRes << '\n';
    throw std::runtime_error("initCUDA() cuCtxGetCurrent() failed");
  }

  cudaErr = cudaStreamCreate(&m_cudaStream);
  if (cudaErr != cudaSuccess)
  {
    std::cerr << "ERROR: initCUDA() cudaStreamCreate() failed: " << cudaErr << '\n';
    throw std::runtime_error("initCUDA() cudaStreamCreate() failed");
  }

  // The ArenaAllocator gets the default Arena size in bytes.
  m_allocator = new cuda::ArenaAllocator(m_sizeArena * 1024 * 1024);
}


void Application::initOptiX()
{
  OptixResult res = initOptiXFunctionTable();
  if (res != OPTIX_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() initOptiXFunctionTable() failed: " << res << '\n';
    throw std::runtime_error("initOptiX() initOptiXFunctionTable() failed");
  }

  OptixDeviceContextOptions options = {};

  options.logCallbackFunction = &Logger::callback;
  options.logCallbackData     = &m_logger;
  options.logCallbackLevel    = 3; // Keep at warning level to suppress the disk cache messages.
#ifndef NDEBUG
  // PERF This incurs significant performance cost and should only be done during development!
  //options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

  res = m_api.optixDeviceContextCreate(m_cudaContext, &options, &m_optixContext);
  if (res != OPTIX_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() optixDeviceContextCreate() failed: " << res << '\n';
    throw std::runtime_error("initOptiX() optixDeviceContextCreate() failed");
  }

  unsigned int numBits = 0;
  OPTIX_CHECK( m_api.optixDeviceContextGetProperty(m_optixContext, OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK, &numBits, sizeof(unsigned int)) );
  MY_ASSERT(numBits != 0);
  m_visibilityMask = (1u << numBits) - 1u;
}


void Application::updateBuffers()
{
  // Set the render resolution.
  m_launchParameters.resolution = m_resolution;

  const size_t numElementsResolution = size_t(m_resolution.x) * size_t(m_resolution.y);

  // Always resize the host output buffer.
  delete[] m_bufferHost;
  m_bufferHost = new float4[numElementsResolution];

  switch (m_interop)
  {
  case INTEROP_OFF:
  default:
    // Resize the native device buffer.
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferAccum)) );
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.bufferAccum), numElementsResolution * sizeof(float4)) );

    // Update the display texture size.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_resolution.x, (GLsizei) m_resolution.y, 0, GL_RGBA, GL_FLOAT, (GLvoid*) m_bufferHost); // RGBA32F
    break;

  case INTEROP_PBO:
    // Resize the OpenGL PBO.
    if (m_cudaGraphicsResource != nullptr)
    {
      CU_CHECK( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
    }
    // Buffer size must be > 0 or OptiX can't create a buffer from it.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, numElementsResolution * sizeof(float) * 4, (void*) 0, GL_DYNAMIC_DRAW); // RGBA32F from byte offset 0 in the pixel unpack buffer.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glFinish(); // Synchronize with following CUDA operations.
    // Keep the PBO buffer registered to only call the faster Map/Unmap around the launches.
    CU_CHECK( cuGraphicsGLRegisterBuffer(&m_cudaGraphicsResource, m_pbo, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) ); 

    // Update the display texture size.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_resolution.x, (GLsizei) m_resolution.y, 0, GL_RGBA, GL_FLOAT, (GLvoid*) m_bufferHost); // RGBA32F
    break;

  case INTEROP_TEX:
    // Resize the native device buffer.
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferAccum)) );
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.bufferAccum), numElementsResolution * sizeof(float4)) );

    if (m_cudaGraphicsResource != nullptr)
    {
      CU_CHECK( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
    }
    // Update the display texture size.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_resolution.x, (GLsizei) m_resolution.y, 0, GL_RGBA, GL_FLOAT, (GLvoid*) m_bufferHost); // RGBA32F
    glFinish(); // Synchronize with following CUDA operations.
    // Keep the texture image registered to only call the faster Map/Unmap around the launches.
    CU_CHECK( cuGraphicsGLRegisterImage(&m_cudaGraphicsResource, m_hdrTexture, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) );
    break;

  case INTEROP_IMG:
    if (m_cudaGraphicsResource != nullptr)
    {
      CU_CHECK( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
    }
    // Update the display texture size.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei) m_resolution.x, (GLsizei) m_resolution.y, 0, GL_RGBA, GL_FLOAT, (GLvoid*) m_bufferHost); // RGBA32F
    glFinish(); // Synchronize with following CUDA operations.
    // Keep the texture image registered.
    CU_CHECK( cuGraphicsGLRegisterImage(&m_cudaGraphicsResource, m_hdrTexture, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST) ); // surface object read/write.
    break;
  }
}


// This handles everything related to scene changes, SRT animation, morphing, skinning.
void Application::updateRenderGraph()
{
  // The scene has been changed inside the GUI. 
  // Rebuild the IAS and the SBT and update the launch parameters.
  if (m_isDirtyScene)
  {
    updateScene(true);        // Rebuild the m_instances vector from instances reachable by the current scene.
    buildInstanceAccel(true); // Rebuild the top-level IAS.
    updateSBT();              // Rebuild the hit records according to the m_instances of the current scene.
    updateLaunchParameters(); // This sets the root m_ias and restarts the accumulation.
 
    m_isDirtyScene = false;
  }
  else if ((m_isPlaying || m_isScrubbing) && m_isAnimated && m_launchParameters.picking.x < 0.0f) // Do not animate while picking.
  {
    if (updateAnimations())
    {
      // update, don't rebuild
      updateScene(false);        // Update the node transforms, morphed and skinned meshes.
      buildInstanceAccel(false); // This updates the top-level IAS with the new matrices.
      //updateSBT();             // No SBT hit record data changes when only updating everything.
      updateLaunchParameters();  // This sets the root m_ias (shouldn't have changed on update) and restarts the accumulation.
    }

    m_isScrubbing = false; // Scrubbing the key frame is a one-shot operation.
  }
}


bool Application::render()
{
  bool repaint = false;

  updateRenderGraph(); // Handle all changes which affect the OptiX render graph.

  if (m_isDirtyLights)
  {
    updateLights();
    m_isDirtyLights = false;
  }

  if (m_cameras[m_indexCamera].getIsDirty() || m_isDirtyResolution)
  {
    updateCamera();
  }

  if (m_isDirtyResolution)
  {
    updateBuffers();
    updateVertexAttributes(); // Calculate new display coordinates when resolution changes.
    m_isDirtyResolution = false;
  }

  switch (m_interop)
  {
    case INTEROP_PBO:
    {
    // INTEROP_PBO renders directly into the linear OpenGL PBO buffer. Map/UnmapResource around optixLaunch calls.
      size_t size = 0;

      CU_CHECK( cuGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
      CU_CHECK( cuGraphicsResourceGetMappedPointer(reinterpret_cast<CUdeviceptr*>(&m_launchParameters.bufferAccum), &size, m_cudaGraphicsResource) ); // The pointer can change on every map!
      MY_ASSERT(m_launchParameters.resolution.x * m_launchParameters.resolution.y * sizeof(float4) <= size);
    }
    break;
    
    case INTEROP_IMG:
    {
      CUarray dstArray = nullptr;

      // Map the texture image surface directly.
      CU_CHECK( cuGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream )); // This is an implicit cuSynchronizeStream().
      CU_CHECK( cuGraphicsSubResourceGetMappedArray(&dstArray, m_cudaGraphicsResource, 0, 0) ); // arrayIndex = 0, mipLevel = 0

      CUDA_RESOURCE_DESC surfDesc{};

      surfDesc.resType = CU_RESOURCE_TYPE_ARRAY;
      surfDesc.res.array.hArray = dstArray;

      CU_CHECK( cuSurfObjectCreate(&m_launchParameters.surface, &surfDesc) );
      break;
    }
  }

  // Update all launch parameters on the device.
  CUDA_CHECK( cudaMemcpyAsync(reinterpret_cast<void*>(m_d_launchParameters), &m_launchParameters, sizeof(LaunchParameters), cudaMemcpyHostToDevice, m_cudaStream) );

  if (0.0f <= m_launchParameters.picking.x)
  {
    //
    // MATERIAL INDEX PICKING
    //
    OPTIX_CHECK( m_api.optixLaunch(m_pipeline, m_cudaStream, reinterpret_cast<CUdeviceptr>(m_d_launchParameters), sizeof(LaunchParameters), &m_sbt, 1, 1, 1) );
    
    m_launchParameters.picking.x = -1.0f; // Disable picking again.

    int32_t indexMaterial = -1;
    CUDA_CHECK( cudaMemcpy((void*) &indexMaterial, (const void*) m_launchParameters.bufferPicking, sizeof(int32_t), cudaMemcpyDeviceToHost) );
    if (0 <= indexMaterial) // Negative means missed all geometry.
    {
      m_indexMaterial = size_t(indexMaterial);
    }
    // repaint == false here! No need to update the rendered image when only picking.
  }
  else
  {
    //
    // RENDERING
    //
    unsigned int iteration = m_launchParameters.iteration;

    if (m_benchmarkMode == SAMPLES_PER_SECOND)
    {
      CUDA_CHECK( cudaDeviceSynchronize() );
      utils::Timer tLaunches;

      for (int i = 0; i < m_launches; ++i)
      {
        // Fill the vector with the iteration indices for the next m_launches.
        m_iterations[i] = iteration++;
        // Only update the iteration from the fixed vector every sub-frame.
        // This makes sure that the asynchronous copy finds the right data on the host when it's executed.
        CUDA_CHECK( cudaMemcpyAsync(reinterpret_cast<void*>(&m_d_launchParameters->iteration), &m_iterations[i], sizeof(unsigned int), cudaMemcpyHostToDevice, m_cudaStream) );
        OPTIX_CHECK( m_api.optixLaunch(m_pipeline, m_cudaStream, reinterpret_cast<CUdeviceptr>(m_d_launchParameters), sizeof(LaunchParameters), &m_sbt, m_resolution.x, m_resolution.y, 1) );
      }

      CUDA_CHECK( cudaDeviceSynchronize() ); // Wait until all kernels finished.

      const float milliseconds = tLaunches.getElapsedMilliseconds();
      const float sps = m_launches * 1000.0f / milliseconds;
      //std::cout << sps << " samples per second (" << m_launches << " launches in " << milliseconds << " ms)\n";

      setBenchmarkValue(sps);
    }
    else
    {
      for (int i = 0; i < m_launches; ++i)
      {
        m_iterations[i] = iteration++; // See comments above.
        CUDA_CHECK( cudaMemcpyAsync(reinterpret_cast<void*>(&m_d_launchParameters->iteration), &m_iterations[i], sizeof(unsigned int), cudaMemcpyHostToDevice, m_cudaStream) );
        OPTIX_CHECK( m_api.optixLaunch(m_pipeline, m_cudaStream, reinterpret_cast<CUdeviceptr>(m_d_launchParameters), sizeof(LaunchParameters), &m_sbt, m_resolution.x, m_resolution.y, 1) );
      }
      CUDA_CHECK( cudaDeviceSynchronize() ); // Wait for all kernels to have finished.
    }

    m_launchParameters.iteration += m_launches; // Skip the number of rendered sub frames inside the host launch parameters.
  
    repaint = true; // Indicate that there is a new image.
  }

  switch (m_interop)
  {
    case INTEROP_PBO:
      CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
      break;
    
    case INTEROP_IMG:
      CU_CHECK( cuSurfObjectDestroy(m_launchParameters.surface) );
      CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
      break;
  }

  return repaint;
}

void Application::display()
{
  glClear(GL_COLOR_BUFFER_BIT); // PERF Do not do this for benchmarks!

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_hdrTexture);

  glBindBuffer(GL_ARRAY_BUFFER, m_vboAttributes);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vboIndices);

  glEnableVertexAttribArray(m_locAttrPosition);
  glEnableVertexAttribArray(m_locAttrTexCoord);

  glUseProgram(m_glslProgram);
  
  glDrawElements(GL_TRIANGLES, (GLsizei) 6, GL_UNSIGNED_INT, (const GLvoid*) 0);

  glUseProgram(0);

  glDisableVertexAttribArray(m_locAttrPosition);
  glDisableVertexAttribArray(m_locAttrTexCoord);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}


void Application::checkInfoLog(const char* /* msg */, GLuint object)
{
  GLint  maxLength;
  GLint  length;
  GLchar *infoLog = nullptr;

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
    "uniform mat4 projection;\n"
    "out vec2 varTexCoord;\n"
    "void main()\n"
    "{\n"
    "  gl_Position = projection * vec4(attrPosition, 0.0, 1.0);\n"
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

      m_locAttrPosition = glGetAttribLocation(m_glslProgram, "attrPosition");
      MY_ASSERT(m_locAttrPosition!= -1);

      m_locAttrTexCoord = glGetAttribLocation(m_glslProgram, "attrTexCoord");
      MY_ASSERT(m_locAttrTexCoord != -1);
      
      m_locProjection = glGetUniformLocation(m_glslProgram, "projection");
      MY_ASSERT(m_locProjection != -1);

      glUniform1i(glGetUniformLocation(m_glslProgram, "samplerHDR"), 0); // Always using texture image unit 0 for the display texture.

      glUniform1f(glGetUniformLocation(m_glslProgram, "invGamma"),       1.0f / m_gamma);
      glUniform3f(glGetUniformLocation(m_glslProgram, "colorBalance"),   m_colorBalance.x, m_colorBalance.y, m_colorBalance.z);
      glUniform1f(glGetUniformLocation(m_glslProgram, "invWhitePoint"),  m_brightness / m_whitePoint);
      glUniform1f(glGetUniformLocation(m_glslProgram, "burnHighlights"), m_burnHighlights);
      glUniform1f(glGetUniformLocation(m_glslProgram, "crushBlacks"),    m_crushBlacks + m_crushBlacks + 1.0f);
      glUniform1f(glGetUniformLocation(m_glslProgram, "saturation"),     m_saturation);

      glUseProgram(0);
    }
  }
}


void Application::updateTonemapper()
{
  glUseProgram(m_glslProgram);

  //glUniform1i(glGetUniformLocation(m_glslProgram, "samplerHDR"), 0); // Always using texture image unit 0 for the display texture.
  glUniform1f(glGetUniformLocation(m_glslProgram, "invGamma"),       1.0f / m_gamma);
  glUniform3f(glGetUniformLocation(m_glslProgram, "colorBalance"),   m_colorBalance.x, m_colorBalance.y, m_colorBalance.z);
  glUniform1f(glGetUniformLocation(m_glslProgram, "invWhitePoint"),  m_brightness / m_whitePoint);
  glUniform1f(glGetUniformLocation(m_glslProgram, "burnHighlights"), m_burnHighlights);
  glUniform1f(glGetUniformLocation(m_glslProgram, "crushBlacks"),    m_crushBlacks + m_crushBlacks + 1.0f);
  glUniform1f(glGetUniformLocation(m_glslProgram, "saturation"),     m_saturation);

  glUseProgram(0);
}


// Needs a valid scene extent.
void Application::guiWindow()
{
  MY_ASSERT(m_fontScale > 0.0f);

  if (!m_isVisibleGUI) // Use SPACE to toggle the display of the GUI window.
  {
    return;
  }

  ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiCond_FirstUseEver);
  if (ImGui::IsWindowCollapsed())
  {
    return;
  }
  
  if (m_font != nullptr)
  {
    ImGui::PushFont(m_font);
  }

  ImGuiWindowFlags window_flags = 0;
  ImGui::Begin("GLTF_renderer", nullptr, window_flags); // No bool flag to omit the close button.
  ImGui::PushItemWidth(-170.0f * m_fontScale); // Right-aligned, keep pixels for the labels.

  if (ImGui::CollapsingHeader("Help"))
  { 
    ImGui::TextWrapped(
      "The rendering window accepts drag-and-drop events for the following file types:\n"
      "  *.gltf or *.glb to load a new asset.\n"
      "  *.hdr files for spherical HDR environment lights (when using the default --miss (-m) 2).\n\n"
      "Interactions\n"
      "  SPACE = Toggle GUI window display\n"
      "  P = Save image as tone-mapped *.png into the working directory.\n"
      "  H = Save image as linear *.hdr into the working directory.\n\n"
      "  LMB drag = Orbit camera\n"
      "  MMB drag = Pan camera\n"
      "  RMB drag = Dolly camera\n"
      "  Mouse Wheel = Zoom (change field of view angle)\n"
      "  Ctrl+LMB = Pick material index from object under mouse cursor.\n\n"
      "Please read the 'GLTF_renderer.exe --help' output and the doc/README.md for more information.\n");
  }
  if (ImGui::CollapsingHeader("System"))
  { 
    ImGui::RadioButton("off", (int*) & m_benchmarkMode, OFF);
    ImGui::SameLine();
    if (ImGui::RadioButton("frames/second", (int*) & m_benchmarkMode, FPS))
    {
      m_benchmarkEntries = 0;
      m_benchmarkCell    = 0;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("samples/second", (int*) & m_benchmarkMode, SAMPLES_PER_SECOND))
    {
      m_benchmarkEntries = 0;
      m_benchmarkCell    = 0;
    }
    ImGui::SameLine();
    ImGui::LabelText("Benchmark", "");

    // Plot the benchmark results.
    if (m_benchmarkMode != OFF)
    {
      float average = 0.0f;
      float maximum = 0.0f;

      for (int i = 0; i < m_benchmarkEntries; ++i)
      {
        const float value = m_benchmarkValues[i];

        average += value;
        maximum  = std::max(maximum, value);
      }
      if (0 < m_benchmarkEntries)
      {
        average /= float(m_benchmarkEntries);
      }

      std::string label;
      std::ostringstream overlay;

      if (m_benchmarkMode == FPS)
      {
        label = "frames/second";
        overlay.precision(2); // Precision is # digits in fraction part.
      }
      else
      {
        label = std::string("samples/second");
        overlay.precision(0); // Precision is # digits in fraction part.
      }

      overlay << std::fixed << std::setw(6) << average << " avg, " << std::setw(6) << maximum << " max";

      ImGui::PlotLines(label.c_str(), m_benchmarkValues.data(), m_benchmarkEntries, 0, overlay.str().c_str(), 0.0f, maximum, ImVec2(0, 50.0f), sizeof(float));
    }
    if (ImGui::InputInt("Launches", &m_launches, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue)) // This requires RETURN to apply a new value.
    {
      m_launches = std::max(1, std::min(m_launches, MAX_LAUNCHES));
      m_benchmarkEntries = 0;
      m_benchmarkCell    = 0;
    }
    if (ImGui::Button("Match")) // Match the rendering resolution to the current client window size.
    {
      m_resolution.x = std::max(1, m_width);
      m_resolution.y = std::max(1, m_height);
      m_isDirtyResolution = true;
    }
    ImGui::SameLine();
    if (ImGui::InputInt2("Resolution", &m_resolution.x, ImGuiInputTextFlags_EnterReturnsTrue)) // This requires RETURN to apply a new value.
    {
      m_resolution.x = std::max(1, m_resolution.x);
      m_resolution.y = std::max(1, m_resolution.y);
      m_isDirtyResolution = true;
    }
    if (ImGui::DragInt2("Path Length (min, max)", reinterpret_cast<int*>(&m_launchParameters.pathLengths), 1.0f, 0, 100))
    {
      m_launchParameters.iteration = 0u; // Restart accumulation.
    }
    if (ImGui::DragFloat("Scene Epsilon", &m_epsilonFactor, 1.0f, 0.0f, 10000.0f))
    {
      m_launchParameters.sceneEpsilon = m_epsilonFactor * SCENE_EPSILON_SCALE;
      m_launchParameters.iteration = 0u; // Restart accumulation.
    }
    // Override all materials to behave as unlit.
    if (ImGui::Checkbox("Force Unlit", &m_forceUnlit))
    {
      m_launchParameters.forceUnlit = (m_forceUnlit) ? 1 : 0;
      m_launchParameters.iteration = 0u; // Restart accumulation.
    }
    if (ImGui::Checkbox("Direct Lighting", &m_useDirectLighting))
    {
      m_launchParameters.directLighting = (m_useDirectLighting) ? 1 : 0;
      m_launchParameters.iteration = 0u; // Restart accumulation.
    }
    // Allow disabling all occlusionTexture effects globally.
    if (ImGui::Checkbox("Ambient Occlusion", &m_useAmbientOcclusion))
    {
      m_launchParameters.ambientOcclusion = (m_useAmbientOcclusion) ? 1 : 0;
      m_launchParameters.iteration = 0u; // Restart accumulation.
    }
    if (ImGui::Checkbox("Show Environment", &m_showEnvironment))
    {
      m_launchParameters.showEnvironment = (m_showEnvironment) ? 1 : 0;
      m_launchParameters.iteration = 0u; // Restart accumulation.
    }
    if (ImGui::Checkbox("Gimbal Lock", &m_isLockedGimbal))
    {
      m_trackball.setGimbalLock(m_isLockedGimbal);
    }
    if (ImGui::DragFloat("Mouse Ratio", &m_mouseSpeedRatio, 0.01f, 0.01f, 10000.0f, "%.2f"))
    {
      if (m_mouseSpeedRatio < 0.01f)
      {
        m_mouseSpeedRatio = 0.01f;
      }
      else if (10000.0f < m_mouseSpeedRatio)
      {
        m_mouseSpeedRatio = 10000.0f;
      }
      m_trackball.setSpeedRatio(m_mouseSpeedRatio);
    }
  }

  if (ImGui::CollapsingHeader("Tonemapper"))
  {
    bool changed = false;
    if (ImGui::ColorEdit3("Balance", (float*) &m_colorBalance))
    {
      changed = true;
    }
    if (ImGui::DragFloat("Gamma", &m_gamma, 0.01f, 0.01f, 10.0f)) // Must not get 0.0f
    {
      changed = true;
    }
    if (ImGui::DragFloat("White Point", &m_whitePoint, 0.01f, 0.01f, 255.0f, "%.2f")) // Must not get 0.0f
    {
      changed = true;
    }
    if (ImGui::DragFloat("Burn Lights", &m_burnHighlights, 0.01f, 0.0f, 10.0f, "%.2f"))
    {
      changed = true;
    }
    if (ImGui::DragFloat("Crush Blacks", &m_crushBlacks, 0.01f, 0.0f, 1.0f, "%.2f"))
    {
      changed = true;
    }
    if (ImGui::DragFloat("Saturation", &m_saturation, 0.01f, 0.0f, 10.0f, "%.2f"))
    {
      changed = true;
    }
    if (ImGui::DragFloat("Brightness", &m_brightness, 0.01f, 0.0f, 100.0f, "%.2f"))
    {
      changed = true;
    }
    if (changed)
    {
      updateTonemapper(); // This doesn't need a renderer restart.
    }
  }

  // Only show the Scenes pane when there is more than one scene inside the asset.
  if (1 < m_asset.scenes.size()) 
  {
    MY_ASSERT(m_indexScene < m_asset.scenes.size())

    if (ImGui::CollapsingHeader("Scenes"))
    {
      // The name of the currently selected scene.
      std::string labelCombo = std::to_string(m_indexScene) + std::string(") ") + std::string(m_asset.scenes[m_indexScene].name);

      if (ImGui::BeginCombo("Scene", labelCombo.c_str()))
      {
        // Add selectable scenes to the combo box.
        for (size_t i = 0; i < m_asset.scenes.size(); ++i)
        {
          bool isSelected = (i == m_indexScene);

          std::string label = std::to_string(i) + std::string(") ") + std::string(m_asset.scenes[i].name);

          if (ImGui::Selectable(label.c_str(), isSelected))
          {
            if (m_indexScene != i)
            {
              m_indexScene = i; 
              // Here the scene has changed and the IAS needs to be rebuild for the selected scene.
              m_isDirtyScene = true;
            }
          }
          if (isSelected)
          {
            ImGui::SetItemDefaultFocus();
          }
        }
        ImGui::EndCombo();
      }
    }
  } // End of Scenes pane.

  // Only show the Cameras pane when there is more than one camere inside the asset.
  if (1 < m_asset.cameras.size())
  {
    MY_ASSERT(m_indexCamera < m_asset.cameras.size());
    
    if (ImGui::CollapsingHeader("Cameras"))
    {
      std::string labelCombo = std::to_string(m_indexCamera) + std::string(") ") + std::string(m_asset.cameras[m_indexCamera].name); // The name of the currently selected camera.

      if (ImGui::BeginCombo("Camera", labelCombo.c_str()))
      {
        // Add selectable cameras to the combo box.
        for (size_t i = 0; i < m_asset.cameras.size(); ++i)
        {
          bool isSelected = (i == m_indexCamera);

          std::string label = std::to_string(i) + std::string(") ") + std::string(m_asset.cameras[i].name);

          if (ImGui::Selectable(label.c_str(), isSelected))
          {
            if (m_indexCamera != i)
            {
              m_indexCamera = i; 
              // Here the scene has changed and the IAS needs to be rebuilt for the selected scene.
              m_cameras[m_indexCamera].setIsDirty(true);
            }
          }
          if (isSelected)
          {
            ImGui::SetItemDefaultFocus();
          }
        }
        ImGui::EndCombo();
      }
    }
  }

  // Only show the Animation GUI when there are animations inside the scene.
  if (!m_animations.empty())
  {
    if (ImGui::CollapsingHeader("Animations"))
    {
      // The animation GUI widgets are disabled while there is no animation enabled.
      if (!m_isAnimated)
      {
        ImGui::BeginDisabled();
      }
      const std::string labelPlay = (m_isPlaying) ? std::string("Stop") : std::string("Play");
      if (ImGui::Button(labelPlay.c_str()))
      {
        m_isPlaying= !m_isPlaying;

        if (m_isPlaying && m_isTimeBased)
        {
          m_timeBase = std::chrono::steady_clock::now();
        }
      }
      ImGui::SameLine();

      if (ImGui::Checkbox("Time-based", &m_isTimeBased))
      {
        // Reset the base time when enabling real-time mode while playing.
        if (m_isPlaying && m_isTimeBased)
        {
          m_timeBase = std::chrono::steady_clock::now();
        }
      }

      ImGui::Separator();

      if (m_isTimeBased)
      {
        std::ostringstream streamMin; 
        streamMin.precision(2); // Precision is # digits in fractional part.
        streamMin << "Start (" << std::fixed << m_timeMinimum << ")";
        const std::string labelMin = streamMin.str();

        const float interval = m_timeMaximum - m_timeMinimum;
        const float stepSlow = std::max(0.01f, interval / 100.0f);
        const float stepFast = std::max(0.01f, interval / 10.0f);

        if (ImGui::InputFloat(labelMin.c_str(), &m_timeStart, stepSlow, stepFast, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue))
        {
          m_timeStart = std::max(0.0f, m_timeStart); // Animations in glTF do not use negative times. Should work though.
          if (m_timeStart >= m_timeEnd)
          {
            m_timeEnd = m_timeStart + 0.01f; // Keep start and end times at least one GUI resolution step apart. 
          }
        }

        // Allow to scrub the curent frame.
        if (ImGui::SliderFloat("Time [s]", &m_timeCurrent, m_timeStart, m_timeEnd, "%.2f", ImGuiSliderFlags_None))
        {
          m_isScrubbing = true;
        }

        std::ostringstream streamMax; 
        streamMax.precision(2);
        streamMax << "End (" << std::fixed << m_timeMaximum << ")";
        const std::string labelMax = streamMax.str();
        if (ImGui::InputFloat(labelMax.c_str(), &m_timeEnd, stepSlow, stepFast, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue))
        {
          m_timeEnd = std::max(m_timeStart + 0.01f, m_timeEnd); // Keep start and end times at least one GUI resolution step apart. 
        }

        // This changes the scaling of time based animations. 1.0f means real-time.
        if (ImGui::SliderFloat("Time Scale", &m_timeScale, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_None))
        {
        }
      }
      else
      {
        const std::string labelStart = std::string("Start (") + std::to_string(m_frameMinimum) + std::string(")");
        if (ImGui::InputInt(labelStart.c_str(), &m_frameStart, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue))
        {
          m_frameStart = std::max(0, m_frameStart);
          if (m_frameStart >= m_frameEnd)
          {
            m_frameEnd = m_frameStart + 1;
          }
        }

        // Allow to scrub the curent frame.
        if (ImGui::SliderInt("Frame", &m_frameCurrent, m_frameStart, m_frameEnd - 1, "%d", ImGuiSliderFlags_None))
        {
          m_isScrubbing = true;
        }

        const std::string labelEnd = std::string("End (") + std::to_string(m_frameMaximum) + std::string(")");
        if (ImGui::InputInt(labelEnd.c_str(), &m_frameEnd, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue))
        {
          m_frameEnd = std::max(m_frameStart + 1, m_frameEnd);
        }

        if (ImGui::InputFloat("Frames/Second", &m_framesPerSecond, 1.0f, 10.0f, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue))
        {
          m_framesPerSecond = std::max(1.0f, m_framesPerSecond);

          m_frameMinimum = std::max(0, static_cast<int>(floorf(m_timeMinimum * m_framesPerSecond)));
          m_frameMaximum = std::max(0, static_cast<int>(ceilf(m_timeMaximum * m_framesPerSecond)));
          m_frameStart = m_frameMinimum;
          m_frameEnd   = m_frameMaximum;
          
          m_frameCurrent = m_frameStart; // Reset the animation to the first frame.
        }
      }
      if (!m_isAnimated)
      {
        ImGui::EndDisabled();
      }

      ImGui::Separator();

      bool isDirtyTimeline = false; // Indicate if the timeline minimum and maximum values are dirty and need to be recalculated.

      // Convenience buttons enabling all or none of the animations.
      if (ImGui::Button("All"))
      {
        for (dev::Animation& animation : m_animations)
        {
          animation.isEnabled = true;
        }
        isDirtyTimeline = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("None"))
      {
        for (dev::Animation& animation : m_animations)
        {
          animation.update(m_nodes, m_timeStart); // Reset animation to start time zero.
          animation.isEnabled = false;
        }
        // All animations are disabled (m_isAnimated will be false), no need to set this.
        //isDirtyTimeline = true;
      }
      
      m_isAnimated = false;
      for (size_t i = 0; i < m_animations.size(); ++i)
      {
        dev::Animation& animation = m_animations[i];

        // Print each animations timeMin and timeMax interval after the animation name.
        std::ostringstream stream; 

        stream.precision(2); // Precision is # digits in fraction part.
        stream << std::fixed << " [" << animation.timeMin << ", " << animation.timeMax << "]";

        std::string label = std::to_string(i) + std::string(") ") + animation.name + stream.str();

        if (ImGui::Checkbox(label.c_str(), &animation.isEnabled))
        {
          if (!animation.isEnabled)
          {
            animation.update(m_nodes, m_timeStart); // Reset animation to start time.
          }
          // When any of the animations is toggled, recalculate the timeline's minimum and maximum times.
          isDirtyTimeline = true;
        }
        // This determines if any animation is enabled.
        if (animation.isEnabled)
        {
          m_isAnimated = true;
        }
      }
      // When all animations are disabled, make sure the m_isPlaying state is reset.
      if (!m_isAnimated)
      {
        m_isPlaying = false;
      }

      // When the animation is enabled and there were changes to the enabled animations,
      // recalculate the minimum and maximum time of all enabled animations.
      if (m_isAnimated && isDirtyTimeline)
      {
        m_timeMinimum =  FLT_MAX;
        m_timeMaximum = -FLT_MAX;

        for (const dev::Animation& animation : m_animations)
        {
          if (animation.isEnabled)
          {
            if (animation.timeMin < m_timeMinimum)
            {
              m_timeMinimum = animation.timeMin;
            }
            if (m_timeMaximum < animation.timeMax)
            {
              m_timeMaximum = animation.timeMax;
            }
          }
        }

        m_timeStart = m_timeMinimum;
        m_timeEnd   = m_timeMaximum;

        m_frameMinimum = std::max(0, static_cast<int>(floorf(m_timeMinimum * m_framesPerSecond)));
        m_frameMaximum = std::max(0, static_cast<int>(ceilf(m_timeMaximum * m_framesPerSecond)));
        m_frameStart = m_frameMinimum;
        m_frameEnd   = m_frameMaximum;

        m_frameCurrent = m_frameStart; // Reset the animation to the first frame.
      }
    }
  }

  // Only show the Variants pane when there are material variants inside the scene.
  if (!m_asset.materialVariants.empty())
  {
    MY_ASSERT(m_indexVariant < m_asset.materialVariants.size());

    if (ImGui::CollapsingHeader("Variants"))
    {
      const size_t previousVariant = m_indexVariant;

      // The name of the currently selected material variant
      std::string labelCombo = std::to_string(m_indexVariant) + std::string(") ") + m_asset.materialVariants[m_indexVariant];

      if (ImGui::BeginCombo("Variant", labelCombo.c_str()))
      {
        for (size_t i = 0; i < m_asset.materialVariants.size(); ++i)
        {
          bool isSelected = (i == m_indexVariant);

          std::string label = std::to_string(i) + std::string(") ") + m_asset.materialVariants[i];

          if (ImGui::Selectable(label.c_str(), isSelected))
          {
            if (m_indexVariant != i)
            {
              m_indexVariant = i;
            }
          }
          if (isSelected)
          {
            ImGui::SetItemDefaultFocus();
          }
        }
        ImGui::EndCombo();
      }
      
      if (previousVariant != m_indexVariant)
      {
        updateVariant();
      }
    }
  }

  // Only show the Materials pane when there are materials inside the asset. Actually when not, there will be default materials.
  if (!m_asset.materials.empty()) // Make sure there is at least one material inside the asset.
  {

    if (ImGui::CollapsingHeader("Materials"))
    {
      MY_ASSERT(m_indexMaterial < m_asset.materials.size())

      // The name of the currently selected material
      std::string labelCombo = std::to_string(m_indexMaterial) + std::string(") ") + std::string(m_asset.materials[m_indexMaterial].name); 

      if (ImGui::BeginCombo("Material", labelCombo.c_str()))
      {
        // Add selectable materials to the combo box.
        for (size_t i = 0; i < m_asset.materials.size(); ++i)
        {
          bool isSelected = (i == m_indexMaterial);

          std::string label = std::to_string(i) + std::string(") ") + std::string(m_asset.materials[i].name);

          if (ImGui::Selectable(label.c_str(), isSelected))
          {
            if (m_indexMaterial != i)
            {
              m_indexMaterial = i; 
            }
          }
          if (isSelected)
          {
            ImGui::SetItemDefaultFocus();
          }
        }
        ImGui::EndCombo();
      }

      // Now display all editable material parameters.
      // (There is a lot of repeated code in here, because otherwise ImGui didn't build unique widgets!)
      const MaterialData& org = m_materialsOrg[m_indexMaterial];
      MaterialData&       cur = m_materials[m_indexMaterial];

      bool changed = false; // Material changed, update the SBT hit records.
      bool rebuild = false; // Material changed in a way which requires to rebuild the AS of all primitives using that material.

      if (ImGui::Button("Reset"))
      {
        // The face culling state is affected by both the doubleSided and the volume state.
        // The only case when face culling is enabled is when the material is not doubleSided and not using the volume extension.
        const bool orgCull = (!org.doubleSided && (org.flags & FLAG_KHR_MATERIALS_VOLUME) == 0);
        const bool curCull = (!cur.doubleSided && (cur.flags & FLAG_KHR_MATERIALS_VOLUME) == 0);
        
        // If the alphaMode changes, the anyhit program invocation for primitives changes.
        rebuild = (curCull != orgCull) || (cur.alphaMode != org.alphaMode);
          
        cur = org; // Reset all material changes to the original values inside the asset.

        changed = true;
      }

      // Generic settings. 
      ImGui::Separator();
      changed |= ImGui::Checkbox("unlit", &cur.unlit);

      // Note that doubleSided and alphaMode changes can trigger AS rebuilds!
      ImGui::Separator();
      if (ImGui::Checkbox("doubleSided", &cur.doubleSided))
      {
        changed = true;
        // If the doubleSided flag changes on materials which are not using the KHR_materials_volume extension,
        // the AS needs to be rebuilt to change the face culling state.
        rebuild = ((cur.flags & FLAG_KHR_MATERIALS_VOLUME) == 0); 
      }

      ImGui::Separator();
      if (ImGui::Combo("alphaMode", reinterpret_cast<int*>(&cur.alphaMode), "OPAQUE\0MASK\0BLEND\0\0"))
      {
        changed = true;
        // Any alphaMode change requires a rebuild because each alpha mode handles anyhit programs differently.
        rebuild = true; 
      }

      // baseColor alpha and alphaCutoff have no effect if for ALPHA_MODE_OPAQUE.
      if (cur.alphaMode != MaterialData::ALPHA_MODE_OPAQUE)
      {
        // This is only one of three factors defining the opacity.
        // There is also the color.w and the baseColorTexture.w.
        changed |= ImGui::SliderFloat("baseAlpha", &cur.baseColorFactor.w, 0.0f, 1.0f); 
      }

      // alphaCutoff is only used with alphaMode == ALPHA_MODE_MASK.
      if (cur.alphaMode == MaterialData::ALPHA_MODE_MASK)
      {
        changed |= ImGui::SliderFloat("alphaCutoff", &cur.alphaCutoff, 0.0f, 1.0f);
      }

      ImGui::Separator();
      changed |= ImGui::ColorEdit3("baseColor", reinterpret_cast<float*>(&cur.baseColorFactor));
      // Only display the texture GUI when the original material defines it.
      if (org.baseColorTexture.object != 0) 
      {
        bool isEnabled = (cur.baseColorTexture.object != 0);
        if (ImGui::Checkbox("baseColorTexture", &isEnabled))
        {
          cur.baseColorTexture.object = (isEnabled) ? org.baseColorTexture.object : 0;
          changed = true;
        }
        // DEBUG If the KHR_texture_transform element should be shown inside the GUI, this code would need to be replicated for all textures.
        // (Moving that into a function requires some item tracking with ImGui::PushId/PopId.)
        // Manipulating the rotation is rather non-intuitive anyway, so keep the clutter out of the GUI
        // and don't offer the texture transforms as editable parameters.
        // Also mind that when using KHR_mesh_quantization with unnormalized texture coordinates,
        // the transform scale is used to normalize them by multiplication with 1.0f/255.0f or 1.0f/65535.0f

        //changed |= ImGui::DragFloat2("baseColorTexture.scale", reinterpret_cast<float*>(&cur.baseColorTexture.scale), 0.01f, -128.0f, 128.0, "%.2f", 1.0f);
        //if (ImGui::SliderFloat("baseColorTexture.rotation", &cur.baseColorTexture.angle, 0.0f, 2.0f * M_PIf)) or with:
        //if (ImGui::SliderAngle("baseColorTexture.rotation", &cur.baseColorTexture.angle, 0.0f, 360.0f)) // While the value is in radians, the display is in degrees.
        //{
        //  cur.baseColorTexture.rotation.x = sinf(cur.baseColorTexture.angle);
        //  cur.baseColorTexture.rotation.y = cosf(cur.baseColorTexture.angle);
        //  changed = true;
        //}
        //changed |= ImGui::DragFloat2("baseColorTexture.translation", reinterpret_cast<float*>(&cur.baseColorTexture.translation), 0.01f, -128.0f, 128.0, "%.2f", 1.0f);
      }

      ImGui::Separator();
      changed |= ImGui::SliderFloat("roughness", &cur.roughnessFactor, 0.0f, 1.0f);
      changed |= ImGui::SliderFloat("metallic", &cur.metallicFactor, 0.0f, 1.0f);
      if (org.metallicRoughnessTexture.object != 0) 
      {
        bool isEnabled = (cur.metallicRoughnessTexture.object != 0);
        if (ImGui::Checkbox("metallicRoughnessTexture", &isEnabled))
        {
          cur.metallicRoughnessTexture.object = (isEnabled) ? org.metallicRoughnessTexture.object : 0;
          changed = true;
        }
      }

      ImGui::Separator();
      changed |= ImGui::SliderFloat("ior", &cur.ior, 1.0f, 5.0f); // Don't allow values below 1.0f here.

      ImGui::Separator();
      changed |= ImGui::SliderFloat("specular", &cur.specularFactor, 0.0f, 1.0f);
      if (org.specularTexture.object != 0) 
      {
        bool isEnabled = (cur.specularTexture.object != 0);
        if (ImGui::Checkbox("specularTexture", &isEnabled))
        {
          cur.specularTexture.object = (isEnabled) ? org.specularTexture.object : 0;
          changed = true;
        }
      }
      changed |= ImGui::ColorEdit3("specularColor", reinterpret_cast<float*>(&cur.specularColorFactor));
      if (org.specularColorTexture.object != 0) 
      {
        bool isEnabled = (cur.specularColorTexture.object != 0);
        if (ImGui::Checkbox("specularColorTexture", &isEnabled))
        {
          cur.specularColorTexture.object = (isEnabled) ? org.specularColorTexture.object : 0;
          changed = true;
        }
      }

      ImGui::Separator();
      changed |= ImGui::SliderFloat("transmission", &cur.transmissionFactor, 0.0f, 1.0f);
      if (org.transmissionTexture.object != 0) 
      {
        bool isEnabled = (cur.transmissionTexture.object != 0);
        if (ImGui::Checkbox("transmissionTexture", &isEnabled))
        {
          cur.transmissionTexture.object = (isEnabled) ? org.transmissionTexture.object : 0;
          changed = true;
        }
      }

      if (org.normalTexture.object != 0) 
      {
        ImGui::Separator();
        bool isEnabled = (cur.normalTexture.object != 0);
        if (ImGui::Checkbox("normalTexture", &isEnabled))
        {
          cur.normalTexture.object = (isEnabled) ? org.normalTexture.object : 0;
          changed = true;
        }
        // normalTextureScale has no effect when there is no normalTexture.
        // Always show the normalTextureScale slider when the original material has a normalTexture
        // because that could be used as cleatcoatNormalTexture with the GUI below and 
        // I don't want the GUI elements to shift when toggling texture enables.
        changed |= ImGui::SliderFloat("normalScale", &cur.normalTextureScale, -10.0f, 10.0f); // Default is 1.0f. What is a suitable range?
      }

      if (m_useAmbientOcclusion)
      {
        if (org.occlusionTexture.object != 0) 
        {
          ImGui::Separator();
          bool isEnabled = (cur.occlusionTexture.object != 0);
          if (ImGui::Checkbox("occlusionTexture", &isEnabled))
          {
            cur.occlusionTexture.object = (isEnabled) ? org.occlusionTexture.object : 0;
            changed = true;
          }
          changed |= ImGui::SliderFloat("occlusionTextureStrength", reinterpret_cast<float*>(&cur.occlusionTextureStrength), 0.0f, 1.0f);
        }
      }

      ImGui::Separator();
      changed |= ImGui::SliderFloat("anisotropy", &cur.anisotropyStrength, 0.0f, 1.0f);
      changed |= ImGui::SliderFloat("anisotropyRotation", &cur.anisotropyRotation, 0.0f, 2.0f * M_PIf);
      if (org.anisotropyTexture.object != 0) 
      {
        bool isEnabled = (cur.anisotropyTexture.object != 0);
        if (ImGui::Checkbox("anisotropyTexture", &isEnabled))
        {
          cur.anisotropyTexture.object = (isEnabled) ? org.anisotropyTexture.object : 0;
          changed = true;
        }
      }

      ImGui::Separator();
      changed |= ImGui::SliderFloat("emissiveStrength", &cur.emissiveStrength, 0.0f, 100.0f); // Default is 1.0f. Modulates emissiveFactor
      changed |= ImGui::ColorEdit3("emissiveColor", reinterpret_cast<float*>(&cur.emissiveFactor));
      if (org.emissiveTexture.object != 0) 
      {
        bool isEnabled = (cur.emissiveTexture.object != 0);
        if (ImGui::Checkbox("emissiveTexture", &isEnabled))
        {
          cur.emissiveTexture.object = (isEnabled) ? org.emissiveTexture.object : 0;
          changed = true;
        }
      }

      ImGui::Separator();
      bool useVolume = ((cur.flags & FLAG_KHR_MATERIALS_VOLUME) != 0);
      if (ImGui::Checkbox("volume", &useVolume))
      {
        if (useVolume)
        {
          cur.flags |= FLAG_KHR_MATERIALS_VOLUME; // Set the volume extension flag inside the material.
        }
        else
        {
          cur.flags &= ~FLAG_KHR_MATERIALS_VOLUME; // Clear the volume extension flag inside the material.
        }

        changed = true;
        
        // If the geometry is not doubleSided then toggling the volume state needs to rebuild the GAS to disable/enable face culling.
        rebuild = !cur.doubleSided;
      }

      // Only show the volume absorption parameters when the volume extension is enabled.
      if (cur.flags & FLAG_KHR_MATERIALS_VOLUME)
      {
        changed |= ImGui::SliderFloat("attenuationDistance", &cur.attenuationDistance, 0.001f, 2.0f * m_sceneExtent.getMaxDimension()); // Must not be 0.0f!
        if (ImGui::ColorEdit3("attenuationColor", reinterpret_cast<float*>(&cur.attenuationColor)))
        {
          // Make sure the logf() for the volume absorption coefficient is never used on zero color components.
          cur.attenuationColor = fmaxf(make_float3(0.001f), cur.attenuationColor);
          changed = true;
        }
        
        // HACK The renderer only evaluates thicknessFactor == 0.0f as thinwalled. It's not using it for the absorption calculation!
        changed |= ImGui::SliderFloat("thickness", &cur.thicknessFactor, 0.0f, 1.0f);
      }

      ImGui::Separator();
      changed |= ImGui::SliderFloat("clearcoat", &cur.clearcoatFactor, 0.0f, 1.0f);
      if (org.clearcoatTexture.object != 0) 
      {
        bool isEnabled = (cur.clearcoatTexture.object != 0);
        if (ImGui::Checkbox("clearcoatTexture", &isEnabled))
        {
          cur.clearcoatTexture.object = (isEnabled) ? org.clearcoatTexture.object : 0;
          changed = true;
        }
      }
      changed |= ImGui::SliderFloat("clearcoatRoughness", &cur.clearcoatRoughnessFactor, 0.0f, 1.0f);
      if (org.clearcoatRoughnessTexture.object != 0) 
      {
        bool isEnabled = (cur.clearcoatRoughnessTexture.object != 0);
        if (ImGui::Checkbox("clearcoatRoughnessTexture", &isEnabled))
        {
          cur.clearcoatRoughnessTexture.object = (isEnabled) ? org.clearcoatRoughnessTexture.object : 0;
          changed = true;
        }
      }
      if (org.clearcoatNormalTexture.object != 0) 
      {
        bool isEnabled = (cur.clearcoatNormalTexture.object != 0);
        if (ImGui::Checkbox("clearcoatNormalTexture", &isEnabled))
        {
          cur.clearcoatNormalTexture.object = (isEnabled) ? org.clearcoatNormalTexture.object : 0;
          changed = true;
        }
      }
      // If the material is not using a clearcoatNormalTexture, but has a normalTexture,
      // allow the user to apply the normalTexture on the clearcoat as well.
      else if (org.normalTexture.object != 0)
      {
        bool useNormalTexture = (cur.clearcoatNormalTexture.object != 0);
        if (ImGui::Checkbox("use normalTexture on clearcoat", &useNormalTexture))
        {
          if (useNormalTexture)
          {
            cur.clearcoatNormalTexture = org.normalTexture; // Use base normalTexture as clearcoatNormalTexture.
            cur.isClearcoatNormalBaseNormal = true;
          }
          else
          {
            cur.clearcoatNormalTexture = org.clearcoatNormalTexture; // clearcoatNormalTexture off.
          }
          changed = true;
        }
      }

      ImGui::Separator();
      changed |= ImGui::ColorEdit3("sheenColor", reinterpret_cast<float*>(&cur.sheenColorFactor));
      if (org.sheenColorTexture.object != 0) 
      {
        bool isEnabled = (cur.sheenColorTexture.object != 0);
        if (ImGui::Checkbox("sheenColorTexture", &isEnabled))
        {
          cur.sheenColorTexture.object = (isEnabled) ? org.sheenColorTexture.object : 0;
          changed = true;
        }
      }
      changed |= ImGui::SliderFloat("sheenRoughness", &cur.sheenRoughnessFactor, 0.0f, 1.0f);
      if (org.sheenRoughnessTexture.object != 0) 
      {
        bool isEnabled = (cur.sheenRoughnessTexture.object != 0);
        if (ImGui::Checkbox("sheenRoughnessTexture", &isEnabled))
        {
          cur.sheenRoughnessTexture.object = (isEnabled) ? org.sheenRoughnessTexture.object : 0;
          changed = true;
        }
      }

      ImGui::Separator();
      changed |= ImGui::SliderFloat("iridescence", &cur.iridescenceFactor, 0.0f, 1.0f);
      if (org.iridescenceTexture.object != 0) 
      {
        bool isEnabled = (cur.iridescenceTexture.object != 0);
        if (ImGui::Checkbox("iridescenceTexture", &isEnabled))
        {
          cur.iridescenceTexture.object = (isEnabled) ? org.iridescenceTexture.object : 0;
          changed = true;
        }
      }
      changed |= ImGui::SliderFloat("iridescenceIor", &cur.iridescenceIor, 1.0f, 5.0f);
      changed |= ImGui::SliderFloat("iridescenceThicknessMin", &cur.iridescenceThicknessMinimum, 0.0f, cur.iridescenceThicknessMaximum);
      changed |= ImGui::SliderFloat("iridescenceThicknessMax", &cur.iridescenceThicknessMaximum, cur.iridescenceThicknessMinimum, 2000.0f);
      if (org.iridescenceThicknessTexture.object != 0) 
      {
        bool isEnabled = (cur.iridescenceThicknessTexture.object != 0);
        if (ImGui::Checkbox("iridescenceThicknessTexture", &isEnabled))
        {
          cur.iridescenceThicknessTexture.object = (isEnabled) ? org.iridescenceThicknessTexture.object : 0;
          changed = true;
        }
      }

      if (changed)
      {
        //debugDumpMaterial(m); // DEBUG
        updateMaterial(static_cast<int>(m_indexMaterial), rebuild);
      }
    }
  }

  if (!m_lightDefinitions.empty())
  {
    if (ImGui::CollapsingHeader("Lights"))
    {
      // If there is an environment light, it's always inside the first element.
      if (m_lightDefinitions[0].typeLight == TYPE_LIGHT_ENV_CONST ||
          m_lightDefinitions[0].typeLight == TYPE_LIGHT_ENV_SPHERE)
      {
        if (ImGui::ColorEdit3("env color", m_colorEnv))
        {
          m_lightDefinitions[0].emission = make_float3(m_colorEnv[0], m_colorEnv[1], m_colorEnv[2]) * m_intensityEnv;
          m_isDirtyLights = true; // Next render() call will update the device side data.
        }
        if (ImGui::DragFloat("env intensity", &m_intensityEnv, 0.001f, 0.0f, 10000.0f))
        {
          m_lightDefinitions[0].emission = make_float3(m_colorEnv[0], m_colorEnv[1], m_colorEnv[2]) * m_intensityEnv;
          m_isDirtyLights = true;
        }
        // If it's a spherical HDR texture environment light, show the environment rotation Euler angles.
        if (m_lightDefinitions[0].typeLight == TYPE_LIGHT_ENV_SPHERE)
        {
          if (ImGui::DragFloat3("env rotation", m_rotationEnv, 1.0f, 0.0f, 360.0f))
          {
            glm::vec3 euler(glm::radians(m_rotationEnv[0]),
                            glm::radians(m_rotationEnv[1]),
                            glm::radians(m_rotationEnv[2]));
          
            glm::quat quatRotation(euler);

            glm::mat4 matRotation    = glm::toMat4(quatRotation);
            glm::mat4 matRotationInv = glm::inverse(matRotation);
          
            for (int i = 0; i < 3; ++i)
            {
              glm::vec4 row = glm::row(matRotation, i);
              m_lightDefinitions[0].matrix[i]    = make_float4(row.x, row.y, row.z, row.w);
              row = glm::row(matRotationInv, i);
              m_lightDefinitions[0].matrixInv[i] = make_float4(row.x, row.y, row.z, row.w);
            }
            m_isDirtyLights = true;
          }
        }
        ImGui::Separator();
      }
      
      if (!m_lights.empty()) // KHR_lights_punctual.
      {
        // For all other lights defined by the KHR_lights_punctual show only the currently selected one.
        // FIXME Implement interactive manipulation of the position and orientation of the current light inside the viewport via the trackball.
        std::string labelCombo = std::to_string(m_indexLight) + std::string(") ") + std::string(m_lights[m_indexLight].name); 

        if (ImGui::BeginCombo("Light", labelCombo.c_str()))
        {
          // Add selectable lights to the combo box.
          for (size_t i = 0; i < m_lights.size(); ++i)
          {
            bool isSelected = (i == m_indexLight);

            std::string label = std::to_string(i) + std::string(") ") + std::string(m_lights[i].name);

            if (ImGui::Selectable(label.c_str(), isSelected))
            {
              if (m_indexLight != i)
              {
                m_indexLight = i; 
              }
            }
            if (isSelected)
            {
              ImGui::SetItemDefaultFocus();
            }
          }
          ImGui::EndCombo();
        }

        // Now show the light parameters of the currently selected light. 
        dev::Light& light = m_lights[m_indexLight];

        if (ImGui::ColorEdit3("color", &light.color.x))
        {
          m_isDirtyLights = true; // Next render() call will update the device side data.
        }
        if (ImGui::DragFloat("intensity", &light.intensity, 0.001f, 0.0f, 10000.0f))
        {
          m_isDirtyLights = true;
        }

        if (light.type != 2) // point or spot
        {
          MY_ASSERT(m_sceneExtent.isValid());
          // Pick a maximum range for the GUI  which is well below the RT_DEFAULT_MAX.
          if (ImGui::DragFloat("range", &light.range, 0.001f, 0.0f, 10.0f * m_sceneExtent.getMaxDimension()))
          {
            m_isDirtyLights = true;
          }
        }
        if (light.type == 1) // spot
        {
          bool isDirtyCone = false;

          //if (ImGui::SliderAngle("inner cone angle", &light.innerConeAngle, 0.0f, glm::degrees(light.outerConeAngle))) // These show only full degrees.
          if (ImGui::SliderFloat("inner cone angle", &light.innerConeAngle, 0.0f, light.outerConeAngle))
          {
            isDirtyCone = true;
            m_isDirtyLights = true;
          }
          //if (ImGui::SliderAngle("outer cone angle", &light.outerConeAngle, glm::degrees(light.innerConeAngle), 90.0f))
          if (ImGui::SliderFloat("outer cone angle", &light.outerConeAngle, light.innerConeAngle, 0.5f * M_PIf))
          {
            isDirtyCone = true;
            m_isDirtyLights = true;
          }
        
          // innerConeAngle must be less than outerConeAngle!
          if (isDirtyCone && light.innerConeAngle >= light.outerConeAngle)
          {
            const float delta = 0.001f;
            if (light.innerConeAngle + delta <= 0.5f * M_PIf) // Room to increase outer angle?
            {
              light.outerConeAngle = light.innerConeAngle + delta;
            }
            else // inner angle to near to maximum cone angle.
            {
              light.innerConeAngle = light.outerConeAngle - delta; // Shrink inner cone angle.
            }
          }
        }
      } // End of m_lights.
    }
  }

  ImGui::PopItemWidth();
  ImGui::End();

  if (m_font != nullptr)
  {
    ImGui::PopFont();
  }
}


void Application::guiEventHandler()
{
  const ImGuiIO& io = ImGui::GetIO();

  if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) // SPACE key toggles the GUI window display.
  {
    m_isVisibleGUI = !m_isVisibleGUI;
  }
  if (ImGui::IsKeyPressed(ImGuiKey_P, false)) // Key P: Save the current output buffer with tonemapping into a *.png file.
  {
    MY_VERIFY( screenshot(true) );
  }
  if (ImGui::IsKeyPressed(ImGuiKey_H, false)) // Key H: Save the current linear output buffer into a *.hdr file.
  {
    MY_VERIFY(screenshot(false));
  }
  if (ImGui::IsKeyPressed(ImGuiKey_W)) // Key W: Camera moves Fwd
  {
    cameraTranslate(0.0f, 0.0f, 1.0f);
  }
  if (ImGui::IsKeyPressed(ImGuiKey_S)) // Key S: Camera moves Back
  {
    cameraTranslate(0.0f, 0.0f, -1.0f);
  }
  if (ImGui::IsKeyPressed(ImGuiKey_A)) // Key A: Camera moves Left
  {
    cameraTranslate(-1.0f, 0.0f, 0.0f);
  }
  if (ImGui::IsKeyPressed(ImGuiKey_D)) // Key D: Camera moves Right
  {
    cameraTranslate(1.0f, 0.0f, 0.0f);
  }
  if (ImGui::IsKeyPressed(ImGuiKey_Q)) // Key Q: Camera moves Down
  {
    cameraTranslate(0.0, -1.0f, 0.0f);
  }
  if (ImGui::IsKeyPressed(ImGuiKey_E)) // Key E: Camera moves Up
  {
    cameraTranslate(0.0f, 1.0f, 0.0f);
  }

  // Client-relative mouse coordinates when ImGuiConfigFlags_ViewportsEnable is off.
  ImVec2 mousePosition = ImGui::GetMousePos();
  // With ImGuiConfigFlags_ViewportsEnable set, mouse coordinates are relative to the primary OS monitor!
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
  {
    // Subtract the main window's client position from the OS mouse position to get the client relative position again.
    mousePosition -= ImGui::GetMainViewport()->Pos;
  }
  const int x = int(mousePosition.x);
  const int y = int(mousePosition.y);
  
  switch (m_guiState)
  {
    case GUI_STATE_NONE:
      if (!io.WantCaptureMouse) // Only allow camera interactions to begin when not interacting with the GUI.
      {
        if (ImGui::IsMouseDown(0)) // LMB down event?
        {
          if (io.KeyCtrl)
          {
            // Any picking.x position >= 0.0f will trigger the material picking inside the next render() call.
            m_launchParameters.picking = getPickingCoordinate(x, y);
          }
          else
          {
            m_trackball.startTracking(x, y);
            m_guiState = GUI_STATE_ORBIT;
          }
        }
        else if (ImGui::IsMouseDown(1)) // RMB down event?
        {
          m_trackball.startTracking(x, y);
          m_guiState = GUI_STATE_DOLLY;
        }
        else if (ImGui::IsMouseDown(2)) // MMB down event?
        {
          m_trackball.startTracking(x, y);
          m_guiState = GUI_STATE_PAN;
        }
        else if (io.MouseWheel != 0.0f) // Mouse wheel event?
        {
          m_trackball.zoom(io.MouseWheel);
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
        m_trackball.setViewMode(dev::Trackball::LookAtFixed);
        m_trackball.orbit(x, y);
      }
      break;

    case GUI_STATE_DOLLY:
      if (ImGui::IsMouseReleased(1)) // RMB released? End of dolly mode.
      {
        m_guiState = GUI_STATE_NONE;
      }
      else
      {
        m_trackball.dolly(x, y);
      }
      break;

    case GUI_STATE_PAN:
      if (ImGui::IsMouseReleased(2)) // MMB released? End of pan mode.
      {
        m_guiState = GUI_STATE_NONE;
      }
      else
      {
        m_trackball.pan(x, y);
      }
      break;
  }
}


std::vector<char> Application::readData(std::string const& filename)
{
  std::ifstream fileStream(filename, std::ios::binary);

  if (fileStream.fail())
  {
    std::cerr << "ERROR: readData() Failed to open file " << filename << '\n';
    return std::vector<char>();
  }

  // Get the size of the file in bytes.
  fileStream.seekg(0, fileStream.end);
  std::streamsize size = fileStream.tellg();
  fileStream.seekg (0, fileStream.beg);

  if (size <= 0)
  {
    std::cerr << "ERROR: readData() File size of " << filename << " is <= 0.\n";
    return std::vector<char>();
  }

  std::vector<char> data(size);

  fileStream.read(data.data(), size);

  if (fileStream.fail())
  {
    std::cerr << "ERROR: readData() Failed to read file " << filename << '\n';
    return std::vector<char>();
  }

  return data;
}





void Application::loadGLTF(const std::filesystem::path& path)
{
  std::cout << "loadGTF(" << path << ")\n"; // DEBUG

  if (!std::filesystem::exists(path))
  {
    std::cerr << "WARNING: loadGLTF() filename " << path << " not found.\n";
    throw std::runtime_error("loadGLTF() File not found");
  }

  m_sceneExtent.toInvalid();

  // Only the material extensions which are enabled inside the parser are actually filled
  // inside the fastgltf::Material and then transferred to the dev::Material inside initMaterials().
  fastgltf::Extensions extensions =
    fastgltf::Extensions::KHR_materials_anisotropy | 
    fastgltf::Extensions::KHR_materials_clearcoat |
    fastgltf::Extensions::KHR_materials_emissive_strength |
    fastgltf::Extensions::KHR_materials_ior |
    fastgltf::Extensions::KHR_materials_iridescence |
    fastgltf::Extensions::KHR_materials_sheen |
    fastgltf::Extensions::KHR_materials_specular |
    fastgltf::Extensions::KHR_materials_transmission |
    fastgltf::Extensions::KHR_materials_unlit |
    fastgltf::Extensions::KHR_materials_variants |
    fastgltf::Extensions::KHR_materials_volume |
    fastgltf::Extensions::KHR_mesh_quantization |
    fastgltf::Extensions::KHR_texture_transform |
    // added for some point-clouds:
    fastgltf::Extensions::EXT_meshopt_compression
    ;

  // The command line parameter --punctual (-p) <int> allows selecting support for the KHR_lights_punctual extension.
  if (m_punctual)
  {
    // point/directional/spot
    extensions |= fastgltf::Extensions::KHR_lights_punctual;
  }

  fastgltf::Parser parser(extensions);

  constexpr auto gltfOptions = fastgltf::Options::None 
    | fastgltf::Options::DontRequireValidAssetMember
    | fastgltf::Options::LoadExternalBuffers
    | fastgltf::Options::DecomposeNodeMatrices
    | fastgltf::Options::LoadExternalImages;

  fastgltf::GltfFileStream data(path);

  const auto type = fastgltf::determineGltfFileType(data);

  fastgltf::Expected<fastgltf::Asset> asset(fastgltf::Error::None);

  std::filesystem::path pathParent = path.parent_path();

  if (pathParent.empty())
  {
    pathParent = std::filesystem::path("./");
  }

  if (type == fastgltf::GltfType::glTF)
  {
    asset = parser.loadGltf(data, pathParent, gltfOptions);
  }
  else if (type == fastgltf::GltfType::GLB)
  {
    asset = parser.loadGltfBinary(data, pathParent, gltfOptions);
  }
  else // if (type == Invalid)
  {
    std::cerr << "ERROR: determineGltfFileType returned Invalid\n";
    throw std::runtime_error("loadGLTF() Invalid file type");
  }

  if (asset.error() != fastgltf::Error::None)
  {
    std::cerr << "ERROR: loadGLTF() failed with error '" << fastgltf::getErrorMessage(asset.error()) << "'\n";
    throw std::runtime_error("loadGLTF() Failed");
  }

//#if !defined(NDEBUG)
//  fastgltf::validate(asset.get());
//  if (asset.error() != fastgltf::Error::None)
//  {
//    std::cerr << "ERROR: loadGLTF() asset validation failed with error '" << fastgltf::getErrorMessage(asset.error()) << "'\n";
//    throw std::runtime_error("loadGLTF() Failed");
//  }
//#endif

  m_asset = std::move(asset.get());
}


void Application::initNodes()
{
  m_nodes.clear();
  m_nodes.reserve(m_asset.nodes.size());

  std::cout << m_asset.nodes.size() << " node(s) to initialize" << std::endl;
  for (const fastgltf::Node& gltf_node : m_asset.nodes)
  {
    dev::Node& node = m_nodes.emplace_back();

    if (gltf_node.skinIndex.has_value())
    {
      node.indexSkin = static_cast<int>(gltf_node.skinIndex.value());
    }
    
    if (gltf_node.meshIndex.has_value())
    {
      node.indexMesh = static_cast<int>(gltf_node.meshIndex.value());

      // Provide a destination for the interpolateWeight() routine.
      const dev::HostMesh& hostMesh = m_hostMeshes[node.indexMesh];

      if (0 < hostMesh.numTargets) // Mesh has any morph targets?
      {
        // Resize the weights to the number of morph targets. This is the stride for the interpolateWeight()!
        // Initialize with zero to get the original attributes by default.
        node.weights.resize(hostMesh.numTargets, 0.0f);
        
        // If the host mesh has weights, copy them here because explicit node.weights have precedence and overwrite these below.
        if (!hostMesh.weights.empty())
        {
          memcpy(node.weights.data(), hostMesh.weights.data(), hostMesh.weights.size() * sizeof(float));
          
          node.morphMode = dev::Node::MORPH_MESH_WEIGHTS; // Can reuse the same morphed DeviceMesh under different nodes.
        }
      }
    }

    // Morph weights on the node, have precedence over mesh.weights.
    // When that happens, there needs to be a unique DeviceMesh GAS for this morphed mesh.
    if (!gltf_node.weights.empty())
    {
      node.weights.resize(gltf_node.weights.size());

      memcpy(node.weights.data(), gltf_node.weights.data(), gltf_node.weights.size() * sizeof(float)); // This overwrites potential weights on the host mesh.

      node.morphMode = dev::Node::MORPH_NODE_WEIGHTS; // Requires a unique DeviceMesh per node!
    }

    if (gltf_node.cameraIndex.has_value())
    {
      node.indexCamera = static_cast<int>(gltf_node.cameraIndex.value());
    }
    
    if (gltf_node.lightIndex.has_value())
    {
      node.indexLight = static_cast<int>(gltf_node.lightIndex.value());
    }
    
    // Matrix and TRS values are mutually exclusive according to the spec.
    if (const fastgltf::math::fmat4x4* matrix = std::get_if<fastgltf::math::fmat4x4>(&gltf_node.transform))
    {
      node.matrix = glm::make_mat4x4(matrix->data());

      node.isDirtyMatrix = false;
    }
    else if (const fastgltf::TRS* transform = std::get_if<fastgltf::TRS>(&gltf_node.transform))
    {
      // Warning: The quaternion to mat4x4 conversion here is not correct with all versions of GLM.
      // glTF provides the quaternion as (x, y, z, w), which is the same layout GLM used up to version 0.9.9.8.
      // However, with commit 59ddeb7 (May 2021) the default order was changed to (w, x, y, z).
      // You could either define GLM_FORCE_QUAT_DATA_XYZW to return to the old layout,
      // or you could use the recently added static factory constructor glm::quat::wxyz(w, x, y, z),
      // which guarantees the parameter order.
      // => 
      // Using GLM version 0.9.9.9 (or newer) and glm::quat::wxyz(w, x, y, z).
      // If this is not compiling your glm version is too old!
      node.translation = glm::make_vec3(transform->translation.data());
      node.rotation    = glm::quat::wxyz(transform->rotation[3], transform->rotation[0], transform->rotation[1], transform->rotation[2]);
      node.scale       = glm::make_vec3(transform->scale.data());

      node.isDirtyMatrix = true;
    }
  }
}


void Application::traverseUpdateSceneExtent(size_t gltfNodeIndex, const glm::mat4x4& mParent)
{
  const fastgltf::Node& gltfNode = m_asset.nodes[gltfNodeIndex];
  const glm::mat4x4 mLocal       = detail::toGLMTransform(gltfNode.transform);
  const glm::mat4x4 toWorld      = mParent * mLocal;

  if (gltfNode.meshIndex.has_value())
  {
    // scan the primitive's vertices, transform to world, update the bbox
    const fastgltf::Mesh& mesh = m_asset.meshes[gltfNode.meshIndex.value()];
    for (const auto& prim : mesh.primitives)
    {
      if (prim.type == fastgltf::PrimitiveType::Points || prim.type == fastgltf::PrimitiveType::Triangles)
      {
        auto it = m_primitiveToHostBuffer.find(&prim);
        if (it == m_primitiveToHostBuffer.end())
        {
          std::cerr << "ERROR: can't find a primitive when computing the scene extent!" << std::endl;
          continue;
        }
        HostBuffer const* buffer = it->second;

        // transform the buffer's positions and update the extent
        const float* pVtx = reinterpret_cast<const float*>(buffer->h_ptr);
        for (uint32_t i = 0; i < buffer->count; ++i, pVtx += 3)
        {
          const glm::vec4 p = toWorld * glm::vec4(pVtx[0], pVtx[1], pVtx[2], 1.0f);
          m_sceneExtent.update(p.x, p.y, p.z);
        }
      }
    }
  }
  // recurse
  for (size_t childId : gltfNode.children)
  {
    traverseUpdateSceneExtent(childId, toWorld);
  }
  if (m_sceneExtent.isValid() == false)
  {
    m_sceneExtent.fixSize();
  }
}

// Update the scene extent: finds out the scene bbox in world space: applies the chain of transforms to
// the positions in the primitives.
// Needs to know which nodes are roots.
void Application::initSceneExtent()
{
  MY_ASSERT(m_indexScene < m_asset.scenes.size());
  const fastgltf::Scene& scene = m_asset.scenes[m_indexScene];
  m_sceneExtent.toInvalid();

  glm::mat4x4 mIdentity = glm::identity<glm::mat4>();
  // all root nodes:
  for (const size_t indexNode : scene.nodeIndices)
  {
    traverseUpdateSceneExtent(indexNode, mIdentity);
  }
  m_sceneExtent.print();
}


void Application::initSkins()
{
  m_skins.clear();
  m_skins.reserve(m_asset.skins.size());

  for (const fastgltf::Skin& gltf_skin : m_asset.skins)
  {
    dev::Skin& skin = m_skins.emplace_back();

    skin.name = gltf_skin.name;
    skin.skeleton = (gltf_skin.skeleton.has_value()) ? static_cast<int>(gltf_skin.skeleton.value()) : -1;

    const int indexAccessor = (gltf_skin.inverseBindMatrices.has_value()) ? static_cast<int>(gltf_skin.inverseBindMatrices.value()) : -1;
    utils::createHostBuffer("inverseBindMatrices", m_asset, indexAccessor, fastgltf::AccessorType::Mat4, fastgltf::ComponentType::Float, 0.0f, skin.inverseBindMatrices);

    skin.joints.reserve(gltf_skin.joints.size());
    for (const size_t joint : gltf_skin.joints)
    {
      skin.joints.push_back(joint);
    }

    skin.matrices.resize(skin.joints.size());
    skin.matricesIT.resize(skin.joints.size());
  }
}


void Application::initImages()
{
  // When calling this more than once, make sure the existing resources are destroyed.
  for (cudaArray_t& image : m_images)
  {
    CUDA_CHECK( cudaFreeArray(image) );
  }
  m_images.clear();

  // FIXME This only supports 8 bit component images!
  // Images. Load all up-front for simplicity.
  for (const fastgltf::Image& image : m_asset.images)
  {
    std::visit(fastgltf::visitor {
      [](const auto& /* arg */) {
      },
      
      [&](const fastgltf::sources::URI& filePath) {
        MY_ASSERT(filePath.fileByteOffset == 0); // No offsets supported with stbi.
        MY_ASSERT(filePath.uri.isLocalPath());   // Loading only local files.
        int width;
        int height;
        int components;

        const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());

        unsigned char* data = stbi_load(path.c_str(), &width, &height, &components, 4);

        if (data != nullptr)
        {
          addImage(width, height, 8, 4, data);
        }
        else
        {
          std::cout << "ERROR: stbi_load() returned nullptr on image " << image.name << '\n';
          const unsigned char texel[4] = { 0xFF, 0x00, 0xFF, 0xFF };
          addImage(1, 1, 8, 4, texel); // DEBUG Error image is 1x1 RGBA8 magenta opaque.
        }
        
        stbi_image_free(data);
      },
      
      [&](const fastgltf::sources::Array& vector) {
        int width;
        int height;
        int components;

        unsigned char* data = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(vector.bytes.data()),
                                                    static_cast<int>(vector.bytes.size()),
                                                    &width, &height, &components, 4);

        if (data != nullptr)
        {
          addImage(width, height, 8, 4, data);
        }
        else
        {
          std::cout << "ERROR: stbi_load() returned nullptr on image " << image.name << '\n';
          const unsigned char texel[4] = { 0xFF, 0x00, 0xFF, 0xFF };
          addImage(1, 1, 8, 4, texel); // DEBUG Error image is 1x1 RGBA8 magenta opaque.
        }

        stbi_image_free(data);
      },

      [&](const fastgltf::sources::BufferView& view) {
        const auto& bufferView = m_asset.bufferViews[view.bufferViewIndex];
        const auto& buffer     = m_asset.buffers[bufferView.bufferIndex];

        std::visit(fastgltf::visitor {
          // We only care about Arrays here, because we specify LoadExternalBuffers, meaning all buffers are already loaded into a vector.
          [](const auto& /* arg */)
          {
          },

          [&](const fastgltf::sources::Array& vector)
          {
            int width;
            int height;
            int components;
            
            unsigned char* data = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(vector.bytes.data()) + bufferView.byteOffset, 
                                                        static_cast<int>(bufferView.byteLength), 
                                                        &width, &height, &components, 4);
       
            if (data != nullptr)
            {
              addImage(width, height, 8, 4, data);
            }
            else
            {
              std::cout << "ERROR: stbi_load() returned nullptr on image " << image.name << '\n';
              const unsigned char texel[4] = { 0xFF, 0x00, 0xFF, 0xFF };
              addImage(1, 1, 8, 4, texel); // DEBUG Error image is 1x1 RGBA8 magenta opaque.
            }
            
            stbi_image_free(data);
          }
        }, buffer.data);
      },
    }, image.data);
  }
}


void Application::initTextures()
{
  // When calling this more than once, make sure the existing resources are destroyed.
  for (cudaTextureObject_t& sampler : m_samplers)
  {
    CUDA_CHECK( cudaDestroyTextureObject(sampler) );
  }
  m_samplers.clear();

  if (m_asset.textures.empty())
  {
    return;
  }

  // glTF requires sRGB for baseColor, specularColor, sheenColor and emissive textures inside the texture interpolation.
  // Doing sRGB adjustments with pow(rgb, 2.2) inside the shader after the texture lookup is too late.
  // TextureLinearInterpolationTest.gltf will only pass with sRGB done inside the texture object itself.
  std::vector<int> sRGB(m_asset.textures.size(), 0); 

  // Run over all materials inside the asset and set the sRGB flag for all textures
  // which are used as baseColorTexture, specularColorTexture, sheenColor, or emissiveTexture.
  for (const fastgltf::Material& material : m_asset.materials)
  {
    if (material.pbrData.baseColorTexture.has_value())
    {
      const fastgltf::TextureInfo& textureInfo = material.pbrData.baseColorTexture.value();
      sRGB[textureInfo.textureIndex] = 1;
    }
    if (material.emissiveTexture.has_value())
    {
      const fastgltf::TextureInfo& textureInfo = material.emissiveTexture.value();
      sRGB[textureInfo.textureIndex] = 1;
    }
    if (material.specular != nullptr && material.specular->specularColorTexture.has_value())
    {
      const fastgltf::TextureInfo& textureInfo = material.specular->specularColorTexture.value();
      sRGB[textureInfo.textureIndex] = 1;
    }
    if (material.sheen != nullptr && material.sheen->sheenColorTexture.has_value())
    {
      const fastgltf::TextureInfo& textureInfo = material.sheen->sheenColorTexture.value();
      sRGB[textureInfo.textureIndex] = 1;
    }
  }

  // Textures. These refer to previously loaded images.
  for (size_t i = 0; i < m_asset.textures.size(); ++i)
  {
    const fastgltf::Texture& texture = m_asset.textures[i];

    // Default to wrap repeat and linear filtering when there is no sampler.
    cudaTextureAddressMode address_s = cudaAddressModeWrap;
    cudaTextureAddressMode address_t = cudaAddressModeWrap;
    cudaTextureFilterMode  filter    = cudaFilterModeLinear;

    if (texture.samplerIndex.has_value())
    {
      MY_ASSERT(texture.samplerIndex.value() < m_asset.samplers.size());
      const auto& sampler = m_asset.samplers[texture.samplerIndex.value()];

      address_s = utils::getTextureAddressMode(sampler.wrapS);
      address_t = utils::getTextureAddressMode(sampler.wrapT);

      if (sampler.minFilter.has_value())
      {
        fastgltf::Filter minFilter = sampler.minFilter.value();

        switch (minFilter)
        {
          // This renderer is not downloading mipmaps. 
          // Pick the filter depending on the 2D filtering which is the first.
          case fastgltf::Filter::Nearest:
          case fastgltf::Filter::NearestMipMapNearest:
          case fastgltf::Filter::NearestMipMapLinear:
            filter = cudaFilterModePoint;
            break;

          case fastgltf::Filter::Linear:
          case fastgltf::Filter::LinearMipMapNearest:
          case fastgltf::Filter::LinearMipMapLinear:
          default:
            filter = cudaFilterModeLinear;
            break;
        }
      }
    }
    
    MY_ASSERT(texture.imageIndex.has_value());
    addSampler(address_s, address_t, filter, texture.imageIndex.value(), sRGB[i]);
  }
}


void Application::initMaterials()
{
  m_materialsOrg.clear();
  m_materials.clear();
  m_indexMaterial = 0; // Reset the GUI indices.
  m_indexVariant  = 0;

  // Materials
  for (size_t index = 0; index < m_asset.materials.size(); ++index)
  {
    //std::cout << "Processing glTF material: '" << material.name << "'\n";

    const fastgltf::Material& material = m_asset.materials[index];

    MaterialData mtl;

    mtl.index = static_cast<int>(index); // To be able to identify the material during picking.

    mtl.doubleSided = material.doubleSided;

    switch (material.alphaMode)
    {
      case fastgltf::AlphaMode::Opaque:
        mtl.alphaMode = MaterialData::ALPHA_MODE_OPAQUE;
        break;

      case fastgltf::AlphaMode::Mask:
        mtl.alphaMode   = MaterialData::ALPHA_MODE_MASK;
        mtl.alphaCutoff = material.alphaCutoff;
        break;

      case fastgltf::AlphaMode::Blend:
        mtl.alphaMode = MaterialData::ALPHA_MODE_BLEND;
        break;

      default:
        std::cerr << "ERROR: Invalid material alpha mode. Using opaque\n";
        mtl.alphaMode = MaterialData::ALPHA_MODE_OPAQUE;
        break;
    }

    mtl.baseColorFactor = make_float4(material.pbrData.baseColorFactor[0], 
                                      material.pbrData.baseColorFactor[1], 
                                      material.pbrData.baseColorFactor[2], 
                                      material.pbrData.baseColorFactor[3]);
    if (material.pbrData.baseColorTexture.has_value())
    {
      detail::parseTextureInfo(m_samplers, material.pbrData.baseColorTexture.value(), mtl.baseColorTexture);
    }

    mtl.metallicFactor  = material.pbrData.metallicFactor;
    mtl.roughnessFactor = material.pbrData.roughnessFactor;
    if (material.pbrData.metallicRoughnessTexture.has_value())
    {
      detail::parseTextureInfo(m_samplers, material.pbrData.metallicRoughnessTexture.value(), mtl.metallicRoughnessTexture);
    }

    if (material.normalTexture.has_value())
    {
      const auto& normalTextureInfo = material.normalTexture.value();
      
      mtl.normalTextureScale = normalTextureInfo.scale;
      detail::parseTextureInfo(m_samplers, normalTextureInfo, mtl.normalTexture);
    }  

    // Ambient occlusion should not really be required with a global illumination renderer,
    // but many glTF models are very low-resolution geometry and details are baked into normal and occlusion maps.
    if (material.occlusionTexture.has_value())
    {
      const auto& occlusionTextureInfo = material.occlusionTexture.value();

      mtl.occlusionTextureStrength = occlusionTextureInfo.strength;
      detail::parseTextureInfo(m_samplers, occlusionTextureInfo, mtl.occlusionTexture);
    }  
    
    mtl.emissiveStrength = material.emissiveStrength; // KHR_materials_emissive_strength
    mtl.emissiveFactor = make_float3(material.emissiveFactor[0],
                                     material.emissiveFactor[1],
                                     material.emissiveFactor[2]);
    if (material.emissiveTexture.has_value())
    {
      detail::parseTextureInfo(m_samplers, material.emissiveTexture.value(), mtl.emissiveTexture);
    }  

    // Set material.flags bits to indicate which Khronos material extension is used and has data.
    // This is only evaluated for the KHH_materials_volume so far because that affects the face culling
    // Volumes require double-sided geometry even when the glTF file didn't specify it.
    mtl.flags = 0;

    // KHR_materials_ior
    // Not handled as optional extension inside fastgltf.
    // It's always present and defaults to 1.5 when not set inside the asset.
    mtl.ior = material.ior;

    // KHR_materials_specular
    if (material.specular != nullptr) 
    {
      mtl.flags |= FLAG_KHR_MATERIALS_SPECULAR;

      mtl.specularFactor = material.specular->specularFactor;
      if (material.specular->specularTexture.has_value())
      {
        detail::parseTextureInfo(m_samplers, material.specular->specularTexture.value(), mtl.specularTexture);
      }
      mtl.specularColorFactor = make_float3(material.specular->specularColorFactor[0],
                                            material.specular->specularColorFactor[1],
                                            material.specular->specularColorFactor[2]);
      if (material.specular->specularColorTexture.has_value())
      {
        detail::parseTextureInfo(m_samplers, material.specular->specularColorTexture.value(), mtl.specularColorTexture);
      }
    }

    // KHR_materials_transmission
    if (material.transmission != nullptr) 
    {
      mtl.flags |= FLAG_KHR_MATERIALS_TRANSMISSION;

      mtl.transmissionFactor = material.transmission->transmissionFactor;
      if (material.transmission->transmissionTexture.has_value())
      {
        detail::parseTextureInfo(m_samplers, material.transmission->transmissionTexture.value(), mtl.transmissionTexture);
      }
    }

    // KHR_materials_volume
    if (material.volume != nullptr) 
    {
      mtl.flags |= FLAG_KHR_MATERIALS_VOLUME;

      // Some glTF models like the IridescenceLamp.gltf use only the thicknessFactor to define the volume.
      // HACK The ray tracer ignores the thicknessFactor except when it's 0.0f,
      // which is one condition for thin-walled materials even when FLAG_KHR_MATERIALS_VOLUME is set.
      mtl.thicknessFactor = material.volume->thicknessFactor;
      //if (material.volume->thicknessTexture.has_value())
      //{
      //  detail::parseTextureInfo(m_samplers, material.volume->thicknessTexture.value(), mtl.thicknessTexture);
      //}
      // The attenuationDistance default is +inf which effectively disables volume absorption.
      // The raytracer only enables volume absorption for attenuationDistance values less than RT_DEFAULT_MAX.
      mtl.attenuationDistance = material.volume->attenuationDistance;
      mtl.attenuationColor = make_float3(material.volume->attenuationColor[0],
                                         material.volume->attenuationColor[1],
                                         material.volume->attenuationColor[2]);
    }

    // KHR_materials_clearcoat
    if (material.clearcoat != nullptr) 
    {
      mtl.flags |= FLAG_KHR_MATERIALS_CLEARCOAT;

      mtl.clearcoatFactor = material.clearcoat->clearcoatFactor;
      if (material.clearcoat->clearcoatTexture.has_value())
      {
        detail::parseTextureInfo(m_samplers, material.clearcoat->clearcoatTexture.value(), mtl.clearcoatTexture);
      }
      mtl.clearcoatRoughnessFactor = material.clearcoat->clearcoatRoughnessFactor;
      if (material.clearcoat->clearcoatRoughnessTexture.has_value())
      {
        detail::parseTextureInfo(m_samplers, material.clearcoat->clearcoatRoughnessTexture.value(), mtl.clearcoatRoughnessTexture);
      }
      if (material.clearcoat->clearcoatNormalTexture.has_value())
      {
        detail::parseTextureInfo(m_samplers, material.clearcoat->clearcoatNormalTexture.value(), mtl.clearcoatNormalTexture);
        
        // If the clearcoatNormalTexture is the same as the normalTexture, then let the shader apply
        // the same normalTextureScale to match the clearcoat normal to the material normal.
        // (The Texture fields are all default initialized, so this comparison always works with valid data.)
        mtl.isClearcoatNormalBaseNormal = (mtl.clearcoatNormalTexture == mtl.normalTexture);
      }
    }

    // KHR_materials_sheen
    if (material.sheen != nullptr) 
    {
      mtl.flags |= FLAG_KHR_MATERIALS_SHEEN;

      mtl.sheenColorFactor = make_float3(material.sheen->sheenColorFactor[0],
                                         material.sheen->sheenColorFactor[1],
                                         material.sheen->sheenColorFactor[2]);
      if (material.sheen->sheenColorTexture.has_value())
      {
        detail::parseTextureInfo(m_samplers, material.sheen->sheenColorTexture.value(), mtl.sheenColorTexture);
      }
      mtl.sheenRoughnessFactor = material.sheen->sheenRoughnessFactor;
      if (material.sheen->sheenRoughnessTexture.has_value())
      {
        detail::parseTextureInfo(m_samplers, material.sheen->sheenRoughnessTexture.value(), mtl.sheenRoughnessTexture);
      }
    }

    // KHR_materials_anisotropy
    if (material.anisotropy != nullptr) 
    {
      mtl.flags |= FLAG_KHR_MATERIALS_ANISOTROPY;

      mtl.anisotropyStrength = material.anisotropy->anisotropyStrength;
      mtl.anisotropyRotation = material.anisotropy->anisotropyRotation;
      if (material.anisotropy->anisotropyTexture.has_value())
      {
        detail::parseTextureInfo(m_samplers, material.anisotropy->anisotropyTexture.value(), mtl.anisotropyTexture);
      }
    }

    // KHR_materials_iridescence
    if (material.iridescence != nullptr) 
    {
      mtl.flags |= FLAG_KHR_MATERIALS_IRIDESCENCE;

      mtl.iridescenceFactor = material.iridescence->iridescenceFactor;
      if (material.iridescence->iridescenceTexture.has_value())
      {
        detail::parseTextureInfo(m_samplers, material.iridescence->iridescenceTexture.value(), mtl.iridescenceTexture);
      }
      mtl.iridescenceIor = material.iridescence->iridescenceIor;
      mtl.iridescenceThicknessMinimum = material.iridescence->iridescenceThicknessMinimum;
      mtl.iridescenceThicknessMaximum = material.iridescence->iridescenceThicknessMaximum;
      if (material.iridescence->iridescenceThicknessTexture.has_value())
      {
        detail::parseTextureInfo(m_samplers, material.iridescence->iridescenceThicknessTexture.value(), mtl.iridescenceThicknessTexture);
      }
    }

    // KHR_materials_unlit
    mtl.unlit = material.unlit;

    //debugDumpMaterial(mtl); // DEBUG 

    m_materialsOrg.push_back(mtl); // The original data inside the asset.
    m_materials.push_back(mtl);    // The materials changed by the GUI.
  }
}




  // Read glTF data and create host buffers.
  // Clear device meshes and reserve host storage for them (device memory is not alloc'd here, see createDeviceMesh()).
  // Take care of morph targets, material index, mappings.
  // Handle triangle and point meshes. TODO lines.
  //
  // While reading, update the scene AABB, needed to compute a scene-dependent sphere radius, before building the AS.
  // After building the AS, we will read m_sceneExtent from device, now with the spheres' sizes into account.

void Application::initMeshes()
{
  m_hostMeshes.clear();
  m_deviceMeshes.clear();
  m_mapKeyTupleToDeviceMeshIndex.clear();
  m_hostMeshes.reserve(m_asset.meshes.size());

  // When each host mesh is only used once, this reduces the DeviceMesh move operator calls during m_deviceMeshes.emplace_back()
  m_deviceMeshes.reserve(m_asset.meshes.size());
  m_sceneExtent.toInvalid();
  m_primitiveToHostBuffer.clear();

  uint32_t nPoints = 0;
  uint32_t nTris = 0;
  uint32_t nOthers = 0;
  uint32_t nPositions = 0;

  // ALL MESHES
  int dbgMeshId = -1;

  for (const fastgltf::Mesh& gltf_mesh : m_asset.meshes)
  {
    dbgMeshId++;
    // Unconditionally create a new empty dev::HostMesh to have the same index as into m_asset.meshes.
    // We'll append host primitives to each host mesh.
    dev::HostMesh& hostMesh = m_hostMeshes.emplace_back();

    hostMesh.name = gltf_mesh.name;
    //std::cout << " Reading glTF mesh " << hostMesh.name << std::endl;
    hostMesh.primitives.reserve(gltf_mesh.primitives.size()); // PERF This might be bigger than needed because only Triangles & Points are handled.

    // Morph weights on the mesh. Only used when the node.weights holding this mesh has no morph weights.
    if (!gltf_mesh.weights.empty())
    {
      hostMesh.weights.resize(gltf_mesh.weights.size());
      memcpy(hostMesh.weights.data(), gltf_mesh.weights.data(), gltf_mesh.weights.size() * sizeof(float));
    }

    // ALL PRIMITIVES IN THE MESH
    int dbgPrimId = -1;
    for (const fastgltf::Primitive& primitive : gltf_mesh.primitives)
    {
      ++dbgPrimId;

      // FIXME Implement all polygonal primitive modes and convert them to independent triangles.
      // FIXME Implement all primitive modes (points and lines) as well and convert them to spheres and linear curves.
      // (That wouldn't handle the "lines render as single pixel" in screen space GLTF specs.)
      // NOTE glTF primitive.type and mode are the same thing.

      const dev::PrimitiveType primitiveType{ detail::toDevPrimitiveType(primitive.type) };

      if (primitiveType == dev::PrimitiveType::Triangles)
        ++nTris;
      else if (primitiveType == dev::PrimitiveType::Points)
        ++nPoints;
      else if (primitiveType == dev::PrimitiveType::Undefined)
      {
        std::cerr << "glTF Primitive " << detail::getDevPrimitiveTypeName(primitive.type) << " not yet implemented" << std::endl;
        ++nOthers;
        continue;
      }
      else
      {
        std::cerr << "ERROR Found unknown primitive type during initMeshes()" << std::endl;
        MY_ASSERT(false);
      }

      // Create various host buffers for the primitive (a primitive can have 1 or more elements e.g. a tri-mesh or a point-cloud subset).
      // glTF specs don't forbid e.g. a point to have tangents, hence here we use the same host data (and C++ class).
      // The difference pops up when building inputs for optixAccelBuild(), we'll use the PrimitiveType then.

      // POSITION attribute must be present!
      auto itPosition = primitive.findAttribute("POSITION");
      if (itPosition == primitive.attributes.end()) // Meshes MUST have a position attribute.
      {
        std::cerr << "ERROR: primitive has no POSITION attribute, skipped.\n";
        continue;
      }

      // If we arrived here, the mesh contains at least one primitive.
      std::string name = "HostPrim_M" + std::to_string(dbgMeshId) + "_P" + std::to_string(dbgPrimId);

      // Append a new dev::HostPrimitive to the dev::HostMesh and fill its data.
      dev::HostPrimitive& hostPrim = hostMesh.createNewPrimitive(primitiveType, name); 

      // Integer indexAccessor type to allow -1 in utils::createHostBuffer() for optional attributes which won't change the DeviceBuffer.
      int indexAccessor = static_cast<int>(itPosition->accessorIndex);
      nPositions += utils::createHostBuffer("POSITION", m_asset, indexAccessor, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float, 1.0f, hostPrim.positions);

      m_primitiveToHostBuffer[ &primitive ] = &hostPrim.positions;
      
      if (primitiveType != dev::PrimitiveType::Points)
      {
        // might have indices
        indexAccessor = (primitive.indicesAccessor.has_value()) ? static_cast<int>(primitive.indicesAccessor.value()) : -1;
        utils::createHostBuffer("INDICES", m_asset, indexAccessor, fastgltf::AccessorType::Scalar, fastgltf::ComponentType::UnsignedInt, 0.0f, hostPrim.indices);
      }

      auto itNormal = primitive.findAttribute("NORMAL");
      indexAccessor = (itNormal != primitive.attributes.end()) ? static_cast<int>(itNormal->accessorIndex) : -1;
      // "When normals are not specified, client implementations MUST calculate flat normals and the provided tangents (if present) MUST be ignored."
      const bool allowTangents = (0 <= indexAccessor);
      utils::createHostBuffer("NORMAL", m_asset, indexAccessor, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float, 0.0f, hostPrim.normals);

      // "When tangents are not specified, client implementations SHOULD calculate tangents using default 
      // MikkTSpace algorithms with the specified vertex positions, normals, and texture coordinates associated with the normal texture."
      auto itTangent = primitive.findAttribute("TANGENT");
      indexAccessor = (itTangent != primitive.attributes.end() && allowTangents) ? static_cast<int>(itTangent->accessorIndex) : -1;
      // Expansion is unused here, but this would mean right-handed.
      utils::createHostBuffer("TANGENT", m_asset, indexAccessor, fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float, 1.0f, hostPrim.tangents);

      auto itColor = primitive.findAttribute("COLOR_0"); // Only supporting one color attribute.
      indexAccessor = (itColor != primitive.attributes.end()) ? static_cast<int>(itColor->accessorIndex) : -1;
      // This also handles alpha expansion of Vec3 colors to Vec4.
      utils::createHostBuffer("COLOR_0", m_asset, indexAccessor, fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float, 1.0f, hostPrim.colors); // Must have expansion == 1.0f!

      for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j)
      {
        const std::string strTexcoord = std::string("TEXCOORD_") + std::to_string(j);
        auto itTexcoord = primitive.findAttribute(strTexcoord);
        indexAccessor = (itTexcoord != primitive.attributes.end()) ? static_cast<int>(itTexcoord->accessorIndex) : -1;
        utils::createHostBuffer(strTexcoord.c_str(), m_asset, indexAccessor, fastgltf::AccessorType::Vec2, fastgltf::ComponentType::Float, 0.0f, hostPrim.texcoords[j]);
      }

      for (int j = 0; j < NUM_ATTR_JOINTS; ++j)
      {
        std::string joints_str = std::string("JOINTS_") + std::to_string(j);
        auto itJoints = primitive.findAttribute(joints_str);
        indexAccessor = (itJoints != primitive.attributes.end()) ? static_cast<int>(itJoints->accessorIndex) : -1;
        utils::createHostBuffer(joints_str.c_str(), m_asset, indexAccessor, fastgltf::AccessorType::Vec4, fastgltf::ComponentType::UnsignedShort, 0.0f, hostPrim.joints[j]);
      }

      for (int j = 0; j < NUM_ATTR_WEIGHTS; ++j)
      {
        std::string weights_str = std::string("WEIGHTS_") + std::to_string(j);
        auto itWeights = primitive.findAttribute(weights_str);
        indexAccessor = (itWeights != primitive.attributes.end()) ? static_cast<int>(itWeights->accessorIndex) : -1;
        utils::createHostBuffer(weights_str.c_str(), m_asset, indexAccessor, fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float, 0.0f, hostPrim.weights[j]);
      }

      // Morph targets.
      if (!primitive.targets.empty())
      {
        // First determine which attributes have morph targets before resizing the target vectors and targetPointers buffer
        hostPrim.maskTargets = 0;

        for (size_t i = 0; i < primitive.targets.size(); ++i)
        {
          auto itPosition = primitive.findTargetAttribute(i, "POSITION");
          hostPrim.maskTargets |= (itPosition != primitive.targets[i].end()) ? ATTR_POSITION : 0;

          auto itTangent = primitive.findTargetAttribute(i, "TANGENT");
          hostPrim.maskTargets |= (itTangent != primitive.targets[i].end()) ? ATTR_TANGENT : 0;

          auto itNormal = primitive.findTargetAttribute(i, "NORMAL");
          hostPrim.maskTargets |= (itNormal != primitive.targets[i].end()) ? ATTR_NORMAL : 0;

          for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j)
          {
            const std::string strTexcoord = std::string("TEXCOORD_") + std::to_string(j);

            auto itTexcoord = primitive.findTargetAttribute(i, strTexcoord);
            hostPrim.maskTargets |= (itTexcoord != primitive.targets[i].end()) ? ATTR_TEXCOORD_0 << j : 0;
          }

          auto itColor = primitive.findTargetAttribute(i, "COLOR_0");
          hostPrim.maskTargets |= (itColor != primitive.targets[i].end()) ? ATTR_COLOR_0 : 0;
        }

        // Only set the number of morph targets when there are some the renderer supports.
        if (hostPrim.maskTargets != 0)
        {
          hostPrim.numTargets = primitive.targets.size();

          // I need this number in case the hostMesh.weights is empty but node.weights are used for morphing.
          hostMesh.numTargets = std::max(hostMesh.numTargets, hostPrim.numTargets);
        }

        if (hostPrim.maskTargets & ATTR_POSITION)
        {
          hostPrim.positionsTarget.resize(hostPrim.numTargets);
        }
        if (hostPrim.maskTargets & ATTR_TANGENT)
        {
          hostPrim.tangentsTarget.resize(hostPrim.numTargets);
        }
        if (hostPrim.maskTargets & ATTR_NORMAL)
        {
          hostPrim.normalsTarget.resize(hostPrim.numTargets);
        }
        if (hostPrim.maskTargets & ATTR_COLOR_0)
        {
          hostPrim.colorsTarget.resize(hostPrim.numTargets);
        }
        for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j)
        {
          if (hostPrim.maskTargets & (ATTR_TEXCOORD_0 << j))
          {
            hostPrim.texcoordsTarget[j].resize(hostPrim.numTargets);
          }
        }

        // Target index. Size must match the morph weights array inside the mesh or node.
        if (hostPrim.maskTargets & ATTR_POSITION)
        {
          for (size_t i = 0; i < hostPrim.numTargets; ++i)
          {
            auto itTarget = primitive.findTargetAttribute(i, "POSITION");
            indexAccessor = (itTarget != primitive.targets[i].end()) ? static_cast<int>(itTarget->accessorIndex) : -1;
            utils::createHostBuffer("(morph) POSITION", m_asset, indexAccessor, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float, 0.0f, hostPrim.positionsTarget[i]);
          }
        }
        if (hostPrim.maskTargets & ATTR_TANGENT)
        {
          for (size_t i = 0; i < hostPrim.numTargets; ++i)
          {
            auto itTarget = primitive.findTargetAttribute(i, "TANGENT");
            indexAccessor = (itTarget != primitive.targets[i].end()) ? static_cast<int>(itTarget->accessorIndex) : -1;
            utils::createHostBuffer("(morph) TANGENT", m_asset, indexAccessor, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float, 0.0f, hostPrim.tangentsTarget[i]); // DAR FIXME PERF Try converting this to Vec4 and expansion == 0.0f)
          }
        }
        if (hostPrim.maskTargets & ATTR_NORMAL)
        {
          for (size_t i = 0; i < hostPrim.numTargets; ++i)
          {
            auto itTarget = primitive.findTargetAttribute(i, "NORMAL");
            indexAccessor = (itTarget != primitive.targets[i].end()) ? static_cast<int>(itTarget->accessorIndex) : -1;
            utils::createHostBuffer("(morph) NORMAL", m_asset, indexAccessor, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float, 0.0f, hostPrim.normalsTarget[i]);
          }
        }
        if (hostPrim.maskTargets & ATTR_COLOR_0)
        {
          for (size_t i = 0; i < hostPrim.numTargets; ++i)
          {
            // "When COLOR_n deltas use an accessor of "VEC3" type, their alpha components MUST be assumed to have a value of 0.0."
            // This is the sole reason for the utils::createHostBuffer() "expansion" argument!
            auto itTarget = primitive.findTargetAttribute(i, "COLOR_0");
            indexAccessor = (itTarget != primitive.targets[i].end()) ? static_cast<int>(itTarget->accessorIndex) : -1;
            utils::createHostBuffer("(morph) COLOR_0", m_asset, indexAccessor, fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float, 0.0f, hostPrim.colorsTarget[i]); // Must have expansion == 0.0f!
          }
        }
        for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j)
        {
          if (hostPrim.maskTargets & (ATTR_TEXCOORD_0 << j))
          {
            const std::string strTarget = std::string("TEXCOORD_") + std::to_string(j);

            for (size_t i = 0; i < hostPrim.numTargets; ++i)
            {
              auto itTarget = primitive.findTargetAttribute(i, strTarget);
              indexAccessor = (itTarget != primitive.targets[i].end()) ? static_cast<int>(itTarget->accessorIndex) : -1;
              utils::createHostBuffer(strTarget.c_str(), m_asset, indexAccessor, fastgltf::AccessorType::Vec2, fastgltf::ComponentType::Float, 0.0f, hostPrim.texcoordsTarget[j][i]);
            }
          }
        }
      }

      hostPrim.indexMaterial = (primitive.materialIndex.has_value()) ? static_cast<int32_t>(primitive.materialIndex.value()) : -1;

      // KHR_materials_variants
      for (size_t i = 0; i < primitive.mappings.size(); ++i)
      {
        const int index = primitive.mappings[i].has_value() ? static_cast<int>(primitive.mappings[i].value()) : hostPrim.indexMaterial;

        hostPrim.mappings.push_back(index);
      }

      // Derive the current material index.
      hostPrim.currentMaterial = (primitive.mappings.empty()) ? hostPrim.indexMaterial : hostPrim.mappings[m_indexVariant];
    } // for primitive
  } // for gltf_mesh

  // Here m_sceneExtent is not valid yet!

  std::cout
    << " *** Meshes found                        " << m_asset.meshes.size() << "\n"
    << " *** Triangle primitives found           " << nTris << "\n"
    << " *** Point       \"       \"               " << nPoints << "\n"
    << " *** Other       \"       \"               " << nOthers << " (ignored)\n"
    << " *** POSITION-s allocated in host memory " << nPositions << std::endl;
}


void Application::initCameras()
{
  m_isDefaultCamera = false;

  // If there is no camera inside the scene, generate a default perspective camera.
  // That simplifies the whole GUI and scene handling.
  if (m_asset.cameras.empty())
  {
    fastgltf::Camera::Perspective perspective;

    perspective.aspectRatio = 1.0f;
    perspective.yfov = 45.0f * M_PIf / 180.0f; // In radians.
    perspective.znear = 0.01f;

    fastgltf::Camera camera;

    camera.camera = perspective;

    m_asset.cameras.push_back(camera);

    m_isDefaultCamera = true; // This triggers an initialization of the default camera position and lookat insde initTrackball()
  }

  m_cameras.clear();
  m_cameras.reserve(m_asset.cameras.size());
  m_indexCamera = 0; // Reset the GUI index.

  for (const fastgltf::Camera& gltf_camera : m_asset.cameras)
  {
    // Just default initialize the camera inside the array to have the same indexing as m_asset.cameras.
    // If there is an error, this will be a default perspective camera.
    dev::Camera& camera = m_cameras.emplace_back();

    // At this time, the camera transformation matrix is unknown.
    // Just initialize position and up vectors with defaults and update them during scene node traversal later.
    if (const auto* pPerspective = std::get_if<fastgltf::Camera::Perspective>(&gltf_camera.camera))
    {
      const glm::vec3 pos(0.0f, 0.0f, 0.0f);
      const glm::vec3 up(0.0f, 1.0f, 0.0f);

      // yfov should be less than PI and must be > 0.0f to work. 
      MY_ASSERT(0.0f < pPerspective->yfov && pPerspective->yfov < M_PIf);
      const float yfov = pPerspective->yfov * 180.0f / M_PIf;

      // This value isn't used anyway because for perspective cameras the viewport defines the aspect ratio.
      const float aspectRatio = (pPerspective->aspectRatio.has_value() && pPerspective->aspectRatio.value() != 0.0f) 
                              ? pPerspective->aspectRatio.value() 
                              : 1.0f;
      
      camera.setPosition(pos);
      camera.setUp(up);
      camera.setFovY(yfov); // In degrees.
      camera.setAspectRatio(aspectRatio);
    }
    else if (const auto* pOrthograhpic = std::get_if<fastgltf::Camera::Orthographic>(&gltf_camera.camera))
    {
      const glm::vec3 pos(0.0f, 0.0f, 0.0f);
      const glm::vec3 up(0.0f, 1.0f, 0.0f);

      // The orthographic projection is always finite inside GLTF because znear and zfar are required.
      // But this defines an infinite projection from a plane at the position.
      camera.setPosition(pos);
      camera.setUp(up);
      camera.setFovY(-1.0f); // <= 0.0f means orthographic projection.
      camera.setMagnification(glm::vec2(pOrthograhpic->xmag, pOrthograhpic->ymag));
    }
    else
    {
      std::cerr << "ERROR: Unexpected camera type.\n";
    }
  }
}


void Application::initLights(const bool first)
{
  m_lights.clear();
  m_lights.reserve(m_asset.lights.size());
  m_indexLight = 0; // Reset the GUI index.
  m_isDirtyLights = true; // The m_lightDefinitions are filled and copied to the device inside updateLights().

  if (m_d_lightDefinitions != 0)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_d_lightDefinitions)) ); 
    m_d_lightDefinitions = 0;
  }

  for (const fastgltf::Light& gltf_light : m_asset.lights)
  {
    dev::Light& light = m_lights.emplace_back();

    // Shared values.
    light.name = gltf_light.name;
    // These types  match the renderer's order of light sampling callable programs.
    switch (gltf_light.type)
    {
      case fastgltf::LightType::Point:
        light.type = 0;
        break;
      case fastgltf::LightType::Spot:
        light.type = 1;
        break;
      case fastgltf::LightType::Directional:
        light.type = 2;
        break;
    }
    light.color = make_float3(gltf_light.color[0], gltf_light.color[1], gltf_light.color[2]);
    light.intensity = gltf_light.intensity;

    switch (gltf_light.type)
    {
      case fastgltf::LightType::Point:
        light.range = (gltf_light.range.has_value()) ? gltf_light.range.value() : RT_DEFAULT_MAX;
        break;

      case fastgltf::LightType::Spot:
        light.range = (gltf_light.range.has_value()) ? gltf_light.range.value() : RT_DEFAULT_MAX;
        light.innerConeAngle = (gltf_light.innerConeAngle.has_value()) ? gltf_light.innerConeAngle.value() : 0.0f;
        light.outerConeAngle = (gltf_light.outerConeAngle.has_value()) ? gltf_light.outerConeAngle.value() : 0.25f * M_PIf;
        break;

      case fastgltf::LightType::Directional:
        // No additional data.
        break;
    }

    light.matrix = glm::mat4(1.0f); // Identity. Updated during traverse() of the scene nodes.
  }

 const size_t numLights = m_lights.size() + ((m_missID == 0) ? 0 : 1);

  m_lightDefinitions.resize(numLights);

  if (first) // No need to reload or recreate the already existing environment light on drag-and-drop of a new glTF asset.
  {
    switch (m_missID)
    {
    case 0: // No environment light. This only makes sense for scenes with emissive materials or KHR_lights_punctual.
      break;

    case 1: // Constant white environment light.
      m_lightDefinitions[0] = createConstantEnvironmentLight();
      break;

    case 2: // Sperical HDR environment light.
      m_lightDefinitions[0] = createSphericalEnvironmentLight();
      break;
    }
  }

  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_d_lightDefinitions), m_lightDefinitions.size() * sizeof(LightDefinition)) );
}


void Application::initAnimations()
{
  m_animations.clear();
  m_animations.reserve(m_asset.animations.size());
  
  for (const fastgltf::Animation& gltf_animation : m_asset.animations)
  {
    dev::Animation& animation = m_animations.emplace_back();

    animation.name = gltf_animation.name;
    // By default, all animations are disabled because skinning and morphing are not handled yet 
    // and would not render a noise free scene although nothing moves.
    // FIXME Maybe add a command line option to enable none, all, or a specific index.
    animation.isEnabled = false; 

    animation.samplers.reserve(gltf_animation.samplers.size());

    for (const fastgltf::AnimationSampler& gltf_animation_sampler : gltf_animation.samplers)
    {
      dev::AnimationSampler& animationSampler = animation.samplers.emplace_back();

      switch (gltf_animation_sampler.interpolation)
      {
        case fastgltf::AnimationInterpolation::Linear:
          animationSampler.interpolation = dev::AnimationSampler::INTERPOLATION_LINEAR;
          break;

        case fastgltf::AnimationInterpolation::Step:
          animationSampler.interpolation = dev::AnimationSampler::INTERPOLATION_STEP;
          break;

        case fastgltf::AnimationInterpolation::CubicSpline:
          animationSampler.interpolation = dev::AnimationSampler::INTERPOLATION_CUBIC_SPLINE;
          break;
      }

      // Read sampler input time values, convert to scalar floats when needed.
      const int indexInputAccessor = static_cast<int>(gltf_animation_sampler.inputAccessor);
      utils::createHostBuffer("sampler input time", m_asset, indexInputAccessor, fastgltf::AccessorType::Scalar, fastgltf::ComponentType::Float, 0.0f, animationSampler.input);

      // Determine start and end time of the inputs.
      // gltf 2.0 specs: "Animation sampler's input accessor MUST have its min and max properties defined."
      const auto* minimum = std::get_if< FASTGLTF_STD_PMR_NS::vector<double> >(&m_asset.accessors[indexInputAccessor].min);
      MY_ASSERT(minimum != nullptr && minimum->size() == 1);
      animationSampler.timeMin = static_cast<float>(minimum->front());

      // Track the minimum time of all samplers in this animation.
      if (animationSampler.timeMin < animation.timeMin)
      {
        animation.timeMin = animationSampler.timeMin;
      }

      const auto* maximum = std::get_if< FASTGLTF_STD_PMR_NS::vector<double> >(&m_asset.accessors[indexInputAccessor].max);
      MY_ASSERT(maximum != nullptr && minimum->size() == 1);
      animationSampler.timeMax = static_cast<float>(maximum->front());

      // Track the maximum time of all samplers in this animation.
      if (animation.timeMax < animationSampler.timeMax)
      {
        animation.timeMax = animationSampler.timeMax;
      }

      // Read sampler output values at these times.
      // These can be scalar floats, vec3 for translations and scale, and vec4 for rotation quaternions.

      const int indexOutputAccessor = static_cast<int>(gltf_animation_sampler.outputAccessor);
      const fastgltf::AccessorType typeSource = m_asset.accessors[indexOutputAccessor].type;
      
      fastgltf::AccessorType typeTarget = fastgltf::AccessorType::Invalid;

      switch (typeSource)
      {
        case fastgltf::AccessorType::Scalar: 
          animationSampler.components = 1;
          typeTarget = typeSource;
          break;

        case fastgltf::AccessorType::Vec3:
          animationSampler.components = 3;
          typeTarget = typeSource;
          break;

        case fastgltf::AccessorType::Vec4:
          animationSampler.components = 4;
          typeTarget = typeSource;
          break;

        default:
          std::cerr << "ERROR: Unexpected animation accessor source type " << (uint16_t) typeSource << '\n';
          MY_ASSERT(!"Unexpected animation accessor source type");
          break;
      }

      utils::createHostBuffer("sampler output values", m_asset, indexOutputAccessor, typeTarget, fastgltf::ComponentType::Float, 0.0f, animationSampler.output);
    }

    // Channels
    for(const fastgltf::AnimationChannel& gltf_channel : gltf_animation.channels)
    {
      dev::AnimationChannel& channel = animation.channels.emplace_back();

      if (gltf_channel.path == fastgltf::AnimationPath::Translation)
      {
        channel.path = dev::AnimationChannel::TRANSLATION;
      }
      else if (gltf_channel.path == fastgltf::AnimationPath::Rotation)
      {
        channel.path = dev::AnimationChannel::ROTATION;
      }
      else if (gltf_channel.path == fastgltf::AnimationPath::Scale)
      {
        channel.path = dev::AnimationChannel::SCALE;
      }
      else if (gltf_channel.path == fastgltf::AnimationPath::Weights)
      {
        channel.path = dev::AnimationChannel::WEIGHTS;
      }
      channel.indexSampler = static_cast<int>(gltf_channel.samplerIndex);
      if (gltf_channel.nodeIndex.has_value())
      {
        channel.indexNode = static_cast<int>(gltf_channel.nodeIndex.value()); // else it stays -1.
        
        // To optimize static base mesh weighting without animation, which doesn't require separate device meshes, 
        // mark nodes which are a morph weight animation channel path for target.
        if (channel.path == dev::AnimationChannel::WEIGHTS)
        {
          m_nodes[channel.indexNode].morphMode = dev::Node::MORPH_ANIMATED_WEIGHTS; // Requires a unique DeviceMesh per node!
        }
      }
    }
  }

  m_isAnimated = false; // All animations are disabled by default.
  // For real-time:
  m_timeStart = m_timeMinimum;
  m_timeEnd   = m_timeMaximum;
  // For key frames:
  m_frameMinimum = std::max(0, static_cast<int>(floorf(m_timeMinimum * m_framesPerSecond)));
  m_frameMaximum = std::max(0, static_cast<int>(ceilf(m_timeMaximum * m_framesPerSecond)));
  m_frameStart = m_frameMinimum;
  m_frameEnd   = m_frameMaximum;
}


bool Application::updateAnimations()
{
  bool animated = false;

  if (m_isTimeBased)
  {
    if (!m_isScrubbing) // Only use the real time when not scrubbing the current time, otherwise just use the slider value.
    {
      std::chrono::steady_clock::time_point timeNow = std::chrono::steady_clock::now();
      std::chrono::duration<double> durationSeconds = timeNow - m_timeBase;
      const float seconds = std::chrono::duration<float>(durationSeconds).count() * m_timeScale;
      // Loop the current time in the user defined interval [m_timeStart, m_timeEnd].
      m_timeCurrent = m_timeStart + fmodf(seconds, m_timeEnd - m_timeStart);
    }
  }
  else
  {
    const float timeStart = static_cast<float>(m_frameStart) / m_framesPerSecond;
    m_timeCurrent = timeStart + static_cast<float>(m_frameCurrent) / m_framesPerSecond;

    if (!m_isScrubbing) // Only advance the frame when not scrubbing, otherwise just use the slider value.
    {
      // FIXME This advances one animation frame with each render() call.
      // That limits the animated images to the maximum number of MAX_LAUNCHES per render call.
      // Only when the animation is stopped the renderer keeps accumulating more samples.
      ++m_frameCurrent; 
      if (m_frameEnd <= m_frameCurrent)
      {
        m_frameCurrent = m_frameStart;
      }
    }
  }

  for (dev::Animation& animation : m_animations)
  {
    animated |= animation.update(m_nodes, m_timeCurrent);
  }

  return animated;
}


void Application::initScene(const int sceneIndex)
{
  // glTF specs: "A glTF asset that does not contain any scenes SHOULD be treated as a library of individual entities such as materials or meshes."
  // That would only make sense if the application would be able to mix assets from different files.
  // If there is no scene defined, this just creates one from all root nodes inside the asset to be able to continue.
  if (m_asset.scenes.empty())
  {
    // Find all root nodes.
    std::vector<bool> isRoot(m_asset.nodes.size(), true);
    
    // Root nodes are all nodes which do not appear inside any node's children.
    for (const fastgltf::Node& node : m_asset.nodes)
    {
      for (size_t child : node.children)
      {
        isRoot[child] = false;
      }
    }

    // Now build a scene which is just the vector of root node indices.
    fastgltf::Scene scene;

    scene.name = std::string("scene_root_nodes");

    for (size_t i = 0; i < isRoot.size(); ++i )
    {
      if (isRoot[i])
      {
        scene.nodeIndices.push_back(i);
      }
    }

    m_asset.scenes.push_back(scene);
  }

  // Determine which scene inside the asset should be used.
  if (sceneIndex < 0)
  {
    m_indexScene = (m_asset.defaultScene.has_value()) ? m_asset.defaultScene.value() : 0;
    m_isDirtyScene = true;
  }
  else if (sceneIndex < m_asset.scenes.size())
  {
    m_indexScene = static_cast<size_t>(sceneIndex);
    m_isDirtyScene = true;
  }
  // else m_indexScene unchanged.
}


void Application::updateScene(const bool rebuild)
{
  if (rebuild) // This is a first time scene initialization or scene switch.
  {
    m_instances.clear();
    // FIXME PERF Without KHR_mesh_gpu_instancing support, there can only be as many instances as there are nodes inside the asset,
    // and not all nodes have meshes assigned, some can be skeletons. Just use the number of nodes to reserve space for the instances.
    m_instances.reserve(m_asset.nodes.size()); 

    m_growMorphWeights.clear();
    m_growSkinMatrices.clear();

    m_growIas.clear();
    m_growIasTemp.clear();
    m_growInstances.clear();
  }

  m_indexInstance = 0; // Global index which is tracked when updating the m_instances matrices during SRT animation. Also needed for the DeviceMesh.

  MY_ASSERT(m_indexScene < m_asset.scenes.size());
  const fastgltf::Scene& scene = m_asset.scenes[m_indexScene];

  // "The node hierarchy MUST be a set of disjoint strict trees. 
  //  That is node hierarchy MUST NOT contain cycles and each node MUST have zero or one parent node."
  // That means it's not possible to instance whole sub-trees in glTF!
  // That also means the node.globalMatrix is unique and must be calculated first to have valid joint matrices.
  for (const size_t indexNode : scene.nodeIndices)
  {
    traverseNodeTrafo(indexNode, glm::mat4(1.0f));
  }

  for (const size_t indexNode : scene.nodeIndices)
  {
    traverseNode(indexNode, rebuild);
  }
}


void Application::updateMorph(const int indexDeviceMesh, const size_t indexNode)
{
  dev::Node& node = m_nodes[indexNode];

  const size_t sizeBytesWeights = node.weights.size() * sizeof(float);

  m_growMorphWeights.grow(sizeBytesWeights);

  CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_growMorphWeights.d_ptr), node.weights.data(), sizeBytesWeights, cudaMemcpyHostToDevice) );

  dev::DeviceMesh& deviceMesh = m_deviceMeshes[indexDeviceMesh];

  for (dev::DevicePrimitive& devicePrim : deviceMesh.primitives)
  {
    device_morphing(m_cudaStream,
                    reinterpret_cast<const float*>(m_growMorphWeights.d_ptr),
                    devicePrim);

    if (devicePrim.maskTargets & ATTR_POSITION)
    {
      deviceMesh.isDirty = true;
    }
  }
}


void Application::updateSkin(const size_t indexNode, const dev::KeyTuple key)
{
  const dev::Node& node = m_nodes[indexNode]; // The parent node.

  const glm::mat4 matrixParentGlobalInverse = glm::inverse(node.matrixGlobal);

  // This value must exist or the indexNode wouldn't have been inside the m_skinnedNodeIndices.
  const int indexSkin = key.idxSkin;
  MY_ASSERT(0 <= indexSkin);

  dev::Skin& skin = m_skins[indexSkin];

  // FIXME This should define a pivot point, but how is that applied?
  // (The Khronos glTF-Sample-Viewer is not using this either.)
  //glm::mat4 matrixSkeletonGlobal = glm::mat4(1.0f);
  //if (0 <= skin.skeleton && skin.skeleton < static_cast<int>(m_nodes.size()))
  //{
  //  const dev::Node& nodeSkeleton = m_nodes[indexNode];
  //  matrixSkeletonGlobal = nodeSkeleton.matrixGlobal;
  //}

  const float* ibm = reinterpret_cast<const float*>(skin.inverseBindMatrices.h_ptr);
  int numSkinMatrices = 0;
  for (const size_t joint : skin.joints)
  {
    // When there are no inverseBindMatrices all of them are identity. Then why are we here?
    glm::mat4 matrixBindInverse(1.0f);

    if (ibm != nullptr)
    {
      // glTF matrices are defined column-major like GLM matrices! 
      matrixBindInverse = glm::mat4(ibm[ 0], ibm[ 1], ibm[ 2], ibm[ 3], 
                                    ibm[ 4], ibm[ 5], ibm[ 6], ibm[ 7], 
                                    ibm[ 8], ibm[ 9], ibm[10], ibm[11], 
                                    ibm[12], ibm[13], ibm[14], ibm[15]);
      ibm += 16;
    }
    
    const glm::mat4 skinMatrix = matrixParentGlobalInverse * m_nodes[joint].matrixGlobal * matrixBindInverse;

    // PERF The GPU transform routines expect row-major data. float4 vectors are rows and the last row is (0, 0, 0, 1) and can be ignored.
    skin.matrices[numSkinMatrices]   = glm::transpose(skinMatrix); // Transpose the column-major matrix to get a row-major matrix. 
    skin.matricesIT[numSkinMatrices] = glm::inverse(skinMatrix);   // The transpose of the inverse transposed matrix is the inverse matrix.

    ++numSkinMatrices;
  }

  // Now recalculate the mesh vertex attributes with the skin's joint matrices.

  //const int indexMesh = node.indexMesh;
  const int indexMesh = key.idxHostMesh; // This is the m_hostMesh index.
  MY_ASSERT(0 <= indexMesh);

  dev::HostMesh& hostMesh = m_hostMeshes[indexMesh];

  std::map<dev::KeyTuple, int>::const_iterator it = m_mapKeyTupleToDeviceMeshIndex.find(key);
  MY_ASSERT(it != m_mapKeyTupleToDeviceMeshIndex.end());
  const int indexDeviceMesh = it->second; // Instanced DeviceMesh.
  dev::DeviceMesh& deviceMesh = m_deviceMeshes[indexDeviceMesh];

  for (size_t indexPrimitive = 0; indexPrimitive < hostMesh.primitives.size(); ++indexPrimitive)
  {
    dev::DevicePrimitive& devicePrim = deviceMesh.primitives[indexPrimitive];

    MY_ASSERT(sizeof(glm::mat4) == 4 * sizeof(float4));
    const size_t sizeInBytesSkinMatrices = skin.matrices.size() * sizeof(glm::mat4);

    m_growSkinMatrices.grow(sizeInBytesSkinMatrices * 2); // For both matrix and matrixIT arrays.

    // Upload the skin matrices and their inverse transpose.
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_growSkinMatrices.d_ptr), skin.matrices.data(), sizeInBytesSkinMatrices, cudaMemcpyHostToDevice) );
    if (devicePrim.normals.d_ptr != 0) // skin.matricesIT aren't used when there are no normals.
    {
      CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_growSkinMatrices.d_ptr + sizeInBytesSkinMatrices), skin.matricesIT.data(), sizeInBytesSkinMatrices, cudaMemcpyHostToDevice) );
    }

    // PERF Native CUDA kernels for skinning animation.
    device_skinning(m_cudaStream,                                              // cudaStream_t stream,
                    reinterpret_cast<const float4*>(m_growSkinMatrices.d_ptr), // const float4* d_skinMatrices,
                    numSkinMatrices,                                           // const unsigned int numSkinMatrices,
                    devicePrim);

    deviceMesh.isDirty = true;
  }
}


void Application::createDevicePrimitive(dev::DevicePrimitive& devicePrim, const dev::HostPrimitive& hostPrim, const int skin)
{
  MY_ASSERT(dev::PrimitiveType::Undefined != hostPrim.getPrimitiveType());

  devicePrim.setPrimitiveType(hostPrim.getPrimitiveType());

  devicePrim.numTargets  = static_cast<int>(hostPrim.numTargets);
  devicePrim.maskTargets = hostPrim.maskTargets;

  utils::createDeviceBuffer(devicePrim.indices, hostPrim.indices); // unsigned int

  // Device Buffers for the base attributes.
  utils::createDeviceBuffer(devicePrim.positions, hostPrim.positions); // float3 (this is the only mandatory attribute!)
  utils::createDeviceBuffer(devicePrim.tangents,  hostPrim.tangents);  // float4 (.w == 1.0 or -1.0 for the handedness)
  utils::createDeviceBuffer(devicePrim.normals,   hostPrim.normals);   // float3
  utils::createDeviceBuffer(devicePrim.colors,    hostPrim.colors);    // float4

  for (int i = 0; i < NUM_ATTR_TEXCOORDS; ++i)
  {
    utils::createDeviceBuffer(devicePrim.texcoords[i], hostPrim.texcoords[i]); // float2
  }
  // These are required on the device for the native CUDA skinning kernels.
  for (int i = 0; i < NUM_ATTR_JOINTS; ++i)
  {
    utils::createDeviceBuffer(devicePrim.joints[i], hostPrim.joints[i]); // ushort4
  }
  for (int i = 0; i < NUM_ATTR_WEIGHTS; ++i)
  {
    utils::createDeviceBuffer(devicePrim.weights[i], hostPrim.weights[i]); // float4
  }

  // Morph targets.
  if (0 < devicePrim.numTargets && devicePrim.maskTargets != 0)
  {
    size_t numMorphedAttributes = 0;

    // First resize the vectors with the DeviceBuffers for the individual morphed attributes to the proper size.
    if (devicePrim.maskTargets & ATTR_POSITION)
    {
      devicePrim.positionsTarget.resize(devicePrim.numTargets);
      ++numMorphedAttributes;
    }
    if (devicePrim.maskTargets & ATTR_TANGENT)
    {
      devicePrim.tangentsTarget.resize(devicePrim.numTargets);
      ++numMorphedAttributes;
    }
    if (devicePrim.maskTargets & ATTR_NORMAL)
    {
      devicePrim.normalsTarget.resize(devicePrim.numTargets);
      ++numMorphedAttributes;
    }
    if (devicePrim.maskTargets & ATTR_COLOR_0)
    {
      devicePrim.colorsTarget.resize(devicePrim.numTargets);
      ++numMorphedAttributes;
    }
    for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j)
    {
      if (devicePrim.maskTargets & (ATTR_TEXCOORD_0 << j))
      {
        devicePrim.texcoordsTarget[j].resize(devicePrim.numTargets);
        ++numMorphedAttributes;
      }
    }

    // Create the device buffers for enabled morph targets.
    // FIXME PERF More optimal would be to interleave the target data per attribute into one device buffer
    // to be able to read them from contiguous addresses during the weighting.
    // The target attributes aren't changing so the interleaving would be a one time operation.
    // That would also remove the need for the targetPointers.

    // Store the target device pointers of all enabled morph attributes into a host buffer.
    HostBuffer hostTargetPointers;

    hostTargetPointers.count = devicePrim.numTargets * numMorphedAttributes;
    hostTargetPointers.size  = hostTargetPointers.count * sizeof(CUdeviceptr);
    hostTargetPointers.h_ptr = new unsigned char[hostTargetPointers.size];

    CUdeviceptr* ptrTarget = reinterpret_cast<CUdeviceptr*>(hostTargetPointers.h_ptr);

    if (devicePrim.maskTargets & ATTR_POSITION) 
    {
      for (int i = 0; i < devicePrim.numTargets; ++i)
      {
        utils::createDeviceBuffer(devicePrim.positionsTarget[i], hostPrim.positionsTarget[i]);
        *ptrTarget++ = devicePrim.positionsTarget[i].d_ptr;
      }
    }
    if (devicePrim.maskTargets & ATTR_TANGENT) 
    {
      for (int i = 0; i < devicePrim.numTargets; ++i)
      {
        // Attention: This is float3! The handedness is not morphed (or skinned)
        // DAR FIXME it might make sense to change this to float4 with .w == 0.0f for easier handling inside the kernel later.
        utils::createDeviceBuffer(devicePrim.tangentsTarget[i], hostPrim.tangentsTarget[i]); 
        *ptrTarget++ = devicePrim.tangentsTarget[i].d_ptr;
      }
    }
    if (devicePrim.maskTargets & ATTR_NORMAL)
    {
      for (int i = 0; i < devicePrim.numTargets; ++i)
      {
        utils::createDeviceBuffer(devicePrim.normalsTarget[i], hostPrim.normalsTarget[i]);
        *ptrTarget++ = devicePrim.normalsTarget[i].d_ptr;
      }
    }
    if (devicePrim.maskTargets & ATTR_COLOR_0)
    {
      for (int i = 0; i < devicePrim.numTargets; ++i)
      {
        utils::createDeviceBuffer(devicePrim.colorsTarget[i], hostPrim.colorsTarget[i]);
        *ptrTarget++ = devicePrim.colorsTarget[i].d_ptr;
      }
    }
    for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j)
    {
      if (devicePrim.maskTargets & (ATTR_TEXCOORD_0 << j))
      {
        for (int i = 0; i < devicePrim.numTargets; ++i)
        {
          utils::createDeviceBuffer(devicePrim.texcoordsTarget[j][i], hostPrim.texcoordsTarget[j][i]);
          *ptrTarget++ = devicePrim.texcoordsTarget[j][i].d_ptr;
        }
      }
    }

    // Create and fill the device buffer with the morph target pointers of the enabled morph attributes.
    utils::createDeviceBuffer(devicePrim.targetPointers, hostTargetPointers);

    // Destination device buffers for the morphed attributes.
    // These can be initialized with the base attributes which would be the same as morph weights == 0.0f.
    // First resize the vectors with the DeviceBuffers for the individual morphed attributes to the proper size.
    if (devicePrim.maskTargets & ATTR_POSITION)
    {
      // The glTF specs define per Mesh weights which are static and per Node weights which are animated
      utils::createDeviceBuffer(devicePrim.positionsMorphed, hostPrim.positions);
    }
    if (devicePrim.maskTargets & ATTR_TANGENT)
    {
      utils::createDeviceBuffer(devicePrim.tangentsMorphed, hostPrim.tangents);
    }
    if (devicePrim.maskTargets & ATTR_NORMAL)
    {
      utils::createDeviceBuffer(devicePrim.normalsMorphed, hostPrim.normals);
    }
    if (devicePrim.maskTargets & ATTR_COLOR_0)
    {
      utils::createDeviceBuffer(devicePrim.colorsMorphed, hostPrim.colors);
    }
    for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j)
    {
      if (devicePrim.maskTargets & (ATTR_TEXCOORD_0 << j))
      {
        utils::createDeviceBuffer(devicePrim.texcoordsMorphed[j], hostPrim.texcoords[j]);
      }
    }
  }

  // Create the destination buffers for the skinned attributes only if the device mesh is under a node with skin index.
  // Otherwise the d_ptr remain null.
  if (0 <= skin)
  {
    // This will only create (and fill) device buffers for the base attributes which are present inside the primitive.
    utils::createDeviceBuffer(devicePrim.positionsSkinned, hostPrim.positions); // float3
    utils::createDeviceBuffer(devicePrim.tangentsSkinned,  hostPrim.tangents);  // float4 (.w == 1.0 or -1.0 for the handedness)
    utils::createDeviceBuffer(devicePrim.normalsSkinned,   hostPrim.normals);   // float3
  }

  // Set the final position attribute pointer. The OptixBuildInput vertexBuffers needs a pointer to that.
  devicePrim.vertexBuffer = devicePrim.getPositionsPtr();

  devicePrim.currentMaterial = hostPrim.currentMaterial;
}


void Application::createDeviceMesh(dev::DeviceMesh& deviceMesh, const dev::KeyTuple key)
{
  // Get the host mesh index and create all required DeviceBuffers.
  const dev::HostMesh& hostMesh = m_hostMeshes[key.idxHostMesh];

  deviceMesh.key = key; // This is unique per DeviceMesh.
  
  deviceMesh.primitives.reserve(hostMesh.primitives.size());

  for (const dev::HostPrimitive& hostPrim : hostMesh.primitives)
  {
    dev::DevicePrimitive& devicePrim = deviceMesh.primitives.emplace_back();
    devicePrim.setName(hostPrim.getName());

    createDevicePrimitive(devicePrim, hostPrim, key.idxSkin);
  }
}


void Application::traverseNodeTrafo(const size_t indexNode, glm::mat4 matrix)
{
  dev::Node& node = m_nodes[indexNode];

  matrix *= node.getMatrix(); // node.getMatrix() is the local transform relative to the parent node.

  node.matrixGlobal = matrix; // The gobal transform of this node, needed for skinning.

  // Traverse all children of this glTF node.
  const fastgltf::Node& gltf_node = m_asset.nodes[indexNode];

  for (size_t child : gltf_node.children)
  {
    traverseNodeTrafo(child, matrix);
  }
}


void Application::traverseNode(const size_t indexNode, const bool rebuild)
{
  // This key maps a tuple of (host-node, skin, mesh) to a DeviceMesh. 
  // Default is (-1, -1, -1) which is an invalid DeviceMesh key.
  dev::KeyTuple key;
  
  dev::Node& node = m_nodes[indexNode];

  // If the node holds morphing weights, an assigned mesh requires a unique DeviceMesh
  // only when the weights are dynamic, not when they are only static on the base mesh. 
  if (!node.weights.empty() && (node.morphMode == dev::Node::MORPH_NODE_WEIGHTS ||
                                node.morphMode == dev::Node::MORPH_ANIMATED_WEIGHTS))
  {
    key.idxNode = static_cast<int>(indexNode);
  }

  if (0 <= node.indexMesh) // -1 when none.
  {
    key.idxHostMesh = node.indexMesh;

    // When a node has a mesh and a skin index, then the mesh is skinned.
    if (0 <= node.indexSkin) // -1 when none.
    {
      key.idxSkin = node.indexSkin;
    }

    dev::HostMesh& hostMesh = m_hostMeshes[node.indexMesh]; // This array has been initialized in initMeshes().

    // If the mesh contains TRIANGLE or POINTS data, add an instance to the scene graph.
    if (!hostMesh.primitives.empty())
    {
      int indexDeviceMesh = -1;

      // Iterator for device meshes.
      std::map<dev::KeyTuple, int>::const_iterator it = m_mapKeyTupleToDeviceMeshIndex.find(key);

      if (rebuild)
      {
        //
        // First time scene initialization.
        //
        if (it == m_mapKeyTupleToDeviceMeshIndex.end())
        {
          indexDeviceMesh = static_cast<int>(m_deviceMeshes.size());

          //
          // NEW DEVICE MESH
          //
          dev::DeviceMesh& deviceMesh = m_deviceMeshes.emplace_back();

          createDeviceMesh(deviceMesh, key);

          m_mapKeyTupleToDeviceMeshIndex[key] = indexDeviceMesh; 

          // One time initialization of morphed attributes with static mesh weights.
          if (!node.weights.empty() && key.idxNode < 0)
          {
            updateMorph(indexDeviceMesh, indexNode);
          }
        }
        else
        {
          // Device mesh already there.
          indexDeviceMesh = it->second; // Instanced DeviceMesh.
        }

        MY_ASSERT(0 <= indexDeviceMesh);

        //
        // NEW INSTANCE CREATION
        //
        dev::Instance& instance = m_instances.emplace_back();
        
        // This is the equivalent of an OpenGL draw command which generates 
        // the morphed and skinned vertex attributes inside the vertex shader.
        // Just that this needs to build/update potentially separate device mesh AS now.
        instance.transform       = node.matrixGlobal;
        instance.indexDeviceMesh = indexDeviceMesh;
      }
      else // Animation update.
      {
        MY_ASSERT(it != m_mapKeyTupleToDeviceMeshIndex.end()); // All device meshes must have been created already!
        
        indexDeviceMesh = it->second; // Existing DeviceMesh.

        // Instance update.
        MY_ASSERT(m_indexInstance < m_instances.size());

        dev::Instance& instance = m_instances[m_indexInstance];

        instance.transform = node.matrixGlobal;
        MY_ASSERT(instance.indexDeviceMesh == indexDeviceMesh);
      }

      if (0 <= key.idxNode) // Node weights or animated morph weights?
      {
        updateMorph(indexDeviceMesh, indexNode);
      }

      if (0 <= key.idxSkin) // Skin on the node with the mesh?
      {
        updateSkin(indexNode, key); // This requires the key to be in m_mapKeyTupleToDeviceMeshIndex already.
      }

      buildDeviceMeshAccel(indexDeviceMesh, rebuild);
      
      // std::cout << "Instance = " << m_indexInstance << " uses Key (mesh = " << key.idxMesh << ", skin = " << key.idxSkin << ", node = " << key.idxNode << ")\n"; // DEBUG 
      // Either adding a new instance or SRT animating existing instances increments the global instance counter.
      ++m_indexInstance;
    }
  }

  if (0 <= node.indexCamera)
  {
    const fastgltf::Camera& gltf_camera = m_asset.cameras[node.indexCamera];
    dev::Camera& camera = m_cameras[node.indexCamera]; // The m_cameras vector is already initialized with default perspective cameras.

    if (const fastgltf::Camera::Perspective* pPerspective = std::get_if<fastgltf::Camera::Perspective>(&gltf_camera.camera))
    {
      // INIT PERSPECTIVE CAMERA
      const glm::vec3 pos     = glm::vec3(node.matrixGlobal * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
      const glm::vec3 up      = glm::vec3(node.matrixGlobal * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));
      const glm::vec3 forward = glm::vec3(node.matrixGlobal * glm::vec4(0.0f, 0.0f, -1.0f, 1.0f));

      const float yfov = pPerspective->yfov * 180.0f / M_PIf;

      float aspectRatio = (pPerspective->aspectRatio.has_value() && pPerspective->aspectRatio.value() != 0.0f) 
                        ? pPerspective->aspectRatio.value() 
                        : 1.0f;

      camera.setPosition(pos);
      camera.setLookat(forward);
      camera.setUp(up);
      camera.setFovY(yfov);
      camera.setAspectRatio(aspectRatio);
    }
    else if (const fastgltf::Camera::Orthographic* pOrthograhpic = std::get_if<fastgltf::Camera::Orthographic>(&gltf_camera.camera))
    {
      // INIT ORTHO CAMERA
      const glm::vec3 pos     = glm::vec3(node.matrixGlobal * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
      const glm::vec3 up      = glm::vec3(node.matrixGlobal * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));
      const glm::vec3 forward = glm::vec3(node.matrixGlobal * glm::vec4(0.0f, 0.0f, -1.0f, 1.0f));

      camera.setPosition(pos);
      camera.setLookat(forward);
      camera.setUp(up);
      camera.setFovY(-1.0f); // <= 0.0f means orthographic projection.
      camera.setMagnification(glm::vec2(pOrthograhpic->xmag, pOrthograhpic->ymag));
    }
    else
    {
      std::cerr << "ERROR: Unexpected camera type.\n";
    }
  }

  // KHR_lights_punctual
  if (0 <= node.indexLight)
  {
    m_lights[node.indexLight].matrix = node.matrixGlobal;

    m_isDirtyLights = true;
  }

  // Traverse all children of this glTF node.
  const fastgltf::Node& gltf_node = m_asset.nodes[indexNode];

  for (size_t child : gltf_node.children)
  {
    // RECURSION
    traverseNode(child, rebuild);
  }
}


void Application::addImage(
  const int32_t width,
  const int32_t height,
  const int32_t bitsPerComponent,
  const int32_t numComponents,
  const void*   data)
{
  // Allocate CUDA array in device memory
  int32_t               pitch;
  cudaChannelFormatDesc channel_desc;

  if (bitsPerComponent == 8)
  {
    pitch = width * numComponents * sizeof(uint8_t);
    channel_desc = cudaCreateChannelDesc<uchar4>();
  }
  else if (bitsPerComponent == 16)
  {
    pitch = width * numComponents * sizeof(uint16_t);
    channel_desc = cudaCreateChannelDesc<ushort4>();
  }
  else
  {
    std::cerr << "ERROR: addImage() Unsupported bitsPerComponent " << bitsPerComponent << '\n';
    throw std::runtime_error("addImage() Unsupported bitsPerComponent");
  }

  cudaArray_t cuda_array = nullptr;

  CUDA_CHECK( cudaMallocArray(&cuda_array, &channel_desc, width, height) );
  CUDA_CHECK( cudaMemcpy2DToArray(cuda_array, 0, 0, data, pitch, pitch, height, cudaMemcpyHostToDevice) );

  m_images.push_back(cuda_array);
}


void Application::addSampler(
  cudaTextureAddressMode address_s,
  cudaTextureAddressMode address_t,
  cudaTextureFilterMode  filter,
  const size_t           image_idx, 
  const int              sRGB)
{
  cudaResourceDesc resDesc = {};

  resDesc.resType = cudaResourceTypeArray;
  MY_ASSERT(image_idx < m_images.size())
  resDesc.res.array.array = m_images[image_idx];

  cudaTextureDesc texDesc = {};

  texDesc.addressMode[0]      = address_s;
  texDesc.addressMode[1]      = address_t;
  texDesc.filterMode          = filter;
  texDesc.readMode            = cudaReadModeNormalizedFloat;
  texDesc.normalizedCoords    = 1;
  texDesc.maxAnisotropy       = 1;
  texDesc.maxMipmapLevelClamp = 0;
  texDesc.minMipmapLevelClamp = 0;
  texDesc.mipmapFilterMode    = cudaFilterModePoint; // No mipmap filtering.
  texDesc.borderColor[0]      = 1.0f; // DEBUG Why is the Khronos glTF-Sample Viewer using white for the border color?
  texDesc.borderColor[1]      = 1.0f;
  texDesc.borderColor[2]      = 1.0f;
  texDesc.borderColor[3]      = 1.0f;
  // glTF uses sRGB for baseColor, specularColor, sheenColor and emissive texture RGB values, all other texture data is linear.
  // TextureLinearInterpolationTest.gltf requires that the texture engine interpolates with sRGB enabled.
  // Doing sRGB adjustments with pow(rgb, 2.2) inside the shader is not producing the correct result because that is after linear texture interpolation.
  texDesc.sRGB                = sRGB;

  // Create texture object.
  cudaTextureObject_t cuda_tex = 0;

  CUDA_CHECK( cudaCreateTextureObject(&cuda_tex, &resDesc, &texDesc, nullptr) );

  m_samplers.push_back(cuda_tex);
}


void Application::cleanup()
{
  // Host and device asset cleanup.
  m_cameras.clear();
  m_lights.clear();
  m_instances.clear();
  m_hostMeshes.clear();
  m_mapKeyTupleToDeviceMeshIndex.clear();
  m_deviceMeshes.clear();
  m_materialsOrg.clear();
  m_materials.clear();   
  for (cudaTextureObject_t& sampler : m_samplers)
  {
    CUDA_CHECK( cudaDestroyTextureObject(sampler) );
  }
  m_samplers.clear();
  for (cudaArray_t& image : m_images)
  {
    CUDA_CHECK( cudaFreeArray(image) );
  }
  m_images.clear();
  m_nodes.clear();       
  m_animations.clear();
  
  // OptiX cleanup.
  if (m_pipeline)
  {
    OPTIX_CHECK( m_api.optixPipelineDestroy(m_pipeline) );
    m_pipeline = 0;
  }

  for (OptixProgramGroup programGroup : m_programGroups)
  {
    OPTIX_CHECK( m_api.optixProgramGroupDestroy(programGroup) );
  }
  m_programGroups.clear();

  for (OptixModule m : m_modules)
  {
    OPTIX_CHECK(m_api.optixModuleDestroy(m));
  }
  m_modules.clear();

  if (m_optixContext)
  {
    OPTIX_CHECK( m_api.optixDeviceContextDestroy(m_optixContext) );
    m_optixContext = 0;
  }

  // CUDA cleanup.
  if (m_cudaGraphicsResource != nullptr)
  {
    CU_CHECK( cuGraphicsUnregisterResource(m_cudaGraphicsResource) );
  }

  if (m_sbt.raygenRecord)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)) );
    m_sbt.raygenRecord = 0;
  }
  if (m_sbt.missRecordBase)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)) );
    m_sbt.missRecordBase = 0;
  }
  if (m_sbt.hitgroupRecordBase)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)) );
    m_sbt.hitgroupRecordBase = 0;
  }

  if (m_launchParameters.bufferAccum != 0 &&
      m_interop != INTEROP_PBO) // For INTEROP_PBO: bufferAccum contains the last PBO mapping, do not call cudaFree() on that.
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferAccum)) );
  }

  if (m_launchParameters.bufferPicking != 0)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_launchParameters.bufferPicking)) );
  }
  
  m_growMorphWeights.clear();
  m_growSkinMatrices.clear();

  m_growIas.clear();
  m_growIasTemp.clear();
  m_growInstances.clear();

  if (m_d_iasAABB != 0)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_d_iasAABB)) );
    m_d_iasAABB = 0;
  }

  if (m_d_lightDefinitions != 0)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_d_lightDefinitions)) );
    m_d_lightDefinitions = 0;
  }

  // OpenGL cleanup:
  if (m_pbo != 0)
  {
    glDeleteBuffers(1, &m_pbo);
  }
  if (m_hdrTexture != 0)
  {
    glDeleteTextures(1, &m_hdrTexture);
  }
  if (m_vboAttributes != 0)
  {
    glDeleteBuffers(1, &m_vboAttributes);
  }
  if (m_vboIndices != 0)
  {
    glDeleteBuffers(1, &m_vboIndices);
  }
  if (m_glslProgram != 0)
  {
    glDeleteProgram(m_glslProgram);
  }

  // Host side allocations.
  if (m_picSheenLUT != nullptr)
  {
    delete m_picSheenLUT;
  }
  if (m_texSheenLUT != nullptr)
  {
    delete m_texSheenLUT;
  }
  if (m_picEnv != nullptr)
  {
    delete m_picEnv;
  }
  if (m_texEnv != nullptr)
  {
    delete m_texEnv;
  }
}


void Application::buildDeviceMeshAccel(const int indexDeviceMesh, const bool rebuild)
{
  // Build input flags depending on the different material configuration assigned to the individual Primitive.
  // Each alphaMode has a different anyhit program handling!
  // Each element 0 has face culling enabled, and element 1 has face culling disabled. 
  //
  // Note that face-culling isn't really compatible with global illumination algorithms! 
  // Materials which are fully transparent on one side and fully opaque on the other aren't physically plausible.
  // Neither reflections nor shadows will look as expected in some test scenes (like NegativeScaleTest.gltf) which 
  // are explicitly built for local lighting in rasterizers.

  // ALPHA_MODE_OPAQUE materials do not need to call into anyhit programs!
  static const unsigned int inputFlagsOpaque[2] =
  {
    OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
    OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING
  };

  // ALPHA_MODE_MASK materials are either fully opaque or fully transparent which is tested 
  // inside the anyhit program by comparing the opacity against the alphaCutoff value.
  static const unsigned int inputFlagsMask[2] =
  {
    OPTIX_GEOMETRY_FLAG_NONE,
    OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING
  };

  // ALPHA_MODE_BLEND materials are using a stochastic opacity threshold which must be evaluated only once per primitive.
  static const unsigned int inputFlagsBlend[2] =
  {
    OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL,
    OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING
  };

  dev::DeviceMesh& deviceMesh = m_deviceMeshes[indexDeviceMesh];

  if (!deviceMesh.isDirty)
  {
    MY_ASSERT(deviceMesh.gas != 0 && deviceMesh.d_gas != 0);
    return; // Nothing to do for this device mesh.
  }

  // The device mesh key is unique, so it can be used to determine if a DeviceMesh is morphed or skinned during animation.
  const bool compact = !(0 <= deviceMesh.key.idxNode || 0 <= deviceMesh.key.idxSkin);

  OptixAccelBuildOptions accelBuildOptions = {};

  // When the AS should be compacted, that means it's not changing dynamically with morphing and/or skinning animations.
  if (compact)
  {
    // Non-animated vertex attributes are built to render as fast as possible.
    accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    //accelBuildOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accelBuildOptions.operation  = OPTIX_BUILD_OPERATION_BUILD; // Always rebuild when reaching this.
  }
  else
  {
    // Meshes with animated vertex attributes (during node graph traversal) are built as fast as possible and allow updates.
    accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    //accelBuildOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accelBuildOptions.operation  = (rebuild) ? OPTIX_BUILD_OPERATION_BUILD : OPTIX_BUILD_OPERATION_UPDATE;
  }

  // This builds one GAS per DeviceMesh but with build input and SBT hit record per DevicePrimitive (with Triangles mode)
  // to be able to use different input flags and material indices.

  MY_ASSERT(m_sceneExtent.isValid());

  std::vector<OptixBuildInput> buildInputs;
  buildInputs.reserve(deviceMesh.primitives.size());

  const auto sceneSize = m_sceneExtent.getDiameter();
  const float allSpheresRadius = m_sphereRadiusFraction * sceneSize;

  for (const dev::DevicePrimitive& devicePrim : deviceMesh.primitives)
  {
    OptixBuildInput buildInput = {};
    // * Set the build input for triangles, points (as OptiX spheres), ...
    // * Set material properties based on the primitive's current material.
    if (devicePrim.setupBuildInput(buildInput, // OUT
                                   m_materials,
                                   inputFlagsOpaque,
                                   inputFlagsMask,
                                   inputFlagsBlend,
                                   allSpheresRadius))
    {

      buildInputs.push_back(buildInput);
      /*if (buildInput.type == OPTIX_BUILD_INPUT_TYPE_SPHERES)
          std::cout << "allSpheresRadius " << allSpheresRadius << std::endl;
      */
    }
  } // for deviceMesh.primitives
    
  if (!buildInputs.empty())
  {
    // If this routine is called with rebuild more than once for a mesh, free the d_gas of this mesh and rebuild it.
    if (rebuild && deviceMesh.d_gas)
    {
      CUDA_CHECK( cudaFree(reinterpret_cast<void*>(deviceMesh.d_gas)) );

      deviceMesh.d_gas = 0;
      deviceMesh.gas   = 0;
    }

    OptixAccelBufferSizes accelBufferSizes = {};

    OPTIX_CHECK( m_api.optixAccelComputeMemoryUsage(m_optixContext,
                                                    &accelBuildOptions,
                                                    buildInputs.data(),
                                                    static_cast<unsigned int>(buildInputs.size()),
                                                    &accelBufferSizes) );
    
    if (compact) // This is always a build operation.
    {
      CUdeviceptr d_gas; // Must be aligned to OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT.

      CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&d_gas), accelBufferSizes.outputSizeInBytes) ); 

      CUdeviceptr d_temp; // Must be aligned to OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT.

      CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&d_temp), accelBufferSizes.tempSizeInBytes) );

      OptixAccelEmitDesc accelEmit = {};

      CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&accelEmit.result), 8) ); // Room for size_t for the compacted size.
      accelEmit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

      OPTIX_CHECK( m_api.optixAccelBuild(m_optixContext,
                                         m_cudaStream,
                                         &accelBuildOptions,
                                         buildInputs.data(),
                                         static_cast<unsigned int>(buildInputs.size()),
                                         d_temp, accelBufferSizes.tempSizeInBytes,
                                         d_gas, accelBufferSizes.outputSizeInBytes, 
                                         &deviceMesh.gas,
                                         &accelEmit, // Emitted property: compacted size
                                         1) );       // Number of emitted properties.

      size_t sizeCompact;

      CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(&sizeCompact), (const void*) accelEmit.result, sizeof(size_t), cudaMemcpyDeviceToHost) );

      CUDA_CHECK( cudaFree(reinterpret_cast<void*>(accelEmit.result)) );

      CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_temp)) );

      // Compact the AS only when possible. This can save more than half the memory on RTX boards.
      if (sizeCompact < accelBufferSizes.outputSizeInBytes)
      {
        CUdeviceptr d_gasCompact;

        CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&d_gasCompact), sizeCompact) );

        OPTIX_CHECK( m_api.optixAccelCompact(m_optixContext,
                                             m_cudaStream,
                                             deviceMesh.gas,
                                             d_gasCompact,
                                             sizeCompact,
                                             &deviceMesh.gas) );

        CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_gas)) );

        deviceMesh.d_gas = d_gasCompact;
      }
      else
      {
        deviceMesh.d_gas = d_gas;
      }
    }
    else // if (!compact) // Means morphing or skinning animation. This can be an initial build or an update operation.
    {
      size_t sizeTemp = accelBufferSizes.tempUpdateSizeInBytes; // Temporary memory required for an update operation.

      if (rebuild)
      {
        sizeTemp = accelBufferSizes.tempSizeInBytes; // Temporary memory required for a full build operation.

        CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&deviceMesh.d_gas), accelBufferSizes.outputSizeInBytes) ); // d_gas has been freed above.
      }

      CUdeviceptr d_temp; // Must be aligned to OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT.

      CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&d_temp), sizeTemp) );

      OPTIX_CHECK( m_api.optixAccelBuild(m_optixContext,
                                         m_cudaStream,
                                         &accelBuildOptions,
                                         buildInputs.data(),
                                         static_cast<unsigned int>(buildInputs.size()),
                                         d_temp, sizeTemp,
                                         deviceMesh.d_gas, accelBufferSizes.outputSizeInBytes,
                                         &deviceMesh.gas,
                                         nullptr,
                                         0) );

      CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_temp)) );
    }
  }

  deviceMesh.isDirty = false;
}


// This is called when changing materials. 
void Application::buildDeviceMeshAccels(const bool rebuild)
{
  for (int indexDeviceMesh = 0; indexDeviceMesh < static_cast<int>(m_deviceMeshes.size()); ++indexDeviceMesh)
  {
    buildDeviceMeshAccel(indexDeviceMesh, rebuild);
  }
}


void Application::buildInstanceAccel(const bool rebuild)
{
  const size_t numInstances = m_instances.size();

  if (numInstances == 0) // Empty scene? 
  {
    // Set the scene AABB to the unit cube.
    m_sceneExtent.toUnity();

    // Set the top-level traversable handle to zero. 
    // This is allowed in optixTrace and will immediately invoke the miss program.
    m_ias = 0;

    return;
  }

  m_sceneExtent.toInvalid();

  if (m_d_iasAABB == 0) // One time allocation of the device buffer receiving the IAS AABB result.
  {
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_d_iasAABB), 6 * sizeof(float)) ); 
  }

  std::vector<OptixInstance> optix_instances(numInstances);
  //std::cout << "buildInstanceAccel " << (rebuild ? "[REBUILD]" : "[UPDATE]") << ", " << numInstances << " instances..." << std::endl;

  unsigned int sbt_offset = 0;

  size_t dbgNumPrims = 0;

  for (size_t i = 0; i < m_instances.size(); ++i)
  {
    const dev::Instance& instance = m_instances[i];

    OptixInstance& optix_instance = optix_instances[i];
    memset(&optix_instance, 0, sizeof(OptixInstance));

    optix_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
    optix_instance.instanceId        = static_cast<unsigned int>(i); // these don't have to be unique, in general
    optix_instance.sbtOffset         = sbt_offset;
    optix_instance.visibilityMask    = m_visibilityMask;
    optix_instance.traversableHandle = m_deviceMeshes[instance.indexDeviceMesh].gas;
    
    utils::setInstanceTransform(
        optix_instance,
        instance.transform);  // Convert from column-major GLM matrices to
                              // row-major OptiX matrices.
 
    sbt_offset += static_cast<unsigned int>(m_deviceMeshes[instance.indexDeviceMesh].primitives.size()) * NUM_RAY_TYPES; // One SBT hit record per GAS build input per RAY_TYPE.
    dbgNumPrims += m_deviceMeshes[instance.indexDeviceMesh].primitives.size();
  }

  const size_t sizeBytesInstances = sizeof(OptixInstance) * numInstances;

  m_growInstances.grow(sizeBytesInstances);

  CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_growInstances.d_ptr), optix_instances.data(), sizeBytesInstances, cudaMemcpyHostToDevice) );

  OptixBuildInput buildInput = {};

  buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;

  buildInput.instanceArray.instances    = m_growInstances.d_ptr;
  buildInput.instanceArray.numInstances = static_cast<unsigned int>(numInstances);

  OptixAccelBuildOptions accelBuildOptions = {};

  // The IAS can always be updated for animations or on some material parameter changes (alphaMode, doubleSided, volume).
  accelBuildOptions.buildFlags = getBuildFlags();
  
  accelBuildOptions.operation   = (rebuild) ? OPTIX_BUILD_OPERATION_BUILD : OPTIX_BUILD_OPERATION_UPDATE;

  OptixAccelBufferSizes accelBufferSizes = {};

  OPTIX_CHECK( m_api.optixAccelComputeMemoryUsage(m_optixContext, &accelBuildOptions, &buildInput, 1, &accelBufferSizes) );

  // This grow() assumes outputSizeInBytes for an update operation is always less or equal to the previous build operation.
  MY_ASSERT(rebuild || (!rebuild && accelBufferSizes.outputSizeInBytes <= m_growIas.size));
  m_growIas.grow(accelBufferSizes.outputSizeInBytes); 

  const size_t tempSizeInBytes = (rebuild) ? accelBufferSizes.tempSizeInBytes : accelBufferSizes.tempUpdateSizeInBytes;

  m_growIasTemp.grow(tempSizeInBytes);

  OptixAccelEmitDesc emitDesc = {};
  
  // Emit the top-level AABB to know the scene size.
  emitDesc.type   = OPTIX_PROPERTY_TYPE_AABBS;
  emitDesc.result = m_d_iasAABB;

  //TODO timer.start();

  OPTIX_CHECK( m_api.optixAccelBuild(m_optixContext,
                                     m_cudaStream,
                                     &accelBuildOptions,
                                     &buildInput,
                                     1, // num build inputs
                                     m_growIasTemp.d_ptr, m_growIasTemp.size,
                                     m_growIas.d_ptr, m_growIas.size,
                                     &m_ias,
                                     &emitDesc,
                                     1));
  //TODO std::cout << "Build time " << timer.getElapsedSeconds() << std::endl;

  glm::vec3 sceneAABB[2];
  CUDA_CHECK( cudaMemcpy(&sceneAABB[0].x, 
                         reinterpret_cast<const void*>(emitDesc.result), 
                         6 * sizeof(float), 
                         cudaMemcpyDeviceToHost) );
  // NOTE: change the scene extent!
  m_sceneExtent.set(sceneAABB);
  MY_ASSERT(m_sceneExtent.isValid());

}

glm::vec3 Application::getSceneCenter() const
{
  return m_sceneExtent.getCenter();
}


void Application::initPipeline()
{
  // Set all module and pipeline options.

  // OptixModuleCompileOptions
  m_mco = {};

  m_mco.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#if USE_DEBUG_EXCEPTIONS
  m_mco.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0; // No optimizations.
  m_mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;     // Full debug. Never profile kernels with this setting!
#else
  m_mco.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3; // All optimizations, is the default.
  // Keep generated line info. (NVCC_OPTIONS use --generate-line-info in CMakeLists.txt)
  m_mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL; // PERF Must use OPTIX_COMPILE_DEBUG_LEVEL_MODERATE to profile code with Nsight Compute!
#endif // USE_DEBUG_EXCEPTIONS

  // OptixPipelineCompileOptions
  m_pco = {};

  m_pco.usesMotionBlur        = 0;
  m_pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  m_pco.numPayloadValues      = 2; // Need only two register for the payload pointer.
  m_pco.numAttributeValues    = 2; // For the two barycentric coordinates of built-in triangles. (The required minimum value.)
#ifndef NDEBUG // USE_DEBUG_EXCEPTIONS
  m_pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
                         OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                         OPTIX_EXCEPTION_FLAG_USER;
#else
  m_pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
  m_pco.pipelineLaunchParamsVariableName = "theLaunchParameters";

  // Supporting built-in tris and spheres at this time.
  m_pco.usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | 
                                                           OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE);

  // OptixPipelineLinkOptions
  m_plo = {};

  m_plo.maxTraceDepth = MAX_TRACE_DEPTH;

  // OptixProgramGroupOptions
  m_pgo = {}; // Just a placeholder.

  // Build the module path names.
  // Starting with OptiX SDK 7.5.0 and CUDA 11.7 either PTX or OptiX IR input can be used to create modules.
  // Just initialize the m_moduleFilenames depending on the definition of USE_OPTIX_IR.
  // That is added to the project definitions inside the CMake script when OptiX SDK 7.5.0 and CUDA 11.7 or newer are found.

  const std::string path(CUDA_PROGRAMS_PATH);

#if defined(USE_OPTIX_IR)
  const std::string extension(".optixir");
#else
  const std::string extension(".ptx");
#endif

  m_moduleFilenames.resize(NUM_MODULE_IDENTIFIERS);

  m_moduleFilenames[MODULE_ID_RAYGENERATION]  = path + std::string("raygen") + extension;
  m_moduleFilenames[MODULE_ID_EXCEPTION]      = path + std::string("exception") + extension;
  m_moduleFilenames[MODULE_ID_MISS]           = path + std::string("miss") + extension;
  m_moduleFilenames[MODULE_ID_HIT]            = path + std::string("hit") + extension;
  m_moduleFilenames[MODULE_ID_LIGHT_SAMPLE]   = path + std::string("light_sample") + extension; // Direct callable programs.

  // Create all modules.

  MY_ASSERT(NUM_RAY_TYPES == 2); // The following code only works for two raytypes.

  m_modules.resize(NUM_MODULE_IDENTIFIERS);

  for (size_t i = 0; i < m_moduleFilenames.size(); ++i)
  {
    std::vector<char> programData = readData(m_moduleFilenames[i]);

    OPTIX_CHECK( m_api.optixModuleCreate(m_optixContext, &m_mco, &m_pco, programData.data(), programData.size(), nullptr, nullptr, &m_modules[i]) );
  }

  // For spheres we need this:
  OptixBuiltinISOptions builtin_is_options = {};

  builtin_is_options.usesMotionBlur = m_pco.usesMotionBlur;
  builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
  builtin_is_options.buildFlags = getBuildFlags();
  OPTIX_CHECK(m_api.optixBuiltinISModuleGet(m_optixContext, &m_mco, &m_pco, &builtin_is_options, &m_moduleBuiltinISSphere));

  // TODO the same for curves.

  // Create the program groups descriptions.

  std::vector<OptixProgramGroupDesc> programGroupDescriptions(NUM_PROGRAM_GROUP_IDS);
  memset(programGroupDescriptions.data(), 0, sizeof(OptixProgramGroupDesc) * programGroupDescriptions.size());

  OptixProgramGroupDesc* pgd = &programGroupDescriptions[PGID_RAYGENERATION];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->raygen.module = m_modules[MODULE_ID_RAYGENERATION];

  if (m_interop != INTEROP_IMG)
  {
    pgd->raygen.entryFunctionName = "__raygen__path_tracer";
  }
  else
  {
    pgd->raygen.entryFunctionName = "__raygen__path_tracer_surface";
  }

  pgd = &programGroupDescriptions[PGID_EXCEPTION];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->exception.module            = m_modules[MODULE_ID_EXCEPTION];
  pgd->exception.entryFunctionName = "__exception__all";

  pgd = &programGroupDescriptions[PGID_MISS_RADIANCE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->miss.module = m_modules[MODULE_ID_MISS];

  switch (m_missID)
  {
    case 0:
    default: // Every other ID means there is no environment light.
      pgd->miss.entryFunctionName = "__miss__env_null";
      break;
    case 1:
      pgd->miss.entryFunctionName = "__miss__env_constant";
      break;
    case 2:
      pgd->miss.entryFunctionName = "__miss__env_sphere";
      break;
  }

  pgd = &programGroupDescriptions[PGID_MISS_SHADOW];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->miss.module            = m_modules[MODULE_ID_MISS];
  pgd->miss.entryFunctionName = "__miss__shadow"; // alphaMode OPAQUE is not using anyhit or closest hit programs for the shadow ray.

  // TRIANGLES: The hit records for the radiance ray.
  pgd = &programGroupDescriptions[PGID_HIT_RADIANCE_TRIANGLES];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleCH = m_modules[MODULE_ID_HIT];
//  pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pgd->hitgroup.moduleAH = m_modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__radiance";

  //  TRIANGLES: The hit records for the shadow ray
  pgd = &programGroupDescriptions[PGID_HIT_SHADOW_TRIANGLES];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleAH = m_modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__shadow";

  //  SPHERES: The hit records for the radiance ray.
  pgd = &programGroupDescriptions[PGID_HIT_RADIANCE_SPHERES];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleCH            = m_modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameCH = "__closesthit__radiance_sphere";
  pgd->hitgroup.moduleAH            = m_modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__radiance_sphere";
  pgd->hitgroup.moduleIS = m_moduleBuiltinISSphere;
  pgd->hitgroup.entryFunctionNameIS = 0;

  // SPHERES: The hit records for the shadow ray.
  pgd = &programGroupDescriptions[PGID_HIT_SHADOW_SPHERES];
  pgd->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->hitgroup.moduleAH = m_modules[MODULE_ID_HIT];
  pgd->hitgroup.entryFunctionNameAH = "__anyhit__shadow_sphere";
  
  // TODO the same for curves

  // Light Sampler
  // Only one of the environment callables will ever be used, but both are required
  // for the proper direct callable index calculation for BXDFs using NUM_LIGHT_TYPES.
  pgd = &programGroupDescriptions[PGID_LIGHT_ENV_CONSTANT];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = m_modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_env_constant";

  pgd = &programGroupDescriptions[PGID_LIGHT_ENV_SPHERE];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = m_modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_env_sphere";

  pgd = &programGroupDescriptions[PGID_LIGHT_POINT];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = m_modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_point";

  pgd = &programGroupDescriptions[PGID_LIGHT_SPOT];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = m_modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_spot";

  pgd = &programGroupDescriptions[PGID_LIGHT_DIRECTIONAL];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->callables.moduleDC            = m_modules[MODULE_ID_LIGHT_SAMPLE];
  pgd->callables.entryFunctionNameDC = "__direct_callable__light_directional";

  // Create the program groups.

  m_programGroups.resize(programGroupDescriptions.size());
  
  OPTIX_CHECK( m_api.optixProgramGroupCreate(m_optixContext,
                                             programGroupDescriptions.data(),
                                             (unsigned int) programGroupDescriptions.size(),
                                             &m_pgo,
                                             nullptr, nullptr,
                                             m_programGroups.data()));
  
  // 3.) Create the pipeline.

  OPTIX_CHECK( m_api.optixPipelineCreate(m_optixContext, &m_pco, &m_plo, m_programGroups.data(), (unsigned int) m_programGroups.size(), nullptr, nullptr, &m_pipeline) );


  // 4.) Calculate the stack size. 
  // This is is always recommended and strictly required when using any direct or continuation callables.
  OptixStackSizes ssp = {}; // Whole pipeline.

  for (OptixProgramGroup pg: m_programGroups)
  {
    OptixStackSizes ss;

#if (OPTIX_VERSION >= 70700)
    OPTIX_CHECK( m_api.optixProgramGroupGetStackSize(pg, &ss, m_pipeline) );
#else
    OPTIX_CHECK( m_api.optixProgramGroupGetStackSize(pg, &ss) );
#endif

    ssp.cssRG = std::max(ssp.cssRG, ss.cssRG);
    ssp.cssMS = std::max(ssp.cssMS, ss.cssMS);
    ssp.cssCH = std::max(ssp.cssCH, ss.cssCH);
    ssp.cssAH = std::max(ssp.cssAH, ss.cssAH);
    ssp.cssIS = std::max(ssp.cssIS, ss.cssIS);
    ssp.cssCC = std::max(ssp.cssCC, ss.cssCC);
    ssp.dssDC = std::max(ssp.dssDC, ss.dssDC);
  }
  
  // Temporaries
  unsigned int cssCCTree           = ssp.cssCC; // Should be 0. No continuation callables in this pipeline. // maxCCDepth == 0
  unsigned int cssCHOrMSPlusCCTree = std::max(ssp.cssCH, ssp.cssMS) + cssCCTree;

  // Arguments
  unsigned int directCallableStackSizeFromTraversal = ssp.dssDC; // maxDCDepth == 1 // FromTraversal: DC is invoked from IS or AH.      // Possible stack size optimizations.
  unsigned int directCallableStackSizeFromState     = ssp.dssDC; // maxDCDepth == 1 // FromState:     DC is invoked from RG, MS, or CH. // Possible stack size optimizations.
  unsigned int continuationStackSize = ssp.cssRG + cssCCTree + cssCHOrMSPlusCCTree * (std::max(1u, m_plo.maxTraceDepth) - 1u) +
                                       std::min(1u, m_plo.maxTraceDepth) * std::max(cssCHOrMSPlusCCTree, ssp.cssAH + ssp.cssIS);
  unsigned int maxTraversableGraphDepth = 2;

  OPTIX_CHECK( m_api.optixPipelineSetStackSize(m_pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState, continuationStackSize, maxTraversableGraphDepth) );
}


void Application::initSBT()
{
  {
    //
    // SBT RAYGEN
    //
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_sbt.raygenRecord), sizeof(dev::EmptyRecord)) );

    dev::EmptyRecord rg_sbt;

    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_RAYGENERATION], &rg_sbt) );

    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_sbt.raygenRecord), &rg_sbt, sizeof(dev::EmptyRecord), cudaMemcpyHostToDevice) );
  }

  {
    //
    // SBT MISS
    //
    const size_t miss_record_size = sizeof(dev::EmptyRecord);

    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_sbt.missRecordBase), miss_record_size * NUM_RAY_TYPES) );

    dev::EmptyRecord ms_sbt[NUM_RAY_TYPES];

    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_MISS_RADIANCE], &ms_sbt[0]) );
    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_MISS_SHADOW],   &ms_sbt[1]) );

    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_sbt.missRecordBase), ms_sbt, miss_record_size * NUM_RAY_TYPES, cudaMemcpyHostToDevice) );
    
    m_sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    m_sbt.missRecordCount = NUM_RAY_TYPES;
  }

  {
    //
    // SBT HITGROUPS {RadianceRay, ShadowRay} x {Triangles, Spheres}
    //

    std::vector<dev::HitGroupRecord> hitGroupRecords;

    for (const dev::Instance& instance : m_instances)
    {
      //std::cout << "Instance " << instance.indexDeviceMesh << std::endl;

      const dev::DeviceMesh& deviceMesh = m_deviceMeshes[instance.indexDeviceMesh];

      for (const dev::DevicePrimitive& devicePrim : deviceMesh.primitives)
      {
        // Geometry and material.
        dev::HitGroupRecord rec = {};

        //std::cout << "\tDevice Primitive " << devicePrim.name << ", type " << devicePrim.getPrimitiveType() << std::endl;

        if(devicePrim.getPrimitiveType() == dev::PrimitiveType::Triangles)
        {
          OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_RADIANCE_TRIANGLES], &rec));
          //std::cout << "[TRIS] optixSbtRecordPackHeader PGID_HIT_RADIANCE_TRIANGLES" << std::endl;

          GeometryData::TriangleMesh triangleMesh = {};

          // Indices
          triangleMesh.indices = reinterpret_cast<uint3*>(devicePrim.indices.d_ptr);
          // Attributes
          triangleMesh.positions = reinterpret_cast<float3*>(devicePrim.getPositionsPtr());
          triangleMesh.normals = reinterpret_cast<float3*>(devicePrim.getNormalsPtr());
          for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j)
          {
            triangleMesh.texcoords[j] = reinterpret_cast<float2*>(devicePrim.getTexcoordsPtr(j));
          }
          triangleMesh.colors = reinterpret_cast<float4*>(devicePrim.getColorsPtr());
          triangleMesh.tangents = reinterpret_cast<float4*>(devicePrim.getTangentsPtr());
          for (int j = 0; j < NUM_ATTR_JOINTS; ++j)
          {
            triangleMesh.joints[j] = reinterpret_cast<ushort4*>(devicePrim.joints[j].d_ptr);
          }
          for (int j = 0; j < NUM_ATTR_WEIGHTS; ++j)
          {
            triangleMesh.weights[j] = reinterpret_cast<float4*>(devicePrim.weights[j].d_ptr);
          }
          //triangleMesh.flagAttributes = getAttributeFlags(devicePrim); // FIXME Currently unused.

          // Note that both trimesh and spheremesh have { positions, normals, colors }
          rec.data.geometryData.setTriangleMesh(triangleMesh);

          if (0 <= devicePrim.currentMaterial)
          {
            rec.data.materialData = m_materials[devicePrim.currentMaterial];
          }
          else
          {
            rec.data.materialData = MaterialData(); // These default materials cannot be edited!
          }

          hitGroupRecords.push_back(rec); // radiance ray - triangles

          OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_SHADOW_TRIANGLES], &rec));
          //std::cout << "[TRIS] optixSbtRecordPackHeader PGID_HIT_SHADOW_TRIANGLES" << std::endl;

          hitGroupRecords.push_back(rec); // shadow ray - triangles
        }
        else if (devicePrim.getPrimitiveType() == dev::PrimitiveType::Points)
        {
          OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_RADIANCE_SPHERES], &rec));
          //OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_RADIANCE_TRIANGLES], &rec));
          //std::cout << "[POINTS] optixSbtRecordPackHeader PGID_HIT_RADIANCE_SPHERES" << std::endl;

          GeometryData::SphereMesh pointMesh = {};

          // Attributes
          pointMesh.positions = reinterpret_cast<float3*>(devicePrim.getPositionsPtr());
          pointMesh.normals = reinterpret_cast<float3*>(devicePrim.getNormalsPtr());
          for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j)
          {
           // pointMesh.texcoords[j] = reinterpret_cast<float2*>(devicePrim.getTexcoordsPtr(j));
          }
          pointMesh.colors = reinterpret_cast<float4*>(devicePrim.getColorsPtr());
          //pointMesh.tangents = reinterpret_cast<float4*>(devicePrim.getTangentsPtr());
          /*for (int j = 0; j < NUM_ATTR_JOINTS; ++j)
          {
            pointMesh.joints[j] = reinterpret_cast<ushort4*>(devicePrim.joints[j].d_ptr);
          }*/
          for (int j = 0; j < NUM_ATTR_WEIGHTS; ++j)
          {
            //pointMesh.weights[j] = reinterpret_cast<float4*>(devicePrim.weights[j].d_ptr);
          }
          //triangleMesh.flagAttributes = getAttributeFlags(devicePrim); // FIXME Currently unused.

          // Note that both trimesh and spheremesh have { positions, normals, colors }
          rec.data.geometryData.setSphereMesh(pointMesh);

          if (0 <= devicePrim.currentMaterial)
          {
            rec.data.materialData = m_materials[devicePrim.currentMaterial];
          }
          else
          {
            rec.data.materialData = MaterialData(); // These default materials cannot be edited!
          }

          hitGroupRecords.push_back(rec); // radiance ray - spheres

          OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_SHADOW_SPHERES], &rec));
          //OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_SHADOW_TRIANGLES], &rec));
          //std::cout << "[POINTS] optixSbtRecordPackHeader PGID_HIT_SHADOW_SPHERES" << std::endl;

          hitGroupRecords.push_back(rec); // shadow ray - spheres
        }
        else
        {
          // ignore the primitive
          MY_ASSERT(false);
        }
      } // all primitives
    } // all instances

    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_sbt.hitgroupRecordBase),
                           hitGroupRecords.size() * sizeof(dev::HitGroupRecord)) );
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase), 
                           hitGroupRecords.data(),
                           hitGroupRecords.size() * sizeof(dev::HitGroupRecord),
                           cudaMemcpyHostToDevice) );

    m_sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(dev::HitGroupRecord));
    m_sbt.hitgroupRecordCount         = static_cast<unsigned int>(hitGroupRecords.size());
  }

  {
    //
    // SBT CALLABLES
    //
    const size_t call_record_size = sizeof(dev::EmptyRecord);

    const int numCallables = NUM_PROGRAM_GROUP_IDS - PGID_LIGHT_ENV_CONSTANT;
    MY_ASSERT(numCallables == 5);

    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_sbt.callablesRecordBase), call_record_size * numCallables) );

    dev::EmptyRecord call_sbt[numCallables];

    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_LIGHT_ENV_CONSTANT], &call_sbt[0]) );
    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_LIGHT_ENV_SPHERE],   &call_sbt[1]) );
    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_LIGHT_POINT],        &call_sbt[2]) );
    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_LIGHT_SPOT],         &call_sbt[3]) );
    OPTIX_CHECK( m_api.optixSbtRecordPackHeader(m_programGroups[PGID_LIGHT_DIRECTIONAL],  &call_sbt[4]) );

    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_sbt.callablesRecordBase), call_sbt, call_record_size * numCallables, cudaMemcpyHostToDevice) );
    
    m_sbt.callablesRecordStrideInBytes = static_cast<uint32_t>(call_record_size);
    m_sbt.callablesRecordCount         = numCallables;
  }
}


// This regenerates all SBT hit records, which is required when switching scenes.
void Application::updateSBT()
{
  if (m_sbt.hitgroupRecordBase)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)) );
    m_sbt.hitgroupRecordBase = 0;
  }

  std::vector<dev::HitGroupRecord> hitGroupRecords;

  for (const dev::Instance& instance : m_instances)
  {
    const dev::DeviceMesh& deviceMesh = m_deviceMeshes[instance.indexDeviceMesh];

    for (const dev::DevicePrimitive& devicePrim : deviceMesh.primitives)
    {
      dev::HitGroupRecord rec = {};
      const bool isTriangles = (devicePrim.getPrimitiveType() == dev::PrimitiveType::Triangles);

      if (isTriangles)
      {
        //std::cout << "\toptixSbtRecordPackHeader TRIS " << devicePrim.name << std::endl;
        OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_RADIANCE_TRIANGLES], &rec));

        GeometryData::TriangleMesh triangleMesh = {};

        // Indices
        triangleMesh.indices = reinterpret_cast<uint3*>(devicePrim.indices.d_ptr);
        // Attributes
        triangleMesh.positions = reinterpret_cast<float3*>(devicePrim.getPositionsPtr());
        triangleMesh.normals = reinterpret_cast<float3*>(devicePrim.getNormalsPtr());
        for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j)
        {
          triangleMesh.texcoords[j] = reinterpret_cast<float2*>(devicePrim.getTexcoordsPtr(j));
        }
        triangleMesh.colors = reinterpret_cast<float4*>(devicePrim.getColorsPtr());
        triangleMesh.tangents = reinterpret_cast<float4*>(devicePrim.getTangentsPtr());
        for (int j = 0; j < NUM_ATTR_JOINTS; ++j)
        {
          triangleMesh.joints[j] = reinterpret_cast<ushort4*>(devicePrim.joints[j].d_ptr);
        }
        for (int j = 0; j < NUM_ATTR_WEIGHTS; ++j)
        {
          triangleMesh.weights[j] = reinterpret_cast<float4*>(devicePrim.weights[j].d_ptr);
        }
        //triangleMesh.flagAttributes = getAttributeFlags(devicePrim); // FIXME Currently unused.
        rec.data.geometryData.setTriangleMesh(triangleMesh);
      }
      else if (devicePrim.getPrimitiveType() == dev::PrimitiveType::Points)
      {
        //std::cout << "\toptixSbtRecordPackHeader SPHERES " << devicePrim.name << std::endl;
        OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_RADIANCE_SPHERES], &rec));

        GeometryData::SphereMesh pointMesh = {};

        // Attributes
        pointMesh.positions = reinterpret_cast<float3*>(devicePrim.getPositionsPtr());
        pointMesh.normals = reinterpret_cast<float3*>(devicePrim.getNormalsPtr());
        for (int j = 0; j < NUM_ATTR_TEXCOORDS; ++j)
        {
         // pointMesh.texcoords[j] = reinterpret_cast<float2*>(devicePrim.getTexcoordsPtr(j));
        }
        pointMesh.colors = reinterpret_cast<float4*>(devicePrim.getColorsPtr());
        //pointMesh.tangents = reinterpret_cast<float4*>(devicePrim.getTangentsPtr());
        for (int j = 0; j < NUM_ATTR_JOINTS; ++j)
        {
          //pointMesh.joints[j] = reinterpret_cast<ushort4*>(devicePrim.joints[j].d_ptr);
        }
        for (int j = 0; j < NUM_ATTR_WEIGHTS; ++j)
        {
          //pointMesh.weights[j] = reinterpret_cast<float4*>(devicePrim.weights[j].d_ptr);
        }
        //triangleMesh.flagAttributes = getAttributeFlags(devicePrim); // FIXME Currently unused.
        rec.data.geometryData.setSphereMesh(pointMesh);
      }
      else
      {
        std::cerr << "ERROR primitive not implemented " << devicePrim.getPrimitiveType() << std::endl;
      }

      if (0 <= devicePrim.currentMaterial)
      {
        rec.data.materialData = m_materials[devicePrim.currentMaterial];
      }
      else
      {
        rec.data.materialData = MaterialData(); // These default materials cannot be edited!
      }
        
      hitGroupRecords.push_back(rec); // radiance ray

      if (isTriangles)
      {
        OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_SHADOW_TRIANGLES], &rec));
      }
      else
      {
        OPTIX_CHECK(m_api.optixSbtRecordPackHeader(m_programGroups[PGID_HIT_SHADOW_SPHERES], &rec));
      }
        
      hitGroupRecords.push_back(rec); // shadow ray
    }
  }

  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_sbt.hitgroupRecordBase), hitGroupRecords.size() * sizeof(dev::HitGroupRecord)) );
  CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase), hitGroupRecords.data(), hitGroupRecords.size() * sizeof(dev::HitGroupRecord), cudaMemcpyHostToDevice) );

  m_sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(dev::HitGroupRecord));
  m_sbt.hitgroupRecordCount         = static_cast<unsigned int>(hitGroupRecords.size());
}


void Application::updateMaterial(const int indexMaterial, const bool rebuild)
{
  MY_ASSERT(m_sbt.hitgroupRecordBase != 0);
  
  dev::HitGroupRecord* rec = reinterpret_cast<dev::HitGroupRecord*>(m_sbt.hitgroupRecordBase);

  for (const dev::Instance& instance : m_instances)
  {
    dev::DeviceMesh& deviceMesh = m_deviceMeshes[instance.indexDeviceMesh];

    for (const dev::DevicePrimitive& devicePrim : deviceMesh.primitives)
    {
      if (indexMaterial == devicePrim.currentMaterial)
      {
        if (!rebuild) // Only update the SBT hit record material data in place when not rebuilding everything anyway.
        {
          // Update the radiance ray hit record material data.
          CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(&rec->data.materialData), &m_materials[indexMaterial], sizeof(MaterialData), cudaMemcpyHostToDevice) );
          ++rec;

          // Update the shadow ray hit record material data.
          CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(&rec->data.materialData), &m_materials[indexMaterial], sizeof(MaterialData), cudaMemcpyHostToDevice) );
          ++rec;
        }
        else
        {
          rec += 2; // Skip two HitGroupRecords.
        }

        m_launchParameters.iteration = 0u; // Restart accumulation when any material in the currently active scene changed.   

        deviceMesh.isDirty = rebuild; // Flag mesh GAS which need to be rebuild.
      }
      else
      {
        rec += 2; // Skip two HitGroupRecords which are not using the m_material[index].
      }
    }
  }

  // When doubleSided or alphaMode changed in a way which requires to rebuild any mesh, update the respective GAS.
  if (rebuild)
  {
    buildDeviceMeshAccels(rebuild);  // This rebuilds only the meshes with isDirty flags.
    buildInstanceAccel(false); // This updates the top-level IAS. The GAS AABBs didn't change on material changes, so updating the IAS is sufficient.
    updateSBT();               // This rebuilds all hit records inside the SBT. Means the above copies aren't required.
    updateLaunchParameters();  // This sets the root m_ias (shouldn't have changed on update) and restarts the accumulation.
  }
}


// Only need to update the SBT hit records with the new material data.
void Application::updateSBTMaterialData()
{
  MaterialData defaultMaterialData = {}; // Default material data in case any primitive has no material assigned.

  MY_ASSERT(m_sbt.hitgroupRecordBase != 0);
  
  dev::HitGroupRecord* rec = reinterpret_cast<dev::HitGroupRecord*>(m_sbt.hitgroupRecordBase);

  for (const dev::Instance& instance : m_instances)
  {
    dev::DeviceMesh& deviceMesh = m_deviceMeshes[instance.indexDeviceMesh];

    for (const dev::DevicePrimitive& devicePrim : deviceMesh.primitives)
    {
      const MaterialData* src = (0 <= devicePrim.currentMaterial) ? &m_materials[devicePrim.currentMaterial] : &defaultMaterialData;

      // Update the radiance ray hit record material data.
      CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(&rec->data.materialData), src, sizeof(MaterialData), cudaMemcpyHostToDevice) );
      ++rec;
      // Update the shadow ray hit record material data.
      CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(&rec->data.materialData), src, sizeof(MaterialData), cudaMemcpyHostToDevice) );
      ++rec;
    }
  }
}


void Application::updateVariant()
{
  bool changed   = false;
  bool rebuildAS = false;

  for (dev::HostMesh& hostMesh : m_hostMeshes)
  {
    for (dev::HostPrimitive& hostPrim : hostMesh.primitives)
    {
      // Variants can only change on this primitive if there are material index mappings available.
      if (!hostPrim.mappings.empty())
      {
        const int indexMaterial = hostPrim.mappings[m_indexVariant]; // m_indexVariant contains the new variant.

        if (indexMaterial != hostPrim.currentMaterial) 
        {
          changed = true; // At least one material index has changed.

          // Check if the material switch requires a rebuild of the AS.
          const MaterialData& cur = m_materials[hostPrim.currentMaterial];
          const MaterialData& var = m_materials[indexMaterial];

          // The face culling state is affected by both the doubleSided and the volume state.
          // The only case when face culling is enabled is when the material is not doubleSided and not using the volume extension.
          const bool curCull = (!cur.doubleSided && (cur.flags & FLAG_KHR_MATERIALS_VOLUME) == 0);
          const bool varCull = (!var.doubleSided && (var.flags & FLAG_KHR_MATERIALS_VOLUME) == 0);
        
          // If the alphaMode changes, the anyhit program invocation for primitives changes.
          const bool rebuild = (curCull != varCull) || (cur.alphaMode != var.alphaMode);

          // Now switch the primitive to the new material index.
          hostPrim.currentMaterial = indexMaterial;

          if (rebuild)
          {
            hostMesh.isDirty = true;
            rebuildAS = true;
          }
        }
      }
    }
  }

  // Migrate the material changes from the host meshes to the device meshes.
  if (changed)
  {
    for (dev::DeviceMesh& deviceMesh : m_deviceMeshes)
    {
      dev::HostMesh& hostMesh = m_hostMeshes[deviceMesh.key.idxHostMesh];

      // Migrate all current material settings to the device mesh using this host mesh.
      for (size_t i = 0; i < hostMesh.primitives.size(); ++i)
      {
        deviceMesh.primitives[i].currentMaterial = hostMesh.primitives[i].currentMaterial;
      }

      deviceMesh.isDirty = hostMesh.isDirty;
    }
    // Reset the isDirty flag on the host meshes after they have been migrated to the device meshes.
    for (dev::HostMesh& hostMesh : m_hostMeshes)
    {
      hostMesh.isDirty = false;
    }
  }

  // While the above code will change the isDirty flags for all valid meshes inside the asset, 
  // the current scene is potentially only using a subset of these.
  // All meshes reached by the currently active m_instances will be rebuilt here.
  // The others are rebuilt automatically when switching scenes.
  if (rebuildAS)
  {
    buildDeviceMeshAccels(true); // This rebuilds only the device meshes with isDirty flags set.
    buildInstanceAccel(false);   // This updates the top-level IAS. The GAS AABBs didn't change, so updating the IAS is sufficient.
    updateSBT();                 // This rebuilds the SBT with all material hit records.
    updateLaunchParameters();    // This sets the root m_ias (shouldn't have changed on update) and restarts the accumulation.
  }
  else if (changed) // No rebuild required, just update all SBT hit records with the new MaterialData.
  {
    updateSBTMaterialData();
    
    m_launchParameters.iteration = 0u; // Restart accumulation when any material in the currently active scene changed.   
  }
}


void Application::initTrackball()
{
  if (m_isDefaultCamera)
  {
    dev::Camera& camera = m_cameras[0];

    MY_ASSERT(m_sceneExtent.isValid());
    camera.setPosition(m_sceneExtent.getCenter() + glm::vec3(0.0f, 0.0f, 1.75f * m_sceneExtent.getMaxDimension()));
    camera.setLookat(m_sceneExtent.getCenter());
  }

  // The trackball does nothing when there is no camera assigned to it.
  m_trackball.setCamera(&m_cameras[m_indexCamera]);
  
  //m_trackball.setMoveSpeed(10.0f);
  
  // This is required to initialize the current longitude and latitude values.
  m_trackball.setReferenceFrame(glm::vec3(1.0f, 0.0f, 0.0f),
                                glm::vec3(0.0f, 1.0f, 0.0f),
                                glm::vec3(0.0f, 0.0f, 1.0f));
  
  m_trackball.setGimbalLock(m_isLockedGimbal); // true keeps models upright when orbiting the trackball, false allows rolling.
}


LightDefinition Application::createConstantEnvironmentLight() const
{
  LightDefinition light = {}; // All unused fields are set to zero.

  light.typeLight = TYPE_LIGHT_ENV_CONST; 

  light.matrix[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrix[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrix[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
  
  light.matrixInv[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrixInv[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrixInv[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);

  light.emission = make_float3(1.0f); // White
 
  light.area        = 4.0f * M_PIf;
  light.invIntegral = 1.0f / light.area;

  return light;
}


LightDefinition Application::createSphericalEnvironmentLight()
{
  if (m_picEnv == nullptr)
  {
    m_picEnv = new Picture();
  }

  bool loadedEnv = false;
  if (!m_pathEnv.empty())
  {
    loadedEnv = m_picEnv->load(m_pathEnv, IMAGE_FLAG_2D);
  }
  if (!loadedEnv)
  {
    //m_picEnv->generateEnvironment(32, 16); // Dummy white environment.
    m_picEnv->generateEnvironmentSynthetic(1024, 512); // Generate HDR environment with some spot light regions.
  }
  
  // Create a new texture to keep the old texture intact in case anything goes wrong.
  Texture *texture = new Texture(m_allocator);

  if (!texture->create(m_picEnv, IMAGE_FLAG_2D | IMAGE_FLAG_ENV))
  {
    delete texture;
    throw std::runtime_error("createSphericalEnvironmentLight() environment map creation failed");
  }

  if (m_texEnv != nullptr)
  {
    delete m_texEnv;
    //m_texEnv = nullptr;
  }

  m_texEnv = texture;

  LightDefinition light = {}; // All unused fields are set to zero.

  light.matrix[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrix[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrix[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
  
  light.matrixInv[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrixInv[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrixInv[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);

  // Textured environment
  light.cdfU = m_texEnv->getCDF_U(); 
  light.cdfV = m_texEnv->getCDF_V();

  // Emisson texture. If not zero, scales emission.
  light.textureEmission = m_texEnv->getTextureObject();

  light.emission = make_float3(1.0f); // Modulates the texture.

  light.typeLight = TYPE_LIGHT_ENV_SPHERE; 
  
  light.area        = 4.0f * M_PIf; // Unused.
  light.invIntegral = 1.0f / m_texEnv->getIntegral();

  // Emission texture width and height. Used to index the CDFs, see above.
  // For mesh lights the width matches the number of triangles and the cdfU is over the triangle areas.
  light.width  = m_texEnv->getWidth(); 
  light.height = m_texEnv->getHeight();

  return light;
}


LightDefinition Application::createPointLight() const
{
  LightDefinition light = {};

  light.matrix[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrix[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrix[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
  
  light.matrixInv[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  light.matrixInv[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  light.matrixInv[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);

  // Set the point light emission depending on the scene size to cancel out the quadratic attenuation.
  light.emission = make_float3(1.0f); // * sqrtf(maxExtent) * 2.0f;

  light.typeLight = TYPE_LIGHT_POINT;
  
  light.area        = 1.0f; // Unused.
  light.invIntegral = 1.0f; // Unused.

  return light;
}

void Application::initLaunchParameters()
{
  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_d_launchParameters), sizeof(LaunchParameters)) );

  m_launchParameters.handle = m_ias; // Root traversable handle of the scene.
  
  // Output buffer for the rendered image (HDR linear color).
  // This is initialized inside updateBuffers() depending on the m_interop state.
  m_launchParameters.bufferAccum = nullptr;
  
  // Output buffer for the picked material index.
  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_launchParameters.bufferPicking), sizeof(int)) );

  m_launchParameters.resolution       = m_resolution; // Independent of the window client area.
  m_launchParameters.picking          = make_float2(-1.0f); // No picking ray.
  m_launchParameters.pathLengths      = make_int2(2, 6);
  m_launchParameters.iteration        = 0u;              // Sub-frame number for the progressive accumulation of results.
  m_launchParameters.sceneEpsilon     = m_epsilonFactor * SCENE_EPSILON_SCALE;
  m_launchParameters.directLighting   = (m_useDirectLighting)   ? 1 : 0;
  m_launchParameters.ambientOcclusion = (m_useAmbientOcclusion) ? 1 : 0;
  m_launchParameters.showEnvironment  = (m_showEnvironment)     ? 1 : 0;
  m_launchParameters.forceUnlit       = (m_forceUnlit)          ? 1 : 0;
  m_launchParameters.textureSheenLUT  = (m_texSheenLUT != nullptr) ? m_texSheenLUT->getTextureObject() : 0;

  m_launchParameters.lightDefinitions = reinterpret_cast<LightDefinition*>(m_d_lightDefinitions);
  m_launchParameters.numLights        = static_cast<int>(m_lightDefinitions.size());

  // All dirty flags are set here and the first render() call will take care to allocate and update all necessary resources.
}


void Application::updateLaunchParameters()
{
  // This is called after acceleration structures have been rebuilt.
  m_launchParameters.handle    = m_ias; // Update the top-level IAS handle.

  m_launchParameters.lightDefinitions = reinterpret_cast<LightDefinition*>(m_d_lightDefinitions);
  m_launchParameters.numLights        = static_cast<int>(m_lightDefinitions.size());

  m_launchParameters.iteration = 0u; // Restart accumulation.
}


void Application::updateLights()
{
  // When there exists an environment light, skip it and start the indexing of m_lightDefinitions at 1.
  int indexDefinition = (m_missID == 0) ? 0 : 1; 

  for (const dev::Light& light : m_lights)
  {
    LightDefinition& lightDefinition = m_lightDefinitions[indexDefinition];

    lightDefinition.emission = light.color * light.intensity;

    switch (light.type)
    {
      case 0: // Point
        lightDefinition.typeLight = TYPE_LIGHT_POINT;
        lightDefinition.range     = light.range;
        break;

      case 1: // Spot
        {
          lightDefinition.typeLight  = TYPE_LIGHT_SPOT;
          lightDefinition.range      = light.range;
          MY_ASSERT(light.innerConeAngle < light.outerConeAngle); // GLTF spec says these must not be equal.
          lightDefinition.cosInner   = cosf(light.innerConeAngle); // Minimum 0.0f, maximum < outerConeAngle.
          lightDefinition.cosOuter   = cosf(light.outerConeAngle); // Maximum M_PIf / 2.0f which is 90 degrees so this is the half-angle.
        }
        break;

      case 2: // Directional
        {
          MY_ASSERT(m_sceneExtent.isValid());
          lightDefinition.typeLight = TYPE_LIGHT_DIRECTIONAL;
          // Directional lights need to know the world size as area to be able to convert lux (lm/m^2).
          const float radius = m_sceneExtent.getMaxDimension() * 0.5f;
          MY_ASSERT(DENOMINATOR_EPSILON < radius); 
          lightDefinition.area = radius * radius * M_PIf;
        }
        break;
    }

    glm::mat4 matInv = glm::inverse(light.matrix);
    for (int i = 0; i < 3; ++i)
    {
      glm::vec4 row = glm::row(light.matrix, i);
      m_lightDefinitions[indexDefinition].matrix[i] = make_float4(row.x, row.y, row.z, row.w);
      row = glm::row(matInv, i);
      m_lightDefinitions[indexDefinition].matrixInv[i] = make_float4(row.x, row.y, row.z, row.w);
    }

    ++indexDefinition;
  }

  // Update all light definition device data.
  CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_d_lightDefinitions), m_lightDefinitions.data(), m_lightDefinitions.size() * sizeof(LightDefinition), cudaMemcpyHostToDevice) );

  m_launchParameters.iteration = 0; // Restart accumulation.
}


void Application::updateCamera()
{
  // Update host side copy of the launch parameters.
  if (m_trackball.getCamera() != &m_cameras[m_indexCamera])
  {
    m_trackball.setCamera(&m_cameras[m_indexCamera]);

    // This is required to initialize the current longitude and latitude values.
    m_trackball.setReferenceFrame(glm::vec3(1.0f, 0.0f, 0.0f),
                                  glm::vec3(0.0f, 1.0f, 0.0f),
                                  glm::vec3(0.0f, 0.0f, 1.0f));

    // This helps keeping models upright when orbiting the trackball.
    m_trackball.setGimbalLock(m_isLockedGimbal);
  }
  
  dev::Camera& camera = m_cameras[m_indexCamera];
    
  // This means the pPerspective->aspectRatio value doesn't matter at all.
  camera.setAspectRatio(static_cast<float>(m_resolution.x) / static_cast<float>(m_resolution.y));
    
  m_launchParameters.cameraType = (0.0f < camera.getFovY()) ? 1 : 0; // 0 == orthographic, 1 == perspective.

  glm::vec3 P = camera.getPosition();
    
  m_launchParameters.cameraP = make_float3(P.x, P.y, P.z);

  glm::vec3 U;
  glm::vec3 V;
  glm::vec3 W;
    
  camera.getUVW(U, V, W);

  // Convert to CUDA float3 vector types.
  m_launchParameters.cameraU = make_float3(U.x, U.y, U.z);
  m_launchParameters.cameraV = make_float3(V.x, V.y, V.z);
  m_launchParameters.cameraW = make_float3(W.x, W.y, W.z);

  m_launchParameters.iteration = 0; // Restart accumulation.

  camera.setIsDirty(false);
}


void Application::update()
{
  MY_ASSERT(!m_isDirtyResolution && m_hdrTexture != 0);
  
  switch (m_interop)
  {
    case INTEROP_OFF:
      // Copy the GPU local render buffer into host and update the HDR texture image from there.
      MY_ASSERT(m_bufferHost != nullptr);
      CUDA_CHECK( cudaMemcpy((void*) m_bufferHost, m_launchParameters.bufferAccum, m_resolution.x * m_resolution.y * sizeof(float4), cudaMemcpyDeviceToHost) );
      // Copy the host buffer to the OpenGL texture image (slowest path, most portable).
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei) m_resolution.x, (GLsizei) m_resolution.y, GL_RGBA, GL_FLOAT, m_bufferHost); // RGBA32F
      break;

    case INTEROP_PBO: 
      // The image was rendered into the linear PBO buffer directly. Just upload to the block-linear texture.
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei) m_resolution.x, (GLsizei) m_resolution.y, GL_RGBA, GL_FLOAT, (GLvoid*) 0); // RGBA32F from byte offset 0 in the pixel unpack buffer.
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      break;

  case INTEROP_TEX:
      {
        CUarray dstArray = nullptr;

        // Map the Texture object directly and copy the output buffer. 
        CU_CHECK( cuGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream )); // This is an implicit cuSynchronizeStream().
        CU_CHECK( cuGraphicsSubResourceGetMappedArray(&dstArray, m_cudaGraphicsResource, 0, 0) ); // arrayIndex = 0, mipLevel = 0

        CUDA_MEMCPY3D params = {};

        params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        params.srcDevice     = reinterpret_cast<CUdeviceptr>(m_launchParameters.bufferAccum);
        params.srcPitch      = m_resolution.x * sizeof(float4); // RGBA32F
        params.srcHeight     = m_resolution.y;

        params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        params.dstArray      = dstArray;
        params.WidthInBytes  = m_resolution.x * sizeof(float4);
        params.Height        = m_resolution.y;
        params.Depth         = 1;

        CU_CHECK( cuMemcpy3D(&params) ); // Copy from linear to array layout.

        CU_CHECK( cuGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream) ); // This is an implicit cuSynchronizeStream().
      }
      break;

  case INTEROP_IMG:
    // Nothing to do. Renders into the m_hdrTexture surface object directly.
    break;
  }
}


void Application::updateBufferHost()
{
  update();
  
  // After the update() call, the m_hdrTexture contains the linear HDR image.
  // When interop is off, the m_bufferHost already contains the current linear HDR image data as well.
  if (m_interop != INTEROP_OFF)
  {
    // Read the m_hdrTexture image into the m_bufferHost.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, (GLvoid*) m_bufferHost);
  }
}


bool Application::screenshot(const bool tonemap)
{
  updateBufferHost(); // Make sure m_bufferHost contains the linear HDR image data.

  ILboolean hasImage = false;

  std::ostringstream path;
   
  path << "img_gltf_" << utils::getDateTime();
  
  unsigned int imageID;

  ilGenImages(1, (ILuint *) &imageID);

  ilBindImage(imageID);
  ilActiveImage(0);
  ilActiveFace(0);

  ilDisable(IL_ORIGIN_SET);

  if (tonemap)
  {
    path << ".png"; // Store a tonemapped RGB8 *.png image

    if (ilTexImage(m_launchParameters.resolution.x, m_launchParameters.resolution.y, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, nullptr))
    {
      uchar3* dst = reinterpret_cast<uchar3*>(ilGetData());

      const float  invGamma       = 1.0f / m_gamma;
      const float3 colorBalance   = make_float3(m_colorBalance.x, m_colorBalance.y, m_colorBalance.z);
      const float  invWhitePoint  = m_brightness / m_whitePoint;
      const float  burnHighlights = m_burnHighlights;
      const float  crushBlacks    = m_crushBlacks + m_crushBlacks + 1.0f;
      const float  saturation     = m_saturation;

      for (int y = 0; y < m_launchParameters.resolution.y; ++y)
      {
        for (int x = 0; x < m_launchParameters.resolution.x; ++x)
        {
          const int idx = m_launchParameters.resolution.x * y + x;

          // Tonemapper. // PERF Add a native CUDA kernel doing this.
          float3 hdrColor = make_float3(m_bufferHost[idx]);

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
    path << ".hdr"; // Store the float4 linear output buffer as *.hdr image.

    hasImage = ilTexImage(m_launchParameters.resolution.x, m_launchParameters.resolution.y, 1, 4, IL_RGBA, IL_FLOAT, (void*) m_bufferHost);
  }

  if (hasImage)
  {
    ilEnable(IL_FILE_OVERWRITE); // By default, always overwrite
    
    std::string filename = path.str();
    utils::convertPath(filename);

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


// Init (if needed) and scale the font depending on screen DPI.
void Application::updateFonts()
{
  // Create or update the font
  ImGuiIO& io = ImGui::GetIO();

  const float fontScale = utils::getFontScale();
  if (fontScale == m_fontScale && m_font != nullptr)
  {
    return;
  }

  // change of DPI detected or no font yet
  m_fontScale = fontScale;
  io.FontGlobalScale = m_fontScale;
  io.FontAllowUserScaling = true;// enable scaling with ctrl + wheel.
  std::cerr << "FontGlobalScale " << io.FontGlobalScale << std::endl;

#if defined(_WIN32)
  static const char* fontName{ "C:/Windows/Fonts/arialbd.ttf" };
#else
  // works on Ubuntu Linux
  static const char* fontName{ "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" };
#endif

  // create and/or scale the font
  if (m_font == nullptr)
  {
    // load the font and create the texture
    io.Fonts->AddFontDefault();
    std::cout << "Loading font " << fontName << std::endl;
    m_font = io.Fonts->AddFontFromFileTTF(fontName, 13.0f);
    glCreateTextures(GL_TEXTURE_2D, 1, &m_fontTexture);
  }

  if (m_font != nullptr)
  {
    // update the texture with scaled font data
    unsigned char* pixels = nullptr;
    int texW, texH;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &texW, &texH);

    // DONT glBindTextures(0, 1, &m_fontTexture); or device update will fail
    glTextureStorage2D(m_fontTexture, 1, GL_RGBA8, texW, texH);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTextureSubImage2D(m_fontTexture, 0, 0, 0, texW, texH, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

#pragma warning( push )
#pragma warning( disable : 4312)
    io.Fonts->SetTexID(reinterpret_cast<ImTextureID>(m_fontTexture));
#pragma warning( pop )
  }
  else
  {
    std::cerr << "ERROR can't load font " << fontName << std::endl;
  }
}

unsigned int Application::getBuildFlags()const
{
  unsigned int flags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  // Prefer fast trace when there are no animations, otherwise prefer fast build.
  flags |= (m_animations.empty()) ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
  // accelBuildOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
  return flags;
}

void Application::cameraTranslate(float dx, float dy, float dz)
{
  MY_ASSERT(m_sceneExtent.isValid());
  const float scale = m_sceneExtent.getDiameter() * CameraSpeedFraction;
  MY_ASSERT(0.0f != scale);

  const glm::vec3 transl{ scale * dx, scale * dy, scale * dz };

  for (auto& cam : m_cameras) {
    const auto fwd   = cam.getDirection();
    const auto right = cam.getRight();
    const auto up    = cam.getUp();
    const auto delta = (fwd * transl.z) + (right * transl.x) + (up * transl.y);
    auto pos = cam.getPosition();
    auto lookat = cam.getLookat();
    pos    += delta;
    lookat += delta;
    cam.setPosition(pos);
    cam.setLookat(lookat);
  }
}
