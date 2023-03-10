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

#include "shaders/config.h"

#include "inc/Rasterizer.h"
#include "inc/MyAssert.h"

#include <algorithm>
#include <iostream>
#include <string.h>


#if USE_TIME_VIEW
static bool generateColorRamp(const std::vector<ColorRampElement>& definition, int size, float *ramp)
{
  if (definition.size() < 1 || !size || !ramp) 
  {
    return false;
  }

  float r;
  float g;
  float b;
  float a = 1.0f; // CUDA doesn't support float3 textures.

  float *p = ramp;

  if (definition.size() == 1)
  {
    // Special case, only one color in the input, means the whole color ramp is that color.
    r = definition[0].c[0];
    g = definition[0].c[1];
    b = definition[0].c[2];

    for (int i = 0; i < size; i++)
    {
      *p++ = r;
      *p++ = g;
      *p++ = b;
      *p++ = a;
    }
    return true;
  }

  // Here definition.size() is at least 2.
  ColorRampElement left;
  ColorRampElement right;
  size_t entry = 0;

  left = definition[entry];
  if (0.0f < left.u)
  {
    left.u = 0.0f;
  }
  else // left.u == 0.0f;
  {
    entry++;
  }
  right = definition[entry++];

  for (int i = 0; i < size; ++i)
  {
    // The 1D coordinate at which we need to calculate the color.
    float u = (float) i / (float) (size - 1);
    
    // Check if it's in the range [left.u, right.u)
    while (!(left.u <= u && u < right.u))
    {
      left = right;
      if (entry < definition.size())
      {
        right = definition[entry++];
      }
      else
      {
        // left is already the last entry, move right.u to the end of the range.
        right.u = 1.0001f; // Make sure we pass 1.0 < right.u in the last iteration.
        break;
      }
    }

    float t = (u - left.u) / (right.u - left.u);
    r = left.c[0] + t * (right.c[0] - left.c[0]);
    g = left.c[1] + t * (right.c[1] - left.c[1]);
    b = left.c[2] + t * (right.c[2] - left.c[2]);

    *p++ = r;
    *p++ = g;
    *p++ = b;
    *p++ = a;
  }
  return true;
}
#endif


Rasterizer::Rasterizer(const int w, const int h, const int interop)
: m_width(w)
, m_height(h)
, m_interop(interop)
, m_widthResolution(w)
, m_heightResolution(h)
, m_numDevices(0)
, m_nodeMask(0)
, m_hdrTexture(0)
, m_pbo(0)
, m_glslProgram(0)
, m_vboAttributes(0)
, m_vboIndices(0)
, m_locAttrPosition(-1)
, m_locAttrTexCoord(-1)
, m_locProjection(-1)
, m_locSamplerHDR(-1)
, m_colorRampTexture(0)
, m_locSamplerColorRamp(-1)
, m_locInvGamma(-1)
, m_locColorBalance(-1)
, m_locInvWhitePoint(-1)
, m_locBurnHighlights(-1)
, m_locCrushBlacks(-1)
, m_locSaturation(-1)
{
  for (int i = 0; i < 8; ++i)
  {
    memset(m_deviceUUID[0], 0, sizeof(m_deviceUUID[0]));
  }
  //memset(m_driverUUID, 0, sizeof(m_driverUUID)); // Unused.
  memset(m_deviceLUID, 0, sizeof(m_deviceLUID));
  
  // Find out which device is running the OpenGL implementation to be able to allocate the PBO peer-to-peer staging buffer on the same device.
  // Needs these OpenGL extensions: 
  // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_external_objects.txt
  // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_external_objects_win32.txt
  // and on CUDA side the CUDA 10.0 Driver API function cuDeviceGetLuid().
  // While the extensions are named EXT_external_objects, the enums and functions are found under name string EXT_memory_object!
  if (GLEW_EXT_memory_object)
  {
    // UUID
    // To determine which devices are used by the current context, first call GetIntegerv with <pname> set to NUM_DEVICE_UUIDS_EXT, 
    // then call GetUnsignedBytei_vEXT with <target> set to DEVICE_UUID_EXT, <index> set to a value in the range [0, <number of device UUIDs>),
    // and <data> set to point to an array of UUID_SIZE_EXT unsigned bytes. 
    glGetIntegerv(GL_NUM_DEVICE_UUIDS_EXT, &m_numDevices); // This is normally 1, but not when multicast is enabled!
    MY_ASSERT(m_numDevices <= 24); // DEBUG m_deviceUUID is only prepared for 24 devices.
    m_numDevices = std::min(m_numDevices, 24);
    
    for (GLint i = 0; i < m_numDevices; ++i)
    {
      glGetUnsignedBytei_vEXT(GL_DEVICE_UUID_EXT, i, m_deviceUUID[i]);
    }
    //glGetUnsignedBytevEXT(GL_DRIVER_UUID_EXT, m_driverUUID); // Not used here.

    // LUID 
    // "The devices in use by the current context may also be identified by an (LUID, node) pair.
    //  To determine the LUID of the current context, call GetUnsignedBytev with <pname> set to DEVICE_LUID_EXT and <data> set to point to an array of LUID_SIZE_EXT unsigned bytes.
    //  Following the call, <data> can be cast to a pointer to an LUID object that will be equal to the locally unique identifier 
    //  of an IDXGIAdapter1 object corresponding to the adapter used by the current context.
    //  To identify which individual devices within an adapter are used by the current context, call GetIntegerv with <pname> set to DEVICE_NODE_MASK_EXT.
    //  A bitfield is returned with one bit set for each device node used by the current context.
    //  The bits set will be subset of those available on a Direct3D 12 device created on an adapter with the same LUID as the current context."
    if (GLEW_EXT_memory_object_win32) 
    {
      // It is not expected that a single context will be associated with multiple DXGI adapters, so only one LUID is returned.
      glGetUnsignedBytevEXT(GL_DEVICE_LUID_EXT, m_deviceLUID);
      glGetIntegerv(GL_DEVICE_NODE_MASK_EXT, &m_nodeMask);
    }
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

  // For batch rendering initialize the texture contents to some default.
  const float texel[4] = { 1.0f, 0.0f, 1.0f, 1.0f }; // Magenta to indicate that the texture has not been initialized.
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, &texel); // RGBA32F

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  if (GLEW_NV_gpu_multicast)
  {
    const char* envMulticast = getenv("GL_NV_GPU_MULTICAST");
    if (envMulticast != nullptr && envMulticast[0] != '0')
    {
      std::cerr << "WARNING: Rasterizer() GL_NV_GPU_MULTICAST is enabled. Primary device needs to be inside the devicesMask to display correctly.\n";
      glTexParameteri(GL_TEXTURE_2D, GL_PER_GPU_STORAGE_NV, GL_TRUE);
    }
  }

  glBindTexture(GL_TEXTURE_2D, 0);

  // The local ImGui sources have been changed to push the GL_TEXTURE_BIT so that this works. 
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  if (m_interop == INTEROP_MODE_PBO)
  {
    // PBO for CUDA-OpenGL interop.
    glGenBuffers(1, &m_pbo);
    MY_ASSERT(m_pbo != 0); 

    // Make sure the buffer is not zero size.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(float) * 4, (GLvoid*) 0, GL_DYNAMIC_DRAW); // RGBA32F from byte offset 0 in the pixel unpack buffer.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  }

  // GLSL shaders objects and program. Must be initialized before using any shader variable locations. 
  initGLSL();
  
  // This initialization is just to generate the vertex buffer objects and bind the VertexAttribPointers.
  // Two hardcoded triangles in the viewport size projection coordinate system with 2D texture coordinates.
  // These get updated to the correct values in reshape() and in setResolution(). The resolution is not known at this point.
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

#if USE_TIME_VIEW
  // Generate the color ramp definition vector.
  std::vector<ColorRampElement> colorRampDefinition;
  
  ColorRampElement cre;

  // Cold to hot: blue, green, red, yellow, white
  cre.u = 0.0f;
  cre.c[0] = 0.0f; // blue
  cre.c[1] = 0.0f;
  cre.c[2] = 1.0f;  
  colorRampDefinition.push_back(cre);
  cre.u = 0.25f;
  cre.c[0] = 0.0f; // green
  cre.c[1] = 1.0f;
  cre.c[2] = 0.0f;  
  colorRampDefinition.push_back(cre);
  cre.u = 0.5f;
  cre.c[0] = 1.0f; // red
  cre.c[1] = 0.0f;
  cre.c[2] = 0.0f;  
  colorRampDefinition.push_back(cre);
  cre.u = 0.75f;
  cre.c[0] = 1.0f; // yellow
  cre.c[1] = 1.0f;
  cre.c[2] = 0.0f;  
  colorRampDefinition.push_back(cre);
  cre.u = 1.0f;
  cre.c[0] = 1.0f; // white
  cre.c[1] = 1.0f;
  cre.c[2] = 1.0f;  
  colorRampDefinition.push_back(cre);

  std::vector<float> texels(256 * 4);

  bool success = generateColorRamp(colorRampDefinition, 256, texels.data());
  if (success)
  {
    glGenTextures(1, &m_colorRampTexture);
    MY_ASSERT(m_colorRampTexture != 0);
    
    glActiveTexture(GL_TEXTURE1); // It's set to texture image unit 1.
    glBindTexture(GL_TEXTURE_1D, m_colorRampTexture);

    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, 256, 0, GL_RGBA, GL_FLOAT, texels.data());

    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glActiveTexture(GL_TEXTURE0);
  }
#endif
}

Rasterizer::~Rasterizer()
{
  glDeleteTextures(1, &m_hdrTexture);

#if USE_TIME_VIEW
  glDeleteTextures(1, &m_colorRampTexture);
#endif
  
  if (m_interop)
  {
    glDeleteBuffers(1, &m_pbo);
  }

  glDeleteBuffers(1, &m_vboAttributes);
  glDeleteBuffers(1, &m_vboIndices);

  glDeleteProgram(m_glslProgram);
}


void Rasterizer::reshape(const int w, const int h)
{
  // No check for zero sizes needed. That's done in Application::reshape()
  if (m_width != w || m_height != h)
  {
    m_width  = w;
    m_height = h;

    glViewport(0, 0, m_width, m_height);

    updateProjectionMatrix();
    updateVertexAttributes();
  }
}

void Rasterizer::display()
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

const int Rasterizer::getNumDevices() const
{
  return m_numDevices;
}

const unsigned char* Rasterizer::getUUID(const unsigned int index) const
{
  MY_ASSERT(index < 24);
  return m_deviceUUID[index];
}

const unsigned char* Rasterizer::getLUID() const
{
  return m_deviceLUID;
}

int Rasterizer::getNodeMask() const
{
  return m_nodeMask;
}

unsigned int Rasterizer::getTextureObject() const
{
  return m_hdrTexture;
}

unsigned int Rasterizer::getPixelBufferObject() const
{
  return m_pbo;
}

void Rasterizer::setResolution(const int w, const int h)
{
  if (m_widthResolution != w || m_heightResolution != h)
  {
    m_widthResolution  = (0 < w) ? w : 1;
    m_heightResolution = (0 < h) ? h : 1;

    updateVertexAttributes();

    // Cannot resize the PBO while it's registered with cuGraphicsGLRegisterBuffer(). Deferred to the Device::render() calls.
  }
}

void Rasterizer::setTonemapper(const TonemapperGUI& tm)
{
#if !USE_TIME_VIEW
  glUseProgram(m_glslProgram);

  glUniform1f(m_locInvGamma, 1.0f / tm.gamma);
  glUniform3f(m_locColorBalance, tm.colorBalance[0], tm.colorBalance[1], tm.colorBalance[2]);
  glUniform1f(m_locInvWhitePoint, tm.brightness / tm.whitePoint);
  glUniform1f(m_locBurnHighlights, tm.burnHighlights);
  glUniform1f(m_locCrushBlacks, tm.crushBlacks + tm.crushBlacks + 1.0f);
  glUniform1f(m_locSaturation, tm.saturation);

  glUseProgram(0);
#endif
}


// Private functions:

void Rasterizer::checkInfoLog(const char* /* msg */, GLuint object)
{
  GLint maxLength = 0;

  const GLboolean isShader = glIsShader(object);
  
  if (isShader)
  {
    glGetShaderiv(object, GL_INFO_LOG_LENGTH, &maxLength);
  }
  else
  {
    glGetProgramiv(object, GL_INFO_LOG_LENGTH, &maxLength);
  }

  if (1 < maxLength) 
  {
    GLchar *infoLog = new GLchar[maxLength];

    if (infoLog != nullptr)
    {
      GLint length = 0;

      if (isShader)
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
      
      delete [] infoLog;
    }
  }
}

void Rasterizer::initGLSL()
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


#if USE_TIME_VIEW
  static const std::string fsSource =
    "#version 330\n"
    "uniform sampler2D samplerHDR;\n"
    "uniform sampler1D samplerColorRamp;\n"
    "in vec2 varTexCoord;\n"
    "layout(location = 0, index = 0) out vec4 outColor;\n"
    "void main()\n"
    "{\n"
    "  float alpha = texture(samplerHDR, varTexCoord).a;\n"
    "  outColor = texture(samplerColorRamp, alpha);\n"
    "}\n";
#else
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
#endif

  GLint vsCompiled = 0;
  GLint fsCompiled = 0;
    
  GLuint glslVS = glCreateShader(GL_VERTEX_SHADER);
  if (glslVS)
  {
    GLsizei len = (GLsizei) vsSource.size();
    const GLchar *vs = vsSource.c_str();
    glShaderSource(glslVS, 1, &vs, &len);
    glCompileShader(glslVS);
    checkInfoLog(vs, glslVS);

    glGetShaderiv(glslVS, GL_COMPILE_STATUS, &vsCompiled);
    MY_ASSERT(vsCompiled);
  }

  GLuint glslFS = glCreateShader(GL_FRAGMENT_SHADER);
  if (glslFS)
  {
    GLsizei len = (GLsizei) fsSource.size();
    const GLchar *fs = fsSource.c_str();
    glShaderSource(glslFS, 1, &fs, &len);
    glCompileShader(glslFS);
    checkInfoLog(fs, glslFS);

    glGetShaderiv(glslFS, GL_COMPILE_STATUS, &fsCompiled);
    MY_ASSERT(fsCompiled);
  }

  m_glslProgram = glCreateProgram();
  if (m_glslProgram)
  {
    GLint programLinked = 0;

    if (glslVS && vsCompiled)
    {
      glAttachShader(m_glslProgram, glslVS);
    }
    if (glslFS && fsCompiled)
    {
      glAttachShader(m_glslProgram, glslFS);
    }

    glLinkProgram(m_glslProgram);
    checkInfoLog("m_glslProgram", m_glslProgram);

    glGetProgramiv(m_glslProgram, GL_LINK_STATUS, &programLinked);
    MY_ASSERT(programLinked);
    
    if (programLinked)
    {
      glUseProgram(m_glslProgram);

      // FIXME Put these into a struct.
      m_locAttrPosition = glGetAttribLocation(m_glslProgram, "attrPosition");
      m_locAttrTexCoord = glGetAttribLocation(m_glslProgram, "attrTexCoord");
      m_locProjection   = glGetUniformLocation(m_glslProgram, "projection");

      MY_ASSERT(m_locAttrPosition != -1);
      MY_ASSERT(m_locAttrTexCoord != -1);
      MY_ASSERT(m_locProjection   != -1);

      m_locSamplerHDR = glGetUniformLocation(m_glslProgram, "samplerHDR");
      MY_ASSERT(m_locSamplerHDR != -1);
      glUniform1i(m_locSamplerHDR, 0); // The rasterizer uses texture image unit 0 to display the HDR image.

#if USE_TIME_VIEW
      m_locSamplerColorRamp = glGetUniformLocation(m_glslProgram, "samplerColorRamp");
      MY_ASSERT(m_locSamplerColorRamp != -1);
      glUniform1i(m_locSamplerColorRamp, 1); // The rasterizer uses texture image unit 1 for the color ramp when USE_TIME_VIEW is enabled.
#else
      m_locInvGamma       = glGetUniformLocation(m_glslProgram, "invGamma");
      m_locColorBalance   = glGetUniformLocation(m_glslProgram, "colorBalance");
      m_locInvWhitePoint  = glGetUniformLocation(m_glslProgram, "invWhitePoint");
      m_locBurnHighlights = glGetUniformLocation(m_glslProgram, "burnHighlights");
      m_locCrushBlacks    = glGetUniformLocation(m_glslProgram, "crushBlacks");
      m_locSaturation     = glGetUniformLocation(m_glslProgram, "saturation");

      MY_ASSERT(m_locInvGamma != -1);
      MY_ASSERT(m_locColorBalance  != -1);
      MY_ASSERT(m_locInvWhitePoint != -1);
      MY_ASSERT(m_locBurnHighlights != -1);
      MY_ASSERT(m_locCrushBlacks != -1);
      MY_ASSERT(m_locSaturation != -1);

      // Set neutral Tonemapper defaults. This will show the linear HDR image.
      glUniform1f(m_locInvGamma, 1.0f);
      glUniform3f(m_locColorBalance, 1.0f, 1.0f, 1.0f);
      glUniform1f(m_locInvWhitePoint, 1.0f);
      glUniform1f(m_locBurnHighlights, 1.0f);
      glUniform1f(m_locCrushBlacks, 1.0f);
      glUniform1f(m_locSaturation, 1.0f);
#endif

      glUseProgram(0);
    }
  }

  if (glslVS)
  {
    glDeleteShader(glslVS);
  }
  if (glslFS)
  {
    glDeleteShader(glslFS);
  }
}


void Rasterizer::updateProjectionMatrix()
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


void Rasterizer::updateVertexAttributes()
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
  
  if (m_widthResolution <= m_width && m_heightResolution <= m_height)
  {
    // Texture fits into viewport without scaling.
    // Calculate the amount of cleared border pixels.
    int w1 = m_width  - m_widthResolution;
    int h1 = m_height - m_heightResolution;
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
  else // Case 
  {
    // Texture needs to be scaled down to fit into client window.
    // Check which extent defines the necessary scaling factor.
    const float wC = float(m_width);
    const float hC = float(m_height);
    const float wR = float(m_widthResolution);
    const float hR = float(m_heightResolution);

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
