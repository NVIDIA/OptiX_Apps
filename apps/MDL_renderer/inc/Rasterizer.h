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
 
#ifndef RASTERIZER_H
#define RASTERIZER_H

#include <GL/glew.h>
#if defined( _WIN32 )
#include <GL/wglew.h>
#endif

#include "inc/Timer.h"
#include "inc/TonemapperGUI.h"

#include <vector>

typedef struct
{
  float u;     // u-coordinate in range [0.0, 1.0]
  float c[3];  // color, components in range [0.0, 1.0]
} ColorRampElement;


class Rasterizer
{
public:
  Rasterizer(const int w, const int h, const int interop);
  ~Rasterizer();

  void reshape(const int w, const int h);
  void display();
  
  const int getNumDevices() const;
  
  const unsigned char* getUUID(const unsigned int index) const;
  const unsigned char* getLUID() const;
  int getNodeMask() const;

  unsigned int getTextureObject() const;
  unsigned int getPixelBufferObject() const;

  void setResolution(const int w, const int h);
  void setTonemapper(const TonemapperGUI& tm);

private:
  void checkInfoLog(const char *msg, GLuint object);
  void initGLSL();
  void updateProjectionMatrix();
  void updateVertexAttributes();

private:
  int m_width;
  int m_height;
  int m_interop;

  int m_widthResolution;
  int m_heightResolution;

  GLint   m_numDevices;                       // Number of OpenGL devices. Normally 1, unless multicast is enabled.
  GLubyte m_deviceUUID[24][GL_UUID_SIZE_EXT]; // Max. 24 devices supported. 16 bytes identifier.
  //GLubyte m_driverUUID[GL_UUID_SIZE_EXT];   // 16 bytes identifier, unused.
  GLubyte m_deviceLUID[GL_LUID_SIZE_EXT];     //  8 bytes identifier.
  GLint   m_nodeMask;                         // Node mask used together with the LUID to identify OpenGL device uniquely.

  GLuint m_hdrTexture;
  GLuint m_pbo;

  GLuint m_colorRampTexture;

  GLuint m_glslProgram;
  
  GLuint m_vboAttributes;
  GLuint m_vboIndices;

  GLint m_locAttrPosition;
  GLint m_locAttrTexCoord;
  GLint m_locProjection;

  GLint m_locSamplerHDR;
  GLint m_locSamplerColorRamp;

  // Rasterizer side of the TonemapperGUI data
  GLint m_locInvGamma;
  GLint m_locColorBalance;
  GLint m_locInvWhitePoint;
  GLint m_locBurnHighlights;
  GLint m_locCrushBlacks;
  GLint m_locSaturation;

  Timer m_timer;
};

#endif // RASTERIZER_H
