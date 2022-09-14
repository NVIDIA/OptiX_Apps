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

#include "inc/SceneGraph.h"

#include "shaders/vector_math.h"

namespace sg
{

  // A simple unit cube built from 12 triangles.
  void Triangles::createBox()
  {
    m_attributes.clear();
    m_indices.clear();

    const float left = -1.0f;
    const float right = 1.0f;
    const float bottom = -1.0f;
    const float top = 1.0f;
    const float back = -1.0f;
    const float front = 1.0f;

    TriangleAttributes attrib;

    // Left.
    attrib.tangent  = make_float3(0.0f, 0.0f, 1.0f);
    attrib.normal   = make_float3(-1.0f, 0.0f, 0.0f);

    attrib.vertex   = make_float3(left, bottom, back);
    attrib.texcoord = make_float3(0.0f, 0.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(left, bottom, front);
    attrib.texcoord = make_float3(1.0f, 0.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(left, top, front);
    attrib.texcoord = make_float3(1.0f, 1.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(left, top, back);
    attrib.texcoord = make_float3(0.0f, 1.0f, 0.0f);
    m_attributes.push_back(attrib);

    // Right.
    attrib.tangent  = make_float3(0.0f, 0.0f, -1.0f);
    attrib.normal   = make_float3(1.0f, 0.0f, 0.0f);

    attrib.vertex   = make_float3(right, bottom, front);
    attrib.texcoord = make_float3(0.0f, 0.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(right, bottom, back);
    attrib.texcoord = make_float3(1.0f, 0.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(right, top, back);
    attrib.texcoord = make_float3(1.0f, 1.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(right, top, front);
    attrib.texcoord = make_float3(0.0f, 1.0f, 0.0f);
    m_attributes.push_back(attrib);

    // Back.  
    attrib.tangent  = make_float3(-1.0f, 0.0f, 0.0f);
    attrib.normal   = make_float3(0.0f, 0.0f, -1.0f);

    attrib.vertex   = make_float3(right, bottom, back);
    attrib.texcoord = make_float3(0.0f, 0.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex = make_float3(left, bottom, back);
    attrib.texcoord = make_float3(1.0f, 0.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex = make_float3(left, top, back);
    attrib.texcoord = make_float3(1.0f, 1.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex = make_float3(right, top, back);
    attrib.texcoord = make_float3(0.0f, 1.0f, 0.0f);
    m_attributes.push_back(attrib);

    // Front.
    attrib.tangent  = make_float3(1.0f, 0.0f, 0.0f);
    attrib.normal   = make_float3(0.0f, 0.0f, 1.0f);

    attrib.vertex   = make_float3(left, bottom, front);
    attrib.texcoord = make_float3(0.0f, 0.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(right, bottom, front);
    attrib.texcoord = make_float3(1.0f, 0.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(right, top, front);
    attrib.texcoord = make_float3(1.0f, 1.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(left, top, front);
    attrib.texcoord = make_float3(0.0f, 1.0f, 0.0f);
    m_attributes.push_back(attrib);

    // Bottom.
    attrib.tangent  = make_float3(1.0f, 0.0f, 0.0f);
    attrib.normal   = make_float3(0.0f, -1.0f, 0.0f);

    attrib.vertex   = make_float3(left, bottom, back);
    attrib.texcoord = make_float3(0.0f, 0.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(right, bottom, back);
    attrib.texcoord = make_float3(1.0f, 0.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(right, bottom, front);
    attrib.texcoord = make_float3(1.0f, 1.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(left, bottom, front);
    attrib.texcoord = make_float3(0.0f, 1.0f, 0.0f);
    m_attributes.push_back(attrib);

    // Top.
    attrib.tangent  = make_float3(1.0f, 0.0f, 0.0f);
    attrib.normal   = make_float3(0.0f, 1.0f, 0.0f);

    attrib.vertex   = make_float3(left, top, front);
    attrib.texcoord = make_float3(0.0f, 0.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(right, top, front);
    attrib.texcoord = make_float3(1.0f, 0.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(right, top, back);
    attrib.texcoord = make_float3(1.0f, 1.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(left, top, back);
    attrib.texcoord = make_float3(0.0f, 1.0f, 0.0f);
    m_attributes.push_back(attrib);

    for (unsigned int i = 0; i < 6; ++i)
    {
      const unsigned int idx = i * 4; // Four m_attributes per box face.

      m_indices.push_back(idx);
      m_indices.push_back(idx + 1);
      m_indices.push_back(idx + 2);

      m_indices.push_back(idx + 2);
      m_indices.push_back(idx + 3);
      m_indices.push_back(idx);
    }
  }

} // namespace sg
