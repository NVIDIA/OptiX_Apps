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

  // Special case for the rectangle light.
  // It's a square centered at the origin with extents in the range [-0.5, 0.5] on the xy-plane with positive z-axis as normal.
  void Triangles::createRect()
  {
    m_attributes.clear();
    m_indices.clear();
    
    TriangleAttributes attrib;

    // Same for all four vertices of the rectangle.
    attrib.tangent  = make_float3(1.0f, 0.0f, 0.0f);
    attrib.normal   = make_float3(0.0f, 0.0f, 1.0f);

    attrib.vertex   = make_float3(-0.5f, -0.5f, 0.0f); // left bottom
    attrib.texcoord = make_float3(0.0f, 0.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(0.5f, -0.5f, 0.0f); // right bottom
    attrib.texcoord = make_float3(1.0f, 0.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(0.5f, 0.5f, 0.0f);  // right top
    attrib.texcoord = make_float3(1.0f, 1.0f, 0.0f);
    m_attributes.push_back(attrib);

    attrib.vertex   = make_float3(-0.5f, 0.5f, 0.0f); // left top
    attrib.texcoord = make_float3(0.0f, 1.0f, 0.0f);
    m_attributes.push_back(attrib);

    m_indices.push_back(0);
    m_indices.push_back(1);
    m_indices.push_back(2);

    m_indices.push_back(2);
    m_indices.push_back(3);
    m_indices.push_back(0);
  }

} // namespace sg
