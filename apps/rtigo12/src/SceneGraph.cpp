/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
#include "shaders/vector_math.h"

#include "inc/SceneGraph.h"

#include <cstring>
#include <iostream>
#include <sstream>

#include "inc/MyAssert.h"

namespace sg
{
  // ========== Node
  Node::Node(const unsigned int id)
  : m_id(id)
  {
  }

  //Node::~Node()
  //{
  //}

  // ========== Group
  Group::Group(const unsigned int id)
  : Node(id)
  {
  }

  //Group::~Group()
  //{
  //}

  sg::NodeType Group::getType() const
  {
    return NT_GROUP;
  }

  void Group::addChild(std::shared_ptr<sg::Instance> instance)
  {
    m_children.push_back(instance);
  }

  size_t Group::getNumChildren() const
  {
    return m_children.size();
  }

  std::shared_ptr<sg::Instance> Group::getChild(const size_t index)
  {
    MY_ASSERT(index < m_children.size());
    return m_children[index];
  }


  // ========== Instance
  Instance::Instance(const unsigned int id)
  : Node(id)
  , m_material(-1) // No material index set by default. Last one >= 0 along a path wins.
  , m_light(-1)    // No light index set by default. Not a light.
  {
    // Set the affine matrix to identity by default.
    memset(m_matrix, 0, sizeof(float) * 12);
    m_matrix[ 0] = 1.0f;
    m_matrix[ 5] = 1.0f;
    m_matrix[10] = 1.0f;
  }

  //Instance::~Instance()
  //{
  //}

  sg::NodeType Instance::getType() const
  {
    return NT_INSTANCE;
  }

  void Instance::setTransform(const float m[12])
  {
    memcpy(m_matrix, m, sizeof(float) * 12);
  }

  const float* Instance::getTransform() const
  {
    return m_matrix;
  }

  void Instance::setChild(std::shared_ptr<sg::Node> node) // Instances can hold all other groups.
  {
    m_child = node;
  }

  std::shared_ptr<sg::Node> Instance::getChild()
  {
    return m_child;
  }

  void Instance::setMaterial(const int index)
  {
    m_material = index;
  }

  int Instance::getMaterial() const
  {
    return m_material;
  }

  void Instance::setLight(const int index)
  {
    m_light = index;
  }

  int Instance::getLight() const
  {
    return m_light;
  }

  // ========== Triangles
  Triangles::Triangles(const unsigned int id)
  : Node(id)
  {
  }

  //Triangles::~Triangles()
  //{
  //}

  sg::NodeType Triangles::getType() const
  {
    return NT_TRIANGLES;
  }

  void Triangles::setAttributes(const std::vector<TriangleAttributes>& attributes)
  {
    m_attributes.resize(attributes.size());
    memcpy(m_attributes.data(), attributes.data(), sizeof(TriangleAttributes) * attributes.size());
  }

  const std::vector<TriangleAttributes>& Triangles::getAttributes() const
  {
    return m_attributes;
  }

  void Triangles::setIndices(const std::vector<unsigned int>& indices)
  {
    m_indices.resize(indices.size());
    memcpy(m_indices.data(), indices.data(), sizeof(unsigned int) * indices.size());
  }
  
  const std::vector<unsigned int>& Triangles::getIndices() const
  {
    return m_indices;
  }

  // Helper function for arbitrary mesh lights.
  // PERF Implement this in a CUDA kernel.
  void Triangles::calculateLightArea(LightGUI& lightGUI) const
  {
    const size_t numTriangles = m_indices.size() / 3;

    lightGUI.cdfAreas.resize(numTriangles + 1);

    float areaSurface = 0.0f;
    
    lightGUI.cdfAreas[0] = areaSurface; // CDF starts with zero. One element more than number of triangles.

    for (size_t i = 0; i < numTriangles; ++i)
    {
      const size_t idx = i * 3;
      
      const unsigned int i0 = m_indices[idx    ];
      const unsigned int i1 = m_indices[idx + 1];
      const unsigned int i2 = m_indices[idx + 2];

      // All in object space.
      const float3 v0 = m_attributes[i0].vertex;
      const float3 v1 = m_attributes[i1].vertex;
      const float3 v2 = m_attributes[i2].vertex;

      // PERF Work in world space to do fewer transforms during explicit light hits.
      dp::math::Vec3f p0(dp::math::Vec4f(v0.x, v0.y, v0.z, 1.0f) * lightGUI.matrix);
      dp::math::Vec3f p1(dp::math::Vec4f(v1.x, v1.y, v1.z, 1.0f) * lightGUI.matrix);
      dp::math::Vec3f p2(dp::math::Vec4f(v2.x, v2.y, v2.z, 1.0f) * lightGUI.matrix);

      dp::math::Vec3f e0 = p1 - p0;
      dp::math::Vec3f e1 = p2 - p0;

      // The triangle area is half of the parallelogram area (length of cross product).
      const float area = dp::math::length(e0 ^ e1) * 0.5f;

      areaSurface += area;

      lightGUI.cdfAreas[i + 1]  = areaSurface; // Store the unnormalized sums of triangle surfaces.
    }

    // Normalize the CDF values. 
    // PERF This means only the lightGUI.area integral value is in world space and the CDF could be reused for instanced mesh lights.
    for (auto& val : lightGUI.cdfAreas)
    {
      val /= areaSurface;
      // The last cdf element will automatically be 1.0f.
      // If this happens to be smaller due to inaccuracies in the floating point calculations, 
      // the clamping to valid triangle indices inside the sample_light_mesh() function will 
      // prevent out of bounds accesses, no need for corrections here.
      // (The corrections would be to set all identical values below 1.0f at the end of this array to 1.0f.)
    }

    lightGUI.area = areaSurface;
  }


} // namespace sg

