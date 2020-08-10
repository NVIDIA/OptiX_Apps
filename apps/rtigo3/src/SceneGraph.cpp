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

  void Triangles::setAttributes(std::vector<TriangleAttributes> const& attributes)
  {
    m_attributes.resize(attributes.size());
    memcpy(m_attributes.data(), attributes.data(), sizeof(TriangleAttributes) * attributes.size());
  }

  std::vector<TriangleAttributes> const& Triangles::getAttributes() const
  {
    return m_attributes;
  }

  void Triangles::setIndices(std::vector<unsigned int> const& indices)
  {
    m_indices.resize(indices.size());
    memcpy(m_indices.data(), indices.data(), sizeof(unsigned int) * indices.size());
  }
  
  std::vector<unsigned int> const& Triangles::getIndices() const
  {
    return m_indices;
  }

} // namespace sg

