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

#ifndef SCENEGRAPH_H
#define SCENEGRAPH_H

// For the vector types.
#include <cuda_runtime.h>

#include "shaders/curve_attributes.h"
#include "shaders/vertex_attributes.h"
#include "shaders/vector_math.h"

#include "inc/LightGUI.h"

#include <memory>
#include <vector>

namespace sg
{

  enum NodeType
  {
    NT_GROUP,
    NT_INSTANCE,
    NT_TRIANGLES,
    NT_CURVES
  };

  class Node
  {
  public:
    Node(const unsigned int id);
    //~Node();

    virtual sg::NodeType getType() const = 0;
    
    unsigned int getId() const
    {
      return m_id;
    }

  private:
    unsigned int m_id;
  };


  class Triangles : public Node
  {
  public:
    Triangles(const unsigned int id);
    //~Triangles();

    sg::NodeType getType() const;

    void createBox();
    void createPlane(const unsigned int tessU, const unsigned int tessV, const unsigned int upAxis);
    void createSphere(const unsigned int tessU, const unsigned int tessV, const float radius, const float maxTheta);
    void createTorus(const unsigned int tessU, const unsigned int tessV, const float innerRadius, const float outerRadius);

    void calculateLightArea(LightGUI& lightGUI) const;

    void setAttributes(const std::vector<TriangleAttributes>& attributes);
    const std::vector<TriangleAttributes>& getAttributes() const;
    
    void setIndices(const std::vector<unsigned int>& indices);
    const std::vector<unsigned int>& getIndices() const;

  private:
    std::vector<TriangleAttributes> m_attributes;
    std::vector<unsigned int>       m_indices; // If m_indices.size() == 0, m_attributes are independent primitives (not actually supported in this renderer implementation!)
  };

  class Curves : public Node
  {
  public:
    Curves(const unsigned int id);
    //~Curves();

    sg::NodeType getType() const;

    bool createHair(std::string const& filename, const float scale);

    void setAttributes(std::vector<CurveAttributes> const& attributes);
    std::vector<CurveAttributes> const& getAttributes() const; 
    
    void setIndices(std::vector<unsigned int> const&);
    std::vector<unsigned int> const& getIndices() const;

  private:
    std::vector<CurveAttributes> m_attributes;
    std::vector<unsigned int>    m_indices;
  };


  class Instance : public Node
  {
  public:
    Instance(const unsigned int id);
    //~Instance();

    sg::NodeType getType() const;

    void setTransform(const float m[12]);
    const float* getTransform() const;
    
    void setChild(std::shared_ptr<sg::Node> node);
    std::shared_ptr<sg::Node> getChild();

    void setMaterial(const int index);
    int  getMaterial() const;

    void setLight(const int index);
    int  getLight() const;

  private:
    int                       m_material;
    int                       m_light;
    float                     m_matrix[12];
    std::shared_ptr<sg::Node> m_child; // An Instance can either hold a Group or Triangles as child.
  };

  class Group : public Node
  {
  public:
    Group(const unsigned int id);
    //~Group();

    sg::NodeType getType() const;

    void addChild(std::shared_ptr<sg::Instance> instance); // Groups can only hold Instances.

    size_t getNumChildren() const;
    std::shared_ptr<sg::Instance> getChild(size_t index);

  private:
    std::vector< std::shared_ptr<sg::Instance> > m_children;
  };

} // namespace sg

#endif // SCENEGRAPH_H
