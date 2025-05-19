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

#pragma once

#include <glm/vec3.hpp>

namespace dev {

  /// @brief Min and Max 3D points of a set of points, with their properties.
  class SceneExtent
  {
  public:

    SceneExtent()
    {
      toInvalid();
    }

    void toInvalid()
    {
      aabb[0] = glm::vec3(Invalid);
      aabb[1] = glm::vec3(-Invalid);
    }

    // set to cube with side length 1
    void toUnity()
    {
      aabb[0] = glm::vec3(-0.5f);
      aabb[1] = glm::vec3(0.5f);
    }

    // data = { pMin, pMax }
    void set(glm::vec3 const* data)
    {
      aabb[0] = data[0];
      aabb[1] = data[1];
    }

    void update(float x, float y, float z)
    {
      aabb[0].x = std::min(aabb[0].x, x);
      aabb[0].y = std::min(aabb[0].y, y);
      aabb[0].z = std::min(aabb[0].z, z);
      aabb[1].x = std::max(aabb[1].x, x);
      aabb[1].y = std::max(aabb[1].y, y);
      aabb[1].z = std::max(aabb[1].z, z);
    }

    // make the extent valid and not empty
    void fixSize()
    {
      const glm::vec3 c = (Invalid == aabb[0].x) ? glm::vec3(0, 0, 0) : getCenter();
      const float eps = 0.5f;
      aabb[0] = c - glm::vec3(eps);
      aabb[1] = c + glm::vec3(eps);
    }

    // Diameter of the bounding sphere.
    // Will be zero if only one point was added!
    float getDiameter() const
    {
      return length(aabb[1] - aabb[0]);
    }

    // The longest dimension (width or height or depth).
    // Will be zero if only one point was added!
    float getMaxDimension() const
    {
      const glm::vec3 d = aabb[1] - aabb[0];
      return fmaxf(fmaxf(d.x, d.y), d.z);
    }

    glm::vec3 getCenter() const
    {
      return (aabb[0] + aabb[1]) * 0.5f;
    }

    // true if update() was called at least once and volume > 0
    bool isValid() const
    {
      return (aabb[0].x == Invalid) ? false : (length(aabb[1] - aabb[0]) > 0.0f);
    }

    void print() const
    {
      std::cout << "Scene extent "
        << aabb[0].x << " " << aabb[0].y << " " << aabb[0].z << " -- "
        << aabb[1].x << " " << aabb[1].y << " " << aabb[1].z << std::endl;
    }

  private:

    const float Invalid = 1e37f;
    glm::vec3   aabb[2]; // min, max
  };
} // namespace dev