//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "Camera.h"
#include "MyAssert.h"

#include <cuda/vector_math.h>

namespace dev
{

  Camera::Camera()
    : m_position(0.0f, 0.0f, 1.0f)
    , m_lookat(0.0f, 0.0, 0.0f)
    , m_up(0.0f, 1.0f, 0.0f)
    , m_fovY(45.0f)
    , m_aspectRatio(1.0f)
    , m_magnification(1.0f, 1.0f)
    , m_isDirty(true)
  {
  }

  float Camera::getDistance() const
  {
    return length(m_lookat - m_position);
  }

  void Camera::setDirection(const glm::vec3& dir)
  {
    m_lookat = m_position + length(m_lookat - m_position) * dir;
    m_isDirty = true;
  }

  glm::vec3 Camera::getDirection() const
  {
    MY_ASSERT(m_position != m_lookat); // This Camera implementation assumes that m_position != m_lookat at all times.
    return normalize(m_lookat - m_position);
  }

  glm::vec3 Camera::getRight() const
  {
    MY_ASSERT(m_position != m_lookat);
    const auto dir = getDirection();
    const auto up  = getUp();
    return normalize(cross(dir,up));
  }

  void Camera::setPosition(const glm::vec3& val)
  {
    m_position = val;
    m_isDirty = true;
  }

  glm::vec3 Camera::getPosition() const
  {
    return m_position;
  }

  void Camera::setLookat(const glm::vec3& val)
  {
    m_lookat = val;
    m_isDirty = true;
  }

  glm::vec3 Camera::getLookat() const
  {
    return m_lookat;
  }

  void Camera::setUp(const glm::vec3& val)
  {
    m_up = val;
    m_isDirty = true;
  }

  glm::vec3 Camera::getUp() const
  {
    return m_up;
  }

  void Camera::setFovY(const float val)
  {
    m_fovY = val; // m_fovY <= 0.0f defines an orthographic camera.
    m_isDirty = true;
  }

  float Camera::getFovY() const
  {
    return m_fovY;
  }

  void Camera::setAspectRatio(const float val)
  {
    MY_ASSERT(0.0f < val);
    m_aspectRatio = val;
    m_isDirty = true;
  }

  float Camera::getAspectRatio() const
  {
    return m_aspectRatio;
  }

  void Camera::setMagnification(const glm::vec2 mag)
  {
    MY_ASSERT(mag.x != 0.0f && mag.y != 0.0f);
    m_magnification = mag;
    m_isDirty = true;
  }

  glm::vec2 Camera::getMagnification() const
  {
    return m_magnification;
  }

  // UVW forms a left-handed orthogonal, but not orthonormal basis!
  void Camera::getUVW(glm::vec3& U, glm::vec3& V, glm::vec3& W) const
  {
    if (0.0f < m_fovY) // perspective camera
    {
      W = m_lookat - m_position; // Do not normalize W, it implies focal length.

      U = normalize(cross(W, m_up)); // W and up-vector are never collinear! The camera manipulator prevents that.
      V = normalize(cross(U, W));

      const float wlen = length(W);
      const float vlen = wlen * tanf(0.5f * m_fovY * M_PIf / 180.0f);
      const float ulen = vlen * m_aspectRatio;

      U *= ulen;
      V *= vlen;
    }
    else // orthographic camera
    {
      W = normalize(m_lookat - m_position);
      U = normalize(cross(W, m_up)); // W and up-vector are never collinear! The camera manipulator prevents that.
      V = normalize(cross(U, W));

      U *= m_magnification.x * m_aspectRatio;
      V *= m_magnification.y;
    }
  }

  bool Camera::getIsDirty() const
  {
    return m_isDirty;
  }

  void Camera::setIsDirty(const bool dirty)
  {
    m_isDirty = dirty;
  }
} // namespace dev
