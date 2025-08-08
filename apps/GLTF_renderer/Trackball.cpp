//
// Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
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
#include "Trackball.h"

#include <cuda/vector_math.h>

#include <cmath>
#include <algorithm>

namespace dev
{

  namespace
  {
    float radians(float degrees)
    {
      return degrees * M_PIf / 180.0f;
    }
    float degrees(float radians)
    {
      return radians * M_1_PIf * 180.0f;
    }
  } // namespace

  void Trackball::setCamera(Camera* camera)
  {
    m_camera = camera;
    m_camera->setIsDirty(true);
    reinitOrientationFromCamera();
  }

  const Camera* Trackball::getCamera() const
  {
    return m_camera;
  }

  void Trackball::setReferenceFrame(const glm::vec3& u, const glm::vec3& v, const glm::vec3& w)
  {
    if (m_camera == nullptr)
    {
      return;
    }

    m_u = u;
    m_v = v;
    m_w = w;
    
    glm::vec3 dirWS = -m_camera->getDirection();
    glm::vec3 dirLocal;
    
    dirLocal.x = dot(dirWS, u);
    dirLocal.y = dot(dirWS, v);
    dirLocal.z = dot(dirWS, w);
    
    m_longitude = atan2f(dirLocal.x, dirLocal.z);
    m_latitude  = asinf(dirLocal.y);
  }

  // Setting the gimbal lock to 'on' will fix the reference frame (i.e., the singularity of the trackball).
  // In most cases this is preferred.
  // For free scene exploration the gimbal lock can be turned off, which causes the trackball's reference frame
  // to be update on every camera update (adopted from the camera).
  void Trackball::setGimbalLock(bool val)
  {
    m_gimbalLock = val;
  }

  bool Trackball::getGimbalLock() const
  {
    return m_gimbalLock;
  }

  void Trackball::setSpeedRatio(float ratio)
  {
    m_speedRatio = ratio;
    if (m_speedRatio < 0.01f)
    {
      m_speedRatio = 0.01f;
    }
    else if (10000.0f < m_speedRatio)
    {
      m_speedRatio = 10000.0f;
    }
  }

  void Trackball::startTracking(int x, int y)
  {
    m_baseX = x;
    m_baseY = y;
  }

  bool Trackball::setDelta(int x, int y)
  {
    if (m_baseX != x || m_baseY != y) 
    {
      m_deltaX = float(x - m_baseX);
      m_deltaY = float(y - m_baseY);

      m_baseX = x;
      m_baseY = y;

      return true; // There was a delta.
    }
    return false;
  }

  void Trackball::orbit(int x, int y)
  {
    if (m_camera == nullptr)
    {
      return;
    }

    if (setDelta(x, y))
    {
      // This is stopping a +-89 degrees to prevent that the view direction becomes collinear with the up-vector, see below.
      m_longitude = radians(fmodf(degrees(m_longitude) - 0.5f * m_deltaX, 360.0f));
      m_latitude  = radians(std::min(89.0f, std::max(-89.0f, degrees(m_latitude) + 0.5f * m_deltaY)));

      // Use latitude-longitude for view definition
      // This is using a right-handed coodinate system with y-up!
      glm::vec3 localDir = glm::vec3(cosf(m_latitude) * sinf(m_longitude), 
                                     sinf(m_latitude),
                                     cosf(m_latitude) * cosf(m_longitude));
    
      glm::vec3 dirWS = m_u * localDir.x + 
                        m_v * localDir.y +
                        m_w * localDir.z;

      float distance = m_camera->getDistance();

      if (m_viewMode == EyeFixed)
      {
        glm::vec3 pos = m_camera->getPosition();

        m_camera->setLookat(pos - dirWS * distance);
      }
      else // if (m_viewMode == LookAtFixed)
      {
        glm::vec3 lookat = m_camera->getLookat();

        m_camera->setPosition(lookat + dirWS * distance);
      }

      if (!m_gimbalLock)
      {
        // This is only working for incremental steps, not exactly 90 degrees up- or downward rotations,
        // which are prevented by clamping to +-89 degrees above.
        // Otherwise the view direction could become collinear with the up-vector and 
        // the cross products inside this function will return NAN and INF values
        // which results in invalid ray exceptions.
        reinitOrientationFromCamera();
      
        m_camera->setUp(m_v);
      }
    }
  }

  void Trackball::reinitOrientationFromCamera()
  {
    if (m_camera == nullptr)
    {
      return;
    }

    m_camera->getUVW(m_u, m_v, m_w);

    // The Camera UVW system is left-handed with y-up!
    // Negate W because the Trackball reference coordinate system is right-handed with y-up.
    m_u = normalize(m_u);
    m_v = normalize(m_v);
    m_w = normalize(-m_w); 
    
    m_longitude = 0.0f;
    m_latitude  = 0.0f;
  }

  Trackball::ViewMode Trackball::getViewMode() const
  {
    return m_viewMode;
  }
    
  void Trackball::setViewMode(Trackball::ViewMode val)
  {
    m_viewMode = val;
  }

  // FIXME Implemnent.
  //void Trackball::rollLeft(float speed)
  //{
  //  glm::vec3 u, v, w;

  //  m_camera->getUVW(u, v, w);
  //  u = normalize(u);
  //  v = normalize(v);

  //  m_camera->setUp(u * cosf(radians(90.0f + speed)) + v * sinf(radians(90.0f + speed)));
  //}

  //void Trackball::rollRight(float speed)
  //{
  //  glm::vec3 u, v, w;

  //  m_camera->getUVW(u, v, w);

  //  u = normalize(u);
  //  v = normalize(v);

  //  m_camera->setUp(u * cosf(radians(90.0f - speed)) + v * sinf(radians(90.0f - speed)));
  //}
  
  void Trackball::dolly(int x, int y)
  {
    if (m_camera == nullptr)
    {
      return;
    }

    if (setDelta(x, y))
    {
      float distance = m_camera->getDistance();

      distance -= distance * m_deltaY / m_speedRatio;
      
      if (distance < 0.001f) // Avoid swapping sides.
      {
        distance = 0.001f;
      }
   
      glm::vec3 lookat    = m_camera->getLookat();
      glm::vec3 direction = m_camera->getDirection();

      m_camera->setPosition(lookat - direction * distance);
    }
  }

  void Trackball::pan(int x, int y)
  {
    if (m_camera == nullptr)
    {
      return;
    }

    if (setDelta(x, y))
    {
      // m_speedRatio pixels will move one vector length.
      float u = m_deltaX / m_speedRatio;
      float v = m_deltaY / m_speedRatio;

      glm::vec3 U;
      glm::vec3 V;
      glm::vec3 W;
      
      m_camera->getUVW(U, V, W);

      glm::vec3 pan = -u * U + v * V;

      glm::vec3 pos    = m_camera->getPosition();
      glm::vec3 lookat = m_camera->getLookat();

      m_camera->setPosition(pos + pan);
      m_camera->setLookat(lookat + pan);
    }
  }


  void Trackball::zoom(float direction)
  {
    if (m_camera == nullptr)
    {
      return;
    }

    float fov = m_camera->getFovY();

    if (0.0f < fov) // Perspective camera.
    {
      fov += direction;

      if (fov < 1.0f)
      {
        fov = 1.0f;
      }
      else if (179.0 < fov)
      {
        fov = 179.0f;
      }

      m_camera->setFovY(fov);
    }
    else // if orthographic camera.
    {
      glm::vec2 mag = m_camera->getMagnification();
      
      mag += mag * (direction * 0.1f); // FIXME Magic number.

      if (0.001f < mag.x && 0.001f < mag.y)
      {
        m_camera->setMagnification(mag);
      }
    }
  }

} // namespace dev
