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

#pragma once

// glm/gtx/component_wise.hpp doesn't compile when not setting GLM_ENABLE_EXPERIMENTAL.
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

namespace dev
{

  class Camera
  {
  public:
    
    Camera();

    float getDistance() const;

    void setDirection(const glm::vec3& dir);

    // normalized
    glm::vec3 getDirection() const;
    
    // normalized
    glm::vec3 getRight() const;

    void setPosition(const glm::vec3& val);
    glm::vec3 getPosition() const;
    
    void setLookat(const glm::vec3& val);
    glm::vec3 getLookat() const;
    
    void setUp(const glm::vec3& val);
    glm::vec3 getUp() const;

    void setFovY(const float val);
    float getFovY() const;

    void setAspectRatio(const float val);
    float getAspectRatio() const;

    void setMagnification(const glm::vec2 val);
    glm::vec2 getMagnification() const;

    // UVW forms a left handed orthogonal, but not orthonormal basis!
    void getUVW(glm::vec3& U, glm::vec3& V, glm::vec3& W) const;

    bool getIsDirty() const;
    void setIsDirty(const bool dirty);

  private:
    glm::vec3 m_position;
    glm::vec3 m_lookat;
    glm::vec3 m_up;
    float     m_fovY;          // In degrees. m_fovY <= 0.0f defines an orthographic cameras. zoom() changes this.
    float     m_aspectRatio;   // Only used inside the perspective camera.
                               // The aspect ratio is implicit with the magnifiers in the orthographic camera.
    glm::vec2 m_magnification; // The orthographic camera defines an x and y scaling factor. zoom() changes this.
    bool      m_isDirty;
  };

} // namespace dev
