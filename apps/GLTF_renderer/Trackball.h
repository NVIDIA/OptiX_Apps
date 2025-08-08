//
// Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

namespace dev
{

  class Camera;

  class Trackball
  {
  public:

    enum ViewMode
    {
      EyeFixed,
      LookAtFixed
    };

    void  setCamera(Camera* camera);
    const Camera* getCamera() const;

    void setReferenceFrame(const glm::vec3& u, const glm::vec3& v, const glm::vec3& w);

    // Setting the gimbal lock to 'on' will fix the reference frame (i.e., the singularity of the trackball).
    // In most cases this is preferred.
    // For free scene exploration the gimbal lock can be turned off, which causes the trackball's reference frame
    // to be update on every camera update (adopted from the camera).
    void setGimbalLock(bool value);
    bool getGimbalLock() const;

    // Adopts the reference frame from the camera.
    // Note that the reference frame of the camera usually has a different 'up' than the 'up' of the camera.
    // Though, typically, it is desired that the trackball's reference frame aligns with the actual up of the camera.
    void reinitOrientationFromCamera();

    void setViewMode(ViewMode val);
    ViewMode getViewMode() const;

    void setSpeedRatio(float ratio);

    void startTracking(int x, int y);

    void orbit(int x, int y); // LMB
    void pan(int x, int y);   // MMB
    void dolly(int x, int y); // RMB
    void zoom(float dir);     // MWHEEL

  private:
    bool setDelta(int x, int y);

  private:
    bool m_gimbalLock = false;

    ViewMode m_viewMode = LookAtFixed;
    
    Camera* m_camera = nullptr;

    float m_speedRatio = 100.0f;

    float m_latitude  = 0.0f; // In radians.
    float m_longitude = 0.0f; // In radians.

    // Mouse tracking.
    // Base coordinates, set by startTracking();
    int m_baseX = 0;
    int m_baseY = 0;

    // Mouse movement deltas set be setDelta().
    float m_deltaX = 0.0f;
    float m_deltaY = 0.0f;

    // The Trackball reference coordinate system is right-handed with y-up.
    // The longitude/latitude polar coordinates get transformed into world space with this basis.
    // This gets overwritten by the first reinitOrientationFromCamera() call.
    glm::vec3 m_u = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 m_v = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 m_w = glm::vec3(0.0f, 0.0f, 1.0f);
  };

} // namespace dev
