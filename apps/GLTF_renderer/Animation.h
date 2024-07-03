/* 
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DEV_ANIMATION_H
#define DEV_ANIMATION_H

#include "DeviceBuffer.h"
#include "Node.h"

#include <vector>
#include <string>

// Just some namespace ("development") to distinguish from fastgltf::Animation.
namespace dev
{

  class AnimationSampler
  {
  public:
    enum TypeInterpolation
    {
      INTERPOLATION_LINEAR,
      INTERPOLATION_STEP,
      INTERPOLATION_CUBIC_SPLINE
    };

    AnimationSampler::AnimationSampler()
      : interpolation(AnimationSampler::INTERPOLATION_LINEAR)
      , timeMin(0.0f)
      , timeMax(1.0f)
      , components(0)
    {
    }

    // This is required because of the DeviceBuffer implementation needs move operators.
    // Move constructor from another AnimationSampler.
    AnimationSampler::AnimationSampler(AnimationSampler&& that) noexcept
    {
      operator=(std::move(that));
    }
    AnimationSampler& operator=(const AnimationSampler&) = delete;
    AnimationSampler& operator=(AnimationSampler&)       = delete;
    AnimationSampler& operator=(AnimationSampler&& that) = default;

  public:
    TypeInterpolation interpolation;

    float timeMin;  // Minimum and maximum time of the input values are
    float timeMax;  // strictly required by the glTF 2.0 specs.

    int components; // Number of components inside the output of the animation sampler. 
                    // 0 == invalid, 1 = float, 3 = float3 scale/translate, 4 = float4 quaternion xyzw

    HostBuffer input;  // Times. (glTF requires regular intervals.)
    HostBuffer output; // Animation values matching the input times. Components defines how this is interpreted, as float, float3, or float4.
  };

  struct AnimationChannel
  {
    enum TypePath
    {
      TRANSLATION,
      ROTATION,
      SCALE,
      WEIGHTS
    };

    TypePath path         = TRANSLATION;
    int      indexNode    = -1;
    int      indexSampler = 0;
  };

  class Animation
  {
  public:
    Animation::Animation()
      : isEnabled(false)
      , timeMin(FLT_MAX)
      , timeMax(-FLT_MAX)
    {
    }

    // This is required because the DeviceBuffer implementation needs move operators.
    // Move constructor from another Animation.
    Animation::Animation(Animation&& that) noexcept
    {
      operator=(std::move(that));
    }
    Animation& operator=(const Animation&) = delete;
    Animation& operator=(Animation&)       = delete;
    Animation& operator=(Animation&& that) = default;

    bool update(std::vector<dev::Node>& nodes, const float time);

  private:
    void interpolateTranslation(
      dev::Node& node,
      const glm::vec3* translations,
      const dev::AnimationSampler::TypeInterpolation interpolation,
      const size_t cell,
      const float t,
      const float delta,
      const bool exact);

    void interpolateRotation(
      dev::Node& node,
      const glm::vec4* rotations,
      const dev::AnimationSampler::TypeInterpolation interpolation,
      const size_t cell,
      const float t,
      const float delta,
      const bool exact);

    void interpolateScale(
      dev::Node& node,
      const glm::vec3* scales,
      const dev::AnimationSampler::TypeInterpolation interpolation,
      const size_t cell,
      const float t,
      const float delta,
      const bool exact);

    void interpolateWeight(
      dev::Node& node,
      const float* weights,
      const dev::AnimationSampler::TypeInterpolation interpolation,
      const size_t cell,
      const float t,
      const float delta,
      const bool exact);

  public:
    std::string name;

    bool isEnabled; // Initially false, toggled inside the GUI.

    float timeMin; // Minimum time of all samplers.
    float timeMax; // Maximum time of all samplers.
    
    std::vector<dev::AnimationSampler> samplers;
    std::vector<dev::AnimationChannel> channels;
  };

} // namespace dev

#endif // DEV_ANIMATION_H

