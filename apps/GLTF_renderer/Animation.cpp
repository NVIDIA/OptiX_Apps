/* 
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "Animation.h"

#include "MyAssert.h"

// Just some namespace ("development") to distinguish from fastgltf::Animation.
namespace dev
{
  // Animation

  bool Animation::update(std::vector<dev::Node>& nodes, const float time)
  {
    bool animated = false;

    if (!isEnabled)
    {
      return animated;
    }

    for (const dev::AnimationChannel& channel : channels)
    {
      dev::AnimationSampler& sampler = samplers[channel.indexSampler];

      MY_ASSERT(2 <= sampler.input.count); // The following algorithm assumes there are at least two input times. 

      const float* times = reinterpret_cast<const float*>(sampler.input.h_ptr);
      MY_ASSERT(times != nullptr);

      // The interpolation values.
      // t and timeDelta aren't used when exact == true.
      int   cell  = 0;     // Use the values times[cell], times[cell + 1] for the interpolation.
      float t     = 0.0f;  // The interpolant in range [0.0f, 1.0f] when exact == false.
      float delta = 1.0f;  // The cubic spline interpolation needs the delta time between the interpolated input keys.
      bool  exact = false; // Do not interpolate, use the output value from the cell directly.

      if (time <= sampler.timeMin) // If the global time is before the animation sampler timeMin, use the first entry.
      {
        //cell = 0;
        exact = true; // Not interpolated!
      }
      else if (sampler.timeMax <= time) // If the global time is after the animation sampler timeMax, use the last entry.
      {
        cell = static_cast<int>(sampler.input.count) - 1;
        exact = true; // Not interpolated!
      }
      else
      {
        // If the time is between two values of the sampler input values,
        // use the properly interpolated value of the output values.
        // FIXME PERF Don't search from the beginning all the time. Store the last cell and start there.
        float timeLo = times[0];
        for (size_t i = 0; i + 1 < sampler.input.count; ++i)
        {
          const float timeHi = times[i + 1];

          if (timeLo <= time && time <= timeHi)
          {
            cell = static_cast<int>(i); // Lower time interval start index.
            // glTF 2.0 specs: 
            // "When the current (requested) timestamp exists in the animation data, its associated property value MUST be used as-is, without interpolation."
            // (That means all interpolation modes inside the glTF 2.0 specs go through the actual output values.)
            exact = (time == timeLo || time == timeHi);
            if (exact)
            {
              if  (time == timeHi)
              {
                ++cell;
              }
            }
            else // timeLo < time < timeHi
            {
              // glTF 2.0 requires that timeLo < timeHi. No need to check if the denominator timeDelta > 0.0f.
              // (Also the intervals should be regular, so this woudn't need to be calculated for each interval.
              delta = timeHi - timeLo;
              t = (time - timeLo) / delta; // Interpolant t in range [0.0f, 1.0f]
            }
            break; // Found the input times interval containing the current time.
          }
          timeLo = timeHi;
        }
      }

      // No call the respective interpolation function.
      dev::Node& node = nodes[channel.indexNode];

      // Interpolant t and timeDelta is not used when exact == true. cell defines the output element to read.
      switch (channel.path)
      {
        case dev::AnimationChannel::TypePath::TRANSLATION:
        {
          const glm::vec3* translations = reinterpret_cast<const glm::vec3*>(sampler.output.h_ptr);
          interpolateTranslation(node, translations, sampler.interpolation, cell, t, delta, exact);
          break;
        }

        case dev::AnimationChannel::TypePath::ROTATION:
        {
          const glm::vec4* rotations = reinterpret_cast<const glm::vec4*>(sampler.output.h_ptr);
          interpolateRotation(node, rotations, sampler.interpolation, cell, t, delta, exact);
          break;
        }

        case dev::AnimationChannel::TypePath::SCALE:
        {
          const glm::vec3* scales = reinterpret_cast<const glm::vec3*>(sampler.output.h_ptr);
          interpolateScale(node, scales, sampler.interpolation, cell, t, delta, exact);
          break;
        }

        case dev::AnimationChannel::TypePath::WEIGHTS:
        {
          const float* weights = reinterpret_cast<const float*>(sampler.output.h_ptr);
          interpolateWeight(node, weights, sampler.interpolation, cell, t, delta, exact);
          break;
        }
      }

      animated = true;
    }
    return animated;
  }


  void Animation::interpolateTranslation(
    dev::Node& node,
    const glm::vec3* translations,
    const dev::AnimationSampler::TypeInterpolation interpolation,
    const size_t cell,
    const float t,
    const float delta,
    const bool exact)
  {
    switch (interpolation)
    {
      case dev::AnimationSampler::TypeInterpolation::INTERPOLATION_LINEAR:
        if (exact)
        {
          node.translation = translations[cell];
        }
        else
        {
          node.translation = glm::mix(translations[cell], translations[cell + 1], t);
        }
        node.isDirtyMatrix = true;
        break;

      case dev::AnimationSampler::TypeInterpolation::INTERPOLATION_STEP:
        node.translation = translations[cell];
        node.isDirtyMatrix = true;
        break;

      case dev::AnimationSampler::TypeInterpolation::INTERPOLATION_CUBIC_SPLINE:
        {
          const size_t k0 = cell * 3;
          if (exact)
          {
            node.translation = translations[k0 + 1]; // property value
          }
          else
          {
            // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#interpolation-cubic

            const size_t k1 = (cell + 1) * 3;

            //const glm::vec3 a0 = translations[k0];     // in-tangent
            const glm::vec3 v0 = translations[k0 + 1]; // property value
            const glm::vec3 b0 = translations[k0 + 2]; // out-tangent

            const glm::vec3 a1 = translations[k1];     // in-tangent
            const glm::vec3 v1 = translations[k1 + 1]; // property value
            //const glm::vec3 b1 = translations[k1 + 2]; // out-tangent

            const float t2 = t * t;
            const float t3 = t * t2;

            node.translation = ( 2.0f * t3 - 3.0f * t2 + 1.0f) * v0 + delta * (t3 - 2.0f * t2 + t) * b0 +
                               (-2.0f * t3 + 3.0f * t2       ) * v1 + delta * (t3 -        t2    ) * a1;
          }
          node.isDirtyMatrix = true;
          break;
        }
    }
  }

  void Animation::interpolateRotation(
    dev::Node& node,
    const glm::vec4* rotations,
    const dev::AnimationSampler::TypeInterpolation interpolation,
    const size_t cell,
    const float t,
    const float delta,
    const bool exact)
  {
    switch (interpolation)
    {
      case dev::AnimationSampler::TypeInterpolation::INTERPOLATION_LINEAR:
      {
        const glm::vec4 r0 = rotations[cell];
        const glm::quat q0 = glm::quat::wxyz(r0.w, r0.x, r0.y, r0.z);

        if (exact)
        {
          node.rotation = glm::normalize(q0);
        }
        else // interpolated
        {
          const glm::vec4 r1 = rotations[cell + 1];
          const glm::quat q1 = glm::quat::wxyz(r1.w, r1.x, r1.y, r1.z);

          node.rotation = glm::normalize(glm::slerp(q0, q1, t));
        }
        // FIXME PERF Only set this when the rotation actually changed. 
        // It won't for repeated calls outside the same side of the sampler [min, max] interval.
        node.isDirtyMatrix = true; 
        break;
      }

      case dev::AnimationSampler::TypeInterpolation::INTERPOLATION_STEP:
      {
        const glm::vec4 r = rotations[cell];
        node.rotation = glm::normalize(glm::quat::wxyz(r.w, r.x, r.y, r.z));
        node.isDirtyMatrix = true;
        break;
      }

      case dev::AnimationSampler::TypeInterpolation::INTERPOLATION_CUBIC_SPLINE:
        {
          const size_t k0 = cell * 3;
          if (exact)
          {
            const glm::vec4 v0 = rotations[k0 + 1]; // property value

            node.rotation = glm::normalize(glm::quat::wxyz(v0.w, v0.x, v0.y, v0.z));
          }
          else // interpolated
          {
            const size_t k1 = (cell + 1) * 3;

            //const glm::vec4 a0 = rotations[k0];     // in-tangent
            const glm::vec4 v0 = rotations[k0 + 1]; // property value
            const glm::vec4 b0 = rotations[k0 + 2]; // out-tangent

            const glm::vec4 a1 = rotations[k1];     // in-tangent
            const glm::vec4 v1 = rotations[k1 + 1]; // property value
            //const glm::vec4 b1 = rotations[k1 + 2]; // out-tangent
          
            const float t2 = t * t;
            const float t3 = t * t2;

            const glm::vec4 vt = ( 2.0f * t3 - 3.0f * t2 + 1.0f) * v0 + delta * (t3 - 2.0f * t2 + t) * b0 +
                                 (-2.0f * t3 + 3.0f * t2       ) * v1 + delta * (t3 -        t2    ) * a1;

            node.rotation = glm::normalize(glm::quat::wxyz(vt.w, vt.x, vt.y, vt.z));
          }
          node.isDirtyMatrix = true;
          break;
        }
    }
  }

  void Animation::interpolateScale(
    dev::Node& node,
    const glm::vec3* scales,
    const dev::AnimationSampler::TypeInterpolation interpolation,
    const size_t cell,
    const float t,
    const float delta,
    const bool exact)
  {
    switch (interpolation)
    {
      case dev::AnimationSampler::TypeInterpolation::INTERPOLATION_LINEAR:
        if (exact)
        {
          node.scale = scales[cell];
        }
        else
        {
          node.scale = glm::mix(scales[cell], scales[cell + 1], t);
        }
        node.isDirtyMatrix = true;
        break;

      case dev::AnimationSampler::TypeInterpolation::INTERPOLATION_STEP:
        node.scale = scales[cell];
        node.isDirtyMatrix = true;
        break;

      case dev::AnimationSampler::TypeInterpolation::INTERPOLATION_CUBIC_SPLINE:
        {
          const size_t k0 = cell * 3;
          if (exact)
          {
            node.scale = scales[k0 + 1]; // property value
          }
          else
          {
            const size_t k1 = (cell + 1) * 3;

            //const glm::vec3 a0 = scales[k0];     // in-tangent
            const glm::vec3 v0 = scales[k0 + 1]; // property value
            const glm::vec3 b0 = scales[k0 + 2]; // out-tangent

            const glm::vec3 a1 = scales[k1];     // in-tangent
            const glm::vec3 v1 = scales[k1 + 1]; // property value
            //const glm::vec3 b1 = scales[k1 + 2]; // out-tangent

            const float t2 = t * t;
            const float t3 = t * t2;

            node.scale = ( 2.0f * t3 - 3.0f * t2 + 1.0f) * v0 + delta * (t3 - 2.0f * t2 + t) * b0 +
                         (-2.0f * t3 + 3.0f * t2       ) * v1 + delta * (t3 -        t2    ) * a1;
          }
          node.isDirtyMatrix = true;
          break;
        }
    }
  }

  void Animation::interpolateWeight(
    dev::Node& node,
    const float* weights,
    const dev::AnimationSampler::TypeInterpolation interpolation,
    const size_t cell,
    const float t,
    const float delta,
    const bool exact)
  {
    // Each cell is morph target many entries in the scalar weight array.
    const size_t stride = node.weights.size(); 
    const size_t base   = cell * stride;

    switch (interpolation)
    {
      case dev::AnimationSampler::TypeInterpolation::INTERPOLATION_LINEAR:
        if (exact)
        {
          for (size_t i = 0; i < stride; ++i)
          {
            node.weights[i] = weights[base + i];
          }
        }
        else
        {
          for (size_t i = 0; i < stride; ++i)
          {
            node.weights[i] = glm::mix(weights[base + i], weights[base + stride + i], t);
          }
        }
        break;

      case dev::AnimationSampler::TypeInterpolation::INTERPOLATION_STEP:
        for (size_t i = 0; i < stride; ++i)
        {
          node.weights[i] = weights[base + i];
        }
        break;

      case dev::AnimationSampler::TypeInterpolation::INTERPOLATION_CUBIC_SPLINE:
        {
          const size_t k0 = cell * 3;

          if (exact)
          {
            for (size_t i = 0; i < stride; ++i)
            {
              node.weights[i] = weights[k0 * stride + i];
            }
          }
          else
          {

            for (size_t i = 0; i < stride; ++i)
            {
              const size_t k1 = (cell + 1) * 3;

              //const float a0 = weights[(k0 * stride) + i];     // in-tangent
              const float v0 = weights[(k0 + 1) * stride + i]; // property value
              const float b0 = weights[(k0 + 2) * stride + i]; // out-tangent

              const float a1 = weights[(k1 * stride) + i];     // in-tangent
              const float v1 = weights[(k1 + 1) * stride + i]; // property value
              //const float b1 = weights[(k1 + 2) * stride + i]; // out-tangent

              const float t2 = t * t;
              const float t3 = t * t2;

              node.weights[i] = 
                ( 2.0f * t3 - 3.0f * t2 + 1.0f) * v0 + delta * (t3 - 2.0f * t2 + t) * b0 +
                (-2.0f * t3 + 3.0f * t2       ) * v1 + delta * (t3 -        t2    ) * a1;
            }
          }
          break;
        }
    }
  }


} // namespace dev


