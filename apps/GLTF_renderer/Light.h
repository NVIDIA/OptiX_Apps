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

#pragma once

#ifndef DEV_LIGHT_H
#define DEV_LIGHT_H

// Always include this before any OptiX headers!
//#include <cuda_runtime.h>
//#include <optix.h>

#include <vector>

// glm/gtx/component_wise.hpp doesn't compile when not setting GLM_ENABLE_EXPERIMENTAL.
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "cuda/vector_math.h"


// Just some namespace ("development") to distinguish from fastgltf::Light.
namespace dev
{

  struct Light
  {
    std::string name;

    // 0 = Point, 1 = Spot, 2 = Directional. Intentionally not matching fastgltf::LightType!
    int type = 0;
    // point, spot: Units are luminous intensity (candela), lumens per square radian (lm/sr)
    // directional: Units are illuminance (lux), lumens per square meter (lm/m^2) 
    float3 color = { 1.0f, 1.0f, 1.0f };
    float  intensity = 1.0f;

    // Point and spot lights can have a range (distance) at which they do not contribute anything.
    float range = RT_DEFAULT_MAX;

    // Spot light cone definition: 
    // Angle in radians from centre of spotlight where falloff begins.
    // Must be greater than or equal to 0 and less than outerConeAngle.
    float innerConeAngle = 0.0f;
    // Angle in radians from centre of spotlight where falloff ends.
    // Must be greater than innerConeAngle and less than or equal to PI / 2.0.
    float outerConeAngle = 0.25f * M_PIf; // Default: PI / 4.0

    glm::mat4 matrix;
  };

} // namespace dev

#endif // DEV_LIGHT_H

