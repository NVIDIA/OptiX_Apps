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

#ifndef DEV_SKIN_H
#define DEV_SKIN_H

#include <string>
#include <vector>

// glm/gtx/component_wise.hpp doesn't compile when not setting GLM_ENABLE_EXPERIMENTAL.
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
//#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
//
//#include "cuda/vector_math.h"

#include "DeviceBuffer.h"

// Just some namespace ("development") to distinguish from fastgltf::Light.
namespace dev
{

  // We need a shadow struct of the fastgltf::Node to track animations.
  class Skin
  {
  public:
    Skin()
      : skeleton(-1)
    {
    }
    // This is required because the DeviceBuffer implementation needs move operators.
    // Move constructor from another Skin
    Skin(Skin&& that) noexcept
    {
      operator=(std::move(that));
    }
    Skin& operator=(const Skin&) = delete;
    Skin& operator=(Skin&)       = delete;
    Skin& operator=(Skin&& that) = default;

  public:
    std::string         name;
    int                 skeleton = -1; // -1 when none given.
    std::vector<size_t> joints;
    HostBuffer          inverseBindMatrices;
    // Derived data:
    std::vector<glm::mat4> matrices;
    std::vector<glm::mat4> matricesIT;
  };

} // namespace dev

#endif // DEV_SKIN_H

