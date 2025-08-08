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

#ifndef DEV_NODE_H
#define DEV_NODE_H

#include <vector>

// glm/gtx/component_wise.hpp doesn't compile when not setting GLM_ENABLE_EXPERIMENTAL.
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "cuda/vector_math.h"

// Just some namespace ("development") to distinguish from fastgltf::Node.
namespace dev
{

  // We need a shadow struct of the fastgltf::Node to track animations.
  struct Node
  {
    enum MorphMode
    {
      MORPH_NONE,
      MORPH_MESH_WEIGHTS,
      MORPH_NODE_WEIGHTS,
      MORPH_ANIMATED_WEIGHTS
    };

    glm::mat4 getMatrix() // This is the local transform relative to the parent.
    {
      if (isDirtyMatrix)
      {
        matrix = glm::translate(glm::mat4(1.0f), translation) * 
                 glm::toMat4(rotation) * 
                 glm::scale(glm::mat4(1.0f), scale);

        isDirtyMatrix = false;
      }
      return matrix;
    }

    bool isDirtyMatrix = true;  // true when any of the translation, rotation, scale has been changed since the last getMatrix() call.

    int indexSkin   = -1; // Index into m_skins when >= 0.
    int indexMesh   = -1; // Index into m_hostMeshes when >= 0.
    int indexCamera = -1; // Index into m_cameras when >= 0.
    int indexLight  = -1; // Index into m_lights when >= 0.

    MorphMode morphMode = MORPH_NONE;

    std::vector<float> weights; // Optional morph weights, initialized from mesh or node weights, potentially animated.

    glm::mat4 matrix;                         // Local transformation relative to the parent node.
    glm::mat4 matrixGlobal = glm::mat4(1.0f); // Global transform of this node, needed for skinning.

    glm::vec3 translation = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::quat rotation    = glm::quat::wxyz(1.0f, 0.0f, 0.0f, 0.0f); // w, x, y, z
    glm::vec3 scale       = glm::vec3(1.0f, 1.0f, 1.0f);
  };

} // namespace dev

#endif // DEV_NODE_H

