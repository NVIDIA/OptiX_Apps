/*
 * Copyright (c) 2013-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "inc/SceneGraph.h"

#include "shaders/vector_math.h"

namespace sg
{

  // DAR FIXME This is pretty redundant when the 3D texture of the SDF is assumes to be placed inside the object space unit cube.
  // It would be possible to put many SDF primitives into one GAS when their AABB placement would be different.
  // Leave this as it is and hardcode the intersection routine assuming the unit cube object space coordinates first.
  // DAR HACK Currently not loading a 3D texture. Implement an intersection shader with am SDF formula first.
  bool SignedDistanceField::createSDF(const float3 minimum,
                                      const float3 maximum,
                                      const float lipschitz,
                                      const unsigned int width, 
                                      const unsigned int height,
                                      const unsigned int depth,
                                      const std::string& filename)
  {
    m_attributes.clear();
    //m_indices.clear();

    SignedDistanceFieldAttributes attrib;

    // Unit cube AABB, plus some epsilon!
    // DANGER: When SDFs which are generating coplanar surface with the unit cube (like fBox(p, make_float3(1.0f))
    // there are rendering artifacts if the AABB is exactly the unit cube as well.
    // The sphere tracing is only done inside the interval between entry and exit points of the ray-AABB intersections.
    // That can miss the SDF and results in circular corruption artifacts on surfaces coplanar with the AABB unit cube.
    // To solve that, adjust the AABB extents outwards to make sure the SDF can always be hit.
    // This is likely to cost some performane, esp. when instancing a lot of SDF tighly together
    // instead of folding the SDF space in a single scaled AABB.

    //attrib.minimum = make_float3(-1.0f - offset);
    //attrib.maximum = make_float3( 1.0f + offset);
    
    // Do not use an offset when using the 3D texture!
    // The transformation from object into texture space assumes the 3D texture ends exactly on the AABB extents.
    attrib.minimum        = minimum;
    attrib.maximum        = maximum;
    attrib.sdfTexture     = 0; // Cannot set this to the CUDA texture object before it has been created on the device and is stored in m_mapTextures.
    attrib.sdfLipschitz   = lipschitz;
    attrib.sdfTextureSize = make_uint3(width, height, depth); // 3D texture size is needed for the central differencing.
    
    m_attributes.push_back(attrib);

    m_filename = filename; // Remember the filename of the SDF 3D texture to be able to look it up inside the m_mapPictures.

    return true; // DAR FIXME This will indicate later if the SDF data could be loaded.
  }

} // namespace sg
