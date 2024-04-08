/* 
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef LIGHT_GUI_H
#define LIGHT_GUI_H

#include <cuda.h>

#include <dp/math/Matmnt.h>

#include "shaders/function_indices.h"

#include <string>

// Host side GUI light parameters, only stores the values not held inside the material already.
struct LightGUI
{
  TypeLight typeLight; // Zero-based light type to select the light sampling and evaluation.

  dp::math::Mat44f matrix;         // object to world
  dp::math::Mat44f matrixInv;      // world to object
  dp::math::Quatf  orientation;    // object to world, rotation only
  dp::math::Quatf  orientationInv; // world to object, rotation only

  unsigned int idGeometry; // Geometry data index for mesh lights. (Supports GAS sharing!)
  unsigned int idMaterial; // The MaterialGUI index which holds the light parameters

  std::vector<float> cdfAreas;      // CDF over the areas of the mesh triangles used for uniform sampling.
  float              area;          // Overall surface area of the mesh light in world space.
};

#endif // LIGHT_GUI_H
