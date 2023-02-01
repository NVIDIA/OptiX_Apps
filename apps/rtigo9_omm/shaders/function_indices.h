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

#ifndef FUNCTION_INDICES_H
#define FUNCTION_INDICES_H

enum TypeRay
{
  TYPE_RAY_RADIANCE,
  TYPE_RAY_SHADOW,
  
  NUM_RAY_TYPES
};

enum TypeLens
{
  TYPE_LENS_PINHOLE,
  TYPE_LENS_FISHEYE,
  TYPE_LENS_SPHERE,

  NUM_LENS_TYPES
};

enum TypeLight
{
  TYPE_LIGHT_ENV_CONST,
  TYPE_LIGHT_ENV_SPHERE,
  TYPE_LIGHT_RECT,
  TYPE_LIGHT_MESH,
  TYPE_LIGHT_POINT,
  TYPE_LIGHT_SPOT,
  TYPE_LIGHT_IES,

  NUM_LIGHT_TYPES
};

enum TypeEDF
{
  TYPE_EDF,
  TYPE_EDF_DIFFUSE,
  TYPE_EDF_SPOT,
  TYPE_EDF_IES,

  NUM_EDF_TYPES
};

enum TypeBXDF
{
  TYPE_BXDF,
  TYPE_BRDF_DIFFUSE,
  TYPE_BRDF_SPECULAR,
  TYPE_BSDF_SPECULAR,
  TYPE_BRDF_GGX_SMITH,
  TYPE_BSDF_GGX_SMITH,

  NUM_BXDF_TYPES
};

#endif // FUNCTION_INDICES_H
