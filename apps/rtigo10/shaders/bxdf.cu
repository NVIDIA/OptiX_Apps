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

#include "config.h"

#include <optix.h>

#include "system_data.h"
#include "per_ray_data.h"
#include "vertex_attributes.h"
#include "function_indices.h"
#include "material_definition.h"
#include "light_definition.h"
#include "shader_common.h"
#include "transform.h"
#include "random_number_generators.h"


extern "C" __constant__ SystemData sysData;


extern "C" __global__ void __closesthit__bxdf()
{
  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  thePrd->distance = optixGetRayTmax(); // Return the current path segment distance, needed for absorption calculations in the integrator.
  
  //thePrd->pos = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
  thePrd->pos += thePrd->wi * thePrd->distance; // DEBUG Check which version is more efficient.

  thePrd->radiance   = make_float3(0.0f);
  thePrd->f_over_pdf = make_float3(0.0f);
  thePrd->pdf        = 0.0f;
  thePrd->flags     |= (FLAG_TERMINATE | FLAG_HIT);
}
