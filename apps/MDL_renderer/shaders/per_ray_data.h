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

#pragma once

#ifndef PER_RAY_DATA_H
#define PER_RAY_DATA_H

#include "config.h"

#include <optix.h>

// For the Bsdf_event_type.
#include <mi/neuraylib/target_code_types.h>

// Set when reaching a closesthit program.
#define FLAG_HIT           0x00000001
// Set when reaching the __anyhit__shadow program. Indicates visibility test failed.
#define FLAG_SHADOW        0x00000002
// Set if the material stack is not empty.
// This is implicit with the idxStack value.
//#define FLAG_VOLUME               0x00000010 
#define FLAG_VOLUME_SCATTERING      0x00000020
#define FLAG_VOLUME_SCATTERING_MISS 0x00000040

// Small 4 entries deep material stack.
#define MATERIAL_STACK_LAST 3
#define MATERIAL_STACK_SIZE 4

// This is the minimal size of the struct. float4 for vectorized access was slower due to more registers used.
struct MaterialStack
{
  float3 ior;     // index of refraction
  float3 sigma_a; // absorption coefficient
  float3 sigma_s; // scattering coefficient
  float  bias;    // directional bias
};


// Note that the fields are ordered by CUDA alignment restrictions.
struct PerRayData
{
  // 16-byte alignment
  
  // 8-byte alignment
  
  // 4-byte alignment
  float3 pos;         // Current surface hit point or volume sample point, in world space
  float  distance;    // Distance from the ray origin to the current position, in world space. Needed for absorption of nested materials.
  
  float3 wo;          // Outgoing direction, to observer, in world space.
  float3 wi;          // Incoming direction, to light, in world space.

  float3 radiance;    // Radiance along the current path segment.
  float  pdf;         // The last BSDF sample's pdf, tracked for multiple importance sampling.
  float3 throughput;  // The current path troughput. Starts white and gets modulated with bsdf_over_pdf with each sample.
  unsigned int flags; // Bitfield with flags. See FLAG_* defines above for its contents.

  float3 sigma_t;     // Extinction coefficient in a homogeneous medium.
  int    walk;        // Number of random walk steps done through scattering volume.
  float3 pdfVolume;   // Volume extinction sample pdf. Used to adjust the throughput along the random walk.

  mi::neuraylib::Bsdf_event_type eventType; // The type of events created by BSDF importance sampling.

  unsigned int seed;  // Random number generator input.
  
  // Small material stack tracking IOR, absorption ansd scattering coefficients of the entered materials. Entry 0 is vacuum.
  int           idxStack; 
  MaterialStack stack[MATERIAL_STACK_SIZE];
};


// Alias the PerRayData pointer and an uint2 for the payload split and merge operations. This generates only move instructions.
typedef union
{
  PerRayData* ptr;
  uint2       dat;
} Payload;

__forceinline__ __device__ uint2 splitPointer(PerRayData* ptr)
{
  Payload payload;

  payload.ptr = ptr;

  return payload.dat;
}

__forceinline__ __device__ PerRayData* mergePointer(unsigned int p0, unsigned int p1)
{
  Payload payload;

  payload.dat.x = p0;
  payload.dat.y = p1;

  return payload.ptr;
}

#endif // PER_RAY_DATA_H
