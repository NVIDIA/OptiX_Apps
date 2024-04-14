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

#include "config.h"

#include <optix.h>

#include "system_data.h"

#include "per_ray_data.h"
#include "random_number_generators.h"
#include "shader_common.h"
#include "transform.h"

extern "C" __constant__ SystemData sysData;


// Note that all light sampling routines return lightSample.direction and lightSample.distance in world space!

extern "C" __device__ LightSample __direct_callable__light_env_constant(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;

  const float2 sample = rng2(prd->seed);

  unitSquareToSphere(sample.x, sample.y, lightSample.direction, lightSample.pdf);
  
  // The emission is constant in all directions.
  // There is no transformation of the object space direction into world space necessary.

  lightSample.distance = RT_DEFAULT_MAX; // Environment light.
  
  lightSample.radiance_over_pdf = light.emission / lightSample.pdf;

  return lightSample;
}


extern "C" __device__ LightSample __direct_callable__light_env_sphere(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;

  lightSample.pdf = 0.0f;

  // Importance-sample the spherical environment light direction in object space.
  // FIXME The binary searches are generating a lot of memory traffic. Replace this with an alias-map lookup.
  const float2 sample = rng2(prd->seed);

  // Note that the marginal CDF is one bigger than the texture height. As index this is the 1.0f at the end of the CDF.
  const float* cdfV = reinterpret_cast<const float*>(light.cdfV);
  const unsigned int idxV = binarySearchCDF(cdfV, light.height, sample.y);

  const float* cdfU = reinterpret_cast<const float*>(light.cdfU);
  cdfU += (light.width + 1) * idxV; // Horizontal CDF is one bigger than the texture width!
  const unsigned int idxU = binarySearchCDF(cdfU, light.width, sample.x);

  // Continuous sampling of the CDF.
  const float cdfLowerU = cdfU[idxU];
  const float cdfUpperU = cdfU[idxU + 1];
  const float du = (sample.x - cdfLowerU) / (cdfUpperU - cdfLowerU);
  const float u = (float(idxU) + du) / float(light.width);

  const float cdfLowerV = cdfV[idxV];
  const float cdfUpperV = cdfV[idxV + 1];
  const float dv = (sample.y - cdfLowerV) / (cdfUpperV - cdfLowerV);
  const float v = (float(idxV) + dv) / float(light.height);

  // Light sample direction vector in object space polar coordinates.
  const float phi   = u * M_PIf * 2.0f;
  const float theta = v * M_PIf; // theta == 0.0f is south pole, theta == M_PIf is north pole.

  const float sinTheta = sinf(theta);

  // All lights shine down the positive z-axis in this renderer.
  // Orient the 2D texture map so that the center (u, v) = (0.5, 0.5) lies exactly on the positive z-axis.
  // Means the seam from u == 1.0 -> 0.0 lies on the negative z-axis and the u range [0.0, 1.0]
  // goes clockwise on the xz-plane when looking from the positive y-axis.
  const float3 dir = make_float3( sinf(phi) * sinTheta,  // Starting on negative z-axis going around clockwise (to positive x-axis).
                                 -cosf(theta),           // From south pole to north pole.
                                 -cosf(phi) * sinTheta); // Starting on negative z-axis.
  
  // Now rotate that normalized object space direction into world space. 
  lightSample.direction = transformVector(light.ori, dir);

  lightSample.distance = RT_DEFAULT_MAX; // Environment light.
  
  // Get the emission from the spherical environment texture.
  const float3 emission = make_float3(tex2D<float4>(light.textureEmission, u, v));
  
  // For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
  // and not the Gaussian-smoothed one used to actually generate the CDFs and uniform sampling in the texel.
  // (Note that this does not contain the light.emission which just modulates the texture.)
  lightSample.pdf = intensity(emission) * light.invIntegral;

  if (DENOMINATOR_EPSILON < lightSample.pdf)
  {
    lightSample.radiance_over_pdf = light.emission * emission / lightSample.pdf;
  }

  return lightSample;
}


extern "C" __device__ LightSample __direct_callable__light_point(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;

  lightSample.pdf = 0.0f; // Default return, invalid light sample (backface, edge on, or too near to the surface)
   
  // Get the world space position from the object to world matrix translation.
  const float3 position = make_float3(light.matrix[0].w, light.matrix[1].w, light.matrix[2].w);

  lightSample.direction = position - prd->pos; // Sample direction from surface point to light sample position.
  
  const float distanceSquared = dot(lightSample.direction, lightSample.direction);

  if (DENOMINATOR_EPSILON < distanceSquared)
  {
    lightSample.distance   = sqrtf(distanceSquared);
    lightSample.direction *= 1.0f / lightSample.distance; // Normalized direction to light.

    // Hardcoded singular lights are defined in visible radiance directly, don't normalize by 0.25f * M_1_PIf.
    float3 emission = light.emission * (1.0f / distanceSquared); // Quadratic attenuation.

    // The emission texture is used as spherical projection around the point light similar to spherical environment lights.
    // By design all lights in this renderer shine down the light's local positive z-Axis, which is the "normal" direction for rect and mesh lights.
    if (light.textureEmission)
    {
      // Transform the direction from light to surface from world space into light object space.
      const float3 R = transformVector(light.oriInv, -lightSample.direction);

      // All lights shine down the positive z-axis in this renderer.
      // Means the spherical texture coordinate seam u == 0.0 == 1.0 is on the negative z-axis direction now.
      const float u = (atan2f(-R.x, R.z) + M_PIf) * 0.5f * M_1_PIf;
      const float v = acosf(-R.y) * M_1_PIf; // Texture is origin at lower left, v == 0.0f is south pole.

      // Modulate the base emission with the emission texture.
      emission *= make_float3(tex2D<float4>(light.textureEmission, u, v));
    }

    lightSample.radiance_over_pdf = emission; // pdf == 1.0f for singular light.

    // Indicate valid light sample (pdf != 0.0f).
    // This value is otherwise unused in a singular light. 
    // The PDF is a Dirac with infinite value for this case.
    lightSample.pdf = 1.0f;
  }

  return lightSample;
}


extern "C" __device__ LightSample __direct_callable__light_spot(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;

  lightSample.pdf = 0.0f;

  // Get the world space position from the world to object matrix translation.
  const float3 position = make_float3(light.matrix[0].w, light.matrix[1].w, light.matrix[2].w);
  lightSample.direction = position - prd->pos; // Sample direction from surface point to light sample position.
  
  const float distanceSquared = dot(lightSample.direction, lightSample.direction);

  if (DENOMINATOR_EPSILON < distanceSquared)
  {
    lightSample.distance   = sqrtf(distanceSquared);
    lightSample.direction *= 1.0f / lightSample.distance; // Normalized direction to light.

    //const float3 normal = normalize(transformNormal(light.matrixInv, make_float3(0.0f, 0.0f, 1.0f)));
    const float3 normal = normalize(make_float3(light.matrixInv[2]));

    // Spot light is aligned to the local z-axis (the normal).
    const float cosTheta  = -dot(lightSample.direction, normal); // Negative because lightSample.direction is from surface to light.
    const float cosSpread = cosf(light.spotAngleHalf);           // Note that the spot light only supports hemispherical distributions.
   
    if (cosSpread <= cosTheta) // Is the lightSample.direction inside the spot light cone?
    {
      // Normalize the hemispherical distribution (half-angle M_PI_2f) to the cone angle (scale by factor: angleHalf / light.spotAngleHalf, range [0.0f, 1.0f]).
      const float cosHemi = cosf(M_PI_2f * acosf(cosTheta) / light.spotAngleHalf);

      // Hardcoded singular lights are defined in visible radiance directly, don't normalize.
      float3 emission = light.emission * (powf(cosHemi, light.spotExponent) / distanceSquared); // Quadratic attenuation.

      // The emission texture is used as projection scaled to the spherical cap inside the spot light cone.
      // By design all lights in this renderer shine down the light's local positive z-Axis, which is the "normal" direction for rect and mesh lights.
      if (light.textureEmission) 
      { 
        // Transform the direction from  light to surface from world space into light object space.
        const float3 R = transformVector(light.oriInv, -lightSample.direction);

        const float u = (acosf(R.x) - M_PI_2f) / light.spotAngleHalf * 0.5f + 0.5f;
        const float v = 0.5f - ((acosf(R.y) - M_PI_2f) / light.spotAngleHalf * 0.5f);

        // Modulate the base emission with the emission texture.
        emission *= make_float3(tex2D<float4>(light.textureEmission, u, v));
      }

      lightSample.radiance_over_pdf = emission; // pdf == 1.0f for singular light.

      // Indicate valid light sample (pdf != 0.0f).
      // This value is otherwise unused in a singular light. 
      // The PDF is a Dirac with infinite value for this case.
      lightSample.pdf = 1.0f;
    }
  }

  return lightSample;
}


extern "C" __device__ LightSample __direct_callable__light_ies(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;

  lightSample.pdf = 0.0f; // Default return, invalid light sample (backface, edge on, or too near to the surface)

  // Get the worls space position from the world to object matrix translation.
  const float3 position = make_float3(light.matrix[0].w, light.matrix[1].w, light.matrix[2].w);
  lightSample.direction = position - prd->pos; // Sample direction from surface point to light sample position.
  
  const float distanceSquared = dot(lightSample.direction, lightSample.direction);

  if (DENOMINATOR_EPSILON < distanceSquared)
  {
    lightSample.distance   = sqrtf(distanceSquared);
    lightSample.direction *= 1.0f / lightSample.distance; // Normalized direction to light.

    // Hardcoded singular lights are defined in visible radiance directly, do not normalize.
    // This just returns the candela values (luminous power per solid angle).
    float3 emission = light.emission * (1.0f / distanceSquared);

    // The emission texture is used as spherical projection around the point light similar to spherical environment lights.
    // By design all lights in this renderer shine down the light's local positive z-Axis, which is the "normal" direction for rect and mesh lights.

    // Transform the direction from light to surface from world into light object space.
    const float3 R = transformVector(light.oriInv, -lightSample.direction);

    // All lights shine down the positive z-axis in this renderer.
    // Means the spherical texture coordinate seam u == 0.0 == 1.0 is on the negative z-axis direction now.
    const float u = (atan2f(-R.x, R.z) + M_PIf) * 0.5f * M_1_PIf;
    const float v = acosf(-R.y) * M_1_PIf; // Texture is origin at lower left, v == 0.0f is south pole.

    if (light.textureProfile)
    {
      // Modulate the base emission with the profile texture (single component float texture, candela)
      emission *= tex2D<float>(light.textureProfile, u, v);
    }

    if (light.textureEmission) // IES light profile can be modulated by emission color texture.
    {
      // Modulate the base emission with the emission texture.
      emission *= make_float3(tex2D<float4>(light.textureEmission, u, v));
    }

    lightSample.radiance_over_pdf = emission; // pdf == 1.0f for singular light.

    // Indicate valid light sample (pdf != 0.0f).
    // This value is otherwise unused in a singular light. 
    // The PDF is a Dirac with infinite value for this case.
    lightSample.pdf = 1.0f;
  }

  return lightSample;
}
