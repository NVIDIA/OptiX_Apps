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
  // There is no transformation of the object space direction into world space necessary.
  // The emission is constant in all directions.

  // Environment lights do not set the light sample position!
  lightSample.distance = RT_DEFAULT_MAX; // Environment light.
  
  // Explicit light sample. Light emision scaled by inverse probabilty to hit this light.
  lightSample.emission = light.emission * float(sysData.numLights);
  
  return lightSample;
}


extern "C" __device__ LightSample __direct_callable__light_env_sphere(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;

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

  // Note that environment lights do not set the light sample position!
  lightSample.distance = RT_DEFAULT_MAX; // Environment light.
  
  // Get the emission from the spherical environment texture.
  const float3 emission = make_float3(tex2D<float4>(light.textureEmission, u, v));
  
  // Explicit light sample. The returned emission must be scaled by the inverse probability to select this light.
  // Modulate the texture emission with the light.emission color for more flexibility.
  lightSample.emission = emission * light.emission * float(sysData.numLights);

  // For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
  // and not the Gaussian-smoothed one used to actually generate the CDFs and uniform sampling in the texel.
  // (Note that this does not contain the light.emission which just modulates the texture.)
  lightSample.pdf = intensity(emission) / light.integral;

  return lightSample;
}

extern "C" __device__ LightSample __direct_callable__light_rect(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;

  lightSample.pdf = 0.0f; // Default return, invalid light sample (backface, edge on, or too near to the surface)

  // Do the check for front or backface first before calculating any expensive sampling point.
  // Just use the center point to calculate the temporary direction.
  // This is sufficient because the rectangle light is planar.
  const float3 center = make_float3(light.matrix[0].w, light.matrix[1].w, light.matrix[2].w);
  //const float3 normal = normalize(transformNormal(light.matrixInv, make_float3(0.0f, 0.0f, 1.0f)));
  const float3 normal = normalize(make_float3(light.matrixInv[2]));

  // Check if the vector from center to surface is in the same hemisphere as the face normal.
  // Only need the sign, no need to normalize the temporary direction.
  if (0.0f < dot(prd->pos - center, normal))
  {
    float pdf = 1.0f; // Neutral factor in case there is no light.texture.

    float2 sample = rng2(prd->seed);

    lightSample.emission = light.emission;

    if (light.textureEmission)
    {
      // Importance-sample the rectangular texture in object space.
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
      sample.x = (float(idxU) + du) / float(light.width); // Texture coordinate and new sample.

      const float cdfLowerV = cdfV[idxV];
      const float cdfUpperV = cdfV[idxV + 1];
      const float dv = (sample.y - cdfLowerV) / (cdfUpperV - cdfLowerV);
      sample.y = (float(idxV) + dv) / float(light.height); // Texture coordinate and new sample.

      // Get the emission from the emission texture.
      const float3 emission = make_float3(tex2D<float4>(light.textureEmission, sample.x, sample.y));
      
      lightSample.emission *= emission; 

      // The pdf to have picked this emission on the texture.
      pdf = intensity(emission) / light.integral; // This must be the emission from the texture only!
    }

    float cutout = 1.0f; // Neutral factor in case there is no cutout.

    if (light.textureCutout) 
    {
      // Clamp the cutout value to the range [0.0f, 1.0f] in case the assigned texture is HDR.
      // This modulates the emission below and must not make it negative or brighter.
      // FIXME This is wasteful because there wouldn't be a need to sample any location which is completely cut away.
      // The cutout condition could be encoded into the CDF directly instead.
      //cutout = min(intensity(make_float3(tex2D<float4>(light.textureCutout, sample.x, sample.y))), 1.0f);
      // Alpha channel of an RGBA texture as used in the OMM baker in the rtigo9_omm example.
      cutout = clamp(tex2D<float4>(light.textureCutout, sample.x, sample.y).w, 0.0f, 1.0f);
    }

    if (0.0f < cutout)
    {
      // Transform the default rectangle in the xy-range [-0.5, 0.5] into world space.
      //const float3 vecU   = transformVector(light.matrix, make_float3(1.0f, 0.0f, 0.0f));
      //const float3 vecV   = transformVector(light.matrix, make_float3(0.0f, 1.0f, 0.0f));
      //const float3 center = transformPoint(light.matrix, make_float3(0.0f));
      // Optimized versions.
      const float3 vecU   = make_float3(light.matrix[0].x, light.matrix[1].x, light.matrix[2].x);
      const float3 vecV   = make_float3(light.matrix[0].y, light.matrix[1].y, light.matrix[2].y);
      const float3 center = make_float3(light.matrix[0].w, light.matrix[1].w, light.matrix[2].w);
  
      lightSample.position  = center + vecU * (sample.x - 0.5f) + vecV * (sample.y - 0.5f); // The light sample position in world coordinates.
      lightSample.direction = lightSample.position - prd->pos; // Sample direction from surface point to light sample position.
      lightSample.distance  = length(lightSample.direction);
  
      if (DENOMINATOR_EPSILON < lightSample.distance)
      {
        lightSample.direction /= lightSample.distance; // Normalized direction to light.

        const float cosTheta = -dot(lightSample.direction, normal); // This must be greater than zero because of the initial frontface check.

        lightSample.emission *= cutout * float(sysData.numLights);  // Explicit light sample, must scale the emission by inverse probabilty to hit this light.

        lightSample.pdf = pdf * lightSample.distance * lightSample.distance / (light.area * cosTheta); // Solid angle pdf. Assumes area != 0.0f.
      }
    }
  }
  return lightSample;
}


extern "C" __device__ LightSample __direct_callable__light_mesh(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;

  lightSample.pdf = 0.0f; // Default return, invalid light sample (backface, edge on, or too near to the surface)

  const float3 sample = rng3(prd->seed); // This light sampler needs three random numbers!

  // Uniformly sample the triangles over their surface area.
  // Note that zero-area triangles (e.g. at the poles of spheres) are automatically never sampled with this method!
  // The cdfU is one bigger than light.width.
  const float* cdfArea = reinterpret_cast<const float*>(light.cdfU);
  const unsigned int idx = binarySearchCDF(cdfArea, light.width, sample.z);

  // Unit square to triangle via barycentric coordinates.
  const float su = sqrtf(sample.x);
  // Barycentric coordinates.
  const float alpha = 1.0f - su;
  const float beta  = sample.y * su;
  const float gamma = 1.0f - alpha - beta; 

  // This cast works because both unsigned int and uint3 have an alignment of 4 bytes.
  const uint3* indices = reinterpret_cast<uint3*>(light.indices);
  const uint3  tri     = indices[idx];

  const TriangleAttributes* attributes = reinterpret_cast<TriangleAttributes*>(light.attributes);

  const TriangleAttributes& attr0 = attributes[tri.x];
  const TriangleAttributes& attr1 = attributes[tri.y];
  const TriangleAttributes& attr2 = attributes[tri.z];

  const float3 texcoord = attr0.texcoord * alpha + attr1.texcoord * beta + attr2.texcoord * gamma;

  float cutout = 1.0f; 

  if (light.textureCutout)
  {
    // Clamp the cutout value to the range [0.0f, 1.0f] in case the assigned texture is HDR.
    // This modulates the emission below and must not make it negative or brighter.
    //cutout = clamp(intensity(make_float3(tex2D<float4>(light.textureCutout, texcoord.x, texcoord.y))), 0.0f, 1.0f);
    // Alpha channel of an RGBA texture as used in the OMM baker in the rtigo9_omm example.
    cutout = clamp(tex2D<float4>(light.textureCutout, texcoord.x, texcoord.y).w, 0.0f, 1.0f);
  }
   
  if (0.0f < cutout)
  { 
    const float3 posObj = attr0.vertex * alpha + attr1.vertex * beta + attr2.vertex * gamma; // Object space sample position.

    lightSample.position  = transformPoint(light.matrix, posObj); // World space sample position.
    lightSample.direction = lightSample.position - prd->pos;      // Sample direction from surface point to light sample position.
    lightSample.distance  = length(lightSample.direction);

    if (DENOMINATOR_EPSILON < lightSample.distance)
    {
      lightSample.direction /= lightSample.distance; // Normalized direction to light.
 
      const float3 ngObj = cross(attr1.vertex - attr0.vertex, attr2.vertex - attr0.vertex); // Object space unnormalized geometry normal.
      
      const float3 normal = normalize(transformNormal(light.matrixInv, ngObj)); // World space geometry normal.

      const float cosTheta = -dot(lightSample.direction, normal);
      if (DENOMINATOR_EPSILON < cosTheta) // Only emit light on the front side.
      {
        // Explicit light sample, must scale the emission by inverse probability to hit this light.
        // This implements edf_diffuse.
        lightSample.emission = light.emission * (cutout * float(sysData.numLights)); 

        if (light.textureEmission)
        {
          // Modulate the base emission with the emission texture. 
          // PERF Could sample first and compare with black.
          lightSample.emission *= make_float3(tex2D<float4>(light.textureEmission, texcoord.x, texcoord.y));
        }

        // Light area to solid angle pdf. Assumes light.area != 0.0f.
        lightSample.pdf = lightSample.distance * lightSample.distance / (light.area * cosTheta); 
      }
    }
  }

  return lightSample;
}


extern "C" __device__ LightSample __direct_callable__light_point(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;

  lightSample.pdf = 0.0f; // Default return, invalid light sample (backface, edge on, or too near to the surface)

  // Get the world space position from the object to world matrix translation.
  lightSample.position  = make_float3(light.matrix[0].w, light.matrix[1].w, light.matrix[2].w);
  lightSample.direction = lightSample.position - prd->pos; // Sample direction from surface point to light sample position.
  lightSample.distance  = length(lightSample.direction);

  if (DENOMINATOR_EPSILON < lightSample.distance)
  {
    lightSample.direction /= lightSample.distance; // Normalized direction to light.

    // Explicit light sample, must scale the emission by inverse probabilty to hit this light.
    // Singular light, pdf is unused. I could actually put it into the pdf to do the attenuation outside. 
    const float attenuation = 1.0f / (lightSample.distance * lightSample.distance);
    // This implements edf_diffuse.
    lightSample.emission = light.emission * (attenuation * float(sysData.numLights));

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
      lightSample.emission *= make_float3(tex2D<float4>(light.textureEmission, u, v));
    }

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

  lightSample.pdf = 0.0f; // Default return, invalid light sample (backface, edge on, or too near to the surface)

  // Get the worls space position from the world to object matrix translation.
  lightSample.position  = make_float3(light.matrix[0].w, light.matrix[1].w, light.matrix[2].w);
  lightSample.direction = lightSample.position - prd->pos; // Sample direction from surface point to light sample position.
  lightSample.distance  = length(lightSample.direction);

  if (DENOMINATOR_EPSILON < lightSample.distance)
  {
    lightSample.direction /= lightSample.distance; // Normalized direction to light.

    //const float3 normal = normalize(transformNormal(light.matrixInv, make_float3(0.0f, 0.0f, 1.0f)));
    const float3 normal = normalize(make_float3(light.matrixInv[2]));

    // Spot light is aligned to the local z-axis (the normal).
    const float cosTheta  = -dot(lightSample.direction, normal); // Negative because lightSample.direction is from surface to light.
    const float cosSpread = cosf(light.spotAngleHalf);           // Note that the spot light only supports hemispherical distributions.
   
    if (cosSpread <= cosTheta) // Is the lightSample.direction inside the spot light cone?
    {
      // Normalize the hemispherical distribution (half-angle M_PI_2f) to the cone angle (scale by factor: angleHalf / light.spotAngleHalf, range [0.0f, 1.0f]).
      const float cosHemi = cosf(M_PI_2f * acosf(cosTheta) / light.spotAngleHalf);

      // Singular light, pdf is unused.
      const float attenuation = powf(cosHemi, light.spotExponent) / (lightSample.distance * lightSample.distance);
      
      // Explicit light sample, must scale the emission by inverse  probabilty to hit this light.
      // This implements edf_diffuse.
      lightSample.emission = light.emission * (attenuation * float(sysData.numLights));

      // The emission texture is used as projection scaled to the spherical cap inside the spot light cone.
      // By design all lights in this renderer shine down the light's local positive z-Axis, which is the "normal" direction for rect and mesh lights.
      if (light.textureEmission) 
      { 
        // Transform the direction from  light to surface from world space into light object space.
        const float3 R = transformVector(light.oriInv, -lightSample.direction);

        const float u = (acosf(R.x) - M_PI_2f) / light.spotAngleHalf * 0.5f + 0.5f;
        const float v = 0.5f - ((acosf(R.y) - M_PI_2f) / light.spotAngleHalf * 0.5f);

        // Modulate the base emission with the emission texture.
        lightSample.emission *= make_float3(tex2D<float4>(light.textureEmission, u, v));
      }

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
  lightSample.position  = make_float3(light.matrix[0].w, light.matrix[1].w, light.matrix[2].w);
  lightSample.direction = lightSample.position - prd->pos; // Sample direction from surface point to light sample position.
  lightSample.distance  = length(lightSample.direction);

  if (DENOMINATOR_EPSILON < lightSample.distance)
  {
    lightSample.direction /= lightSample.distance; // Normalized direction to light.

    // Explicit light sample, must scale the emission by inverse probabilty to hit this light.
    // Singular light, pdf is unused.
    const float attenuation = 1.0f / (lightSample.distance * lightSample.distance);
    // This implements edf_diffuse.
    lightSample.emission = light.emission * (attenuation * float(sysData.numLights));

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
      lightSample.emission *= tex2D<float>(light.textureProfile, u, v);
    }

    if (light.textureEmission) // IES light profile can be modulated by emission color texture.
    {
      // Modulate the base emission with the emission texture.
      lightSample.emission *= make_float3(tex2D<float4>(light.textureEmission, u, v));
    }

    // Indicate valid light sample (pdf != 0.0f).
    // This value is otherwise unused in a singular light. 
    // The PDF is a Dirac with infinite value for this case.
    lightSample.pdf = 1.0f;
  }

  return lightSample;
}
