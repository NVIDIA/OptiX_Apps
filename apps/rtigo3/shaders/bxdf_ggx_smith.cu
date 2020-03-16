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

#include "per_ray_data.h"
#include "material_definition.h"
#include "shader_common.h"
#include "random_number_generators.h"

// "Microfacet Models for Refraction through Rough Surfaces" - Walter, Marschner, Li, Torrance. 2007
// "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs" - Eric Heitz

// This function evaluates a Fresnel dielectric function when the transmitting cosine ("cost")
// is unknown and the incident index of refraction is assumed to be 1.0f.
// \param et     The transmitted index of refraction.
// \param costIn The cosine of the angle between the incident direction and normal direction.
__forceinline__ __device__ float evaluateFresnelDielectric(const float et, const float cosIn)
{
  const float cosi = fabsf(cosIn);

  float sint = 1.0f - cosi * cosi;
  sint = (0.0f < sint) ? sqrtf(sint) / et : 0.0f;

  // Handle total internal reflection.
  if (1.0f < sint)
  {
    return 1.0f;
  }

  float cost = 1.0f - sint * sint;
  cost = (0.0f < cost) ? sqrtf(cost) : 0.0f;

  const float et_cosi = et * cosi;
  const float et_cost = et * cost;

  const float rPerpendicular = (cosi - et_cost) / (cosi + et_cost);
  const float rParallel      = (et_cosi - cost) / (et_cosi + cost);

  const float result = (rParallel * rParallel + rPerpendicular * rPerpendicular) * 0.5f;

  return (result <= 1.0f) ? result : 1.0f;
}


// Optimized version to calculate D and PDF reusing shared calculations.
__forceinline__ __device__ float2 distribution_d_pdf(const float ax, const float ay, float3 const& wm)
{
  if (DENOMINATOR_EPSILON < wm.z) // Heaviside function: X_plus(wm * wg). (wm is in tangent space.)
  {
    const float cosThetaSqr = wm.z * wm.z;
    const float tanThetaSqr = (1.0f - cosThetaSqr) / cosThetaSqr;

    const float phiM    = atan2f(wm.y, wm.x);
    const float cosPhiM = cosf(phiM);
    const float sinPhiM = sinf(phiM);

    const float term = 1.0f + tanThetaSqr * ((cosPhiM * cosPhiM) / (ax * ax) + (sinPhiM * sinPhiM) / (ay * ay));

    const float d   = 1.0f / (M_PIf * ax * ay * cosThetaSqr * cosThetaSqr * term * term); // Heitz, Formula (85)
    const float pdf = d * wm.z; // PDF with respect to the half-direction.
      
    return make_float2(d, pdf);
  }
  return make_float2(0.0f);
}

// Return a sample direction in local tangent space coordinates.
__forceinline__ __device__ float3 distribution_sample(const float ax, const float ay, const float u1, const float u2)
{
  // Made isotropic to ay. Output vector scales .x accordingly.
  const float theta    = atanf(ay * sqrtf(u1) / sqrtf(1.0f - u1)); // Walter, Formula (35).
  const float phi      = 2.0f * M_PIf * u2;                        // Walter, Formula (36).
  const float sinTheta = sinf(theta);
  return normalize(make_float3(cosf(phi) * sinTheta * ax / ay,     // Heitz, Formula (77)
                               sinf(phi) * sinTheta,
                               cosf(theta)));
}

// "Microfacet Models for Refraction through Rough Surfaces" - Walter, Marschner, Li, Torrance.
// PERF Using this because it's faster than the approximation below.
__forceinline__ __device__ float smith_G1(const float alpha, float3 const& w, float3 const& wm)
{
  const float w_wm = dot(w, wm);
  if (w_wm * w.z <= 0.0f) // X_plus(v * m / v * n) from Walter, Formula (34). // PERF Checking the sign with a multiplication here.
  {
    return 0.0f;
  }
  const float cosThetaSqr = w.z * w.z;
  const float sinThetaSqr = 1.0f - cosThetaSqr;
  //const float tanTheta = (0.0f < sinThetaSqr) ? sqrtf(sinThetaSqr) / w.z : 0.0f; // PERF Remove the sqrtf() by calculating tanThetaSqr here
  //const float invA = alpha * tanTheta;                                           // because this is squared below: invASqr = alpha * alpha * tanThetaSqr;
  //const float lambda = (-1.0f + sqrtf(1.0f + invA * invA)) * 0.5f; // Heitz, Formula (86)
  //return 1.0f / (1.0f + lambda);                                   // Heitz, below Formula (69)
  const float tanThetaSqr = (0.0f < sinThetaSqr) ? sinThetaSqr / cosThetaSqr : 0.0f;
  const float invASqr = alpha * alpha * tanThetaSqr;                                           
  return 2.0f / (1.0f + sqrtf(1.0f + invASqr));                     // Optimized version is Walter, Formula (34)
}

// Approximation from "Microfacet Models for Refraction through Rough Surfaces" - Walter, Marschner, Li, Torrance.
//__forceinline__ __device__ float smith_G1(const float alpha, float3 const& w, float3 const& wm)
//{
//  const float w_wm = optix::dot(w, wm);
//  if (w_wm * w.z <= 0.0f) // X_plus(v * m / v * n) from Walter, Formula (34). // PERF Checking the sign with a multiplication here.
//  {
//    return 0.0f;
//  }
//  const float t        = 1.0f - w.z * w.z; 
//  const float tanTheta = (0.0f < t) ? sqrtf(t) / w.z : 0.0f;
//  if (tanTheta == 0.0f)
//  {
//    return 1.0f;
//  }
//  const float a = 1.0f / (tanTheta * alpha);
//  if (1.6f <= a)
//  {
//    return 1.0f;
//  }
//  const float aSqr = a * a;
//  return (3.535f * a + 2.181f * aSqr) / (1.0f + 2.276f * a + 2.577f * aSqr); // Walter, Formula (27) used for Heitz, Formula (83)
//}

__forceinline__ __device__ float distribution_G(const float ax, const float ay, float3 const& wo, float3 const& wi, float3 const& wm)
{
  float phi   = atan2f(wo.y, wo.x);
  float c     = cosf(phi);
  float s     = sinf(phi);
  float alpha = sqrtf(c * c * ax * ax + s * s * ay * ay); // Heitz, Formula (80) for wo

  const float g = smith_G1(alpha, wo, wm);

  phi   = atan2f(wi.y, wi.x);
  c     = cosf(phi);
  s     = sinf(phi);
  alpha = sqrtf(c * c * ax * ax + s * s * ay * ay); // Heitz, Formula (80) for wi.

  return g * smith_G1(alpha, wi, wm);
}

// ########## BRDF GGX with Smith shadowing

extern "C" __device__ void __direct_callable__sample_brdf_ggx_smith(MaterialDefinition const& material, State const& state, PerRayData* prd)
{
  // Sample a microfacet normal in local space, which effectively is a tangent space coordinate.
  const float2 sample = rng2(prd->seed);

  const float3 wm = distribution_sample(material.roughness.x, 
                                        material.roughness.y, 
                                        sample.x,
                                        sample.y);

  const TBN tangentSpace(state.tangent, state.normal); // Tangent space transformation, handles anisotropic rotation. 
  
  const float3 wh = tangentSpace.transformToWorld(wm); // wh is the microfacet normal in world space coordinates!
 
  prd->wi = reflect(-prd->wo, wh);

  if (dot(prd->wi, state.normalGeo) <= 0.0f) // Do not sample opaque materials below the geometric surface.
  {
    prd->flags |= FLAG_TERMINATE;
    return;
  }

  const float3 wo = tangentSpace.transformToLocal(prd->wo);
  const float3 wi = tangentSpace.transformToLocal(prd->wi);

  const float wi_wh = dot(prd->wi, wh);

  if (wo.z <= 0.0f || wi.z <= 0.0f || wi_wh <= 0.0f) 
  {
    prd->flags |= FLAG_TERMINATE;
    return;
  }

  const float2 D_PDF = distribution_d_pdf(material.roughness.x,
                                          material.roughness.y,
                                          wm);
  if (D_PDF.y <= 0.0f)
  {
    prd->flags |= FLAG_TERMINATE;
    return;
  }

  const float G = distribution_G(material.roughness.x,
                                 material.roughness.y,
                                 wo, wi, wm);
    
  // Watch out: PBRT2 puts the factor 1.0f / (4.0f * cosThetaH) into the pdf() functions.
  //            This is the density function with respect to the light vector.
  prd->pdf = D_PDF.y / (4.0f * wi_wh);
  //prd->f_over_pdf = state.albedo * (fabsf(dot(prd->wi, state->normal)) * D_PDF.x * G / (4.0f * wo.z * wi.z * prd->pdf));
  prd->f_over_pdf = state.albedo * (G * D_PDF.x * wi_wh / (D_PDF.y * wo.z)); // Optimized version with all factors canceled out.

  prd->flags |= FLAG_DIFFUSE; // Can handle direct lighting.
}

// When reaching this function, the roughness values are clamped to a minimal working value already,
// so that anisotropic roughness can simply be calculated without additional checks!
extern "C" __device__ float4 __direct_callable__eval_brdf_ggx_smith(MaterialDefinition const& material, State const& state, PerRayData* const prd, const float3 wiL)
{
  const TBN tangentSpace(state.tangent, state.normal); // Tangent space transformation, handles anisotropic rotation. 

  const float3 wo = tangentSpace.transformToLocal(prd->wo);
  const float3 wi = tangentSpace.transformToLocal(wiL);

  if (wo.z <= 0.0f || wi.z <= 0.0f) // Either vector on the other side of the node.normal hemisphere?
  {
    return make_float4(0.0f);
  }

  float3 wm = wo + wi; // The half-vector is the microfacet normal, in tangent space
  if (isNull(wm)) // Collinear in opposing directions?
  {
    return make_float4(0.0f);
  }

  wm = normalize(wm);

  const float2 D_PDF = distribution_d_pdf(material.roughness.x,
                                          material.roughness.y,
                                          wm);

  const float G = distribution_G(material.roughness.x,
                                 material.roughness.y,
                                 wo, wi, wm);

  const float3 f = state.albedo * (D_PDF.x * G / (4.0f * wo.z * wi.z));
  
  // Watch out: PBRT2 puts the factor 1.0f / (4.0f * cosThetaH) into the pdf() functions.
  //            This is the density function with respect to the light vector.
  const float pdf = D_PDF.y / (4.0f * dot(wi, wm));

  return make_float4(f, pdf);
}

// ########## BSDF GGX with Smith shadowing

extern "C" __device__ void __direct_callable__sample_bsdf_ggx_smith(MaterialDefinition const& material, State const& state, PerRayData* prd)
{
  // Return the current material's absorption coefficient and ior to the integrator to be able to support nested materials.
  prd->absorption_ior = make_float4(material.absorption, material.ior);

  // Need to figure out here which index of refraction to use if the ray is already inside some refractive medium.
  // This needs to happen with the original FLAG_FRONTFACE condition to find out from which side of the geometry we're looking!
  // ior.xy are the current volume's IOR and the surrounding volume's IOR.
  // Thin-walled materials have no volume, always use the frontface eta for them!
  const float eta = (prd->flags & (FLAG_FRONTFACE | FLAG_THINWALLED))
                  ? prd->absorption_ior.w / prd->ior.x 
                  : prd->ior.y / prd->absorption_ior.w;
  
  // Sample a microfacet normal in local space, which effectively is a tangent space coordinate.
  const float2 sample = rng2(prd->seed);

  const float3 wm = distribution_sample(material.roughness.x, 
                                        material.roughness.y, 
                                        sample.x,
                                        sample.y);

  const TBN tangentSpace(state.tangent, state.normal); // Tangent space transformation, handles anisotropic rotation. 
  
  const float3 wh = tangentSpace.transformToWorld(wm); // wh is the microfacet normal in world space coordinates!


  const float3 R = reflect(-prd->wo, wh);

  float reflective = 1.0f;
  if (refract(prd->wi, -prd->wo, wh, eta))
  {
    if (prd->flags & FLAG_THINWALLED)
    {
      // DAR FIXME The resulting vector isn't necessarily on the other side of the geometric normal, but should be!
      prd->wi = reflect(R, state.normal); // Flip the vector to the other side of the normal.
    }
    // Note, not using fabs() on the cosine to get the refract side correct.
    // Total internal reflection will leave this reflection probability at 1.0f.
    reflective = evaluateFresnelDielectric(eta, dot(prd->wo, wh));
  }

  const float pseudo = rng(prd->seed);
  if (pseudo < reflective)
  {
    prd->wi = R; // Fresnel reflection or total internal reflection.
  }
  else if (!(prd->flags & FLAG_THINWALLED)) // Only non-thinwalled materials have a volume and transmission events.
  {
    prd->flags |= FLAG_TRANSMISSION;
  }

  // No Fresnel factor here. The probability to pick one or the other side took care of that.
  prd->f_over_pdf = state.albedo;
  prd->pdf        = 1.0f; // Not 0.0f to make sure the path is not terminated. Otherwise unused for specular events.
}

//extern "C" __device__ float4 __direct_callable__eval_bsdf_ggx_smith(MaterialDefinition const& material, State const& state, PerRayData* const prd, const float3 wiL)
//{
//  // The implementation handles this as specular continuation. Can reuse the eval_brdf_specular() implementation.
//  return make_float4(0.0f);
//}
