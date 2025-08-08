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

#ifndef BXDF_COMMON_H
#define BXDF_COMMON_H

#include "config.h"

#include "vector_math.h"

// For the updateMaterialStack() function.
#include "per_ray_data.h"


__forceinline__ __device__ float3 mix_rgb(const float3 base, const float3 layer, const float3 factor)
{
  return (1.0f - fmaxf(factor)) * base + factor * layer;
}


// Fresnel equation for an equal mix of polarization.
__forceinline__ __device__ float ior_fresnel(
  const float eta, // refracted / reflected ior
  const float kh)  // cosine of angle between normal/half-vector and direction
{
  float costheta = 1.0f - (1.0f - kh * kh) / (eta * eta);
  
  if (costheta <= 0.0f)
  {
    return 1.0f;
  }

  costheta = sqrtf(costheta); // refracted angle cosine

  const float n1t1 = kh;
  const float n1t2 = costheta;
  const float n2t1 = kh * eta;
  const float n2t2 = costheta * eta;
  const float r_p = (n1t2 - n2t1) / (n1t2 + n2t1);
  const float r_o = (n1t1 - n2t2) / (n1t1 + n2t2);
  
  const float fres = 0.5f * (r_p * r_p + r_o * r_o);

  return clamp(fres, 0.0f, 1.0f);
}


//__forceinline__ __device__ float3 ior_fresnel(
//  const float3 eta, // refracted / reflected ior
//  const float  kh)  // cosine between of angle normal/half-vector and direction
//{
//  float3 result;
//
//  result.x =                               ior_fresnel(eta.x, kh);
//  result.y = (eta.y == eta.x) ? result.x : ior_fresnel(eta.y, kh);
//  result.z = (eta.z == eta.x) ? result.x : ior_fresnel(eta.z, kh);
//
//  return result;
//}


// Compute refraction direction.
__forceinline__ __device__ float3 refract(
  const float3 k,  // direction (pointing from surface)
  const float3 n,  // normal
  const float  b,  // (reflected side IOR) / (transmitted side IOR)
  const float  nk, // dot(n, k)
  bool& tir)       // total internal reflection
{
  const float refraction = b * b * (1.0f - nk * nk);

  tir = (1.0f <= refraction);

  return (tir) ? (n * (nk + nk) - k) : normalize((-k * b + n * (b * nk - sqrtf(1.0f - refraction))));
}


// Check for total internal reflection.
__forceinline__ __device__ bool isTIR(const float2 ior, const float kh)
{
  const float b = ior.x / ior.y;

  return (1.0f < (b * b * (1.0f - kh * kh)));
}


// Compute half vector (convention: pointing to outgoing direction, like shading normal)
__forceinline__ __device__ float3 compute_half_vector(
  const float3 k1,
  const float3 k2,
  const float3 normal,
  const float2 ior,
  const float  nk2,
  const bool   transmission,
  const bool   thinwalled)
{
  float3 h;

  if (transmission)
  {
    if (thinwalled) // No refraction!
    {
      h = k1 + (normal * (nk2 + nk2) + k2); // Use corresponding reflection direction.
    }
    else
    {
      h = k2 * ior.y + k1 * ior.x; // Points into thicker medium.

      if (ior.y > ior.x)
      {
        h *= -1.0f; // Make pointing to outgoing direction's medium.
      }
    }
  }
  else
  {
    h = k1 + k2; // unnormalized half-vector
  }
  
  return normalize(h);
}


// Smith-masking for anisotropic GGX
__forceinline__ __device__ float smith_shadow_mask(
  const float3 k,
  const float2 roughness)
{
  const float ax = k.x * roughness.x;
  const float ay = k.y * roughness.y;

  const float inv_a2 = (ax * ax + ay * ay) / (k.z * k.z);

  return 2.0f / (1.0f + sqrtf(1.0f + inv_a2));
}


__forceinline__ __device__ float ggx_smith_shadow_mask(
  float& G1,
  float& G2,
  const float3 k1,
  const float3 k2,
  const float2 roughness)
{
  G1 = smith_shadow_mask(k1, roughness);
  G2 = smith_shadow_mask(k2, roughness);

  return G1 * G2;
}

// Sample visible (Smith-masked) half-vector according to the anisotropic GGX distribution.
// (See Eric Heitz - A Simpler and Exact Sampling Routine for the GGX Distribution of Visible Normals)
__forceinline__ __device__ float3 hvd_ggx_sample_vndf(
  const float3 k,
  const float2 roughness,
  const float2 xi)
{ 
  // Stretch view.
  const float3 v = normalize(make_float3(k.x * roughness.x,
                                         k.y * roughness.y,
                                         k.z));
  // Orthonormal basis.
  const float3 t1 = (v.z < 0.99999f)
                  ? normalize(cross(v, make_float3(0.0f, 0.0f, 1.0f))) 
                  : make_float3(1.0f, 0.0f, 0.0f); // tangent
  const float3 t2 = cross(t1, v);                  // bitangent

  // Sample point with polar coordinates (r, phi)
  const float a = 1.0f / (1.0f + v.z);

  const float r   = sqrtf(xi.x);
  const float phi = (xi.y < a) ? xi.y / a * M_PIf : M_PIf + (xi.y - a) / (1.0f - a) * M_PIf;
  
  const float p1 = r * cosf(phi);
  const float p2 = r * sinf(phi) * ((xi.y < a) ? 1.0f : v.z);

  // Compute normal.
  float3 h = p1 * t1 + p2 * t2 + sqrtf(fmaxf(0.0f, 1.0f - p1 * p1 - p2 * p2)) * v;

  // Unstretch.
  h.x *= roughness.x;
  h.y *= roughness.y;
  h.z  = fmaxf(0.0f, h.z);

  return normalize(h);
}


// Evaluate anisotropic GGX distribution on the non-projected hemisphere.
__forceinline__ __device__ float hvd_ggx_eval(
  const float2 invAlpha,
  const float3 h) // == make_float3(dot(tangent, h), dot(bitangent, h), dot(normal, h))
 {
  const float x = h.x * invAlpha.x;
  const float y = h.y * invAlpha.y;
  
  const float aniso = x * x + y * y;

  const float f = aniso + h.z * h.z;

  return M_1_PIf * invAlpha.x * invAlpha.y * h.z / (f * f);
}


__forceinline__ __device__ float3 flip(const float3 h, const float3 k, float xi)
{
  const float a = h.z * k.z;
  const float b = h.x * k.x + h.y * k.y;

  const float kh   = fmaxf(0.0f, a + b);
  const float kh_f = fmaxf(0.0f, a - b);

  const float p_flip = kh_f / (kh + kh_f);

  // PERF xi is not used after this operation by the only caller brdf_sheen_sample(),
  // so there is no need to scale the sample.
  //if (xi < p_flip)
  //{
  //  xi /= p_flip;
  //  return make_float3(-h.x, -h.y, h.z);
  //}
  //else
  //{
  //  xi = (xi - p_flip) / (1.0f - p_flip);
  //  return h;
  //}

  return (xi < p_flip) ? make_float3(-h.x, -h.y, h.z) : h;
}


// Cook-Torrance style v-cavities masking term.
__forceinline__ __device__ float vcavities_mask(
  const float nh, // abs(dot(normal, half))
  const float kh, // abs(dot(dir, half))
  const float nk) // abs(dot(normal, dir))
{
  return fminf(2.0f * nh * nk / kh, 1.0f);
}


__forceinline__ __device__ float vcavities_shadow_mask(
  float& G1,
  float& G2,
  const float nh,
  const float3 k1, const float k1h,
  const float3 k2, const float k2h)
{
    G1 = vcavities_mask(nh, k1h, k1.z); // In my renderer the z-coordinate is the normal!
    G2 = vcavities_mask(nh, k2h, k2.z);

    //return (refraction) ? fmaxf(0.0f, G1 + G2 - 1.0f) : fminf(G1, G2);
    return fminf(G1, G2); // PERF Need reflection only.
}


// Sample half-vector according to anisotropic sheen distribution.
__forceinline__ __device__ float3 hvd_sheen_sample(const float2 xi, 
                                                   const float invRoughness)
{
  const float phi = 2.0f * M_PIf * xi.x;

  float sinPhi = sinf(phi);
  float cosPhi = cosf(phi);

  const float sinTheta = powf(1.0f - xi.y, 1.0f / (invRoughness + 2.0f));
  const float cosTheta = sqrtf(1.0f - sinTheta * sinTheta);

  return normalize(make_float3(cosPhi * sinTheta,
                               sinPhi * sinTheta,
                               cosTheta)); // In my renderer the z-coordinate is the normal!
}


// Evaluate anisotropic sheen half-vector distribution on the non-projected hemisphere.
__forceinline__ __device__ float hvd_sheen_eval(const float invRoughness,
                                                const float nh) // dot(shading_normal, h)
{
  const float sinTheta2 = fmaxf(0.0f, 1.0f - nh * nh);
  const float sinTheta  = sqrtf(sinTheta2);

  return (invRoughness + 2.0f) * powf(sinTheta, invRoughness) * 0.5f * M_1_PIf * nh;
}

// Iridescence implementation

template<typename T> 
__forceinline__ __device__ T sqr(T v)
{
  return v * v;
}

// Compute squared norm of s/p polarized Fresnel reflection coefficients and phase shifts in complex unit circle.
// Born/Wolf - "Principles of Optics", section 13.4
__forceinline__ __device__ float2 fresnel_conductor(
  float2& phase_shift_sin,
  float2& phase_shift_cos,
  const float n_a,
  const float n_b,
  const float k_b,
  const float cos_a,
  const float sin_a_sqd)
{
  const float k_b2 = k_b * k_b;
  const float n_b2 = n_b * n_b;
  const float n_a2 = n_a * n_a;
  const float tmp0 = n_b2 - k_b2;
  const float half_U = 0.5f * (tmp0 - n_a2 * sin_a_sqd);
  const float half_V = sqrtf(fmaxf(0.0f, half_U * half_U + k_b2 * n_b2));

  const float u_b2 = half_U + half_V;
  const float v_b2 = half_V - half_U;
  const float u_b = sqrtf(fmaxf(0.0f, u_b2));
  const float v_b = sqrtf(fmaxf(0.0f, v_b2));

  const float tmp1 = tmp0 * cos_a;
  const float tmp2 = n_a * u_b;
  const float tmp3 = (2.0f * n_b * k_b) * cos_a;
  const float tmp4 = n_a * v_b;
  const float tmp5 = n_a * cos_a;

  const float tmp6 = (2.0f * tmp5) * v_b;
  const float tmp7 = (u_b2 + v_b2) - tmp5 * tmp5;

  const float tmp8 = (2.0f * tmp5) * ((2.0f * n_b * k_b) * u_b - tmp0 * v_b);
  const float tmp9 = sqr((n_b2 + k_b2) * cos_a) - n_a2 * (u_b2 + v_b2);

  const float tmp67 = tmp6 * tmp6 + tmp7 * tmp7;
  const float inv_sqrt_x = (0.0f < tmp67) ? (1.0f / sqrtf(tmp67)) : 0.0f;
  const float tmp89 = tmp8 * tmp8 + tmp9 * tmp9;
  const float inv_sqrt_y = (0.0f < tmp89) ? (1.0f / sqrtf(tmp89)) : 0.0f;

  phase_shift_cos = make_float2(tmp7 * inv_sqrt_x, tmp9 * inv_sqrt_y);
  phase_shift_sin = make_float2(tmp6 * inv_sqrt_x, tmp8 * inv_sqrt_y);

  return make_float2(
      (sqr(tmp5 - u_b) + v_b2) / (sqr(tmp5 + u_b) + v_b2),
      (sqr(tmp1 - tmp2) + sqr(tmp3 - tmp4)) / (sqr(tmp1 + tmp2) + sqr(tmp3 + tmp4)));
}


// Simplified for dielectric, no phase shift computation.
__forceinline__ __device__ float2 fresnel_dielectric(
  const float n_a,
  const float n_b,
  const float cos_a,
  const float cos_b)
{
  const float naca = n_a * cos_a;
  const float nbcb = n_b * cos_b;
  const float r_s = (naca - nbcb) / (naca + nbcb);

  const float nacb = n_a * cos_b;
  const float nbca = n_b * cos_a;
  const float r_p = (nbca - nacb) / (nbca + nacb);

  return make_float2(r_s * r_s, r_p * r_p);
}


__forceinline__ __device__ float3 thin_film_factor(
  float coating_thickness,
  const float coating_ior,
  const float base_ior,
  const float incoming_ior,
  const float kh)
{
  coating_thickness = fmaxf(0.0f, coating_thickness);

  const float sin0_sqr = fmaxf(0.0f, 1.0f - kh * kh);
  const float eta01 = incoming_ior / coating_ior;
  const float eta01_sqr = eta01 * eta01;
  const float sin1_sqr = eta01_sqr * sin0_sqr;

  if (1.0f < sin1_sqr) // TIR at first interface
  {
    return make_float3(1.0f);
  }

  const float cos1 = sqrtf(fmaxf(0.0f, 1.0f - sin1_sqr));
  const float2 R01 = fresnel_dielectric(incoming_ior, coating_ior, kh, cos1);

  float2 phi12_sin, phi12_cos;
  const float2 R12 = fresnel_conductor(phi12_sin, phi12_cos, coating_ior, base_ior, /* base_k = */ 0.0f, cos1, sin1_sqr);

  const float tmp = (4.0f * M_PIf) * coating_ior * coating_thickness * cos1;

  const float R01R12_s = fmaxf(0.0f, R01.x * R12.x);
  const float r01r12_s = sqrtf(R01R12_s);
  
  const float R01R12_p = fmaxf(0.0f, R01.y * R12.y);
  const float r01r12_p = sqrtf(R01R12_p);

  float3 xyz = make_float3(0.0f);

  //!! using low res color matching functions here
  constexpr float lambda_min  = 400.0f;
  constexpr float lambda_step = ((700.0f - 400.0f) / 16.0f);

  static const float3 cie_xyz[16] = 
  {
    {0.02986f, 0.00310f, 0.13609f},
    {0.20715f, 0.02304f, 0.99584f},
    {0.36717f, 0.06469f, 1.89550f},
    {0.28549f, 0.13661f, 1.67236f},
    {0.08233f, 0.26856f, 0.76653f},
    {0.01723f, 0.48621f, 0.21889f},
    {0.14400f, 0.77341f, 0.05886f},
    {0.40957f, 0.95850f, 0.01280f},
    {0.74201f, 0.97967f, 0.00060f},
    {1.03325f, 0.84591f, 0.00000f},
    {1.08385f, 0.62242f, 0.00000f},
    {0.79203f, 0.36749f, 0.00000f},
    {0.38751f, 0.16135f, 0.00000f},
    {0.13401f, 0.05298f, 0.00000f},
    {0.03531f, 0.01375f, 0.00000f},
    {0.00817f, 0.00317f, 0.00000f}
  };

  float lambda = lambda_min + 0.5f * lambda_step;

  for (unsigned int i = 0; i < 16; ++i)
  {
    const float phi = tmp / lambda;

    float phi_s = sinf(phi);
    float phi_c = cosf(phi);

    const float cos_phi_s = phi_c * phi12_cos.x - phi_s * phi12_sin.x; // cos(a+b) = cos(a) * cos(b) - sin(a) * sin(b)
    const float tmp_s = 2.0f * r01r12_s * cos_phi_s;
    const float R_s = (R01.x + R12.x + tmp_s) / (1.0f + R01R12_s + tmp_s);

    const float cos_phi_p = phi_c * phi12_cos.y - phi_s * phi12_sin.y; // cos(a+b) = cos(a) * cos(b) - sin(a) * sin(b)
    const float tmp_p = 2.0f * r01r12_p * cos_phi_p;
    const float R_p = (R01.y + R12.y + tmp_p) / (1.0f + R01R12_p + tmp_p);

    xyz += cie_xyz[i] * (0.5f * (R_s + R_p));

    lambda += lambda_step;
  }

  xyz *= (1.0f / 16.0f);

  // ("normalized" such that the loop for no shifted wave gives reflectivity (1,1,1))
  return clamp(
    make_float3(
      xyz.x * ( 3.2406f / 0.433509f) + 
      xyz.y * (-1.5372f / 0.433509f) +
      xyz.z * (-0.4986f / 0.433509f),

      xyz.x * (-0.9689f / 0.341582f) +
      xyz.y * ( 1.8758f / 0.341582f) +
      xyz.z * ( 0.0415f / 0.341582f),

      xyz.x * ( 0.0557f / 0.32695f) +
      xyz.y * (-0.204f  / 0.32695f) +
      xyz.z * ( 1.057f  / 0.32695f)
    ), 
    0.0f, 1.0f);
}



#endif // BXDF_COMMON_H
