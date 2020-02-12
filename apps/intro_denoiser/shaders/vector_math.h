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

#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

#include "app_config.h"

#include <cuda_runtime.h>


#if defined(__CUDACC__) || defined(__CUDABE__)
#define VECTOR_MATH_API __forceinline__ __host__ __device__
#else
#include <math.h>
#define VECTOR_MATH_API inline
#endif


#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288419716939937510
#endif
#ifndef M_Ef
#define M_Ef 2.71828182845904523536f
#endif
#ifndef M_LOG2Ef
#define M_LOG2Ef 1.44269504088896340736f
#endif
#ifndef M_LOG10Ef
#define M_LOG10Ef 0.434294481903251827651f
#endif
#ifndef M_LN2f
#define M_LN2f 0.693147180559945309417f
#endif
#ifndef M_LN10f
#define M_LN10f 2.30258509299404568402f
#endif
#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif
#ifndef M_PI_2f
#define M_PI_2f 1.57079632679489661923f
#endif
#ifndef M_PI_4f
#define M_PI_4f 0.785398163397448309616f
#endif
#ifndef M_1_PIf
#define M_1_PIf 0.318309886183790671538f
#endif
#ifndef M_2_PIf
#define M_2_PIf 0.636619772367581343076f
#endif
#ifndef M_2_SQRTPIf
#define M_2_SQRTPIf 1.12837916709551257390f
#endif
#ifndef M_SQRT2f
#define M_SQRT2f 1.41421356237309504880f
#endif
#ifndef M_SQRT1_2f
#define M_SQRT1_2f 0.707106781186547524401f
#endif

#if !defined(__CUDACC__)

VECTOR_MATH_API int max(int a, int b)
{
  return (a > b) ? a : b;
}

VECTOR_MATH_API int min(int a, int b)
{
  return (a < b) ? a : b;
}

VECTOR_MATH_API long long max(long long a, long long b)
{
  return (a > b) ? a : b;
}

VECTOR_MATH_API long long min(long long a, long long b)
{
  return (a < b) ? a : b;
}

VECTOR_MATH_API unsigned int max(unsigned int a, unsigned int b)
{
  return (a > b) ? a : b;
}

VECTOR_MATH_API unsigned int min(unsigned int a, unsigned int b)
{
  return (a < b) ? a : b;
}

VECTOR_MATH_API unsigned long long max(unsigned long long a, unsigned long long b)
{
  return (a > b) ? a : b;
}

VECTOR_MATH_API unsigned long long min(unsigned long long a, unsigned long long b)
{
  return (a < b) ? a : b;
}

#endif

/** lerp */
VECTOR_MATH_API float lerp(const float a, const float b, const float t)
{
  return a + t * (b - a);
}

/** bilerp */
VECTOR_MATH_API float bilerp(const float x00, const float x10, const float x01, const float x11, 
                             const float u, const float v)
{
  return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

/** clamp */
VECTOR_MATH_API float clamp(const float f, const float a, const float b)
{
  return fmaxf(a, fminf(f, b));
}


/* float2 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
VECTOR_MATH_API float2 make_float2(const float s)
{
  return make_float2(s, s);
}
VECTOR_MATH_API float2 make_float2(const int2& a)
{
  return make_float2(float(a.x), float(a.y));
}
VECTOR_MATH_API float2 make_float2(const uint2& a)
{
  return make_float2(float(a.x), float(a.y));
}
/** @} */

/** negate */
VECTOR_MATH_API float2 operator-(const float2& a)
{
  return make_float2(-a.x, -a.y);
}

/** min
* @{
*/
VECTOR_MATH_API float2 fminf(const float2& a, const float2& b)
{
  return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}
VECTOR_MATH_API float fminf(const float2& a)
{
  return fminf(a.x, a.y);
}
/** @} */

/** max
* @{
*/
VECTOR_MATH_API float2 fmaxf(const float2& a, const float2& b)
{
  return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}
VECTOR_MATH_API float fmaxf(const float2& a)
{
  return fmaxf(a.x, a.y);
}
/** @} */

/** add
* @{
*/
VECTOR_MATH_API float2 operator+(const float2& a, const float2& b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}
VECTOR_MATH_API float2 operator+(const float2& a, const float b)
{
  return make_float2(a.x + b, a.y + b);
}
VECTOR_MATH_API float2 operator+(const float a, const float2& b)
{
  return make_float2(a + b.x, a + b.y);
}
VECTOR_MATH_API void operator+=(float2& a, const float2& b)
{
  a.x += b.x;
  a.y += b.y;
}
/** @} */

/** subtract
* @{
*/
VECTOR_MATH_API float2 operator-(const float2& a, const float2& b)
{
  return make_float2(a.x - b.x, a.y - b.y);
}
VECTOR_MATH_API float2 operator-(const float2& a, const float b)
{
  return make_float2(a.x - b, a.y - b);
}
VECTOR_MATH_API float2 operator-(const float a, const float2& b)
{
  return make_float2(a - b.x, a - b.y);
}
VECTOR_MATH_API void operator-=(float2& a, const float2& b)
{
  a.x -= b.x;
  a.y -= b.y;
}
/** @} */

/** multiply
* @{
*/
VECTOR_MATH_API float2 operator*(const float2& a, const float2& b)
{
  return make_float2(a.x * b.x, a.y * b.y);
}
VECTOR_MATH_API float2 operator*(const float2& a, const float s)
{
  return make_float2(a.x * s, a.y * s);
}
VECTOR_MATH_API float2 operator*(const float s, const float2& a)
{
  return make_float2(a.x * s, a.y * s);
}
VECTOR_MATH_API void operator*=(float2& a, const float2& s)
{
  a.x *= s.x;
  a.y *= s.y;
}
VECTOR_MATH_API void operator*=(float2& a, const float s)
{
  a.x *= s;
  a.y *= s;
}
/** @} */

/** divide
* @{
*/
VECTOR_MATH_API float2 operator/(const float2& a, const float2& b)
{
  return make_float2(a.x / b.x, a.y / b.y);
}
VECTOR_MATH_API float2 operator/(const float2& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
VECTOR_MATH_API float2 operator/(const float s, const float2& a)
{
  return make_float2(s / a.x, s / a.y);
}
VECTOR_MATH_API void operator/=(float2& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

/** lerp */
VECTOR_MATH_API float2 lerp(const float2& a, const float2& b, const float t)
{
  return a + t * (b - a);
}

/** bilerp */
VECTOR_MATH_API float2 bilerp(const float2& x00, const float2& x10, const float2& x01, const float2& x11,
                              const float u, const float v)
{
  return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

/** clamp
* @{
*/
VECTOR_MATH_API float2 clamp(const float2& v, const float a, const float b)
{
  return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

VECTOR_MATH_API float2 clamp(const float2& v, const float2& a, const float2& b)
{
  return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** dot product */
VECTOR_MATH_API float dot(const float2& a, const float2& b)
{
  return a.x * b.x + a.y * b.y;
}

/** length */
VECTOR_MATH_API float length(const float2& v)
{
  return sqrtf(dot(v, v));
}

/** normalize */
VECTOR_MATH_API float2 normalize(const float2& v)
{
  float invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}

/** floor */
VECTOR_MATH_API float2 floor(const float2& v)
{
  return make_float2(::floorf(v.x), ::floorf(v.y));
}

/** reflect */
VECTOR_MATH_API float2 reflect(const float2& i, const float2& n)
{
  return i - 2.0f * n * dot(n, i);
}

/** Faceforward
* Returns N if dot(i, nref) > 0; else -N;
* Typical usage is N = faceforward(N, -ray.dir, N);
* Note that this is opposite of what faceforward does in Cg and GLSL */
VECTOR_MATH_API float2 faceforward(const float2& n, const float2& i, const float2& nref)
{
  return n * copysignf(1.0f, dot(i, nref));
}

/** exp */
VECTOR_MATH_API float2 expf(const float2& v)
{
  return make_float2(::expf(v.x), ::expf(v.y));
}

/** pow **/
VECTOR_MATH_API float2 powf(const float2& v, const float e)
{
  return make_float2(::powf(v.x, e), ::powf(v.y, e));
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API float getByIndex(const float2& v, int i)
{
  return ((float*) (&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API void setByIndex(float2& v, int i, float x)
{
  ((float*) (&v))[i] = x;
}


/* float3 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
VECTOR_MATH_API float3 make_float3(const float s)
{
  return make_float3(s, s, s);
}
VECTOR_MATH_API float3 make_float3(const float2& a)
{
  return make_float3(a.x, a.y, 0.0f);
}
VECTOR_MATH_API float3 make_float3(const int3& a)
{
  return make_float3(float(a.x), float(a.y), float(a.z));
}
VECTOR_MATH_API float3 make_float3(const uint3& a)
{
  return make_float3(float(a.x), float(a.y), float(a.z));
}
/** @} */

/** negate */
VECTOR_MATH_API float3 operator-(const float3& a)
{
  return make_float3(-a.x, -a.y, -a.z);
}

/** min
* @{
*/
VECTOR_MATH_API float3 fminf(const float3& a, const float3& b)
{
  return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
VECTOR_MATH_API float fminf(const float3& a)
{
  return fminf(fminf(a.x, a.y), a.z);
}
/** @} */

/** max
* @{
*/
VECTOR_MATH_API float3 fmaxf(const float3& a, const float3& b)
{
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
VECTOR_MATH_API float fmaxf(const float3& a)
{
  return fmaxf(fmaxf(a.x, a.y), a.z);
}
/** @} */

/** add
* @{
*/
VECTOR_MATH_API float3 operator+(const float3& a, const float3& b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
VECTOR_MATH_API float3 operator+(const float3& a, const float b)
{
  return make_float3(a.x + b, a.y + b, a.z + b);
}
VECTOR_MATH_API float3 operator+(const float a, const float3& b)
{
  return make_float3(a + b.x, a + b.y, a + b.z);
}
VECTOR_MATH_API void operator+=(float3& a, const float3& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
/** @} */

/** subtract
* @{
*/
VECTOR_MATH_API float3 operator-(const float3& a, const float3& b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
VECTOR_MATH_API float3 operator-(const float3& a, const float b)
{
  return make_float3(a.x - b, a.y - b, a.z - b);
}
VECTOR_MATH_API float3 operator-(const float a, const float3& b)
{
  return make_float3(a - b.x, a - b.y, a - b.z);
}
VECTOR_MATH_API void operator-=(float3& a, const float3& b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
VECTOR_MATH_API float3 operator*(const float3& a, const float3& b)
{
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
VECTOR_MATH_API float3 operator*(const float3& a, const float s)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}
VECTOR_MATH_API float3 operator*(const float s, const float3& a)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}
VECTOR_MATH_API void operator*=(float3& a, const float3& s)
{
  a.x *= s.x;
  a.y *= s.y;
  a.z *= s.z;
}
VECTOR_MATH_API void operator*=(float3& a, const float s)
{
  a.x *= s;
  a.y *= s;
  a.z *= s;
}
/** @} */

/** divide
* @{
*/
VECTOR_MATH_API float3 operator/(const float3& a, const float3& b)
{
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
VECTOR_MATH_API float3 operator/(const float3& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
VECTOR_MATH_API float3 operator/(const float s, const float3& a)
{
  return make_float3(s / a.x, s / a.y, s / a.z);
}
VECTOR_MATH_API void operator/=(float3& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

/** lerp */
VECTOR_MATH_API float3 lerp(const float3& a, const float3& b, const float t)
{
  return a + t * (b - a);
}

/** bilerp */
VECTOR_MATH_API float3 bilerp(const float3& x00, const float3& x10, const float3& x01, const float3& x11,
                              const float u, const float v)
{
  return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

/** clamp
* @{
*/
VECTOR_MATH_API float3 clamp(const float3& v, const float a, const float b)
{
  return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

VECTOR_MATH_API float3 clamp(const float3& v, const float3& a, const float3& b)
{
  return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** dot product */
VECTOR_MATH_API float dot(const float3& a, const float3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/** cross product */
VECTOR_MATH_API float3 cross(const float3& a, const float3& b)
{
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

/** length */
VECTOR_MATH_API float length(const float3& v)
{
  return sqrtf(dot(v, v));
}

/** normalize */
VECTOR_MATH_API float3 normalize(const float3& v)
{
  float invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}

/** floor */
VECTOR_MATH_API float3 floor(const float3& v)
{
  return make_float3(::floorf(v.x), ::floorf(v.y), ::floorf(v.z));
}

/** reflect */
VECTOR_MATH_API float3 reflect(const float3& i, const float3& n)
{
  return i - 2.0f * n * dot(n, i);
}

/** Faceforward
* Returns N if dot(i, nref) > 0; else -N;
* Typical usage is N = faceforward(N, -ray.dir, N);
* Note that this is opposite of what faceforward does in Cg and GLSL */
VECTOR_MATH_API float3 faceforward(const float3& n, const float3& i, const float3& nref)
{
  return n * copysignf(1.0f, dot(i, nref));
}

/** exp */
VECTOR_MATH_API float3 expf(const float3& v)
{
  return make_float3(::expf(v.x), ::expf(v.y), ::expf(v.z));
}

/** pow **/
VECTOR_MATH_API float3 powf(const float3& v, const float e)
{
  return make_float3(::powf(v.x, e), ::powf(v.y, e), ::powf(v.z, e));
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API float getByIndex(const float3& v, int i)
{
  return ((float*) (&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API void setByIndex(float3& v, int i, float x)
{
  ((float*) (&v))[i] = x;
}

/* float4 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
VECTOR_MATH_API float4 make_float4(const float s)
{
  return make_float4(s, s, s, s);
}
VECTOR_MATH_API float4 make_float4(const float3& a)
{
  return make_float4(a.x, a.y, a.z, 0.0f);
}
VECTOR_MATH_API float4 make_float4(const int4& a)
{
  return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
VECTOR_MATH_API float4 make_float4(const uint4& a)
{
  return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
/** @} */

/** negate */
VECTOR_MATH_API float4 operator-(const float4& a)
{
  return make_float4(-a.x, -a.y, -a.z, -a.w);
}

/** min
* @{
*/
VECTOR_MATH_API float4 fminf(const float4& a, const float4& b)
{
  return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}
VECTOR_MATH_API float fminf(const float4& a)
{
  return fminf(fminf(a.x, a.y), fminf(a.z, a.w));
}
/** @} */

/** max
* @{
*/
VECTOR_MATH_API float4 fmaxf(const float4& a, const float4& b)
{
  return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}
VECTOR_MATH_API float fmaxf(const float4& a)
{
  return fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w));
}
/** @} */

/** add
* @{
*/
VECTOR_MATH_API float4 operator+(const float4& a, const float4& b)
{
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
VECTOR_MATH_API float4 operator+(const float4& a, const float b)
{
  return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
VECTOR_MATH_API float4 operator+(const float a, const float4& b)
{
  return make_float4(a + b.x, a + b.y, a + b.z, a + b.w);
}
VECTOR_MATH_API void operator+=(float4& a, const float4& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
/** @} */

/** subtract
* @{
*/
VECTOR_MATH_API float4 operator-(const float4& a, const float4& b)
{
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
VECTOR_MATH_API float4 operator-(const float4& a, const float b)
{
  return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}
VECTOR_MATH_API float4 operator-(const float a, const float4& b)
{
  return make_float4(a - b.x, a - b.y, a - b.z, a - b.w);
}
VECTOR_MATH_API void operator-=(float4& a, const float4& b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
}
/** @} */

/** multiply
* @{
*/
VECTOR_MATH_API float4 operator*(const float4& a, const float4& s)
{
  return make_float4(a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w);
}
VECTOR_MATH_API float4 operator*(const float4& a, const float s)
{
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
VECTOR_MATH_API float4 operator*(const float s, const float4& a)
{
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
VECTOR_MATH_API void operator*=(float4& a, const float4& s)
{
  a.x *= s.x; a.y *= s.y; a.z *= s.z; a.w *= s.w;
}
VECTOR_MATH_API void operator*=(float4& a, const float s)
{
  a.x *= s;
  a.y *= s;
  a.z *= s;
  a.w *= s;
}
/** @} */

/** divide
* @{
*/
VECTOR_MATH_API float4 operator/(const float4& a, const float4& b)
{
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
VECTOR_MATH_API float4 operator/(const float4& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
VECTOR_MATH_API float4 operator/(const float s, const float4& a)
{
  return make_float4(s / a.x, s / a.y, s / a.z, s / a.w);
}
VECTOR_MATH_API void operator/=(float4& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

/** lerp */
VECTOR_MATH_API float4 lerp(const float4& a, const float4& b, const float t)
{
  return a + t * (b - a);
}

/** bilerp */
VECTOR_MATH_API float4 bilerp(const float4& x00, const float4& x10, const float4& x01, const float4& x11,
                                          const float u, const float v)
{
  return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

/** clamp
* @{
*/
VECTOR_MATH_API float4 clamp(const float4& v, const float a, const float b)
{
  return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

VECTOR_MATH_API float4 clamp(const float4& v, const float4& a, const float4& b)
{
  return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** dot product */
VECTOR_MATH_API float dot(const float4& a, const float4& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/** length */
VECTOR_MATH_API float length(const float4& r)
{
  return sqrtf(dot(r, r));
}

/** normalize */
VECTOR_MATH_API float4 normalize(const float4& v)
{
  float invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}

/** floor */
VECTOR_MATH_API float4 floor(const float4& v)
{
  return make_float4(::floorf(v.x), ::floorf(v.y), ::floorf(v.z), ::floorf(v.w));
}

/** reflect */
VECTOR_MATH_API float4 reflect(const float4& i, const float4& n)
{
  return i - 2.0f * n * dot(n, i);
}

/**
* Faceforward
* Returns N if dot(i, nref) > 0; else -N;
* Typical usage is N = faceforward(N, -ray.dir, N);
* Note that this is opposite of what faceforward does in Cg and GLSL
*/
VECTOR_MATH_API float4 faceforward(const float4& n, const float4& i, const float4& nref)
{
  return n * copysignf(1.0f, dot(i, nref));
}

/** exp */
VECTOR_MATH_API float4 expf(const float4& v)
{
  return make_float4(::expf(v.x), ::expf(v.y), ::expf(v.z), ::expf(v.w));
}

/** pow */
VECTOR_MATH_API float4 powf(const float4& v, const float e)
{
  return make_float4(::powf(v.x, e), ::powf(v.y, e), ::powf(v.z, e), ::powf(v.w, e));
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API float getByIndex(const float4& v, int i)
{
  return ((float*) (&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API void setByIndex(float4& v, int i, float x)
{
  ((float*) (&v))[i] = x;
}


/* int functions */
/******************************************************************************/

/** clamp */
VECTOR_MATH_API int clamp(const int f, const int a, const int b)
{
  return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API int getByIndex(const int1& v, int i)
{
  return ((int*) (&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API void setByIndex(int1& v, int i, int x)
{
  ((int*) (&v))[i] = x;
}


/* int2 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
VECTOR_MATH_API int2 make_int2(const int s)
{
  return make_int2(s, s);
}
VECTOR_MATH_API int2 make_int2(const float2& a)
{
  return make_int2(int(a.x), int(a.y));
}
/** @} */

/** negate */
VECTOR_MATH_API int2 operator-(const int2& a)
{
  return make_int2(-a.x, -a.y);
}

/** min */
VECTOR_MATH_API int2 min(const int2& a, const int2& b)
{
  return make_int2(min(a.x, b.x), min(a.y, b.y));
}

/** max */
VECTOR_MATH_API int2 max(const int2& a, const int2& b)
{
  return make_int2(max(a.x, b.x), max(a.y, b.y));
}

/** add
* @{
*/
VECTOR_MATH_API int2 operator+(const int2& a, const int2& b)
{
  return make_int2(a.x + b.x, a.y + b.y);
}
VECTOR_MATH_API void operator+=(int2& a, const int2& b)
{
  a.x += b.x;
  a.y += b.y;
}
/** @} */

/** subtract
* @{
*/
VECTOR_MATH_API int2 operator-(const int2& a, const int2& b)
{
  return make_int2(a.x - b.x, a.y - b.y);
}
VECTOR_MATH_API int2 operator-(const int2& a, const int b)
{
  return make_int2(a.x - b, a.y - b);
}
VECTOR_MATH_API void operator-=(int2& a, const int2& b)
{
  a.x -= b.x;
  a.y -= b.y;
}
/** @} */

/** multiply
* @{
*/
VECTOR_MATH_API int2 operator*(const int2& a, const int2& b)
{
  return make_int2(a.x * b.x, a.y * b.y);
}
VECTOR_MATH_API int2 operator*(const int2& a, const int s)
{
  return make_int2(a.x * s, a.y * s);
}
VECTOR_MATH_API int2 operator*(const int s, const int2& a)
{
  return make_int2(a.x * s, a.y * s);
}
VECTOR_MATH_API void operator*=(int2& a, const int s)
{
  a.x *= s;
  a.y *= s;
}
/** @} */

/** clamp
* @{
*/
VECTOR_MATH_API int2 clamp(const int2& v, const int a, const int b)
{
  return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}

VECTOR_MATH_API int2 clamp(const int2& v, const int2& a, const int2& b)
{
  return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality
* @{
*/
VECTOR_MATH_API bool operator==(const int2& a, const int2& b)
{
  return a.x == b.x && a.y == b.y;
}

VECTOR_MATH_API bool operator!=(const int2& a, const int2& b)
{
  return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API int getByIndex(const int2& v, int i)
{
  return ((int*) (&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API void setByIndex(int2& v, int i, int x)
{
  ((int*) (&v))[i] = x;
}


/* int3 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
VECTOR_MATH_API int3 make_int3(const int s)
{
  return make_int3(s, s, s);
}
VECTOR_MATH_API int3 make_int3(const float3& a)
{
  return make_int3(int(a.x), int(a.y), int(a.z));
}
/** @} */

/** negate */
VECTOR_MATH_API int3 operator-(const int3& a)
{
  return make_int3(-a.x, -a.y, -a.z);
}

/** min */
VECTOR_MATH_API int3 min(const int3& a, const int3& b)
{
  return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

/** max */
VECTOR_MATH_API int3 max(const int3& a, const int3& b)
{
  return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

/** add
* @{
*/
VECTOR_MATH_API int3 operator+(const int3& a, const int3& b)
{
  return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
VECTOR_MATH_API void operator+=(int3& a, const int3& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
/** @} */

/** subtract
* @{
*/
VECTOR_MATH_API int3 operator-(const int3& a, const int3& b)
{
  return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

VECTOR_MATH_API void operator-=(int3& a, const int3& b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
VECTOR_MATH_API int3 operator*(const int3& a, const int3& b)
{
  return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
VECTOR_MATH_API int3 operator*(const int3& a, const int s)
{
  return make_int3(a.x * s, a.y * s, a.z * s);
}
VECTOR_MATH_API int3 operator*(const int s, const int3& a)
{
  return make_int3(a.x * s, a.y * s, a.z * s);
}
VECTOR_MATH_API void operator*=(int3& a, const int s)
{
  a.x *= s;
  a.y *= s;
  a.z *= s;
}
/** @} */

/** divide
* @{
*/
VECTOR_MATH_API int3 operator/(const int3& a, const int3& b)
{
  return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}
VECTOR_MATH_API int3 operator/(const int3& a, const int s)
{
  return make_int3(a.x / s, a.y / s, a.z / s);
}
VECTOR_MATH_API int3 operator/(const int s, const int3& a)
{
  return make_int3(s / a.x, s / a.y, s / a.z);
}
VECTOR_MATH_API void operator/=(int3& a, const int s)
{
  a.x /= s;
  a.y /= s;
  a.z /= s;
}
/** @} */

/** clamp
* @{
*/
VECTOR_MATH_API int3 clamp(const int3& v, const int a, const int b)
{
  return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

VECTOR_MATH_API int3 clamp(const int3& v, const int3& a, const int3& b)
{
  return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality
* @{
*/
VECTOR_MATH_API bool operator==(const int3& a, const int3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

VECTOR_MATH_API bool operator!=(const int3& a, const int3& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API int getByIndex(const int3& v, int i)
{
  return ((int*) (&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API void setByIndex(int3& v, int i, int x)
{
  ((int*) (&v))[i] = x;
}


/* int4 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
VECTOR_MATH_API int4 make_int4(const int s)
{
  return make_int4(s, s, s, s);
}
VECTOR_MATH_API int4 make_int4(const float4& a)
{
  return make_int4((int) a.x, (int) a.y, (int) a.z, (int) a.w);
}
/** @} */

/** negate */
VECTOR_MATH_API int4 operator-(const int4& a)
{
  return make_int4(-a.x, -a.y, -a.z, -a.w);
}

/** min */
VECTOR_MATH_API int4 min(const int4& a, const int4& b)
{
  return make_int4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

/** max */
VECTOR_MATH_API int4 max(const int4& a, const int4& b)
{
  return make_int4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

/** add
* @{
*/
VECTOR_MATH_API int4 operator+(const int4& a, const int4& b)
{
  return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
VECTOR_MATH_API void operator+=(int4& a, const int4& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
/** @} */

/** subtract
* @{
*/
VECTOR_MATH_API int4 operator-(const int4& a, const int4& b)
{
  return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

VECTOR_MATH_API void operator-=(int4& a, const int4& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply
* @{
*/
VECTOR_MATH_API int4 operator*(const int4& a, const int4& b)
{
  return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
VECTOR_MATH_API int4 operator*(const int4& a, const int s)
{
  return make_int4(a.x * s, a.y * s, a.z * s, a.w * s);
}
VECTOR_MATH_API int4 operator*(const int s, const int4& a)
{
  return make_int4(a.x * s, a.y * s, a.z * s, a.w * s);
}
VECTOR_MATH_API void operator*=(int4& a, const int s)
{
  a.x *= s;
  a.y *= s;
  a.z *= s;
  a.w *= s;
}
/** @} */

/** divide
* @{
*/
VECTOR_MATH_API int4 operator/(const int4& a, const int4& b)
{
  return make_int4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
VECTOR_MATH_API int4 operator/(const int4& a, const int s)
{
  return make_int4(a.x / s, a.y / s, a.z / s, a.w / s);
}
VECTOR_MATH_API int4 operator/(const int s, const int4& a)
{
  return make_int4(s / a.x, s / a.y, s / a.z, s / a.w);
}
VECTOR_MATH_API void operator/=(int4& a, const int s)
{
  a.x /= s;
  a.y /= s;
  a.z /= s;
  a.w /= s;
}
/** @} */

/** clamp
* @{
*/
VECTOR_MATH_API int4 clamp(const int4& v, const int a, const int b)
{
  return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

VECTOR_MATH_API int4 clamp(const int4& v, const int4& a, const int4& b)
{
  return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality
* @{
*/
VECTOR_MATH_API bool operator==(const int4& a, const int4& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

VECTOR_MATH_API bool operator!=(const int4& a, const int4& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API int getByIndex(const int4& v, int i)
{
  return ((int*) (&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API void setByIndex(int4& v, int i, int x)
{
  ((int*) (&v))[i] = x;
}


/* uint functions */
/******************************************************************************/

/** clamp */
VECTOR_MATH_API unsigned int clamp(const unsigned int f, const unsigned int a, const unsigned int b)
{
  return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API unsigned int getByIndex(const uint1& v, unsigned int i)
{
  return ((unsigned int*) (&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API void setByIndex(uint1& v, int i, unsigned int x)
{
  ((unsigned int*) (&v))[i] = x;
}


/* uint2 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
VECTOR_MATH_API uint2 make_uint2(const unsigned int s)
{
  return make_uint2(s, s);
}
VECTOR_MATH_API uint2 make_uint2(const float2& a)
{
  return make_uint2((unsigned int) a.x, (unsigned int) a.y);
}
/** @} */

/** min */
VECTOR_MATH_API uint2 min(const uint2& a, const uint2& b)
{
  return make_uint2(min(a.x, b.x), min(a.y, b.y));
}

/** max */
VECTOR_MATH_API uint2 max(const uint2& a, const uint2& b)
{
  return make_uint2(max(a.x, b.x), max(a.y, b.y));
}

/** add
* @{
*/
VECTOR_MATH_API uint2 operator+(const uint2& a, const uint2& b)
{
  return make_uint2(a.x + b.x, a.y + b.y);
}
VECTOR_MATH_API void operator+=(uint2& a, const uint2& b)
{
  a.x += b.x;
  a.y += b.y;
}
/** @} */

/** subtract
* @{
*/
VECTOR_MATH_API uint2 operator-(const uint2& a, const uint2& b)
{
  return make_uint2(a.x - b.x, a.y - b.y);
}
VECTOR_MATH_API uint2 operator-(const uint2& a, const unsigned int b)
{
  return make_uint2(a.x - b, a.y - b);
}
VECTOR_MATH_API void operator-=(uint2& a, const uint2& b)
{
  a.x -= b.x;
  a.y -= b.y;
}
/** @} */

/** multiply
* @{
*/
VECTOR_MATH_API uint2 operator*(const uint2& a, const uint2& b)
{
  return make_uint2(a.x * b.x, a.y * b.y);
}
VECTOR_MATH_API uint2 operator*(const uint2& a, const unsigned int s)
{
  return make_uint2(a.x * s, a.y * s);
}
VECTOR_MATH_API uint2 operator*(const unsigned int s, const uint2& a)
{
  return make_uint2(a.x * s, a.y * s);
}
VECTOR_MATH_API void operator*=(uint2& a, const unsigned int s)
{
  a.x *= s;
  a.y *= s;
}
/** @} */

/** clamp
* @{
*/
VECTOR_MATH_API uint2 clamp(const uint2& v, const unsigned int a, const unsigned int b)
{
  return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}

VECTOR_MATH_API uint2 clamp(const uint2& v, const uint2& a, const uint2& b)
{
  return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality
* @{
*/
VECTOR_MATH_API bool operator==(const uint2& a, const uint2& b)
{
  return a.x == b.x && a.y == b.y;
}

VECTOR_MATH_API bool operator!=(const uint2& a, const uint2& b)
{
  return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API unsigned int getByIndex(const uint2& v, unsigned int i)
{
  return ((unsigned int*) (&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API void setByIndex(uint2& v, int i, unsigned int x)
{
  ((unsigned int*) (&v))[i] = x;
}


/* uint3 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
VECTOR_MATH_API uint3 make_uint3(const unsigned int s)
{
  return make_uint3(s, s, s);
}
VECTOR_MATH_API uint3 make_uint3(const float3& a)
{
  return make_uint3((unsigned int) a.x, (unsigned int) a.y, (unsigned int) a.z);
}
/** @} */

/** min */
VECTOR_MATH_API uint3 min(const uint3& a, const uint3& b)
{
  return make_uint3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

/** max */
VECTOR_MATH_API uint3 max(const uint3& a, const uint3& b)
{
  return make_uint3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

/** add
* @{
*/
VECTOR_MATH_API uint3 operator+(const uint3& a, const uint3& b)
{
  return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
VECTOR_MATH_API void operator+=(uint3& a, const uint3& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
/** @} */

/** subtract
* @{
*/
VECTOR_MATH_API uint3 operator-(const uint3& a, const uint3& b)
{
  return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

VECTOR_MATH_API void operator-=(uint3& a, const uint3& b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
VECTOR_MATH_API uint3 operator*(const uint3& a, const uint3& b)
{
  return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
VECTOR_MATH_API uint3 operator*(const uint3& a, const unsigned int s)
{
  return make_uint3(a.x * s, a.y * s, a.z * s);
}
VECTOR_MATH_API uint3 operator*(const unsigned int s, const uint3& a)
{
  return make_uint3(a.x * s, a.y * s, a.z * s);
}
VECTOR_MATH_API void operator*=(uint3& a, const unsigned int s)
{
  a.x *= s;
  a.y *= s;
  a.z *= s;
}
/** @} */

/** divide
* @{
*/
VECTOR_MATH_API uint3 operator/(const uint3& a, const uint3& b)
{
  return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}
VECTOR_MATH_API uint3 operator/(const uint3& a, const unsigned int s)
{
  return make_uint3(a.x / s, a.y / s, a.z / s);
}
VECTOR_MATH_API uint3 operator/(const unsigned int s, const uint3& a)
{
  return make_uint3(s / a.x, s / a.y, s / a.z);
}
VECTOR_MATH_API void operator/=(uint3& a, const unsigned int s)
{
  a.x /= s;
  a.y /= s;
  a.z /= s;
}
/** @} */

/** clamp
* @{
*/
VECTOR_MATH_API uint3 clamp(const uint3& v, const unsigned int a, const unsigned int b)
{
  return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

VECTOR_MATH_API uint3 clamp(const uint3& v, const uint3& a, const uint3& b)
{
  return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality
* @{
*/
VECTOR_MATH_API bool operator==(const uint3& a, const uint3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

VECTOR_MATH_API bool operator!=(const uint3& a, const uint3& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory
*/
VECTOR_MATH_API unsigned int getByIndex(const uint3& v, unsigned int i)
{
  return ((unsigned int*) (&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory
*/
VECTOR_MATH_API void setByIndex(uint3& v, int i, unsigned int x)
{
  ((unsigned int*) (&v))[i] = x;
}


/* uint4 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
VECTOR_MATH_API uint4 make_uint4(const unsigned int s)
{
  return make_uint4(s, s, s, s);
}
VECTOR_MATH_API uint4 make_uint4(const float4& a)
{
  return make_uint4((unsigned int) a.x, (unsigned int) a.y, (unsigned int) a.z, (unsigned int) a.w);
}
/** @} */

/** min
* @{
*/
VECTOR_MATH_API uint4 min(const uint4& a, const uint4& b)
{
  return make_uint4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}
/** @} */

/** max
* @{
*/
VECTOR_MATH_API uint4 max(const uint4& a, const uint4& b)
{
  return make_uint4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}
/** @} */

/** add
* @{
*/
VECTOR_MATH_API uint4 operator+(const uint4& a, const uint4& b)
{
  return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
VECTOR_MATH_API void operator+=(uint4& a, const uint4& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
/** @} */

/** subtract
* @{
*/
VECTOR_MATH_API uint4 operator-(const uint4& a, const uint4& b)
{
  return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

VECTOR_MATH_API void operator-=(uint4& a, const uint4& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply
* @{
*/
VECTOR_MATH_API uint4 operator*(const uint4& a, const uint4& b)
{
  return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
VECTOR_MATH_API uint4 operator*(const uint4& a, const unsigned int s)
{
  return make_uint4(a.x * s, a.y * s, a.z * s, a.w * s);
}
VECTOR_MATH_API uint4 operator*(const unsigned int s, const uint4& a)
{
  return make_uint4(a.x * s, a.y * s, a.z * s, a.w * s);
}
VECTOR_MATH_API void operator*=(uint4& a, const unsigned int s)
{
  a.x *= s;
  a.y *= s;
  a.z *= s;
  a.w *= s;
}
/** @} */

/** divide
* @{
*/
VECTOR_MATH_API uint4 operator/(const uint4& a, const uint4& b)
{
  return make_uint4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
VECTOR_MATH_API uint4 operator/(const uint4& a, const unsigned int s)
{
  return make_uint4(a.x / s, a.y / s, a.z / s, a.w / s);
}
VECTOR_MATH_API uint4 operator/(const unsigned int s, const uint4& a)
{
  return make_uint4(s / a.x, s / a.y, s / a.z, s / a.w);
}
VECTOR_MATH_API void operator/=(uint4& a, const unsigned int s)
{
  a.x /= s;
  a.y /= s;
  a.z /= s;
  a.w /= s;
}
/** @} */

/** clamp
* @{
*/
VECTOR_MATH_API uint4 clamp(const uint4& v, const unsigned int a, const unsigned int b)
{
  return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

VECTOR_MATH_API uint4 clamp(const uint4& v, const uint4& a, const uint4& b)
{
  return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality
* @{
*/
VECTOR_MATH_API bool operator==(const uint4& a, const uint4& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

VECTOR_MATH_API bool operator!=(const uint4& a, const uint4& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory
*/
VECTOR_MATH_API unsigned int getByIndex(const uint4& v, unsigned int i)
{
  return ((unsigned int*) (&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory
*/
VECTOR_MATH_API void setByIndex(uint4& v, int i, unsigned int x)
{
  ((unsigned int*) (&v))[i] = x;
}

/* long long functions */
/******************************************************************************/

/** clamp */
VECTOR_MATH_API long long clamp(const long long f, const long long a, const long long b)
{
  return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API long long getByIndex(const longlong1& v, int i)
{
  return ((long long*) (&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API void setByIndex(longlong1& v, int i, long long x)
{
  ((long long*) (&v))[i] = x;
}


/* longlong2 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
VECTOR_MATH_API longlong2 make_longlong2(const long long s)
{
  return make_longlong2(s, s);
}
VECTOR_MATH_API longlong2 make_longlong2(const float2& a)
{
  return make_longlong2(int(a.x), int(a.y));
}
/** @} */

/** negate */
VECTOR_MATH_API longlong2 operator-(const longlong2& a)
{
  return make_longlong2(-a.x, -a.y);
}

/** min */
VECTOR_MATH_API longlong2 min(const longlong2& a, const longlong2& b)
{
  return make_longlong2(min(a.x, b.x), min(a.y, b.y));
}

/** max */
VECTOR_MATH_API longlong2 max(const longlong2& a, const longlong2& b)
{
  return make_longlong2(max(a.x, b.x), max(a.y, b.y));
}

/** add
* @{
*/
VECTOR_MATH_API longlong2 operator+(const longlong2& a, const longlong2& b)
{
  return make_longlong2(a.x + b.x, a.y + b.y);
}
VECTOR_MATH_API void operator+=(longlong2& a, const longlong2& b)
{
  a.x += b.x;
  a.y += b.y;
}
/** @} */

/** subtract
* @{
*/
VECTOR_MATH_API longlong2 operator-(const longlong2& a, const longlong2& b)
{
  return make_longlong2(a.x - b.x, a.y - b.y);
}
VECTOR_MATH_API longlong2 operator-(const longlong2& a, const long long b)
{
  return make_longlong2(a.x - b, a.y - b);
}
VECTOR_MATH_API void operator-=(longlong2& a, const longlong2& b)
{
  a.x -= b.x;
  a.y -= b.y;
}
/** @} */

/** multiply
* @{
*/
VECTOR_MATH_API longlong2 operator*(const longlong2& a, const longlong2& b)
{
  return make_longlong2(a.x * b.x, a.y * b.y);
}
VECTOR_MATH_API longlong2 operator*(const longlong2& a, const long long s)
{
  return make_longlong2(a.x * s, a.y * s);
}
VECTOR_MATH_API longlong2 operator*(const long long s, const longlong2& a)
{
  return make_longlong2(a.x * s, a.y * s);
}
VECTOR_MATH_API void operator*=(longlong2& a, const long long s)
{
  a.x *= s;
  a.y *= s;
}
/** @} */

/** clamp
* @{
*/
VECTOR_MATH_API longlong2 clamp(const longlong2& v, const long long a, const long long b)
{
  return make_longlong2(clamp(v.x, a, b), clamp(v.y, a, b));
}

VECTOR_MATH_API longlong2 clamp(const longlong2& v, const longlong2& a, const longlong2& b)
{
  return make_longlong2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality
* @{
*/
VECTOR_MATH_API bool operator==(const longlong2& a, const longlong2& b)
{
  return a.x == b.x && a.y == b.y;
}

VECTOR_MATH_API bool operator!=(const longlong2& a, const longlong2& b)
{
  return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API long long getByIndex(const longlong2& v, int i)
{
  return ((long long*) (&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API void setByIndex(longlong2& v, int i, long long x)
{
  ((long long*) (&v))[i] = x;
}


/* longlong3 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
VECTOR_MATH_API longlong3 make_longlong3(const long long s)
{
  return make_longlong3(s, s, s);
}
VECTOR_MATH_API longlong3 make_longlong3(const float3& a)
{
  return make_longlong3((long long) a.x, (long long) a.y, (long long) a.z);
}
/** @} */

/** negate */
VECTOR_MATH_API longlong3 operator-(const longlong3& a)
{
  return make_longlong3(-a.x, -a.y, -a.z);
}

/** min */
VECTOR_MATH_API longlong3 min(const longlong3& a, const longlong3& b)
{
  return make_longlong3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

/** max */
VECTOR_MATH_API longlong3 max(const longlong3& a, const longlong3& b)
{
  return make_longlong3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

/** add
* @{
*/
VECTOR_MATH_API longlong3 operator+(const longlong3& a, const longlong3& b)
{
  return make_longlong3(a.x + b.x, a.y + b.y, a.z + b.z);
}
VECTOR_MATH_API void operator+=(longlong3& a, const longlong3& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
/** @} */

/** subtract
* @{
*/
VECTOR_MATH_API longlong3 operator-(const longlong3& a, const longlong3& b)
{
  return make_longlong3(a.x - b.x, a.y - b.y, a.z - b.z);
}

VECTOR_MATH_API void operator-=(longlong3& a, const longlong3& b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
VECTOR_MATH_API longlong3 operator*(const longlong3& a, const longlong3& b)
{
  return make_longlong3(a.x * b.x, a.y * b.y, a.z * b.z);
}
VECTOR_MATH_API longlong3 operator*(const longlong3& a, const long long s)
{
  return make_longlong3(a.x * s, a.y * s, a.z * s);
}
VECTOR_MATH_API longlong3 operator*(const long long s, const longlong3& a)
{
  return make_longlong3(a.x * s, a.y * s, a.z * s);
}
VECTOR_MATH_API void operator*=(longlong3& a, const long long s)
{
  a.x *= s;
  a.y *= s;
  a.z *= s;
}
/** @} */

/** divide
* @{
*/
VECTOR_MATH_API longlong3 operator/(const longlong3& a, const longlong3& b)
{
  return make_longlong3(a.x / b.x, a.y / b.y, a.z / b.z);
}
VECTOR_MATH_API longlong3 operator/(const longlong3& a, const long long s)
{
  return make_longlong3(a.x / s, a.y / s, a.z / s);
}
VECTOR_MATH_API longlong3 operator/(const long long s, const longlong3& a)
{
  return make_longlong3(s / a.x, s / a.y, s / a.z);
}
VECTOR_MATH_API void operator/=(longlong3& a, const long long s)
{
  a.x /= s;
  a.y /= s;
  a.z /= s;
}
/** @} */

/** clamp
* @{
*/
VECTOR_MATH_API longlong3 clamp(const longlong3& v, const long long a, const long long b)
{
  return make_longlong3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

VECTOR_MATH_API longlong3 clamp(const longlong3& v, const longlong3& a, const longlong3& b)
{
  return make_longlong3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality
* @{
*/
VECTOR_MATH_API bool operator==(const longlong3& a, const longlong3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

VECTOR_MATH_API bool operator!=(const longlong3& a, const longlong3& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API long long getByIndex(const longlong3& v, int i)
{
  return ((long long*) (&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API void setByIndex(longlong3& v, int i, int x)
{
  ((long long*) (&v))[i] = x;
}


/* longlong4 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
VECTOR_MATH_API longlong4 make_longlong4(const long long s)
{
  return make_longlong4(s, s, s, s);
}
VECTOR_MATH_API longlong4 make_longlong4(const float4& a)
{
  return make_longlong4((long long) a.x, (long long) a.y, (long long) a.z, (long long) a.w);
}
/** @} */

/** negate */
VECTOR_MATH_API longlong4 operator-(const longlong4& a)
{
  return make_longlong4(-a.x, -a.y, -a.z, -a.w);
}

/** min */
VECTOR_MATH_API longlong4 min(const longlong4& a, const longlong4& b)
{
  return make_longlong4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

/** max */
VECTOR_MATH_API longlong4 max(const longlong4& a, const longlong4& b)
{
  return make_longlong4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

/** add
* @{
*/
VECTOR_MATH_API longlong4 operator+(const longlong4& a, const longlong4& b)
{
  return make_longlong4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
VECTOR_MATH_API void operator+=(longlong4& a, const longlong4& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
/** @} */

/** subtract
* @{
*/
VECTOR_MATH_API longlong4 operator-(const longlong4& a, const longlong4& b)
{
  return make_longlong4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

VECTOR_MATH_API void operator-=(longlong4& a, const longlong4& b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
}
/** @} */

/** multiply
* @{
*/
VECTOR_MATH_API longlong4 operator*(const longlong4& a, const longlong4& b)
{
  return make_longlong4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
VECTOR_MATH_API longlong4 operator*(const longlong4& a, const long long s)
{
  return make_longlong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
VECTOR_MATH_API longlong4 operator*(const long long s, const longlong4& a)
{
  return make_longlong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
VECTOR_MATH_API void operator*=(longlong4& a, const long long s)
{
  a.x *= s;
  a.y *= s;
  a.z *= s;
  a.w *= s;
}
/** @} */

/** divide
* @{
*/
VECTOR_MATH_API longlong4 operator/(const longlong4& a, const longlong4& b)
{
  return make_longlong4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
VECTOR_MATH_API longlong4 operator/(const longlong4& a, const long long s)
{
  return make_longlong4(a.x / s, a.y / s, a.z / s, a.w / s);
}
VECTOR_MATH_API longlong4 operator/(const long long s, const longlong4& a)
{
  return make_longlong4(s / a.x, s / a.y, s / a.z, s / a.w);
}
VECTOR_MATH_API void operator/=(longlong4& a, const long long s)
{
  a.x /= s;
  a.y /= s;
  a.z /= s;
  a.w /= s;
}
/** @} */

/** clamp
* @{
*/
VECTOR_MATH_API longlong4 clamp(const longlong4& v, const long long a, const long long b)
{
  return make_longlong4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

VECTOR_MATH_API longlong4 clamp(const longlong4& v, const longlong4& a, const longlong4& b)
{
  return make_longlong4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality
* @{
*/
VECTOR_MATH_API bool operator==(const longlong4& a, const longlong4& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

VECTOR_MATH_API bool operator!=(const longlong4& a, const longlong4& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API long long getByIndex(const longlong4& v, int i)
{
  return ((long long*) (&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API void setByIndex(longlong4& v, int i, long long x)
{
  ((long long*) (&v))[i] = x;
}

/* ulonglong functions */
/******************************************************************************/

/** clamp */
VECTOR_MATH_API unsigned long long clamp(const unsigned long long f, const unsigned long long a, const unsigned long long b)
{
  return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API unsigned long long getByIndex(const ulonglong1& v, unsigned int i)
{
  return ((unsigned long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API void setByIndex(ulonglong1& v, int i, unsigned long long x)
{
  ((unsigned long long*)(&v))[i] = x;
}


/* ulonglong2 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
VECTOR_MATH_API ulonglong2 make_ulonglong2(const unsigned long long s)
{
  return make_ulonglong2(s, s);
}
VECTOR_MATH_API ulonglong2 make_ulonglong2(const float2& a)
{
  return make_ulonglong2((unsigned long long)a.x, (unsigned long long)a.y);
}
/** @} */

/** min */
VECTOR_MATH_API ulonglong2 min(const ulonglong2& a, const ulonglong2& b)
{
  return make_ulonglong2(min(a.x, b.x), min(a.y, b.y));
}

/** max */
VECTOR_MATH_API ulonglong2 max(const ulonglong2& a, const ulonglong2& b)
{
  return make_ulonglong2(max(a.x, b.x), max(a.y, b.y));
}

/** add
* @{
*/
VECTOR_MATH_API ulonglong2 operator+(const ulonglong2& a, const ulonglong2& b)
{
  return make_ulonglong2(a.x + b.x, a.y + b.y);
}
VECTOR_MATH_API void operator+=(ulonglong2& a, const ulonglong2& b)
{
  a.x += b.x;
  a.y += b.y;
}
/** @} */

/** subtract
* @{
*/
VECTOR_MATH_API ulonglong2 operator-(const ulonglong2& a, const ulonglong2& b)
{
  return make_ulonglong2(a.x - b.x, a.y - b.y);
}
VECTOR_MATH_API ulonglong2 operator-(const ulonglong2& a, const unsigned long long b)
{
  return make_ulonglong2(a.x - b, a.y - b);
}
VECTOR_MATH_API void operator-=(ulonglong2& a, const ulonglong2& b)
{
  a.x -= b.x;
  a.y -= b.y;
}
/** @} */

/** multiply
* @{
*/
VECTOR_MATH_API ulonglong2 operator*(const ulonglong2& a, const ulonglong2& b)
{
  return make_ulonglong2(a.x * b.x, a.y * b.y);
}
VECTOR_MATH_API ulonglong2 operator*(const ulonglong2& a, const unsigned long long s)
{
  return make_ulonglong2(a.x * s, a.y * s);
}
VECTOR_MATH_API ulonglong2 operator*(const unsigned long long s, const ulonglong2& a)
{
  return make_ulonglong2(a.x * s, a.y * s);
}
VECTOR_MATH_API void operator*=(ulonglong2& a, const unsigned long long s)
{
  a.x *= s;
  a.y *= s;
}
/** @} */

/** clamp
* @{
*/
VECTOR_MATH_API ulonglong2 clamp(const ulonglong2& v, const unsigned long long a, const unsigned long long b)
{
  return make_ulonglong2(clamp(v.x, a, b), clamp(v.y, a, b));
}

VECTOR_MATH_API ulonglong2 clamp(const ulonglong2& v, const ulonglong2& a, const ulonglong2& b)
{
  return make_ulonglong2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality
* @{
*/
VECTOR_MATH_API bool operator==(const ulonglong2& a, const ulonglong2& b)
{
  return a.x == b.x && a.y == b.y;
}

VECTOR_MATH_API bool operator!=(const ulonglong2& a, const ulonglong2& b)
{
  return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API unsigned long long getByIndex(const ulonglong2& v, unsigned int i)
{
  return ((unsigned long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
VECTOR_MATH_API void setByIndex(ulonglong2& v, int i, unsigned long long x)
{
  ((unsigned long long*)(&v))[i] = x;
}


/* ulonglong3 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
VECTOR_MATH_API ulonglong3 make_ulonglong3(const unsigned long long s)
{
  return make_ulonglong3(s, s, s);
}
VECTOR_MATH_API ulonglong3 make_ulonglong3(const float3& a)
{
  return make_ulonglong3((unsigned long long)a.x, (unsigned long long)a.y, (unsigned long long)a.z);
}
/** @} */

/** min */
VECTOR_MATH_API ulonglong3 min(const ulonglong3& a, const ulonglong3& b)
{
  return make_ulonglong3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

/** max */
VECTOR_MATH_API ulonglong3 max(const ulonglong3& a, const ulonglong3& b)
{
  return make_ulonglong3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

/** add
* @{
*/
VECTOR_MATH_API ulonglong3 operator+(const ulonglong3& a, const ulonglong3& b)
{
  return make_ulonglong3(a.x + b.x, a.y + b.y, a.z + b.z);
}
VECTOR_MATH_API void operator+=(ulonglong3& a, const ulonglong3& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
/** @} */

/** subtract
* @{
*/
VECTOR_MATH_API ulonglong3 operator-(const ulonglong3& a, const ulonglong3& b)
{
  return make_ulonglong3(a.x - b.x, a.y - b.y, a.z - b.z);
}

VECTOR_MATH_API void operator-=(ulonglong3& a, const ulonglong3& b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
VECTOR_MATH_API ulonglong3 operator*(const ulonglong3& a, const ulonglong3& b)
{
  return make_ulonglong3(a.x * b.x, a.y * b.y, a.z * b.z);
}
VECTOR_MATH_API ulonglong3 operator*(const ulonglong3& a, const unsigned long long s)
{
  return make_ulonglong3(a.x * s, a.y * s, a.z * s);
}
VECTOR_MATH_API ulonglong3 operator*(const unsigned long long s, const ulonglong3& a)
{
  return make_ulonglong3(a.x * s, a.y * s, a.z * s);
}
VECTOR_MATH_API void operator*=(ulonglong3& a, const unsigned long long s)
{
  a.x *= s;
  a.y *= s;
  a.z *= s;
}
/** @} */

/** divide
* @{
*/
VECTOR_MATH_API ulonglong3 operator/(const ulonglong3& a, const ulonglong3& b)
{
  return make_ulonglong3(a.x / b.x, a.y / b.y, a.z / b.z);
}
VECTOR_MATH_API ulonglong3 operator/(const ulonglong3& a, const unsigned long long s)
{
  return make_ulonglong3(a.x / s, a.y / s, a.z / s);
}
VECTOR_MATH_API ulonglong3 operator/(const unsigned long long s, const ulonglong3& a)
{
  return make_ulonglong3(s / a.x, s / a.y, s / a.z);
}
VECTOR_MATH_API void operator/=(ulonglong3& a, const unsigned long long s)
{
  a.x /= s;
  a.y /= s;
  a.z /= s;
}
/** @} */

/** clamp
* @{
*/
VECTOR_MATH_API ulonglong3 clamp(const ulonglong3& v, const unsigned long long a, const unsigned long long b)
{
  return make_ulonglong3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

VECTOR_MATH_API ulonglong3 clamp(const ulonglong3& v, const ulonglong3& a, const ulonglong3& b)
{
  return make_ulonglong3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality
* @{
*/
VECTOR_MATH_API bool operator==(const ulonglong3& a, const ulonglong3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

VECTOR_MATH_API bool operator!=(const ulonglong3& a, const ulonglong3& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory
*/
VECTOR_MATH_API unsigned long long getByIndex(const ulonglong3& v, unsigned int i)
{
  return ((unsigned long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory
*/
VECTOR_MATH_API void setByIndex(ulonglong3& v, int i, unsigned long long x)
{
  ((unsigned long long*)(&v))[i] = x;
}


/* ulonglong4 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
VECTOR_MATH_API ulonglong4 make_ulonglong4(const unsigned long long s)
{
  return make_ulonglong4(s, s, s, s);
}
VECTOR_MATH_API ulonglong4 make_ulonglong4(const float4& a)
{
  return make_ulonglong4((unsigned long long)a.x, (unsigned long long)a.y, (unsigned long long)a.z, (unsigned long long)a.w);
}
/** @} */

/** min
* @{
*/
VECTOR_MATH_API ulonglong4 min(const ulonglong4& a, const ulonglong4& b)
{
  return make_ulonglong4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}
/** @} */

/** max
* @{
*/
VECTOR_MATH_API ulonglong4 max(const ulonglong4& a, const ulonglong4& b)
{
  return make_ulonglong4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}
/** @} */

/** add
* @{
*/
VECTOR_MATH_API ulonglong4 operator+(const ulonglong4& a, const ulonglong4& b)
{
  return make_ulonglong4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
VECTOR_MATH_API void operator+=(ulonglong4& a, const ulonglong4& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
/** @} */

/** subtract
* @{
*/
VECTOR_MATH_API ulonglong4 operator-(const ulonglong4& a, const ulonglong4& b)
{
  return make_ulonglong4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

VECTOR_MATH_API void operator-=(ulonglong4& a, const ulonglong4& b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
}
/** @} */

/** multiply
* @{
*/
VECTOR_MATH_API ulonglong4 operator*(const ulonglong4& a, const ulonglong4& b)
{
  return make_ulonglong4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
VECTOR_MATH_API ulonglong4 operator*(const ulonglong4& a, const unsigned long long s)
{
  return make_ulonglong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
VECTOR_MATH_API ulonglong4 operator*(const unsigned long long s, const ulonglong4& a)
{
  return make_ulonglong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
VECTOR_MATH_API void operator*=(ulonglong4& a, const unsigned long long s)
{
  a.x *= s;
  a.y *= s;
  a.z *= s;
  a.w *= s;
}
/** @} */

/** divide
* @{
*/
VECTOR_MATH_API ulonglong4 operator/(const ulonglong4& a, const ulonglong4& b)
{
  return make_ulonglong4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
VECTOR_MATH_API ulonglong4 operator/(const ulonglong4& a, const unsigned long long s)
{
  return make_ulonglong4(a.x / s, a.y / s, a.z / s, a.w / s);
}
VECTOR_MATH_API ulonglong4 operator/(const unsigned long long s, const ulonglong4& a)
{
  return make_ulonglong4(s / a.x, s / a.y, s / a.z, s / a.w);
}
VECTOR_MATH_API void operator/=(ulonglong4& a, const unsigned long long s)
{
  a.x /= s;
  a.y /= s;
  a.z /= s;
  a.w /= s;
}
/** @} */

/** clamp
* @{
*/
VECTOR_MATH_API ulonglong4 clamp(const ulonglong4& v, const unsigned long long a, const unsigned long long b)
{
  return make_ulonglong4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

VECTOR_MATH_API ulonglong4 clamp(const ulonglong4& v, const ulonglong4& a, const ulonglong4& b)
{
  return make_ulonglong4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality
* @{
*/
VECTOR_MATH_API bool operator==(const ulonglong4& a, const ulonglong4& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

VECTOR_MATH_API bool operator!=(const ulonglong4& a, const ulonglong4& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory
*/
VECTOR_MATH_API unsigned long long getByIndex(const ulonglong4& v, unsigned int i)
{
  return ((unsigned long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory
*/
VECTOR_MATH_API void setByIndex(ulonglong4& v, int i, unsigned long long x)
{
  ((unsigned long long*)(&v))[i] = x;
}


/******************************************************************************/

/** Narrowing functions
* @{
*/
VECTOR_MATH_API int2 make_int2(const int3& v0)
{
  return make_int2(v0.x, v0.y);
}
VECTOR_MATH_API int2 make_int2(const int4& v0)
{
  return make_int2(v0.x, v0.y);
}
VECTOR_MATH_API int3 make_int3(const int4& v0)
{
  return make_int3(v0.x, v0.y, v0.z);
}
VECTOR_MATH_API uint2 make_uint2(const uint3& v0)
{
  return make_uint2(v0.x, v0.y);
}
VECTOR_MATH_API uint2 make_uint2(const uint4& v0)
{
  return make_uint2(v0.x, v0.y);
}
VECTOR_MATH_API uint3 make_uint3(const uint4& v0)
{
  return make_uint3(v0.x, v0.y, v0.z);
}
VECTOR_MATH_API longlong2 make_longlong2(const longlong3& v0)
{
  return make_longlong2(v0.x, v0.y);
}
VECTOR_MATH_API longlong2 make_longlong2(const longlong4& v0)
{
  return make_longlong2(v0.x, v0.y);
}
VECTOR_MATH_API longlong3 make_longlong3(const longlong4& v0)
{
  return make_longlong3(v0.x, v0.y, v0.z);
}
VECTOR_MATH_API ulonglong2 make_ulonglong2(const ulonglong3& v0)
{
  return make_ulonglong2(v0.x, v0.y);
}
VECTOR_MATH_API ulonglong2 make_ulonglong2(const ulonglong4& v0)
{
  return make_ulonglong2(v0.x, v0.y);
}
VECTOR_MATH_API ulonglong3 make_ulonglong3(const ulonglong4& v0)
{
  return make_ulonglong3(v0.x, v0.y, v0.z);
}
VECTOR_MATH_API float2 make_float2(const float3& v0)
{
  return make_float2(v0.x, v0.y);
}
VECTOR_MATH_API float2 make_float2(const float4& v0)
{
  return make_float2(v0.x, v0.y);
}
VECTOR_MATH_API float2 make_float2(const uint3& v0)
{
  return make_float2(float(v0.x), float(v0.y));
}
VECTOR_MATH_API float3 make_float3(const float4& v0)
{
  return make_float3(v0.x, v0.y, v0.z);
}
/** @} */

/** Assemble functions from smaller vectors
* @{
*/
VECTOR_MATH_API int3 make_int3(const int v0, const int2& v1)
{
  return make_int3(v0, v1.x, v1.y);
}
VECTOR_MATH_API int3 make_int3(const int2& v0, const int v1)
{
  return make_int3(v0.x, v0.y, v1);
}
VECTOR_MATH_API int4 make_int4(const int v0, const int v1, const int2& v2)
{
  return make_int4(v0, v1, v2.x, v2.y);
}
VECTOR_MATH_API int4 make_int4(const int v0, const int2& v1, const int v2)
{
  return make_int4(v0, v1.x, v1.y, v2);
}
VECTOR_MATH_API int4 make_int4(const int2& v0, const int v1, const int v2)
{
  return make_int4(v0.x, v0.y, v1, v2);
}
VECTOR_MATH_API int4 make_int4(const int v0, const int3& v1)
{
  return make_int4(v0, v1.x, v1.y, v1.z);
}
VECTOR_MATH_API int4 make_int4(const int3& v0, const int v1)
{
  return make_int4(v0.x, v0.y, v0.z, v1);
}
VECTOR_MATH_API int4 make_int4(const int2& v0, const int2& v1)
{
  return make_int4(v0.x, v0.y, v1.x, v1.y);
}
VECTOR_MATH_API uint3 make_uint3(const unsigned int v0, const uint2& v1)
{
  return make_uint3(v0, v1.x, v1.y);
}
VECTOR_MATH_API uint3 make_uint3(const uint2& v0, const unsigned int v1)
{
  return make_uint3(v0.x, v0.y, v1);
}
VECTOR_MATH_API uint4 make_uint4(const unsigned int v0, const unsigned int v1, const uint2& v2)
{
  return make_uint4(v0, v1, v2.x, v2.y);
}
VECTOR_MATH_API uint4 make_uint4(const unsigned int v0, const uint2& v1, const unsigned int v2)
{
  return make_uint4(v0, v1.x, v1.y, v2);
}
VECTOR_MATH_API uint4 make_uint4(const uint2& v0, const unsigned int v1, const unsigned int v2)
{
  return make_uint4(v0.x, v0.y, v1, v2);
}
VECTOR_MATH_API uint4 make_uint4(const unsigned int v0, const uint3& v1)
{
  return make_uint4(v0, v1.x, v1.y, v1.z);
}
VECTOR_MATH_API uint4 make_uint4(const uint3& v0, const unsigned int v1)
{
  return make_uint4(v0.x, v0.y, v0.z, v1);
}
VECTOR_MATH_API uint4 make_uint4(const uint2& v0, const uint2& v1)
{
  return make_uint4(v0.x, v0.y, v1.x, v1.y);
}
VECTOR_MATH_API longlong3 make_longlong3(const long long v0, const longlong2& v1)
{
  return make_longlong3(v0, v1.x, v1.y);
}
VECTOR_MATH_API longlong3 make_longlong3(const longlong2& v0, const long long v1)
{
  return make_longlong3(v0.x, v0.y, v1);
}
VECTOR_MATH_API longlong4 make_longlong4(const long long v0, const long long v1, const longlong2& v2)
{
  return make_longlong4(v0, v1, v2.x, v2.y);
}
VECTOR_MATH_API longlong4 make_longlong4(const long long v0, const longlong2& v1, const long long v2)
{
  return make_longlong4(v0, v1.x, v1.y, v2);
}
VECTOR_MATH_API longlong4 make_longlong4(const longlong2& v0, const long long v1, const long long v2)
{
  return make_longlong4(v0.x, v0.y, v1, v2);
}
VECTOR_MATH_API longlong4 make_longlong4(const long long v0, const longlong3& v1)
{
  return make_longlong4(v0, v1.x, v1.y, v1.z);
}
VECTOR_MATH_API longlong4 make_longlong4(const longlong3& v0, const long long v1)
{
  return make_longlong4(v0.x, v0.y, v0.z, v1);
}
VECTOR_MATH_API longlong4 make_longlong4(const longlong2& v0, const longlong2& v1)
{
  return make_longlong4(v0.x, v0.y, v1.x, v1.y);
}
VECTOR_MATH_API ulonglong3 make_ulonglong3(const unsigned long long v0, const ulonglong2& v1)
{
  return make_ulonglong3(v0, v1.x, v1.y);
}
VECTOR_MATH_API ulonglong3 make_ulonglong3(const ulonglong2& v0, const unsigned long long v1)
{
  return make_ulonglong3(v0.x, v0.y, v1);
}
VECTOR_MATH_API ulonglong4 make_ulonglong4(const unsigned long long v0, const unsigned long long v1, const ulonglong2& v2)
{
  return make_ulonglong4(v0, v1, v2.x, v2.y);
}
VECTOR_MATH_API ulonglong4 make_ulonglong4(const unsigned long long v0, const ulonglong2& v1, const unsigned long long v2)
{
  return make_ulonglong4(v0, v1.x, v1.y, v2);
}
VECTOR_MATH_API ulonglong4 make_ulonglong4(const ulonglong2& v0, const unsigned long long v1, const unsigned long long v2)
{
  return make_ulonglong4(v0.x, v0.y, v1, v2);
}
VECTOR_MATH_API ulonglong4 make_ulonglong4(const unsigned long long v0, const ulonglong3& v1)
{
  return make_ulonglong4(v0, v1.x, v1.y, v1.z);
}
VECTOR_MATH_API ulonglong4 make_ulonglong4(const ulonglong3& v0, const unsigned long long v1)
{
  return make_ulonglong4(v0.x, v0.y, v0.z, v1);
}
VECTOR_MATH_API ulonglong4 make_ulonglong4(const ulonglong2& v0, const ulonglong2& v1)
{
  return make_ulonglong4(v0.x, v0.y, v1.x, v1.y);
}
VECTOR_MATH_API float3 make_float3(const float2& v0, const float v1)
{
  return make_float3(v0.x, v0.y, v1);
}
VECTOR_MATH_API float3 make_float3(const float v0, const float2& v1)
{
  return make_float3(v0, v1.x, v1.y);
}
VECTOR_MATH_API float4 make_float4(const float v0, const float v1, const float2& v2)
{
  return make_float4(v0, v1, v2.x, v2.y);
}
VECTOR_MATH_API float4 make_float4(const float v0, const float2& v1, const float v2)
{
  return make_float4(v0, v1.x, v1.y, v2);
}
VECTOR_MATH_API float4 make_float4(const float2& v0, const float v1, const float v2)
{
  return make_float4(v0.x, v0.y, v1, v2);
}
VECTOR_MATH_API float4 make_float4(const float v0, const float3& v1)
{
  return make_float4(v0, v1.x, v1.y, v1.z);
}
VECTOR_MATH_API float4 make_float4(const float3& v0, const float v1)
{
  return make_float4(v0.x, v0.y, v0.z, v1);
}
VECTOR_MATH_API float4 make_float4(const float2& v0, const float2& v1)
{
  return make_float4(v0.x, v0.y, v1.x, v1.y);
}
/** @} */

#endif // VECTOR_MATH_H
