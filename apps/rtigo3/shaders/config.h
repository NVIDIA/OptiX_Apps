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

// This header with defines is included in all shaders
// to be able to switch different code paths at a central location.
// Changing any setting here will rebuild the whole solution.

#pragma once

#ifndef CONFIG_H
#define CONFIG_H

#define RT_DEFAULT_MAX 1.e27f

// Scales the m_sceneEpsilonFactor to give the effective SystemData::sceneEpsilon.
#define SCENE_EPSILON_SCALE 1.0e-7f
#define CLOCK_FACTOR_SCALE  1.0e-9f

// Prevent that division by very small floating point values results in huge values, for example dividing by pdf.
#define DENOMINATOR_EPSILON 1.0e-6f

// If both anisotropic roughness values fall below this threshold, the BSDF switches to specular.
#define MICROFACET_MIN_ROUGHNESS 0.0014142f

// 0 == Brute force path tracing without next event estimation (direct lighting). // Debug setting to compare lighting results.
// 1 == Next event estimation per path vertex (direct lighting) and using MIS with power heuristic. // Default.
#define USE_NEXT_EVENT_ESTIMATION 1

// 0 == All debug features disabled. Code optimization level on maximum. (Benchmark only in this mode!)
// 1 == All debug features enabled. Code generated with full debug info. (Really only for debugging, big performance hit!)
#define USE_DEBUG_EXCEPTIONS 0

// 0 == Disable clock() usage and time view display.
// 1 == Enable clock() usage and time view display.
#define USE_TIME_VIEW 0

#define INTEROP_MODE_OFF 0
#define INTEROP_MODE_TEX 1
#define INTEROP_MODE_PBO 2

#endif // CONFIG_H
