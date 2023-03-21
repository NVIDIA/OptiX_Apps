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

// This header with defines is included in all shaders
// to be able to switch different code paths at a central location.
// Changing any setting here will rebuild the whole solution.

#pragma once

#ifndef CONFIG_H
#define CONFIG_H

// Used to shoot rays without distance limit.
#define RT_DEFAULT_MAX 1.e27f

// Scales the m_sceneEpsilonFactor to give the effective SystemData::sceneEpsilon.
#define SCENE_EPSILON_SCALE 1.0e-7f

// Prevent that division by very small floating point values results in huge values, for example dividing by pdf.
#define DENOMINATOR_EPSILON 1.0e-6f

// 0 == All debug features disabled. Code optimization level on maximum. (Benchmark only in this mode!)
// 1 == All debug features enabled. Code generated with full debug info. (Really only for debugging, big performance hit!)
#define USE_DEBUG_EXCEPTIONS 0

// 0 == Disable clock() usage and time view display.
// 1 == Enable clock() usage and time view display.
#define USE_TIME_VIEW 0
// The m_clockFactor GUI value is scaled by this. 
// With a default of m_clockFactor = 1000 this means a million clocks will be value 1.0 in the alpha channel
// which is used as 1D texture coordinate inside the GLSL display shader and results in white in the temperature ramp texture.
#define CLOCK_FACTOR_SCALE 1.0e-9f

// These defines are used in Application, Rasterizer, Raytracer and Device. This is the only header included by all.
#define INTEROP_MODE_OFF 0
#define INTEROP_MODE_TEX 1
#define INTEROP_MODE_PBO 2

// The number of supported texture coordinate slots inside MDL shaders (state::texture_*(i)).
// This makes the Mdl_state bigger which will cost performance!
// It's also less convenient to use the TBN ortho-normal basis.
// hair_bsdf() requires two texture coordinates to communicate the intersection results
// and a per fiber texture coordinate which can be used to color each hair individually. 
// That's the whole reason for this define.
// HACK The renderer's vertex attributes are not acctually managing two distinct texture coordinates.
// The Mdl_state will simply duplicate the data of texture coordinate slot 0 to slot 1 everywhere except for the hair_bsdf().
#define NUM_TEXTURE_SPACES 2

// The number of float4 elements inside the texture_results cache. Default is 16.
// Used to configure the MDL backend's code generation with set_option("num_texture_results", ...)
// This value influences how many things can be precalculated inside the init() function.
// If the number of result elements in this array is lower than what is required,
// the expressions for the remaining results will be compiled into the sample() and evaluate() functions
// which will make the compilation and runtime performance slower. 
// For very resource-heavy materials, experiment with bigger values.
#define NUM_TEXTURE_RESULTS 16

#endif // CONFIG_H
