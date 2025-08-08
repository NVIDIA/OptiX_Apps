/* 
 * Copyright (c) 2013-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef CHECK_MACROS_H
#define CHECK_MACROS_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <sstream>

#include "MyAssert.h"

#define CU_CHECK(call) \
{ \
  const CUresult result = call; \
  if (result != CUDA_SUCCESS) \
  { \
    const char* name; \
    cuGetErrorName(result, &name); \
    const char *error; \
    cuGetErrorString(result, &error); \
    std::ostringstream message; \
    message << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " (" << result << ") " << name << ": " << error; \
    MY_ASSERT(!"CU_CHECK"); \
    throw std::runtime_error(message.str()); \
  } \
}

#define CU_CHECK_NO_THROW(call) \
{ \
  const CUresult result = call; \
  if (result != CUDA_SUCCESS) \
  { \
    const char* name; \
    cuGetErrorName(result, &name); \
    const char *error; \
    cuGetErrorString(result, &error); \
    std::cerr << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " (" << result << ") " << name << ": " << error << '\n'; \
    MY_ASSERT(!"CU_CHECK_NO_THROW"); \
  } \
}

#define CUDA_CHECK(call) \
{ \
  const cudaError_t result = call; \
  if (result != cudaSuccess) \
  { \
    const char* name  = cudaGetErrorName(result); \
    const char* error = cudaGetErrorString(result); \
    std::ostringstream message; \
    message << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " (" << result << ") " << name << ": " << error; \
    MY_ASSERT(!"CUDA_CHECK"); \
    throw std::runtime_error(message.str()); \
  } \
}

#define CUDA_CHECK_NO_THROW(call) \
{ \
  const cudaError_t result = call; \
  if (result != cudaSuccess) \
  { \
    const char* name  = cudaGetErrorName(result); \
    const char* error = cudaGetErrorString(result); \
    std::cerr << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " (" << result << ") " << name << ": " << error << '\n'; \
    MY_ASSERT(!"CUDA_CHECK_NO_THROW"); \
  } \
}

#define OPTIX_CHECK(call) \
{ \
  const OptixResult result = call; \
  if (result != OPTIX_SUCCESS) \
  { \
    std::ostringstream message; \
    message << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " (" << result << ")"; \
    MY_ASSERT(!"OPTIX_CHECK"); \
    throw std::runtime_error(message.str()); \
  } \
}

#define OPTIX_CHECK_NO_THROW(call) \
{ \
  const OptixResult result = call; \
  if (result != OPTIX_SUCCESS) \
  { \
    std::cerr << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " (" << result << ")\n"; \
    MY_ASSERT(!"OPTIX_CHECK_NO_THROW"); \
  } \
}


#endif // CHECK_MACROS_H

