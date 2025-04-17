//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.#pragma once

#pragma once
#include <fastgltf/types.hpp>

// Abstract function arguments for the conversion routines from the Accessor,
// in order to read the SparseAccessor indices and values data with these as well.
struct ConversionArguments
{
  size_t                  srcByteOffset;     // == accessor.byteOffset
  fastgltf::AccessorType  srcType;           // == accessor.type  
  fastgltf::ComponentType srcComponentType;  // == accessor.componentType
  size_t                  srcCount;          // == accessor.count
  bool                    srcNormalized;     // == accessor.normalized
  fastgltf::BufferView*   srcBufferView = nullptr;     // nullptr when the accessor has no buffer view index. Can happen with sparse accessors.
  fastgltf::Buffer*       srcBuffer = nullptr;         // nullptr when the accessor has no buffer view index. Can happen with sparse accessors.

  fastgltf::AccessorType  dstType;
  fastgltf::ComponentType dstComponentType;
  float                   dstExpansion;      // Vec3 to Vec4 expansion value (1.0f or 0.0f). Color attributes and color morph targets need that distinction!
  unsigned char*          dstPtr = nullptr;
};
