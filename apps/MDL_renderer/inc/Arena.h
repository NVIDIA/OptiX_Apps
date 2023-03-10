/* 
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef ARENA_H
#define ARENA_H

#include <cuda.h>

#include "inc/CheckMacros.h"
#include "inc/MyAssert.h"

#include <vector>
#include <list>
#include <map>

namespace cuda
{
  enum Usage
  {
    USAGE_STATIC,
    USAGE_TEMP
  };

  class Arena;

  struct Block
  {
    Block(cuda::Arena* arena = nullptr, const CUdeviceptr addr = 0, const size_t size = 0, const CUdeviceptr ptr = 0);
    
    bool isValid() const;
    bool operator<(const cuda::Block& rhs);

    cuda::Arena* m_arena; // The arena which owns this block.
    CUdeviceptr  m_addr;  // The internal address inside the arena of the memory containing the aligned block at ptr.
    size_t       m_size;  // The internal size of the allocation starting at m_addr.
    CUdeviceptr  m_ptr;   // The aligned pointer to size bytes the user requested.
  };


  class Arena
  {
    public:
      Arena(const size_t size);
      ~Arena();

      bool allocBlock(cuda::Block& block, const size_t size, const size_t alignment, const cuda::Usage usage);
      void freeBlock(const cuda::Block& block);

    private:
      CUdeviceptr            m_addr;       // The start pointer of the CUDA memory for this arena. This is 256 bytes aligned due to cuMalloc().
      size_t                 m_size;       // The overall size of the arena in bytes.
      std::list<cuda::Block> m_blocksFree; // A list of free blocks inside this arena. This is normally very short because static and temporary allocations are not interleaved.
  };


  class ArenaAllocator
  {
  public:
    ArenaAllocator(const size_t sizeArenaBytes);
    ~ArenaAllocator();

    CUdeviceptr alloc(const size_t size, const size_t alignment, const cuda::Usage usage = cuda::USAGE_STATIC);
    void free(const CUdeviceptr ptr);

    size_t getSizeMemoryAllocated() const;

  private:
    size_t                             m_sizeArenaBytes;      // The minimum size an arena is allocated with. If calls alloc() with a bigger size, that will be used.
    size_t                             m_sizeMemoryAllocated; // The number of bytes which have been allocated (with alignment).
    std::vector<Arena*>                m_arenas;              // A number of Arenas managing a CUdeviceptr with at least m_sizeArenaBytes each.
    std::map<CUdeviceptr, cuda::Block> m_blocksAllocated;     // Map the user pointer to the internal block. This abstracts Arena and Block from the user. (Kind of expensive.)
  };

} // namespace cuda

#endif // ARENA_H
