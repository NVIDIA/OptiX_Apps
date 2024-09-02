/* 
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda.h>

#include "inc/Arena.h"

#include "inc/CheckMacros.h"
#include "inc/MyAssert.h"

#include <iostream>
#include <vector>
#include <list>
#include <map>
#include <algorithm>


#if !defined(NDEBUG)
// Only used inside a MY_ASSERT(). Prevent unused function warning in debug targets.
static bool isPowerOfTwo(const size_t s)
{
  return (s != 0) && ((s & (s - 1)) == 0);
}
#endif


namespace cuda
{

  Block::Block(cuda::Arena* arena, const CUdeviceptr addr, const size_t size, const CUdeviceptr ptr)
  : m_arena(arena)
  , m_addr(addr)
  , m_size(size)
  , m_ptr(ptr)
  {
  }

  bool Block::isValid() const
  {
    return (m_ptr != 0);
  }

  bool Block::operator<(const cuda::Block& rhs)
  {
    return (m_addr < rhs.m_addr);
  }
  

  Arena::Arena(const size_t size)
  : m_size((size + 255) & ~255) // Make sure the Arena has a multiple of 256 bytes size.
  {
    // This can fail with CUDA OOM errors and then none of the further allocations will work.
    // Same as if there would be no arena.
    CU_CHECK( cuMemAlloc(&m_addr, m_size) );

    // Make sure the base address of the arena is 256-byte aligned.
    MY_ASSERT((m_addr & 255) == 0);
    
    // When the Arena is created, there is one big free block of the full size.
    // Free blocks do not have a user pointer assigned, only allocated blocks are valid.
    m_blocksFree.push_back(cuda::Block(this, m_addr, m_size, 0)); 
  }

  Arena::~Arena()
  {
    // By design there is no need to clear this here.
    // These are not pointers and there is no Block destructor.
    // m_blocksFree.clear();
    
    // Wipe the arena.
    // All active blocks after this point are invalid and must not be in use anymore.
    CU_CHECK_NO_THROW( cuMemFree(m_addr) ); 
  }
   
  bool Arena::allocBlock(cuda::Block& block, const size_t size, const size_t alignment, const cuda::Usage usage)
  {
    const size_t maskAligned = alignment - 1;

    for (std::list<cuda::Block>::iterator it = m_blocksFree.begin(); it != m_blocksFree.end(); ++it)
    {
      if (usage == cuda::USAGE_STATIC) // Static blocks are allocated at the front of free blocks.
      {
        const size_t offset = it->m_addr & maskAligned;          // If the address of this free block is not on the required alignment
        const size_t adjust = (offset) ? alignment - offset : 0; // the size needs to be adjusted by this many bytes.
      
        const size_t sizeAdjusted = size + adjust; // The resulting block needs to be this big to fit the requested size with the proper alignment.

        if (sizeAdjusted <= it->m_size)
        {
          // The new block address starts at the beginning of the free block. 
          // The adjusted address is the properly aligned pointer in that free block.
          block = cuda::Block(this, it->m_addr, sizeAdjusted, it->m_addr + adjust);
          MY_ASSERT((block.m_ptr & maskAligned) == 0); // DEBUG
        
          it->m_addr += sizeAdjusted; // Advance the free block pointer.
          it->m_size -= sizeAdjusted; // Reduce the free block size. 

          if (it->m_size == 0) // If the block was fully used, remove it from the list of free blocks.
          {
            m_blocksFree.erase(it);
          }

          return true;
        }
      }
      else // if (usage == cuda::USAGE_TEMP) // Temporary allocations are placed at the end of free blocks to reduce fragmentation inside the arena.
      {
        const CUdeviceptr addrEnd = it->m_addr + it->m_size; // The poiner behind the last byte of the free block.
        CUdeviceptr       addrTmp = addrEnd - size;          // The pointer to the start of the allocated block if the alignment fits.
        
        const size_t adjust = addrTmp & maskAligned; // If the start address is not properly aligned, this is the amount of bytes we need to start earlier to get an aligned pointer.
        
        const size_t sizeAdjusted = size + adjust;   // For performance, do not split the free block into size and adjust, although there will be adjust many bytes free at the end of the block.

        if (sizeAdjusted <= it->m_size)
        {
          addrTmp -= adjust; // The block address needs to be that many bytes ealier to be aligned.
          MY_ASSERT(it->m_addr <= addrTmp); // DEBUG Cannot happen with the size check above.
  
          // The new block starts at the aligned address at the end of the free block.
          block = cuda::Block(this, addrTmp, sizeAdjusted, addrTmp);
          MY_ASSERT((block.m_ptr & maskAligned) == 0); // DEBUG Check the user pointer for proper alignment.
        
          it->m_size -= sizeAdjusted; // Reduce the free block size. The address stays the same because we allocated at the end.

          if (it->m_size == 0) // If the block was fully used, remove it from the list of free blocks.
          {
            MY_ASSERT(it->m_addr == addrTmp); // If we used the whole size, these two addresses must match.
            m_blocksFree.erase(it);
          }

          return true;
        }
      }
    }

    return false;
  }

  void Arena::freeBlock(const cuda::Block& block)
  {
    // Search for the list element which has the next higher m_addr than the block.
    std::list<Block>::iterator itNext = m_blocksFree.begin();

    while (itNext != m_blocksFree.end() && itNext->m_addr < block.m_addr)
    {
      ++itNext;
    }

    // Insert block before itNext. Returned iterator "it" points to block.
    std::list<Block>::iterator it = m_blocksFree.insert(itNext, block); 
    
    // If itNext is not end(), then it points to a free block and "it" is directly before that.
    if (itNext != m_blocksFree.end())
    {
      if (it->m_addr + it->m_size == itNext->m_addr) // Check if the memory blocks are adjacent.
      {
        it->m_size += itNext->m_size; // Merge the two blocks to the first 
        m_blocksFree.erase(itNext);   // and erase the second.
      }
    }

    // If "it" is not begin(), then there is an element before it which could be adjacent.
    if (it != m_blocksFree.begin())
    {
      itNext = it--; // Now "it" can be at least begin() and itNext is always the element directly after it.
      if (it->m_addr + it->m_size == itNext->m_addr)
      {
        it->m_size += itNext->m_size; // Merge the two blocks to the first
        m_blocksFree.erase(itNext);   // and erase the second.
      }
    }
  }


  ArenaAllocator::ArenaAllocator(const size_t sizeArenaBytes) 
  : m_sizeArenaBytes(sizeArenaBytes)
  , m_sizeMemoryAllocated(0)
  {
  }

  ArenaAllocator::~ArenaAllocator()
  {
    for (auto arena : m_arenas)
    {
      delete arena;
    }
  }

  CUdeviceptr ArenaAllocator::alloc(const size_t size, const size_t alignment, const cuda::Usage usage)
  {
    // All memory alignments needed in this implementation are at max 256 and power-of-two
    // which means adjustments can be done with bitmasks instead of modulo operators.
    MY_ASSERT(0 < size && alignment <= 256 && isPowerOfTwo(alignment));

    // This allocator does not support allocating a pointer with zero bytes capacity. (cuMemAlloc() doesn't either.)
    // That would break the uniqueness of the user pointer inside the m_blocksAllocated because the free block wouldn't advance its address.
    // If really needed, override the zero size here.
    if (size == 0)
    {
      return 0;
    }

    cuda::Block block;

    // PERF Normally the biggest free block is inside the most recently created arena. 
    // Means if the search starts from the back of the vector, the chance to find a free block is much higher.
    size_t i = m_arenas.size();
    while (0 < i--)
    {
      if (m_arenas[i]->allocBlock(block, size, alignment, usage))
      {
        break;
      }
    }

    // If none of the existing Arenas had a sufficient contiguous memory block, create a new Arena which can hold the size. 
    if (!block.isValid())
    {
      try
      {
        const size_t sizeArenaBytes = std::max(m_sizeArenaBytes, size);

        // Allocate a new arena which can hold the requested size of bytes.
        // No user pointer adjustment is required if m_sizeArenaBytes <= size. This is always aligned to 256 bytes.
        Arena* arena = new Arena(sizeArenaBytes); // This can fail with a CUDA out-of-memory error!

        m_arenas.push_back(arena); // Append it to the vector of arenas.

        (void) arena->allocBlock(block, size, alignment, usage);
        MY_ASSERT(block.isValid()); // This allocation should not fail.
      }
      catch (const std::exception& e)
      {
        std::cerr << e.what() << '\n';
      }
    }

    if (block.isValid())
    {
      // DEBUG Make sure the user pointer is unique. (It wasn't when using size == 0.)
      //std::map<CUdeviceptr, cuda::Block>::const_iterator it = m_blocksAllocated.find(block.m_ptr); 
      //if (it != m_blocksAllocated.end())
      //{
      //  MY_ASSERT(false);
      //}
      
      m_blocksAllocated[block.m_ptr] = block; // Track all successful allocations inside the ArenaAllocator with the user pointer as key.

      m_sizeMemoryAllocated += block.m_size;  // Track the overall number of bytes allocated.
    }

    return block.m_ptr; // This is 0 when the block is invalid which can only happen with a CUDA OOM error.
  }

  void ArenaAllocator::free(const CUdeviceptr ptr)
  {
    // Allow free() to be called with nullptr. This actually happens on purpose.
    if (ptr == 0)
    {
      return;
    }

    std::map<CUdeviceptr, cuda::Block>::const_iterator it = m_blocksAllocated.find(ptr);
    if (it != m_blocksAllocated.end())
    {
      const cuda::Block& block = it->second;
      
      MY_ASSERT(block.m_size <= m_sizeMemoryAllocated);
      m_sizeMemoryAllocated -= block.m_size; // Track overall number of byts allocated.

      block.m_arena->freeBlock(block); // Merge this block to the list of free blocks in this arena.

      m_blocksAllocated.erase(it); // Remove it from the map of allocated blocks.
    }
    else
    {
      std::cerr << "ERROR: ArenaAllocator::free() failed to find the pointer " << ptr << "\n";
    }
  }

  size_t ArenaAllocator::getSizeMemoryAllocated() const
  {
    return m_sizeMemoryAllocated;
  }

} // namespace cuda
