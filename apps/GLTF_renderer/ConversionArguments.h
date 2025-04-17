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
