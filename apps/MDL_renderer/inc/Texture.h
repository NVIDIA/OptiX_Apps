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

#pragma once

#ifndef TEXTURE_H
#define TEXTURE_H

// Always include this before any OptiX headers.
#include <cuda.h>
#include <cuda_runtime.h>

#include "inc/Picture.h"

#include <string>
#include <vector>

// Bitfield encoding of the texture channels.
// These are used to remap user format and user data to the internal format.
// Each four bits hold the channel index of red, green, blue, alpha, and luminance. 
// (encoding >> ENC_*_SHIFT) & ENC_MASK gives the channel index if the result is less than 4.
// That encoding allows to automatically swap red and blue, map luminance to RGB (not the other way round though!),
// fill in alpha with input data or force it to one if required.
// 49 remapper functions take care to convert the data types including fixed-point adjustments.

#define ENC_MASK    0xF

#define ENC_RED_SHIFT   0
#define ENC_RED_0       ( 0u << ENC_RED_SHIFT)
#define ENC_RED_1       ( 1u << ENC_RED_SHIFT)
#define ENC_RED_2       ( 2u << ENC_RED_SHIFT)
#define ENC_RED_3       ( 3u << ENC_RED_SHIFT)
#define ENC_RED_NONE    (15u << ENC_RED_SHIFT)

#define ENC_GREEN_SHIFT 4
#define ENC_GREEN_0     ( 0u << ENC_GREEN_SHIFT)
#define ENC_GREEN_1     ( 1u << ENC_GREEN_SHIFT)
#define ENC_GREEN_2     ( 2u << ENC_GREEN_SHIFT)
#define ENC_GREEN_3     ( 3u << ENC_GREEN_SHIFT)
#define ENC_GREEN_NONE  (15u << ENC_GREEN_SHIFT)

#define ENC_BLUE_SHIFT  8
#define ENC_BLUE_0      ( 0u << ENC_BLUE_SHIFT)
#define ENC_BLUE_1      ( 1u << ENC_BLUE_SHIFT)
#define ENC_BLUE_2      ( 2u << ENC_BLUE_SHIFT)
#define ENC_BLUE_3      ( 3u << ENC_BLUE_SHIFT)
#define ENC_BLUE_NONE   (15u << ENC_BLUE_SHIFT)

#define ENC_ALPHA_SHIFT 12
#define ENC_ALPHA_0     ( 0u << ENC_ALPHA_SHIFT)
#define ENC_ALPHA_1     ( 1u << ENC_ALPHA_SHIFT)
#define ENC_ALPHA_2     ( 2u << ENC_ALPHA_SHIFT)
#define ENC_ALPHA_3     ( 3u << ENC_ALPHA_SHIFT)
#define ENC_ALPHA_NONE  (15u << ENC_ALPHA_SHIFT)

#define ENC_LUM_SHIFT   16
#define ENC_LUM_0       ( 0u << ENC_LUM_SHIFT)
#define ENC_LUM_1       ( 1u << ENC_LUM_SHIFT)
#define ENC_LUM_2       ( 2u << ENC_LUM_SHIFT)
#define ENC_LUM_3       ( 3u << ENC_LUM_SHIFT)
#define ENC_LUM_NONE    (15u << ENC_LUM_SHIFT)

#define ENC_CHANNELS_SHIFT 20
#define ENC_CHANNELS_1     (1u << ENC_CHANNELS_SHIFT)
#define ENC_CHANNELS_2     (2u << ENC_CHANNELS_SHIFT)
#define ENC_CHANNELS_3     (3u << ENC_CHANNELS_SHIFT)
#define ENC_CHANNELS_4     (4u << ENC_CHANNELS_SHIFT)

#define ENC_TYPE_SHIFT          24
// These are indices into the remapper table.
#define ENC_TYPE_CHAR           ( 0u << ENC_TYPE_SHIFT)
#define ENC_TYPE_UNSIGNED_CHAR  ( 1u << ENC_TYPE_SHIFT)
#define ENC_TYPE_SHORT          ( 2u << ENC_TYPE_SHIFT)
#define ENC_TYPE_UNSIGNED_SHORT ( 3u << ENC_TYPE_SHIFT)
#define ENC_TYPE_INT            ( 4u << ENC_TYPE_SHIFT)
#define ENC_TYPE_UNSIGNED_INT   ( 5u << ENC_TYPE_SHIFT)
#define ENC_TYPE_FLOAT          ( 6u << ENC_TYPE_SHIFT)
#define ENC_TYPE_UNDEFINED      (15u << ENC_TYPE_SHIFT)

// Flags to indicate that special handling is required.
#define ENC_MISC_SHIFT  28
#define ENC_FIXED_POINT (1u << ENC_MISC_SHIFT)
#define ENC_ALPHA_ONE   (2u << ENC_MISC_SHIFT)
// Highest bit set means invalid encoding.
#define ENC_INVALID     (8u << ENC_MISC_SHIFT)


class Device;

class Texture
{
public:
  Texture(Device* device);
  ~Texture();

  size_t destroy(Device* device);

  void setTextureDescription(const CUDA_TEXTURE_DESC& descr);

  void setAddressMode(CUaddress_mode s, CUaddress_mode t, CUaddress_mode r);
  void setFilterMode(CUfilter_mode filter, CUfilter_mode filterMipmap);
  void setReadMode(bool asInteger);
  void setSRGB(bool srgb);
  void setBorderColor(float r, float g, float b, float a);
  void setNormalizedCoords(bool normalized);
  void setMaxAnisotropy(unsigned int aniso);
  void setMipmapLevelBiasMinMax(float bias, float minimum, float maximum);
 
  bool create(const Picture* picture, const unsigned int flags);
  bool create(const Texture* shared);
  
  bool update(const Picture* picture);

  Device* getOwner() const;

  unsigned int getWidth() const;
  unsigned int getHeight() const;
  unsigned int getDepth() const;

  cudaTextureObject_t getTextureObject() const;

  size_t getSizeBytes() const; // Texture memory tracking of CUarrays and CUmipmappedArrays in bytes sent to the cuMemCpy3D().

  // Specific to emission texture handling.
  void calculateSphericalCDF(const float* rgba);   // For importance sampling of spherical environment lights.
  void calculateRectangularCDF(const float* rgba); // For importance-sampling of textured rectangle lights.

  CUdeviceptr getCDF_U() const;
  CUdeviceptr getCDF_V() const;
  float       getIntegral() const;

private:
  bool create1D(const Picture* picture);
  bool create2D(const Picture* picture);
  bool create3D(const Picture* picture);
  bool createCube(const Picture* picture);

  bool update1D(const Picture* picture);
  bool update2D(const Picture* picture);
  bool update3D(const Picture* picture);
  bool updateCube(const Picture* picture);

private:
  Device* m_owner; // The device which created this Texture. Needed for peer-to-peer sharing, resp. for proper destruction.

  unsigned int m_width;
  unsigned int m_height;
  unsigned int m_depth;

  unsigned int m_flags;

  unsigned int m_encodingHost;
  unsigned int m_encodingDevice;
  
  CUDA_ARRAY3D_DESCRIPTOR m_descArray3D;
  size_t                  m_sizeBytesPerElement;

  CUDA_RESOURCE_DESC m_resourceDescription; // For the final texture object creation.

  CUDA_TEXTURE_DESC m_textureDescription; // This contains all texture parameters which can be set individually or as a whole.
  
  // Note that the CUarray or CUmipmappedArray are shared among peer devices, not the texture object!
  // This needs to be created per device, which happens in the two create() functions.
  CUtexObject       m_textureObject;
  
  // Only one of these is ever used per texture.
  CUarray          m_d_array;
  CUmipmappedArray m_d_mipmappedArray;

  // How much memory the CUarray or CUmipmappedArray required in bytes input to cuMemcpy3d()
  // (without m_deviceAttribute.textureAlignment or potential row padding on the device).
  size_t           m_sizeBytesArray;

  // Specific to emission textures on spherical environment and rectangular lights.
  CUdeviceptr m_d_cdfU;
  CUdeviceptr m_d_cdfV;
  float       m_integral;
};

#endif // TEXTURE_H
