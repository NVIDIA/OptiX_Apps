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

// Code in these classes is based on the ILTexLoader.h/.cpp routines inside the NVIDIA nvpro-pipeline ILTexLoader plugin:
// https://github.com/nvpro-pipeline/pipeline/blob/master/dp/sg/io/IL/Loader/ILTexLoader.cpp

#pragma once

#ifndef PICTURE_H
#define PICTURE_H

#include <IL/il.h>

#include "inc/LoaderIES.h"

#include <string>
#include <vector>

// Bits for the image handling and texture creation flags.
#define IMAGE_FLAG_1D     0x00000001
#define IMAGE_FLAG_2D     0x00000002
#define IMAGE_FLAG_3D     0x00000004
#define IMAGE_FLAG_CUBE   0x00000008
// Modifier bits on the above types.
// Layered image (not applicable to 3D)
#define IMAGE_FLAG_LAYER  0x00000010
// Mipmapped image (ignored when there are no mipmaps provided).
#define IMAGE_FLAG_MIPMAP 0x00000020

// FIXME See if there is a better method to store pictures and flags separately
// to be able to build different samplers (texture objects) from the same input image.

// Special flags for emission textures. Used with IMAGE_FLAG_2D only.
// Only area lights (env, rect) are importance sampled in this unidirectional path tracer.
// Spherical environment map. Wrap clamp on poles.
#define IMAGE_FLAG_ENV    0x00000100
// Spherical omnidirectional projection map assigned to point lights. Wrap clamp on poles.
#define IMAGE_FLAG_POINT  0x00000200
// Spherical cap projection assigned to spot lights (not importance sampled). Wrap clamp.
#define IMAGE_FLAG_SPOT   0x00000400
// Emission texture on a rectangle light (importance sampled). Wrap clamp.
#define IMAGE_FLAG_RECT   0x00000800
// Spherical (polar grid) emission texture from an IES light profile, applied to point lights. Wrap clamp on poles.
#define IMAGE_FLAG_IES    0x00001000

struct Image
{
  Image(unsigned int width, unsigned int height, unsigned int depth, int format, int type);
  ~Image();

  unsigned int m_width;
  unsigned int m_height;
  unsigned int m_depth;

  int          m_format; // DevIL image format.
  int          m_type;   // DevIL image component type.

  // Derived values.
  unsigned int m_bpp; // bytes per pixel
  unsigned int m_bpl; // bytes per scanline
  unsigned int m_bps; // bytes per slice (plane)
  unsigned int m_nob; // number of bytes (complete image)

  unsigned char* m_pixels; // The pixel data of one image.
};


class Picture
{
public:
  Picture();
  ~Picture();

  bool load(const std::string& filename, const unsigned int flags);
  void clear();
  
  // Add an empty new vector of images. Each vector can hold one mipmap chain. Returns the new image index.
  unsigned int addImages();
  
  // Add a mipmap chain as new vector of images. Returns the new image index.
  // pixels and extents are for the LOD 0. mipmaps can be empty.
  unsigned int addImages(const void* pixels, 
                         const unsigned int width, const unsigned int height, const unsigned int depth,
                         const int format, const int type,
                         const std::vector<const void*>& mipmaps, const unsigned int flags); 

  // Append a new image LOD to the existing vector of images building a mipmap chain. (No mipmap consistency check here.)
  unsigned int addLevel(const unsigned int index, 
                        const void* pixels, 
                        const unsigned int width, const unsigned int height, const unsigned int depth, 
                        const int format, const int type);

  unsigned int getFlags() const;
  void setFlags(const unsigned int flags);
  void addFlags(const unsigned int flags); 

  unsigned int getNumberOfImages() const;
  unsigned int getNumberOfLevels(unsigned int indexImage) const;
  const Image* getImageLevel(unsigned int indexImage, unsigned int indexLevel) const;
  bool         isCubemap() const;

  // This is needed when generating cubemaps without loading them via DevIL.
  void setIsCubemap(const bool isCube);

  bool createIES(const IESData& iesData);

  // DEBUG Function to generate all 14 texture targets with RGBA8 images.
  void generateRGBA8(unsigned int width, unsigned int height, unsigned int depth, const unsigned int flags);

private:
  void mirrorX(unsigned int index);
  void mirrorY(unsigned int index);
  
  void clearImages(); // Delete all pixel data, all Image pointers and clear the m_images vector.
  
  bool generateIES(const IESData& ies,
                   const std::vector<float>& horizontalAngles,
                   const std::vector<float>& candelas);

private:
  unsigned int m_flags;  // The image flags with which this Picture has been loaded. 
  bool         m_isCube; // Track if the picture is a cube map.
  std::vector< std::vector<Image*> > m_images;
};

#endif // PICTURE_H
