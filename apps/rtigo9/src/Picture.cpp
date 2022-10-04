/* 
 * Copyright (c) 2013-2022, NVIDIA CORPORATION. All rights reserved.
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


#include "inc/Picture.h"

#include "dp/math/math.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <deque>

#include "inc/MyAssert.h"


static unsigned int numberOfComponents(int format)
{
  switch (format)
  {
    case IL_RGB:
    case IL_BGR:
      return 3;

    case IL_RGBA:
    case IL_BGRA:
      return 4;

    case IL_LUMINANCE: 
    case IL_ALPHA:
      return 1;

    case IL_LUMINANCE_ALPHA:
      return 2;

    default:
      MY_ASSERT(!"Unsupported image data format.");
      return 0;
  }
}

static unsigned int sizeOfComponents(int type)
{
  switch (type)
  {
    case IL_BYTE:
    case IL_UNSIGNED_BYTE:
      return 1;

    case IL_SHORT:
    case IL_UNSIGNED_SHORT:
      return 2;

    case IL_INT:
    case IL_UNSIGNED_INT:
    case IL_FLOAT:
      return 4;

    default:
      MY_ASSERT(!"Unsupported image data type.");
      return 0;
  }
}

Image::Image(unsigned int width,
             unsigned int height,
             unsigned int depth,
             int          format,
             int          type)
: m_width(width)
, m_height(height)
, m_depth(depth)
, m_format(format)
, m_type(type)
, m_pixels(nullptr)
{
  m_bpp = numberOfComponents(m_format) * sizeOfComponents(m_type);
  m_bpl = m_width  * m_bpp;
  m_bps = m_height * m_bpl;
  m_nob = m_depth  * m_bps;
}

Image::~Image()
{
  if (m_pixels != nullptr)
  {
    delete[] m_pixels;
    m_pixels = nullptr;
  }
}

static int determineFace(int i, bool isCubemapDDS)
{
  int face = i;
  
  // If this is a cubemap in a DDS file, exchange the z-negative and z-positive images to match OpenGL and what's used here for OptiX.
  if (isCubemapDDS)
  {
    if (i == 4)
    {
      face = 5;
    }
    else if (i == 5)
    {
      face = 4;
    }
  }
  
  return face;
}


Picture::Picture()
: m_flags(0)
, m_isCube(false)
{
}

Picture::~Picture()
{
  clearImages();
}

unsigned int Picture::getFlags() const
{
  return m_flags;
}

void Picture::setFlags(const unsigned int  flags)
{
  m_flags = flags;
}

void Picture::addFlags(const unsigned int flags)
{
  m_flags |= flags;
}

unsigned int Picture::getNumberOfImages() const
{
  return static_cast<unsigned int>(m_images.size());
}

unsigned int Picture::getNumberOfLevels(unsigned int index) const
{
  MY_ASSERT(index < m_images.size());
  return static_cast<unsigned int>(m_images[index].size());
}

const Image* Picture::getImageLevel(unsigned int index, unsigned int level) const
{
  if (index < m_images.size() && level < m_images[index].size())
  {
    return m_images[index][level];
  }
  return nullptr;
}

bool Picture::isCubemap() const
{
  return m_isCube;
}

void Picture::setIsCubemap(const bool isCube)
{
  m_isCube = isCube;
}


// Returns true if the input extents were already the smallest possible mipmap level.
static bool calculateNextExtents(unsigned int &w, unsigned int &h, unsigned int& d, const unsigned int flags)
{
  bool done = false;

  // Calculate the expected LOD image extents.
  if (flags & IMAGE_FLAG_LAYER)
  {
    if (flags & IMAGE_FLAG_1D)
    {
      // 1D layered mipmapped.
      done = (w == 1);
      w = (1 < w) ? w >> 1 : 1;
      // height is 1
      // depth is the number of layers and must not change.
    }
    else if (flags & (IMAGE_FLAG_2D | IMAGE_FLAG_CUBE))
    {
      // 2D or cubemap layered mipmapped
      done = (w == 1 && h == 1);
      w = (1 < w) ? w >> 1 : 1;
      h = (1 < h) ? h >> 1 : 1;
      // depth is the number of layers (* 6 for cubemaps) and must not change.
    }
  }
  else
  {
    // Standard mipmap chain.
    done = (w == 1 && h == 1 && d == 1);
    w = (1 < w) ? w >> 1 : 1;
    h = (1 < h) ? h >> 1 : 1;
    d = (1 < d) ? d >> 1 : 1;
  }
  return done;
}


void Picture::clearImages()
{
  for (size_t i = 0; i < m_images.size(); ++i)
  {
    for (size_t lod = 0; lod < m_images[i].size(); ++lod)
    {
      delete m_images[i][lod];
      m_images[i][lod] = nullptr;
    }
    m_images[i].clear();
  }
  m_images.clear();
}


bool Picture::load(const std::string& filename, const unsigned int flags)
{
  bool success = false;

  clearImages(); // Each load() wipes previously loaded image data.

  if (filename.empty())
  {
    std::cerr << "ERROR: Picture::load() " << filename << " empty\n";
    MY_ASSERT(!"Picture::load() filename empty");
    return success;
  }

  if (!std::filesystem::exists(filename))
  {
    std::cerr << "ERROR: Picture::load() " << filename << " not found\n";
    MY_ASSERT(!"Picture::load() File not found");
    return success;
  }

  m_flags = flags; // Track the flags with which this picture was loaded.

  std::string ext;
  std::string::size_type last = filename.find_last_of('.');
  if (last != std::string::npos) 
  { 
    ext = filename.substr(last, std::string::npos);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
  }

  bool isDDS = (ext == std::string(".dds")); // .dds images need special handling
  m_isCube = false;
  
  unsigned int imageID;

  ilGenImages(1, (ILuint *) &imageID);
  ilBindImage(imageID);

  // Let DevIL handle the proper orientation during loading.
  if (isDDS)
  {
    ilEnable(IL_ORIGIN_SET);
    ilOriginFunc(IL_ORIGIN_UPPER_LEFT); // DEBUG What happens when I set IL_ORIGIN_LOWER_LEFT all the time?
  }
  else
  {
    ilEnable(IL_ORIGIN_SET);
    ilOriginFunc(IL_ORIGIN_LOWER_LEFT);
  }

  // Load the image from file. This loads all data.
  if (ilLoadImage((const ILstring) filename.c_str()))
  {
    std::vector<const void*> mipmaps; // All mipmaps excluding the LOD 0.

    ilBindImage(imageID);  
    ilActiveImage(0); // Get the frst image, potential LOD 0.
    MY_ASSERT(IL_NO_ERROR == ilGetError()); 
    
    // Get the size of the LOD 0 image.
    unsigned int w = ilGetInteger(IL_IMAGE_WIDTH);
    unsigned int h = ilGetInteger(IL_IMAGE_HEIGHT);
    unsigned int d = ilGetInteger(IL_IMAGE_DEPTH);

    // Querying for IL_NUM_IMAGES returns the number of images following the current one. Add 1 for the correct image count!
    int numImages = ilGetInteger(IL_NUM_IMAGES) + 1;

    int numMipmaps = 0; // Default to no mipmap handling.

    if (flags & IMAGE_FLAG_MIPMAP) // Only handle mipmaps if we actually want to load them.
    {
      numMipmaps = ilGetInteger(IL_NUM_MIPMAPS); // Excluding the current image which becomes LOD 0.

      // Special check to see if the number of top-level images build a 1D, 2D, or 3D mipmap chain, if there are no mipmaps in the LOD 0 image.
      if (1 < numImages && !numMipmaps)
      {
        bool isMipmapChain = true; // Indicates if the images in this file build a standard mimpmap chain.

        for (int i = 1; i < numImages; ++i) // Start check at LOD 1.
        {
          ilBindImage(imageID);  
          ilActiveImage(i);
          MY_ASSERT(IL_NO_ERROR == ilGetError()); 

          // Next image extents.
          const unsigned int ww = ilGetInteger(IL_IMAGE_WIDTH);
          const unsigned int hh = ilGetInteger(IL_IMAGE_HEIGHT);
          const unsigned int dd = ilGetInteger(IL_IMAGE_DEPTH);

          calculateNextExtents(w, h, d, flags); // Calculates the extents of the next mipmap level, taking layered textures into account!

          if (ww == w && hh == h && dd == d) // Criteria for next mipmap level match.
          { 
            // Top-level image actually is the i-th mipmap level. Remember the data. 
            // This doesn't get overwritten below, because the standard mipmap handling is based on numMipmaps != 0.
            mipmaps.push_back(ilGetData());
          }
          else
          {
            // Could not identify top-level image as a mipmap level, no further testing required.
            // Test failed, means the number of images do not build a mipmap chain.
            isMipmapChain = false; 
            mipmaps.clear();
            break; 
          }
        }

        if (isMipmapChain)
        {
          // Consider only the very first image in the file in the following code.
          numImages = 1;
        }
      }
    }

    m_isCube = (ilGetInteger(IL_IMAGE_CUBEFLAGS) != 0);

    // If the file isn't identified as cubemap already,
    // check if there are six square images of the same extents in the file and handle them as cubemap.
    if (!m_isCube && numImages == 6) 
    {
      bool isCube = true;

      unsigned int w0 = 0;
      unsigned int h0 = 0;
      unsigned int d0 = 0;
      
      for (int image = 0; image < numImages && isCube; ++image)
      {
        ilBindImage(imageID);
        ilActiveImage(image);
        MY_ASSERT(IL_NO_ERROR == ilGetError()); 

        if (image == 0)
        {
          w0 = ilGetInteger(IL_IMAGE_WIDTH);
          h0 = ilGetInteger(IL_IMAGE_HEIGHT);
          d0 = ilGetInteger(IL_IMAGE_DEPTH);

          MY_ASSERT(0 < d0); // This case of no image data is handled later.

          if (w0 != h0)
          {
            isCube = false; // Not square.
          }
        }
        else
        {
          unsigned int w1 = ilGetInteger(IL_IMAGE_WIDTH);
          unsigned int h1 = ilGetInteger(IL_IMAGE_HEIGHT);
          unsigned int d1 = ilGetInteger(IL_IMAGE_DEPTH);
          
          // All LOD 0 faces must be the same size.
          if (w0 != w1 || h0 != h1)
          {
            isCube = false;
          }
          // If this should be interpreted as layered cubemap, all images must have the same number of layers.
          if ((flags & IMAGE_FLAG_LAYER) && d0 != d1)
          {
            isCube = false;
          }
        }
      }
      m_isCube = isCube;
    }

    for (int image = 0; image < numImages; ++image)
    {
      // Cubemap faces within DevIL philosophy are organized like this:
      // image -> 1st face -> face index 0
      // face1 -> 2nd face -> face index 1
      // ...
      // face5 -> 6th face -> face index 5

      const int numFaces = ilGetInteger(IL_NUM_FACES) + 1;

      for (int f = 0; f < numFaces; ++f)
      {
        // Need to juggle with the faces to get them aligned with how OpenGL expects cube faces. Using the same layout in OptiX.
        const int face = determineFace(f, m_isCube && isDDS);

        // DevIL frequently loses track of the current state.
        ilBindImage(imageID);  
        ilActiveImage(image);
        ilActiveFace(face);
        MY_ASSERT(IL_NO_ERROR == ilGetError()); 

        // pixel format
        int format = ilGetInteger(IL_IMAGE_FORMAT);

        if (IL_COLOR_INDEX == format)
        {
          // Convert color index to whatever the base type of the palette is.
          if (!ilConvertImage(ilGetInteger(IL_PALETTE_BASE_TYPE), IL_UNSIGNED_BYTE))
          {
            // Free all resources associated with the DevIL image.
            ilDeleteImages(1, &imageID);
            MY_ASSERT(IL_NO_ERROR == ilGetError());
            return false;
          }
          // Now query format of the converted image.
          format = ilGetInteger(IL_IMAGE_FORMAT);
        }

        const int type = ilGetInteger(IL_IMAGE_TYPE);

        // Image dimension of the LOD 0 in pixels.
        unsigned int width  = ilGetInteger(IL_IMAGE_WIDTH);
        unsigned int height = ilGetInteger(IL_IMAGE_HEIGHT);
        unsigned int depth  = ilGetInteger(IL_IMAGE_DEPTH);

        if (width == 0 || height == 0 || depth == 0) // There must be at least a single pixel.
        {
          std::cerr << "ERROR Picture::load() " << filename << ": image " << image << " face " << f << " extents (" << width << ", " << height << ", " << depth << ")\n";
          MY_ASSERT(!"Picture::load() Image with zero extents.");

          // Free all resources associated with the DevIL image.
          ilDeleteImages(1, &imageID);
          MY_ASSERT(IL_NO_ERROR == ilGetError());
          return false;
        }

        // Get the remaining mipmaps for this image.
        // Note that the special case handled above where multiple images built a mipmap chain
        // will not enter this because that was only checked when numMipmaps == 0.
        if (0 < numMipmaps)
        {
          mipmaps.clear(); // Clear this for currently processed image.

          for (int j = 1; j <= numMipmaps; ++j) // Mipmaps are starting at LOD 1.
          { 
            // DevIL frequently loses track of the current state.
            ilBindImage(imageID);
            ilActiveImage(image);
            ilActiveFace(face);
            ilActiveMipmap(j);

            // Not checking consistency of the individual LODs here.
            mipmaps.push_back(ilGetData());
          }

          // Look at LOD 0 of this image again for the next ilGetData().
          // DevIL frequently loses track of the current state.
          ilBindImage(imageID);
          ilActiveImage(image);
          ilActiveFace(face);
          ilActiveMipmap(0);
        }

        // Add a a new vector of images with the whole mipmap chain.
        // Mind that there will be six of these for a cubemap image!
        unsigned int index = addImages(ilGetData(), width, height, depth, format, type, mipmaps, flags);

        if (m_isCube && isDDS)
        {
          // WARNING:
          // This piece of code MUST NOT be visited twice for the same image,
          // because this would falsify the desired effect!
          // The images at this position are flipped at the x-axis (due to DevIL)
          // flipping at x-axis will result in original image
          // mirroring at y-axis will result in rotating the image 180 degree
          if (face == 0 || face == 1 || face == 4 || face == 5) // px, nx, pz, nz
          {
            mirrorY(index); // mirror over y-axis
          }
          else // py, ny
          {
            mirrorX(index); // flip over x-axis
          }
        }

        ILint origin = ilGetInteger(IL_IMAGE_ORIGIN);
        if (!m_isCube && origin == IL_ORIGIN_UPPER_LEFT)
        {
          // OpenGL expects origin at lower left, so the image has to be flipped at the x-axis
          // for DDS cubemaps we handle the separate face rotations above
          // DEBUG This should only happen for DDS images. 
          // All others are flipped by DevIL because I set the origin to lower left. Handle DDS images the same?
          mirrorX(index); // reverse rows 
        }
      }
    }
    success = true;
  }
  else
  {
    std::cerr << "ERROR: Picture::load() ilLoadImage(" << filename << " failed with error " << ilGetError() << '\n';
    MY_ASSERT(!"Picture::load() ilLoadImage failed");
  }

  // Free all resources associated with the DevIL image
  ilDeleteImages(1, &imageID);
  MY_ASSERT(IL_NO_ERROR == ilGetError());
  
  return success;
}

void Picture::clear()
{
  m_images.clear();
}

// Append a new empty vector of images. Returns the new image index.
unsigned int Picture::addImages()
{
  const unsigned int index = static_cast<unsigned int>(m_images.size());

  m_images.push_back(std::vector<Image*>()); // Append a new empty vector of image pointers. Each vector can hold one mipmap chain.

  return index;
}

// Add a vector of images and fill it with the LOD 0 pixels and the optional mipmap chain.
unsigned int Picture::addImages(const void* pixels, 
                                const unsigned int width, const unsigned int height, const unsigned int depth,
                                const int format, const int type,
                                const std::vector<const void*>& mipmaps, const unsigned int flags)
{
  const unsigned int index = static_cast<unsigned int>(m_images.size());

  m_images.push_back(std::vector<Image*>()); // Append a new empty vector of image pointers.

  Image* image = new Image(width, height, depth, format, type); // LOD 0
  
  image->m_pixels = new unsigned char[image->m_nob];
  memcpy(image->m_pixels, pixels, image->m_nob);

  m_images[index].push_back(image); // LOD 0

  unsigned int w = width;
  unsigned int h = height;
  unsigned int d = depth;

  for (size_t i = 0; i < mipmaps.size(); ++i)
  {
    MY_ASSERT(mipmaps[i]); // No nullptr expected.

    calculateNextExtents(w, h, d, flags); // Mind that the flags let this work for layered mipmap chains!

    image = new Image(w, h, d, format, type); // LOD 1 to N.

    image->m_pixels = new unsigned char[image->m_nob];
    memcpy(image->m_pixels, mipmaps[i], image->m_nob);

    m_images[index].push_back(image); // LOD 1 - N
  }

  return index;
}

// Append a new image LOD to the images in index.
unsigned int Picture::addLevel(const unsigned int index, const void* pixels,
                               const unsigned int width, const unsigned int height, const unsigned int depth, 
                               const int format, const int type)
{
  MY_ASSERT(index < m_images.size());
  MY_ASSERT(pixels != nullptr);
  MY_ASSERT((0 < width) && (0 < height) && (0 < depth));

  Image* image = new Image(width, height, depth, format, type);

  image->m_pixels = new unsigned char[image->m_nob];
  memcpy(image->m_pixels, pixels, image->m_nob);
  
  const unsigned int level = static_cast<unsigned int>(m_images[index].size());

  m_images[index].push_back(image);

  return level;
}


void Picture::mirrorX(unsigned int index)
{
  MY_ASSERT(index < m_images.size());

  // Flip all images upside down.
  for (size_t i = 0; i < m_images[index].size(); ++i)
  {
    Image* image = m_images[index][i];

    const unsigned char* srcPixels = image->m_pixels;
    unsigned char*       dstPixels = new unsigned char[image->m_nob];

    for (unsigned int z = 0; z < image->m_depth; ++z) 
    {
      for (unsigned int y = 0; y < image->m_height; ++y) 
      {
        const unsigned char* srcLine = srcPixels + z * image->m_bps + y * image->m_bpl;
        unsigned char*       dstLine = dstPixels + z * image->m_bps + (image->m_height - 1 - y) * image->m_bpl;
      
        memcpy(dstLine, srcLine, image->m_bpl);
      }
    }
    delete[] image->m_pixels;
    image->m_pixels = dstPixels;
  }
}

void Picture::mirrorY(unsigned int index)
{
  MY_ASSERT(index < m_images.size());

  // Mirror all images left to right.
  for (size_t i = 0; i < m_images[index].size(); ++i)
  {
    Image* image = m_images[index][i];

    const unsigned char* srcPixels = image->m_pixels;
    unsigned char*       dstPixels = new unsigned char[image->m_nob];

    for (unsigned int z = 0; z < image->m_depth; ++z) 
    {
      for (unsigned int y = 0; y < image->m_height; ++y) 
      {
        const unsigned char* srcLine = srcPixels + z * image->m_bps + y * image->m_bpl;
        unsigned char*       dstLine = dstPixels + z * image->m_bps + y * image->m_bpl;

        for (unsigned int x = 0; x < image->m_width; ++x) 
        {
          const unsigned char* srcPixel = srcLine + x * image->m_bpp;
          unsigned char*       dstPixel = dstLine + (image->m_width - 1 - x) * image->m_bpp;

          memcpy(dstPixel, srcPixel, image->m_bpp);
        }
      }
    }

    delete[] image->m_pixels;
    image->m_pixels = dstPixels;
  }
}


void Picture::generateRGBA8(unsigned int width, unsigned int height, unsigned int depth, const unsigned int flags)
{
  clearImages();

  const unsigned char colors[14][4] = 
  {
    { 0xFF, 0x00, 0x00, 0xFF }, // red 
    { 0x00, 0xFF, 0x00, 0xFF }, // green
    { 0x00, 0x00, 0xFF, 0xFF }, // blue
    { 0xFF, 0xFF, 0x00, 0xFF }, // yellow
    { 0x00, 0xFF, 0xFF, 0xFF }, // cyan
    { 0xFF, 0x00, 0xFF, 0xFF }, // magenta
    { 0xFF, 0xFF, 0xFF, 0xFF }, // white

    { 0x7F, 0x00, 0x00, 0xFF }, // dark red 
    { 0x00, 0x7F, 0x00, 0xFF }, // dark green
    { 0x00, 0x00, 0x7F, 0xFF }, // dark blue
    { 0x7F, 0x7F, 0x00, 0xFF }, // dark yellow
    { 0x00, 0x7F, 0x7F, 0xFF }, // dark cyan
    { 0x7F, 0x00, 0x7F, 0xFF }, // dark magenta
    { 0x7F, 0x7F, 0x7F, 0xFF }  // grey
  };
  
  m_flags = flags;

  m_isCube = ((flags & IMAGE_FLAG_CUBE) != 0);

  unsigned char* rgba = new unsigned char[width * height * depth * 4]; // Enough to hold the LOD 0.
  
  const unsigned int numFaces = (m_isCube) ? 6 : 1; // Cubemaps put each face in a new images vector.
  
  for (unsigned int face = 0; face < numFaces; ++face)
  {
    const unsigned int index = addImages(); // New mipmap chain.
    MY_ASSERT(index == face);

    // calculateNextExtents() changes the w, h, d values. Restore them for each face.
    unsigned int w = width;
    unsigned int h = height;
    unsigned int d = depth;
    
    bool done = false; // Indicates if one mipmap chain is done.

    unsigned int idx = face; // Color index. Each face gets a different color.

    while (!done) 
    {
      unsigned char* p = rgba;

      for (unsigned int z = 0; z < d; ++z) 
      {
        for (unsigned int y = 0; y < h; ++y) 
        {
          for (unsigned int x = 0; x < w; ++x) 
          {
            p[0] = colors[idx][0];
            p[1] = colors[idx][1];
            p[2] = colors[idx][2];
            p[3] = colors[idx][3];
            p += 4;
          }
        }
      }

      idx = (idx + 1) % 14; // Next color index. Each mipmap level gets a different color.
    
      const unsigned int level = addLevel(index, rgba, w, h, d, IL_RGBA, IL_UNSIGNED_BYTE);

      if ((flags & IMAGE_FLAG_MIPMAP) == 0)
      {
        done = true;
      }
      else
      {
        done = calculateNextExtents(w, h, d, flags);
      }
    }
  }
  
  delete[] rgba;
}



// Functions related to IES light profiles.

// This is the "Type C point symmetry" case.
// Horizontal max angle == 0.
bool generateTypeC_PointSymmetry(IESData const& ies,
                                 std::vector<float>& horizontalAngles,
                                 std::vector<float>& candelas)
{
  // There need to be at least two values for 0 and 360 degrees.
  if (ies.photometric.numHorizontalAngles != 1 ||
      ies.photometric.horizontalAngles[0] != 0.0f)
  {
    MY_ASSERT(!"generateTypeC_PointSymmetry failed.")
    return false;
  }

  // Generate data for horizontal angles 0 and 360 degrees.
  horizontalAngles.resize(2);
  candelas.resize(2 * ies.photometric.numVerticalAngles);

  horizontalAngles[0] = 0.0f;
  horizontalAngles[1] = 360.0f;
  
  for (int j = 0; j < ies.photometric.numVerticalAngles; ++j)
  {
    candelas[j]                                     = ies.photometric.candela[j]; // copy 0 
    candelas[ies.photometric.numVerticalAngles + j] = ies.photometric.candela[j]; // symmetry 360 
  }

  return true;
}

// This is the Type C bilateral symmetry along the 0-180 degree photometric plane.
// This algorithm generates a new array of horizontal angle to cover the full circle from 0-360 phi.
// Everything after this routine can be handled as no symmetry case.
bool generateTypeC_BilateralSymmetryX(IESData const& ies,
                                      std::vector<float>& horizontalAngles,
                                      std::vector<float>& candelas)
{
  // There need to be at least two values for 0 and 180 degrees.
  if (ies.photometric.numHorizontalAngles < 2 ||
      ies.photometric.horizontalAngles[0] != 0.0f ||
      ies.photometric.horizontalAngles[ies.photometric.numHorizontalAngles - 1] != 180.0f)
  {
    MY_ASSERT(!"generateTypeC_BilateralSymmetryX failed.")
    return false;
  }

  // Calculate the new number of horizontal angle entries after the bilateral symmetry operation.
  // 360 degrees copies from 0 degrees. 
  // 180 degrees is not mirrored!
  // Means there are at least three values for 0, 180, and 360.
  // All values between 0 and 180 dregrees appear two times.
  const int n = 3 + (ies.photometric.numHorizontalAngles - 2) * 2;
  
  horizontalAngles.resize(n);
  candelas.resize(n * ies.photometric.numVerticalAngles);

  // Now calculate the new horizontal angles which build the full circle.
  int i    = 0;
  int iSym = n - 1;

  while (i < ies.photometric.numHorizontalAngles)
  {
    const float phi = ies.photometric.horizontalAngles[i];

    horizontalAngles[i   ] = phi;
    horizontalAngles[iSym] = 360.0f - phi; // 180 degrees entry is written twice, but that's ok.

    // Copy and mirror the candela value arrays per horizontal angle.
    const int ii = i    * ies.photometric.numVerticalAngles;
    const int is = iSym * ies.photometric.numVerticalAngles;
    
    for (int j = 0; j < ies.photometric.numVerticalAngles; ++j)
    {
      const float value = ies.photometric.candela[ii + j];

      candelas[ii + j] = value; // copy
      candelas[is + j] = value; // symmetry
    }

    i++;
    --iSym;
  }

  return true;
}


// This is the "Type C bilateral symmetry" along the 90-270 degree photometric plane.
bool generateTypeC_BilateralSymmetryY(IESData const& ies,
                                      std::vector<float>& horizontalAngles,
                                      std::vector<float>& candelas)
{
  // There need to be at least two values for 0 and 180 degrees.
  if (ies.photometric.numHorizontalAngles < 2 ||
      ies.photometric.horizontalAngles[0] != 90.0f ||
      ies.photometric.horizontalAngles[ies.photometric.numHorizontalAngles - 1] != 270.0f)
  {
    MY_ASSERT(!"generateTypeC_BilateralSymmetryY failed.")
    return false;
  }

  // Use a deque to generate the proper symmetry.
  // No need for complicated size calculations with special cases for 180 degrees or arbitrary horizontal angle distributions.
  std::deque<float>                dequeAngles;
  std::deque< std::vector<float> > dequeCandelas;
  
  std::vector<float> tempCandela; // Temporary copy of the vertical candela data array per horizontal angle.
  tempCandela.resize(ies.photometric.numVerticalAngles);

  // First populate the deques with the original data
  bool has180 = false; // Symmetry from 180 to 0 and 360 degrees requires interpolation if this stays false.
  for (int i = 0; i < ies.photometric.numHorizontalAngles; ++i) 
  {
    const float angle = ies.photometric.horizontalAngles[i];
    if (angle == 180.0f)
    {
      has180 = true; // Simple case: Indicates that there doesn't need to be an interpolation for the symmetry from 180 to 0 and 30 degrees.
    }
    dequeAngles.push_back(angle);

    const int idxSrc = i * ies.photometric.numVerticalAngles;
    for (int j = 0; j < ies.photometric.numVerticalAngles; ++j)
    {
      tempCandela[j] = ies.photometric.candela[idxSrc + j];
    }
    dequeCandelas.push_back(tempCandela);
  }

  // Now handle the symmetry along the 90-270 axis.
  // Fill the first quadrant from 90 - 0 degrees.
  // Going forward in the original data goes backward at the front of the deque, from 90 - 0 degrees.
  for (int i = 1; i < ies.photometric.numHorizontalAngles - 1; ++i) // Exclude the 90 and 270 degrees entries.
  {
    const float angle = ies.photometric.horizontalAngles[i];
    if (angle <= 180.0f) // angle in range (90, 180]
    {
      dequeAngles.push_front(180.0f - angle);

      const int idxSrc = i * ies.photometric.numVerticalAngles;
      for (int j = 0; j < ies.photometric.numVerticalAngles; ++j)
      {
        tempCandela[j] = ies.photometric.candela[idxSrc + j];
      }
      dequeCandelas.push_front(tempCandela);
    }
    else
    {
      break; // 180 < angle, no symmetry for that quadrant in this loop.
    }
  }
  // Fill the fourth quadrant from 270 to 360 degrees.
  // Going backwards in the original data goes forward at the back of the deque from 270 - 360.
  for (int i = ies.photometric.numHorizontalAngles - 2; 0 < i; --i) // Exclude the 90 and 270 degrees entries.
  {
    const float angle = ies.photometric.horizontalAngles[i];
    if (180.0f <= angle) // angle in range [180, 270)
    {
      dequeAngles.push_back(270.0f + (270.0f - angle));

      const int idxSrc = i * ies.photometric.numVerticalAngles;
      for (int j = 0; j < ies.photometric.numVerticalAngles; ++j)
      {
        tempCandela[j] = ies.photometric.candela[idxSrc + j];
      }
      dequeCandelas.push_back(tempCandela);
    }
    else // angle < 180, handled already in the first loop.
    {
      break;
    }
  }

  // If there is no direct entry for 180 degrees, the 0 and 360 degrees do not exist, yet.
  // The candela values for them need to be interpolated!
  if (!has180)
  {
    // Find the indices around them and the interpolant to fill the 0 and 360 degrees candela arrays later.
    int indexLo; 
    int indexHi;
    float t_180 = 0.0f;
    // Always entered because ies.photometric.numHorizontalAngles is at least 2.
    for (int i = 0; i < ies.photometric.numHorizontalAngles - 1; ++i)
    {
      indexLo = i;
      indexHi = i + 1;
      
      const float phiLo = ies.photometric.horizontalAngles[indexLo];
      const float phiHi = ies.photometric.horizontalAngles[indexHi];
      
      if (phiLo <= 180.0f && 180.0f < phiHi)
      {
        t_180 = (180.0f - phiLo) / (phiHi - phiLo);
        break;
      }
    }

    MY_ASSERT(indexLo != -1 && indexHi != -1);

    // There can't be entries for 0 and 360 degrees if no value for 180 degrees was in the list of horizontal angles.
    MY_ASSERT(0.0f < dequeAngles.front());
    MY_ASSERT(dequeAngles.back() < 360.0f);

    // Calculate the interpolated vertical angles.
    indexLo *= ies.photometric.numVerticalAngles;
    indexHi *= ies.photometric.numVerticalAngles;
    
    for (int j = 0; j < ies.photometric.numVerticalAngles; ++j)
    {
      const float candelaLo = ies.photometric.candela[indexLo + j];
      const float candelaHi = ies.photometric.candela[indexHi + j];

      // The interpolated array of vertical candela values for the 180 degrees horizontal angle.
      // This array is mirrored from 180 to both the 0 and 360 degrees entries.
      tempCandela[j] = dp::math::lerp(t_180, candelaLo, candelaHi);
    }

    dequeAngles.push_front(0.0f);
    dequeCandelas.push_front(tempCandela);

    dequeAngles.push_back(360.0f);
    dequeCandelas.push_back(tempCandela);
  }

  // Copy the results back to the output data.
  const size_t sizeAngles = dequeAngles.size();
  horizontalAngles.resize(sizeAngles);
  candelas.resize(sizeAngles * ies.photometric.numVerticalAngles);

  size_t idx = 0; 
  for (std::deque<float>::const_iterator it = dequeAngles.begin(); it != dequeAngles.end(); ++it)
  {
    horizontalAngles[idx++] = *it;
  }
  
  idx = 0; 
  for (std::deque< std::vector<float> >::const_iterator it = dequeCandelas.begin(); it != dequeCandelas.end(); ++it)
  {
    const size_t idxDst = idx++ * ies.photometric.numVerticalAngles;
    for (size_t j = 0; j < ies.photometric.numVerticalAngles; ++j)
    {
      candelas[idxDst + j] = (*it)[j];
    }
  }

  return true;
}

// This is the "Type C quadrant symmetry" case.
bool generateTypeC_QuadrantSymmetry(IESData const& ies,
                                    std::vector<float>& horizontalAngles,
                                    std::vector<float>& candelas)
{
  // There need to be at least two values for 0 and 90 degrees.
  if (ies.photometric.numHorizontalAngles < 2 ||
      ies.photometric.horizontalAngles[0] != 0.0f ||
      ies.photometric.horizontalAngles[ies.photometric.numHorizontalAngles - 1] != 90.0f)
  {
    MY_ASSERT(!"generateTypeC_QuadrantSymmetry failed.")
    return false;
  }

  // Calculate the new number of horizontal angle entries after the quadrant symmetry operation.
  // 180 degrees copies from 0 degrees.
  // 270 degrees copies from 90 degrees.
  // 360 degrees copies from 0 degrees.
  // Means there are at least five values for 0, 90, 180, 270 and 360.
  // All inner values between 0 and 90 degrees are present four times.
  const int n = 5 + (ies.photometric.numHorizontalAngles - 2) * 4;
  
  horizontalAngles.resize(n);
  candelas.resize(n * ies.photometric.numVerticalAngles);

  // Now calculate the new horizontal angles which build the full circle.
  // Indices are quadrants 1 to 4.
  int i1 = 0;                                             // Quadrant 1: 0 to 90, the original data.
  int i2 = 2 * (ies.photometric.numHorizontalAngles - 1); // Quadrant 2: 180 to 90
  int i3 = i2;                                            // Quadrant 3: 180 to 270
  int i4 = n - 1;                                         // Quadrant 4: 360 to 270

  while (i1 < ies.photometric.numHorizontalAngles)
  {
    const float phi = ies.photometric.horizontalAngles[i1];

    horizontalAngles[i1] = phi;
    horizontalAngles[i2] = 180.0f - phi;
    horizontalAngles[i3] = 180.0f + phi;
    horizontalAngles[i4] = 360.0f - phi;

    // Copy and mirror the candela value arrays per horizontal angle.
    const int idx1 = i1 * ies.photometric.numVerticalAngles;
    const int idx2 = i2 * ies.photometric.numVerticalAngles;
    const int idx3 = i3 * ies.photometric.numVerticalAngles;
    const int idx4 = i4 * ies.photometric.numVerticalAngles;
    
    for (int j = 0; j < ies.photometric.numVerticalAngles; ++j)
    {
      const float value = ies.photometric.candela[idx1 + j];

      candelas[idx1 + j] = value;
      candelas[idx2 + j] = value;
      candelas[idx3 + j] = value;
      candelas[idx4 + j] = value;
    }
    
    ++i1;
    --i2;
    ++i3;
    --i4;
  }

  return true;
}

// This is the "Type C no symmetry" case.
// Horizontal min angle < 90 and horizontal max angle > 180, normally 0 to 360.
// FIXME Only supporting the 0 to 360 case here.
bool generateTypeC_NoSymmetry(IESData const& ies,
                              std::vector<float>& horizontalAngles,
                              std::vector<float>& candelas)
{
  // There need to be at least two values for 0 and 360 degrees.
  if (ies.photometric.numHorizontalAngles < 2 ||
      ies.photometric.horizontalAngles[0] != 0.0f ||
      ies.photometric.horizontalAngles[ies.photometric.numHorizontalAngles - 1] != 360.0f)
  {
    MY_ASSERT(!"generateTypeC_NoSymmetry failed.")
    return false;
  }

  // Just copying the data over to the output arrays.
  horizontalAngles.resize(ies.photometric.numHorizontalAngles);
  candelas.resize(ies.photometric.numHorizontalAngles * ies.photometric.numVerticalAngles);

  for (int i = 0; i < ies.photometric.numHorizontalAngles; ++i)
  {
    horizontalAngles[i] = ies.photometric.horizontalAngles[i];

    const int idx = i * ies.photometric.numVerticalAngles;
    for (int j = 0; j < ies.photometric.numVerticalAngles; ++j)
    {
      candelas[idx + j] = ies.photometric.candela[idx + j]; // copy
    }
  }
  return true;
}

// This is the "Type A or B lateral symmetry" case.
// Horizontal min angle == 0 and horizontal max angle == 90.
bool generateTypeAB_BilateralSymmetry(IESData const& ies,
                                      std::vector<float>& horizontalAngles,
                                      std::vector<float>& candelas)
{
  // There need to be at least two values for 0 and 90 degrees.
  if (ies.photometric.numHorizontalAngles < 2 ||
      ies.photometric.horizontalAngles[0] != 0.0f ||
      ies.photometric.horizontalAngles[ies.photometric.numHorizontalAngles - 1] != 90.0f)
  {
    MY_ASSERT(!"generateTypeAB_BilateralSymmetry failed.")
    return false;
  }

  // Calculate the new number of horizontal angle entries after the bilateral symmetry operation.
  // -90 degrees copies from 90 degrees. 
  // 0 degrees is not mirrored!
  // Means there are at least three values for -90, 0, and 90.
  // All values between 0 and 90 dregrees appear two times.
  const int n = 3 + (ies.photometric.numHorizontalAngles - 2) * 2;
  
  horizontalAngles.resize(n);
  candelas.resize(n * ies.photometric.numVerticalAngles);

  // Now calculate the new horizontal angles which build the hemisphere
  int iCopy = ies.photometric.numHorizontalAngles - 1; // Start at the center with the 0 degrees entry.
  int iSym  = iCopy;                                   // iSym runs backwards and gets the negative angles

  for (int i = 0; i < ies.photometric.numHorizontalAngles; ++i)
  {
    const float phi = ies.photometric.horizontalAngles[i];

    horizontalAngles[iSym]  = -phi; // Symmetry.
    horizontalAngles[iCopy] =  phi; // Copy. 0 degrees entry is written twice, but that's ok. (Keep the positive 0.)

    // Copy and mirror the candela value arrays per horizontal angle.
    const int ii = i     * ies.photometric.numVerticalAngles;
    const int ic = iCopy * ies.photometric.numVerticalAngles;
    const int is = iSym  * ies.photometric.numVerticalAngles;
    
    for (int j = 0; j < ies.photometric.numVerticalAngles; ++j)
    {
      const float value = ies.photometric.candela[ii + j];

      candelas[is + j] = value; // Symmetry.
      candelas[ic + j] = value; // Copy.
    }
    
    ++iCopy;
    --iSym;
  }

  return true;
}


// This is the "Type A or B no symmetry" case.
// Horizontal min angle == -90 and horizontal max angle == 90.
bool generateTypeAB_NoSymmetry(IESData const& ies,
                               std::vector<float>& horizontalAngles,
                               std::vector<float>& candelas)
{
  // There need to be at least two values for -90 and 90 degrees.
  if (ies.photometric.numHorizontalAngles < 2 ||
      ies.photometric.horizontalAngles[0] != -90.0f ||
      ies.photometric.horizontalAngles[ies.photometric.numHorizontalAngles - 1] != 90.0f)
  {
    MY_ASSERT(!"generateTypeAB_NoSymmetry failed.")
    return false;
  }

  // Just copying the data over to the output arrays.
  horizontalAngles.resize(ies.photometric.numHorizontalAngles);
  candelas.resize(ies.photometric.numHorizontalAngles * ies.photometric.numVerticalAngles);

  for (int i = 0; i < ies.photometric.numHorizontalAngles; ++i)
  {
    horizontalAngles[i] = ies.photometric.horizontalAngles[i];

    const int idx = i * ies.photometric.numVerticalAngles;
    for (int j = 0; j < ies.photometric.numVerticalAngles; ++j)
    {
      candelas[idx + j] = ies.photometric.candela[idx + j]; // copy
    }
  }
  return true;
}

bool Picture::generateIES(const IESData& ies,
                          const std::vector<float>& horizontalAngles,
                          const std::vector<float>& candelas)
{
  // This is the replacement for ies.photometric.numHorizontalAngles after the symmetry operations have been applied.
  const size_t sizeHorizontalAngles = horizontalAngles.size();

  if (sizeHorizontalAngles < 2 ||
      ies.photometric.numVerticalAngles < 2 || 
      candelas.size() != sizeHorizontalAngles * ies.photometric.numVerticalAngles)
  {
    MY_ASSERT(!"generateIES() failed.")
    return false;
  }

  // Multiply the lamp multiplier, ballast factor and ballast-lamp photometric factor
  float multiplier = (0.0f < ies.lamp.multiplier) ? ies.lamp.multiplier : 1.0f;
  multiplier *= ies.electrical.ballastFactor * ies.electrical.ballastLampPhotometricFactor;

  std::vector<float> pixels; // This receives the full spherical polar grid IES data.

#if 0
  // FIXME Using the IES data as-is with linear texturing doesn't get the angles correct
  // because texels are sampled in the center but the data is for the lower left corner.
  // E.g. with horizontalAngularResolution == 45 the light would be rotated by 22.5 degrees by the texture sampling.
  // This would also not place the spherical coordinate (0, 0) at the desired location.
  // Manual interpolation inside the device code would need nearest filtering and knowledge about the goniometer type.
  // Possible but that would require multiple IES light sampling functions.
  // 
  // Be pragmatic and always expand the data to a fixed size spherical float texture, which also simplifies support
  // for goniometer type A and B which both only define a hemispherical distribution around polar coordinate (0, 0).
  // With the resolution 720x360 the horizontal deviation from the original input is only 0.5 degress at maximum.
  // That's the #else clause.

  // First check if both horizontal and vertical angles have equidistant distributions of angles.
  bool horizontalRegular = true;
  const float horizontalAngularResolution = horizontalAngles[1] - horizontalAngles[0];
  for (int i = 1; horizontalRegular && i < sizeHorizontalAngles - 1; ++i)
  {
    if (0.01f < fabsf(horizontalAngularResolution - (horizontalAngles[i + 1] - horizontalAngles[i])))
    {
      horizontalRegular = false;
    }
  }

  bool verticalRegular = true;
  const float verticalAngularResolution = ies.photometric.verticalAngles[1] - ies.photometric.verticalAngles[0];
  for (int i = 1; verticalRegular && i < ies.photometric.numVerticalAngles - 1; ++i)
  {
    if (0.01f < fabsf(verticalAngularResolution - (ies.photometric.verticalAngles[i + 1] - ies.photometric.verticalAngles[i])))
    {
      verticalRegular = false;
    }
  }

  // This is the case most often encountered with real measured data where the angular grid is in regular intervals.
  if (horizontalRegular && verticalRegular)
  {
    if (ies.photometric.verticalAngles[0] == 0.0f &&
        ies.photometric.verticalAngles[ies.photometric.numVerticalAngles - 1] == 180.0f)
    {
      // Full definition with regular grid. Just copy the data.
      width  = (unsigned int) (sizeHorizontalAngles - 1); // phi == 360 is not uploaded! Texture repeats in phi direction.
      height = ies.photometric.numVerticalAngles;         // Poles are uploaded!

      pixels.resize(m_width * m_height);

      for (int y = 0; y < (int) m_height; ++y)
      {
        for (int x = 0; x < (int) m_width; ++x)
        {
          const float candela = candelas[x * ies.photometric.numVerticalAngles + y];
          m_texels[y * m_width + x] = candela * multiplier;
        }
      }
    }
    else if (ies.photometric.verticalAngles[0] == 0.0f &&
             ies.photometric.verticalAngles[ies.photometric.numVerticalAngles - 1] == 90.0f)
    {
      // Lower hemispher with regular regular grid. Increase the size and fill upper hemisphere with zero.
      m_width  = (unsigned int) (sizeHorizontalAngles - 1); // phi == 360 is not uploaded! Texture repeats in phi direction.
      m_height = ies.photometric.numVerticalAngles * 2 - 1; // Poles are uploaded! -1 for the 90 degrees entry which is not duplicated.
      m_depth  = 1;

      m_texels.resize(m_width * m_height);

      for (int y = 0; y < (int) m_height; ++y)
      {
        for (int x = 0; x < (int) m_width; ++x)
        {
          float candela = 0.0f;
          if (y < ies.photometric.numVerticalAngles)
          {
            candela = candelas[x * ies.photometric.numVerticalAngles + y];
          }
          m_texels[y * m_width + x] = candela * multiplier;
        }
      }
    }
    else if (ies.photometric.verticalAngles[0] == 90.0f &&
             ies.photometric.verticalAngles[ies.photometric.numVerticalAngles - 1] == 180.0f)
    {
      // Upper hemisphere with regular regular grid. Increase the size and fill lower hemisphere with zero.
      m_width  = int(sizeHorizontalAngles - 1);                  // phi == 360 is not uploaded! Texture repeats in phi direction.
      m_height = int(ies.photometric.numVerticalAngles * 2 - 1); // Poles are uploaded! -1 for the 90 degrees entry which is not duplicated.
      m_depth  = 1;

      m_texels.resize(m_width * m_height);

      for (int y = 0; y < (int) m_height; ++y)
      {
        for (int x = 0; x < (int) m_width; ++x)
        {
          float candela = 0.0f;
          if (ies.photometric.numVerticalAngles - 1 <= y)
          {
            candela = candelas[x * ies.photometric.numVerticalAngles + y - (ies.photometric.numVerticalAngles - 1)];
          }
          // I want the vertical angle 0 to be at lookup position v == 1.0f. Invert the y-index!
          m_texels[y * m_width + x] = candela * multiplier;
        }
      }
    }
  }
  else // The polar grid is not regular.
  {
#endif

  // Use a hardcoded 0.5 degrees resolution (720x360 image) to generate the full spherical polar grid texture.
  // This resolution covers most of the angular grids in IES files I've seen. 
  // This texture is about a megabyte in size.
  // Note that since the textures are interpolated in the texel center and the polar grid is defined on the corners, 
  // the horizontal angle is 0.25 degrees off during rendering.
  const unsigned int width  = 720;
  const unsigned int height = 361; // 360 + 1 to upload the poles as well.

  pixels.resize(width * height);
  
  // Scale value to get the actual phi and theta values per texel coordinate.
  const float scaleX = 360.0f / float(width);
  const float scaleY = 180.0f / float(height - 1);

  for (unsigned int y = 0; y < height; ++y)
  {
    float theta = float(y) * scaleY; // [0, 180]
      
    if (ies.photometric.goniometerType != TYPE_C) // Type A and B have vertical angles in the range [-90, 90].
    {
      theta -= 90.0f; // [-90, 90]
    }

    // Find the vertical angle interval which contains this theta. // PERF Make this incremental.
    int   iTheta = -1;
    float tTheta = 0.0f;
    for (int i = 0; i < ies.photometric.numVerticalAngles - 1; ++i)
    {
      const float thetaLo = ies.photometric.verticalAngles[i];
      const float thetaHi = ies.photometric.verticalAngles[i + 1];
      if (thetaLo <= theta && theta <= thetaHi && thetaLo != thetaHi)
      {
        iTheta = i;
        tTheta = (theta - thetaLo) / (thetaHi - thetaLo);
        break;
      }
    }

    for (unsigned int x = 0; x < width; ++x)
    {
      float candela = 0.0f; // Black when no interval in the source data contains this (phi, theta) coordinate.

      if (0 <= iTheta) // Only walk through the horizontal angles when there actually was data for this theta.
      {
        float phi = float(x) * scaleX; // [0, 359.5]
          
        if (ies.photometric.goniometerType == TYPE_C)
        {
          // According to documentation I found, the goniometer type C has the angle oriented counter-clockwise
          // compared to the goniometer A and B types where the horizontal range [-90, 90] is clockwise.
          // Well, at least that is mathematically positive, but that inconsistency is rather disturbing.
          // Invert the angle and adjust the Type C goniometer to have the (0, 90) coordinate in the center of the image.
          phi = 540.0f - phi; // == 360.0f - phi + 180.0f; 
          if (360.0f <= phi)
          {
            phi -= 360.0f;
          }
        }
        else // Goniometer types A and B  have horizontal angles only in the range [-90, 90]. The rest is black.
        {
          // Keep goniometer type A and B coordinate (0, 0) at the correct spherical location (center of the texture).
          phi -= 180.0f; // [-180.0, 179.5]
        }

        // Find the horizontal angle interval which contains this phi.
        // PERF Make this incremental. There are generally very few horizontal angles though.
        int   iPhi = -1;
        float tPhi = 0.0f;
        for (int i = 0; i < int(sizeHorizontalAngles - 1); ++i)
        {
          const float phiLo = horizontalAngles[i];
          const float phiHi = horizontalAngles[i + 1];
          if (phiLo <= phi && phi <= phiHi && phiLo != phiHi)
          {
            iPhi = i;
            tPhi = (phi - phiLo) / (phiHi - phiLo);
            break;
          }
        }

        if (0 <= iPhi && 0 <= iTheta)
        {
          const int idxLL =  iPhi      * ies.photometric.numVerticalAngles + iTheta;
          const int idxLR = (iPhi + 1) * ies.photometric.numVerticalAngles + iTheta;
          const int idxUL =  idxLL + 1;
          const int idxUR =  idxLR + 1;

          const float candelaLower = dp::math::lerp(tPhi, candelas[idxLL], candelas[idxLR]);
          const float candelaUpper = dp::math::lerp(tPhi, candelas[idxUL], candelas[idxUR]);
          
          candela = dp::math::lerp(tTheta, candelaLower, candelaUpper) * multiplier;
        }
      }

      pixels[y * width + x] = candela;
    }
  }
  
  // Set the data in pixels as 1-component luminance float image.
  clear();
  const unsigned int index = addImages();
  (void) addLevel(index, pixels.data(), width, height, 1, IL_LUMINANCE, IL_FLOAT);
  
  // Indicate that this Picture contains IES data.
  // The Texture needs to be kept as 1-component float and use  wrap clamp in vertical direction.
  m_flags = IMAGE_FLAG_2D | IMAGE_FLAG_IES; 

  return true;
}

bool Picture::createIES(const IESData& ies)
{
  bool success = false;

  // Resulting vectors with new horizontal angles and candela entries after symmetry operations.
  std::vector<float> horizontalAngles;
  std::vector<float> candelas;

  MY_ASSERT(0 < ies.photometric.numHorizontalAngles);

  const float minHorizontalAngle = ies.photometric.horizontalAngles[0];
  const float maxHorizontalAngle = ies.photometric.horizontalAngles[ies.photometric.numHorizontalAngles - 1];

  // Figure out symmetry.
  switch (ies.photometric.goniometerType)
  {
  case TYPE_A:
  case TYPE_B:
    if (minHorizontalAngle == 0.0f && maxHorizontalAngle == 90.0f)
    {
      success = generateTypeAB_BilateralSymmetry(ies, horizontalAngles, candelas);
    }
    else
    {
      success = generateTypeAB_NoSymmetry(ies, horizontalAngles, candelas);
    }
    break;

  case TYPE_C:
    if (maxHorizontalAngle <= 0.0f)
    {
      success = generateTypeC_PointSymmetry(ies, horizontalAngles, candelas);
    }
    else if (maxHorizontalAngle == 90.0f)
    {
      success = generateTypeC_QuadrantSymmetry(ies, horizontalAngles, candelas);
    }
    else if (maxHorizontalAngle == 180.0f)
    {
      success = generateTypeC_BilateralSymmetryX(ies, horizontalAngles, candelas);
    }
    else if (minHorizontalAngle == 90.0f && maxHorizontalAngle == 270.0f)
    {
      success = generateTypeC_BilateralSymmetryY(ies, horizontalAngles, candelas);
    }
    else
    {
      success = generateTypeC_NoSymmetry(ies, horizontalAngles, candelas);
    }
    break;
  }

  // After any of the above symmetry functions return, the horizontalAngles define the full range from 0 to 360 degrees.
  // Means a single function to convert this to a lookup texture.
  if (success)
  {
    success = generateIES(ies, horizontalAngles, candelas);
  }

  //m_goniometerType = ies.photometric.goniometerType;

  return success;
}
