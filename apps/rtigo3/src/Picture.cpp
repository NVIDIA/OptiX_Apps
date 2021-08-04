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


#include "inc/Picture.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>

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
: m_isCube(false)
{
}

Picture::~Picture()
{
  clearImages();
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


bool Picture::load(std::string const& filename, const unsigned int flags)
{
  bool success = false;

  clearImages(); // Each load() wipes previously loaded image data.

  std::string foundFile = filename; // FIXME Search at least the current working directory.
  if (foundFile.empty())
  {
    std::cerr << "ERROR Picture::load(): " << filename << " not found\n";
    MY_ASSERT(!"Picture not found");
    return success;
  }

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
  if (ilLoadImage((const ILstring) foundFile.c_str()))
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
          std::cerr << "ERROR Image::load(): " << filename << ": image " << image << " face " << f << " extents (" << width << ", " << height << ", " << depth << ")\n";
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

  if (!success)
  {
    std::cerr << "ERROR Picture::load(): " << filename << " not loaded\n";
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

  m_images.push_back(std::vector<Image*>()); // Append a new empty vector of image pointers. Each vector holds a mipmap chain.

  return index;
}

// Add a vector of images and fill it with the LOD 0 pixels and the optional mipmap chain.
unsigned int Picture::addImages(const void* pixels, 
                                const unsigned int width, const unsigned int height, const unsigned int depth,
                                const int format, const int type,
                                std::vector<const void*> const& mipmaps, const unsigned int flags)
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


// DEBUG
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
  
  m_isCube = (flags & IMAGE_FLAG_CUBE) != 0;

  unsigned char* rgba = new unsigned char[width * height * depth * 4]; // Enough to hold the LOD 0.
  
  const unsigned int numFaces = (flags & IMAGE_FLAG_CUBE) ? 6 : 1; // Cubemaps put each face in a new images vector.
  
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
