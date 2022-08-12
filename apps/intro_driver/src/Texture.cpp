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

#include "shaders/app_config.h"

#include "shaders/vector_math.h"

#include "inc/Texture.h"
#include "inc/CheckMacros.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>

#include "inc/MyAssert.h"



// The ENC_RED|GREEN|BLUE|ALPHA|LUM codes define from which source channel is read when writing R, G, B, A or L.
static unsigned int determineHostEncoding(int format, int type) // format and type are DevIL defines.
{
  unsigned int encoding;

  switch (format)
  {
    case IL_RGB:
      encoding = ENC_RED_0 | ENC_GREEN_1 | ENC_BLUE_2 | ENC_ALPHA_NONE | ENC_LUM_NONE | ENC_CHANNELS_3;
      break;
    case IL_RGBA:
      encoding = ENC_RED_0 | ENC_GREEN_1 | ENC_BLUE_2 | ENC_ALPHA_3 | ENC_LUM_NONE | ENC_CHANNELS_4;
      break;
    case IL_BGR:
      encoding = ENC_RED_2 | ENC_GREEN_1 | ENC_BLUE_0 | ENC_ALPHA_NONE | ENC_LUM_NONE | ENC_CHANNELS_3;
      break;
    case IL_BGRA:
      encoding = ENC_RED_2 | ENC_GREEN_1 | ENC_BLUE_0 | ENC_ALPHA_3 | ENC_LUM_NONE | ENC_CHANNELS_4;
      break;
    case IL_LUMINANCE: 
      // encoding = ENC_RED_NONE | ENC_GREEN_NONE | ENC_BLUE_NONE | ENC_ALPHA_NONE | ENC_LUM_0 | ENC_CHANNELS_1;
      encoding = ENC_RED_0 | ENC_GREEN_0 | ENC_BLUE_0 | ENC_ALPHA_NONE | ENC_LUM_NONE | ENC_CHANNELS_1; // Source RGB from L to expand to (L, L, L, 1).
      break;
    case IL_ALPHA:
      encoding = ENC_RED_NONE | ENC_GREEN_NONE | ENC_BLUE_NONE | ENC_ALPHA_0 | ENC_LUM_NONE | ENC_CHANNELS_1;
      break;
    case IL_LUMINANCE_ALPHA:
      // encoding = ENC_RED_NONE | ENC_GREEN_NONE | ENC_BLUE_NONE | ENC_ALPHA_1 | ENC_LUM_0 | ENC_CHANNELS_2;
      encoding = ENC_RED_0 | ENC_GREEN_0 | ENC_BLUE_0 | ENC_ALPHA_1 | ENC_LUM_NONE | ENC_CHANNELS_2; // Source RGB from L to expand to (L, L, L, A).
      break;
    default:
      MY_ASSERT(!"Unsupported user pixel format.");
      encoding = ENC_RED_NONE | ENC_GREEN_NONE | ENC_BLUE_NONE | ENC_ALPHA_NONE | ENC_LUM_NONE | ENC_INVALID; // Error! Invalid encoding.
      break;
  }

  switch (type)
  {
    case IL_UNSIGNED_BYTE:
      encoding |= ENC_TYPE_UNSIGNED_CHAR;
      break;
    case IL_UNSIGNED_SHORT:
      encoding |= ENC_TYPE_UNSIGNED_SHORT;
      break;
    case IL_UNSIGNED_INT:
      encoding |= ENC_TYPE_UNSIGNED_INT;
      break;
    case IL_BYTE:
      encoding |= ENC_TYPE_CHAR;
      break;
    case IL_SHORT:
      encoding |= ENC_TYPE_SHORT;
      break;
    case IL_INT:
      encoding |= ENC_TYPE_INT;
      break;
    case IL_FLOAT:
      encoding |= ENC_TYPE_FLOAT;
      break;
    default:
      MY_ASSERT(!"Unsupported user data format.");
      encoding |= ENC_INVALID; // Error! Invalid encoding.
      break;
  }
        
  return encoding;
}

// For OpenGL interop these formats are supported by CUDA according to the current manual on cudaGraphicsGLRegisterImage:
// GL_RED, GL_RG, GL_RGBA, GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA, GL_INTENSITY
// {GL_R, GL_RG, GL_RGBA} x {8, 16, 16F, 32F, 8UI, 16UI, 32UI, 8I, 16I, 32I}
// {GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA, GL_INTENSITY} x {8, 16, 16F_ARB, 32F_ARB, 8UI_EXT, 16UI_EXT, 32UI_EXT, 8I_EXT, 16I_EXT, 32I_EXT}

// The following mapping is done for host textures. RGB formats will be expanded to RGBA.

// While single and dual channel textures can easily be uploaded, the texture doesn't know what the destination format actually is,
// that is, a LUMINANCE_ALPHA texture returns the luminance in the red channel and the alpha in the green channel.
// That doesn't work the same way as OpenGL which copies luminance to all three RGB channels automatically.
// DEBUG Check how the tex*<>(obj, ...) templates react when asking for more data than in the texture.
static unsigned int determineDeviceEncoding(int format, int type) // format and type are DevIL defines.
{
  unsigned int encoding;

  switch (format)
  {
    case IL_RGB:
    case IL_BGR:
      encoding = ENC_RED_0 | ENC_GREEN_1 | ENC_BLUE_2 | ENC_ALPHA_3 | ENC_LUM_NONE | ENC_CHANNELS_4 | ENC_ALPHA_ONE; // (R, G, B, 1)
      break;
    case IL_RGBA:
    case IL_BGRA:
      encoding = ENC_RED_0 | ENC_GREEN_1 | ENC_BLUE_2 | ENC_ALPHA_3 | ENC_LUM_NONE | ENC_CHANNELS_4; // (R, G, B, A)
      break;
    case IL_LUMINANCE: 
      //encoding = ENC_RED_NONE | ENC_GREEN_NONE | ENC_BLUE_NONE | ENC_ALPHA_NONE | ENC_LUM_0 | ENC_CHANNELS_1; // L in R
      encoding = ENC_RED_0 | ENC_GREEN_1 | ENC_BLUE_2 | ENC_ALPHA_3 | ENC_LUM_NONE | ENC_CHANNELS_4 | ENC_ALPHA_ONE; // Expands to (L, L, L, 1)
      break;
    case IL_ALPHA:
      //encoding = ENC_RED_NONE | ENC_GREEN_NONE | ENC_BLUE_NONE | ENC_ALPHA_0 | ENC_LUM_NONE | ENC_CHANNELS_1; // A in R
      encoding = ENC_RED_0 | ENC_GREEN_1 | ENC_BLUE_2 | ENC_ALPHA_3 | ENC_LUM_NONE | ENC_CHANNELS_4; // Expands to (0, 0, 0, A)
      break;
    case IL_LUMINANCE_ALPHA:
      //encoding = ENC_RED_NONE | ENC_GREEN_NONE | ENC_BLUE_NONE | ENC_ALPHA_1 | ENC_LUM_0 | ENC_CHANNELS_2; // LA in RG
      encoding = ENC_RED_0 | ENC_GREEN_1 | ENC_BLUE_2 | ENC_ALPHA_3 | ENC_LUM_NONE | ENC_CHANNELS_4; // Expands to (L, L, L, A)
      break;
    default:
      MY_ASSERT(!"Unsupported user pixel format.");
      encoding = ENC_RED_NONE | ENC_GREEN_NONE | ENC_BLUE_NONE | ENC_ALPHA_NONE | ENC_LUM_NONE | ENC_INVALID; // Error! Invalid encoding.
      break;
  }

  switch (type)
  {
    case IL_BYTE:
      encoding |= ENC_TYPE_CHAR | ENC_FIXED_POINT;
      break;
    case IL_UNSIGNED_BYTE:
      encoding |= ENC_TYPE_UNSIGNED_CHAR | ENC_FIXED_POINT;
      break;
    case IL_SHORT:
      encoding |= ENC_TYPE_SHORT | ENC_FIXED_POINT;
      break;
    case IL_UNSIGNED_SHORT:
      encoding |= ENC_TYPE_UNSIGNED_SHORT | ENC_FIXED_POINT;
      break;
    case IL_INT:
      encoding |= ENC_TYPE_INT | ENC_FIXED_POINT;
      break;
    case IL_UNSIGNED_INT:
      encoding |= ENC_TYPE_UNSIGNED_INT | ENC_FIXED_POINT;
      break;
    case IL_FLOAT:
      encoding |= ENC_TYPE_FLOAT;
      break;
    // FIXME Add IL_HALF for EXR images. Why are they loaded as IL_FLOAT?
    default:
      MY_ASSERT(!"Unsupported user data format.");
      encoding |= ENC_INVALID; // Error! Invalid encoding.
      break;
  }

  return encoding;
}

// Helper function calculating the CUarray_format 
static void determineFormatChannels(const unsigned int deviceEncoding, CUarray_format& format, unsigned int& numChannels)
{
  const unsigned int type = deviceEncoding & (ENC_MASK << ENC_TYPE_SHIFT);

  switch (type)
  {
    case ENC_TYPE_CHAR:
      format = CU_AD_FORMAT_SIGNED_INT8;
      break;
    case ENC_TYPE_UNSIGNED_CHAR:
      format = CU_AD_FORMAT_UNSIGNED_INT8;
      break;
    case ENC_TYPE_SHORT:
      format = CU_AD_FORMAT_SIGNED_INT16;
      break;
    case ENC_TYPE_UNSIGNED_SHORT:
      format = CU_AD_FORMAT_UNSIGNED_INT16;
      break;
    case ENC_TYPE_INT:
      format = CU_AD_FORMAT_SIGNED_INT32;
      break;
    case ENC_TYPE_UNSIGNED_INT:
      format = CU_AD_FORMAT_UNSIGNED_INT32;
      break;
    //case ENC_TYPE_HALF: // FIXME Implement.
    //  format = CU_AD_FORMAT_HALF;
    //  break;
    case ENC_TYPE_FLOAT:
      format = CU_AD_FORMAT_FLOAT;
      break;
    default:
      MY_ASSERT(!"determineFormatChannels() Unexpected data type.");
      break;
  }

  numChannels = (deviceEncoding >> ENC_CHANNELS_SHIFT) & ENC_MASK;
}


static unsigned int getElementSize(const unsigned int deviceEncoding)
{
  unsigned int bytes = 0;

  const unsigned int type = deviceEncoding & (ENC_MASK << ENC_TYPE_SHIFT);
  switch (type)
  {
    case ENC_TYPE_CHAR:
    case ENC_TYPE_UNSIGNED_CHAR:
      bytes = 1;
      break;
    case ENC_TYPE_SHORT:
    case ENC_TYPE_UNSIGNED_SHORT:
    //case ENC_TYPE_HALF: // FIXME Implement.
    bytes = 2;
      break;
    case ENC_TYPE_INT:
    case ENC_TYPE_UNSIGNED_INT:
    case ENC_TYPE_FLOAT:
      bytes = 4;
      break;
    default:
      MY_ASSERT(!"getElementSize() Unexpected data type.");
      break;
  }
  
  const unsigned int numChannels = (deviceEncoding >> ENC_CHANNELS_SHIFT) & ENC_MASK;

  return bytes * numChannels;
}


// Texture format conversion routines.

template<typename T> 
T getAlphaOne()
{
  return (std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::max() : T(1));
}

// Fixed point adjustment for integer data D and S.
template<typename D, typename S>
D adjust(S value)
{
  int dstBits = int(sizeof(D)) * 8;
  int srcBits = int(sizeof(S)) * 8;

  D result = D(0); // Clear bits to allow OR operations.

  if (std::numeric_limits<D>::is_signed)
  {
    if (std::numeric_limits<S>::is_signed)
    {
      // D signed, S signed
      if (dstBits <= srcBits)
      {
        // More bits provided than needed. Use the most significant bits of value.
        result = D(value >> (srcBits - dstBits));
      }
      else
      {
        // Shift value into the most significant bits of result and replicate value into the lower bits until all are touched.
        int shifts = dstBits - srcBits;
        result = D(value << shifts);            // This sets the destination sign bit as well.
        value &= std::numeric_limits<S>::max(); // Clear the sign bit inside the source value.
        srcBits--;                              // Reduce the number of srcBits used to replicate the remaining data.
        shifts -= srcBits;                      // Subtracting the now one smaller srcBits from shifts means the next shift will fill up with the remaining non-sign bits as intended.
        while (0 <= shifts)
        {
          result |= D(value << shifts);
          shifts -= srcBits;
        }
        if (shifts < 0) // There can be one to three empty bits left blank in the result now.
        {
          result |= D(value >> -shifts); // Shift to the right to get the most significant bits of value into the least significant destination bits.
        }
      }
    }
    else
    {
      // D signed, S unsigned
      if (dstBits <= srcBits)
      {
        // More bits provided than needed. Use the most significant bits of value.
        result = D(value >> (srcBits - dstBits + 1)); // + 1 because the destination is signed and the value needs to remain positive.
      }
      else
      {
        // Shift value into the most significant bits of result, keep the sign clear, and replicate value into the lower bits until all are touched.
        int shifts = dstBits - srcBits - 1; // - 1 because the destination is signed and the value needs to remain positive.
        while (0 <= shifts)
        {
          result |= D(value << shifts);
          shifts -= srcBits;
        }
        if (shifts < 0)
        {
          result |= D(value >> -shifts);
        }
      }
    }
  }
  else
  {
    if (std::numeric_limits<S>::is_signed)
    {
      // D unsigned, S signed
      value = std::max(S(0), value); // Only the positive values will be transferred.
      srcBits--;                     // Skip the sign bit. Means equal bit size won't happen here.
      if (dstBits <= srcBits)        // When it's really bigger it has at least 7 bits more, no need to care for dangling bits
      {
        result = D(value >> (srcBits - dstBits));
      }
      else
      {
        int shifts = dstBits - srcBits;
        while (0 <= shifts)
        {
          result |= D(value << shifts);
          shifts -= srcBits;
        }
        if (shifts < 0)
        {
          result |= D(value >> -shifts);
        }
      }
    }
    else
    {
      // D unsigned, S unsigned
      if (dstBits <= srcBits)
      {
        // More bits provided than needed. Use the most significant bits of value.
        result = D(value >> (srcBits - dstBits));
      }
      else
      {
        // Shift value into the most significant bits of result and replicate into the lower ones until all bits are touched.
        int shifts = dstBits - srcBits;
        while (0 <= shifts) 
        {
          result |= D(value << shifts);
          shifts -= srcBits;
        } 
        // Both bit sizes are even multiples of 8, there are no trailing bits here.
        MY_ASSERT(shifts == -srcBits);
      }
    }
  }
  return result;
}


template<typename D, typename S>
void remapAdjust(void *dst, unsigned int dstEncoding, const void *src, unsigned int srcEncoding, size_t count)
{
  const S *psrc = reinterpret_cast<const S *>(src);
  D *pdst = reinterpret_cast<D *>(dst);
  unsigned int dstChannels = (dstEncoding >> ENC_CHANNELS_SHIFT) & ENC_MASK;
  unsigned int srcChannels = (srcEncoding >> ENC_CHANNELS_SHIFT) & ENC_MASK;
  bool fixedPoint = !!(dstEncoding & ENC_FIXED_POINT);
  bool alphaOne   = !!(dstEncoding & ENC_ALPHA_ONE);

  while (count--)
  {
    unsigned int shift = ENC_RED_SHIFT;
    for (unsigned int i = 0; i < 5; ++i, shift += 4) // Five possible channels: R, G, B, A, L
    {
      unsigned int d = (dstEncoding >> shift) & ENC_MASK;
      if (d < 4) // This data channel exists inside the destination.
      {
        unsigned int s = (srcEncoding >> shift) & ENC_MASK;
        // If destination alpha was added to support this format or if no source data is given for alpha, fill it with 1.
        if (shift == ENC_ALPHA_SHIFT && (alphaOne || 4 <= s))
        {
          pdst[d] = getAlphaOne<D>();
        }
        else
        {
          if (s < 4) // There is data for this channel inside the source. (This could be a luminance to RGB mapping as well).
          {
            S value = psrc[s];
            pdst[d] = (fixedPoint) ? adjust<D>(value) : D(value);
          }
          else // no value provided
          {
            pdst[d] = D(0);
          }
        }
      }
    }
    pdst += dstChannels;
    psrc += srcChannels;
  }
}

// Straight channel copy with no adjustment. Since the data types match, fixed point doesn't matter.
template<typename T>
void remapCopy(void *dst, unsigned int dstEncoding, const void *src, unsigned int srcEncoding, size_t count)
{
  const T *psrc = reinterpret_cast<const T *>(src);
  T *pdst = reinterpret_cast<T *>(dst);
  unsigned int dstChannels = (dstEncoding >> ENC_CHANNELS_SHIFT) & ENC_MASK;
  unsigned int srcChannels = (srcEncoding >> ENC_CHANNELS_SHIFT) & ENC_MASK;
  bool alphaOne = !!(dstEncoding & ENC_ALPHA_ONE);

  while (count--)
  {
    unsigned int shift = ENC_RED_SHIFT;
    for (unsigned int i = 0; i < 5; ++i, shift += 4) // Five possible channels: R, G, B, A, L
    {
      unsigned int d = (dstEncoding >> shift) & ENC_MASK;
      if (d < 4) // This data channel exists inside the destination.
      {
        unsigned int s = (srcEncoding >> shift) & ENC_MASK;
        if (shift == ENC_ALPHA_SHIFT && (alphaOne || 4 <= s))
        {
          pdst[d] = getAlphaOne<T>();
        }
        else
        {
          pdst[d] = (s < 4) ? psrc[s] : T(0);
        }
      }
    }
    pdst += dstChannels;
    psrc += srcChannels;
  }
}

template<typename D>
void remapFromFloat(void *dst, unsigned int dstEncoding, const void *src, unsigned int srcEncoding, size_t count)
{
  const float *psrc = reinterpret_cast<const float *>(src);
  D *pdst = reinterpret_cast<D *>(dst);
  unsigned int dstChannels = (dstEncoding >> ENC_CHANNELS_SHIFT) & ENC_MASK;
  unsigned int srcChannels = (srcEncoding >> ENC_CHANNELS_SHIFT) & ENC_MASK;
  bool fixedPoint = !!(dstEncoding & ENC_FIXED_POINT);
  bool alphaOne   = !!(dstEncoding & ENC_ALPHA_ONE);

  while (count--)
  {
    unsigned int shift = ENC_RED_SHIFT;
    for (unsigned int i = 0; i < 5; ++i, shift += 4)
    {
      unsigned int d = (dstEncoding >> shift) & ENC_MASK;
      if (d < 4) // This data channel exists inside the destination.
      {
        unsigned int s = (srcEncoding >> shift) & ENC_MASK;
        if (shift == ENC_ALPHA_SHIFT && (alphaOne || 4 <= s))
        {
          pdst[d] = getAlphaOne<D>();
        }
        else
        {
          if (s < 4) // This data channel exists inside the source.
          {
            float value = psrc[s];
            if (fixedPoint)
            {
              MY_ASSERT(std::numeric_limits<D>::is_integer); // Destination with float format cannot be fixed point.

              float minimum = (std::numeric_limits<D>::is_signed) ? -1.0f : 0.0f;
              value = std::min(std::max(minimum, value), 1.0f);
              pdst[d] = D(std::numeric_limits<D>::max() * value); // Scaled copy.
            }
            else // element type, clamped copy.
            {
              float maximum = float(std::numeric_limits<D>::max()); // This will run out of precision for int and unsigned int.
              float minimum = -maximum;
              pdst[d] = D(std::min(std::max(minimum, value), maximum));
            }
          }
          else // no value provided
          {
            pdst[d] = D(0);
          }
        }
      }
    }
    pdst += dstChannels;
    psrc += srcChannels;
  }
}

template<typename S>
void remapToFloat(void *dst, unsigned int dstEncoding, const void *src, unsigned int srcEncoding, size_t count)
{
  const S *psrc = reinterpret_cast<const S *>(src);
  float *pdst = reinterpret_cast<float *>(dst);
  unsigned int dstChannels = (dstEncoding >> ENC_CHANNELS_SHIFT) & ENC_MASK;
  unsigned int srcChannels = (srcEncoding >> ENC_CHANNELS_SHIFT) & ENC_MASK;
  bool alphaOne = !!(dstEncoding & ENC_ALPHA_ONE);

  while (count--)
  {
    unsigned int shift = ENC_RED_SHIFT;
    for (unsigned int i = 0; i < 5; ++i, shift += 4)
    {
      unsigned int d = (dstEncoding >> shift) & ENC_MASK;
      if (d < 4) // This data channel exists inside the destination.
      {
        unsigned int s = (srcEncoding >> shift) & ENC_MASK;
        if (shift == ENC_ALPHA_SHIFT && (alphaOne || 4 <= s))
        {
          pdst[d] = 1.0f;
        }
        else
        {
          // If there is data for this channel just cast it straight in.
          // This will run out of precision for int and unsigned int source data.
          pdst[d] = (s < 4) ? float(psrc[s]) : 0.0f;
        }
      }
    }
    pdst += dstChannels;
    psrc += srcChannels;
  }
}


typedef void (*PFNREMAP)(void *dst, unsigned int dstEncoding, const void *src, unsigned int srcEncoding, size_t count);

// Function table with 49 texture format conversion routines from loaded image data to supported CUDA texture formats.
// Index is [destination type][source type]
PFNREMAP remappers[7][7] = 
{
  {
    remapCopy<char>,
    remapAdjust<char, unsigned char>,
    remapAdjust<char, short>,
    remapAdjust<char, unsigned short>,
    remapAdjust<char, int>,
    remapAdjust<char, unsigned int>,
    remapFromFloat<char>
  },
  { 
    remapAdjust<unsigned char, char>,
    remapCopy<unsigned char>,
    remapAdjust<unsigned char, short>,
    remapAdjust<unsigned char, unsigned short>,
    remapAdjust<unsigned char, int>,
    remapAdjust<unsigned char, unsigned int>,
    remapFromFloat<unsigned char>
  },
  { 
    remapAdjust<short, char>,
    remapAdjust<short, unsigned char>,
    remapCopy<short>,
    remapAdjust<short, unsigned short>,
    remapAdjust<short, int>,
    remapAdjust<short, unsigned int>,
    remapFromFloat<short>
  },
  {
    remapAdjust<unsigned short, char>,
    remapAdjust<unsigned short, unsigned char>,
    remapAdjust<unsigned short, short>,
    remapCopy<unsigned short>,
    remapAdjust<unsigned short, int>,
    remapAdjust<unsigned short, unsigned int>,
    remapFromFloat<unsigned short>
  },
  { 
    remapAdjust<int, char>,
    remapAdjust<int, unsigned char>,
    remapAdjust<int, short>,
    remapAdjust<int, unsigned short>,
    remapCopy<int>,
    remapAdjust<int, unsigned int>,
    remapFromFloat<int>
  },
  {
    remapAdjust<unsigned int, char>,
    remapAdjust<unsigned int, unsigned char>,
    remapAdjust<unsigned int, short>,
    remapAdjust<unsigned int, unsigned short>,
    remapAdjust<unsigned int, int>,
    remapCopy<unsigned int>,
    remapFromFloat<unsigned int>
  },
  {
    remapToFloat<char>,
    remapToFloat<unsigned char>,
    remapToFloat<short>,
    remapToFloat<unsigned short>,
    remapToFloat<int>,
    remapToFloat<unsigned int>,
    remapCopy<float>
  }
};


// Finally the function which converts any loaded image into a texture format supported by CUDA (1, 2, 4 channels only).
static void convert(void *dst, unsigned int deviceEncoding, const void *src, unsigned int hostEncoding, size_t elements)
{
  // Only destination encoding knows about the fixed-point encoding. For straight data memcpy() cases that is irrelevant.
  // PERF Avoid this conversion altogether when it's just a memcpy()!
  if ((deviceEncoding & ~ENC_FIXED_POINT) == hostEncoding)
  {
    memcpy(dst, src, elements * getElementSize(deviceEncoding)); // The fastest path.
  }
  else
  {
    unsigned int dstType = (deviceEncoding >> ENC_TYPE_SHIFT) & ENC_MASK;
    unsigned int srcType = (hostEncoding   >> ENC_TYPE_SHIFT) & ENC_MASK;
    MY_ASSERT(dstType < 7 && srcType < 7); 
          
    PFNREMAP pfn = remappers[dstType][srcType];

    (*pfn)(dst, deviceEncoding, src, hostEncoding, elements);
  }
}

Texture::Texture()
: m_width(0)
, m_height(0)
, m_depth(0)
, m_hostEncoding(ENC_INVALID)
, m_deviceEncoding(ENC_INVALID)
, m_sizeBytesPerElement(0)
, m_textureObject(0)
, m_d_array(0)
, m_d_mipmappedArray(0)
, m_d_envCDF_U(0)
, m_d_envCDF_V(0)
, m_integral(1.0f)
{
  m_descArray3D.Width       = 0;
  m_descArray3D.Height      = 0;
  m_descArray3D.Depth       = 0;
  m_descArray3D.Format      = CU_AD_FORMAT_UNSIGNED_INT8;
  m_descArray3D.NumChannels = 0;
  m_descArray3D.Flags       = 0;

  // Setup CUDA_TEXTURE_DESC defaults. 
  
  // Note that cuTexObjectCreate() fails if the "int reserved[12]" data is not zero! Not mentioned in the CUDA documentation.
  memset(&m_textureDescription, 0, sizeof(CUDA_TEXTURE_DESC)); 

  // The developer can override these at will before calling Texture::create().
  // If the flag CU_TRSF_NORMALIZED_COORDINATES is not set, the only supported address mode is CU_TR_ADDRESS_MODE_CLAMP.
  m_textureDescription.addressMode[0] = CU_TR_ADDRESS_MODE_WRAP;
  m_textureDescription.addressMode[1] = CU_TR_ADDRESS_MODE_WRAP;
  m_textureDescription.addressMode[2] = CU_TR_ADDRESS_MODE_WRAP;

  m_textureDescription.filterMode = CU_TR_FILTER_MODE_LINEAR;

  // Possible flags: CU_TRSF_READ_AS_INTEGER, CU_TRSF_NORMALIZED_COORDINATES, CU_TRSF_SRGB
  m_textureDescription.flags = CU_TRSF_NORMALIZED_COORDINATES;

  m_textureDescription.maxAnisotropy = 1;

  // LOD 0 only by default.
  // This means when using mipmaps it's the developer's responsibility to set at least 
  // maxMipmapLevelClamp > 0.0f before calling Texture::create() to make sure mipmaps can be sampled!
  m_textureDescription.mipmapFilterMode    = CU_TR_FILTER_MODE_POINT; // Bilinear filtering by default.
  m_textureDescription.mipmapLevelBias     = 0.0f;
  m_textureDescription.minMipmapLevelClamp = 0.0f;
  m_textureDescription.maxMipmapLevelClamp = 0.0f; // This should be set to Picture::getNumberOfLevels() when using mipmaps.

  m_textureDescription.borderColor[0] = 0.0f;
  m_textureDescription.borderColor[1] = 0.0f;
  m_textureDescription.borderColor[2] = 0.0f;
  m_textureDescription.borderColor[3] = 0.0f;
}


Texture::~Texture()
{
  if (m_d_array)
  {
    CU_CHECK( cuArrayDestroy(m_d_array) );
  }
  if (m_d_mipmappedArray)
  {
    CU_CHECK( cuMipmappedArrayDestroy(m_d_mipmappedArray) );
  }
  if (m_d_envCDF_U)
  {
    CU_CHECK( cuMemFree(m_d_envCDF_U) );
  }
  if (m_d_envCDF_V)
  {
    CU_CHECK( cuMemFree(m_d_envCDF_V) );
  }
  if (m_textureObject)
  {
    CU_CHECK( cuTexObjectDestroy(m_textureObject) );
  }
}

// For all functions changing the m_textureDescription values,
// make sure they are called before the texture object has been created,
// otherwise the texture object would need to be recreated.

// Set the whole description.
void Texture::setTextureDescription(CUDA_TEXTURE_DESC const& descr)
{
  m_textureDescription = descr;
}

void Texture::setAddressMode(CUaddress_mode s, CUaddress_mode t, CUaddress_mode r)
{
  MY_ASSERT(m_textureObject == 0);

  m_textureDescription.addressMode[0] = s;
  m_textureDescription.addressMode[1] = t;
  m_textureDescription.addressMode[2] = r;
}

void Texture::setFilterMode(CUfilter_mode filter, CUfilter_mode filterMipmap)
{
  MY_ASSERT(m_textureObject == 0);

  m_textureDescription.filterMode = filter;
  m_textureDescription.filterMode = filterMipmap;
}

void Texture::setBorderColor(float r, float g, float b, float a)
{
  MY_ASSERT(m_textureObject == 0);

  m_textureDescription.borderColor[0] = r;
  m_textureDescription.borderColor[1] = g;
  m_textureDescription.borderColor[2] = b;
  m_textureDescription.borderColor[3] = a;
}

void Texture::setMaxAnisotropy(unsigned int aniso)
{
  MY_ASSERT(m_textureObject == 0);

  m_textureDescription.maxAnisotropy = aniso;
}

void Texture::setMipmapLevelBiasMinMax(float bias, float minimum, float maximum)
{
  MY_ASSERT(m_textureObject == 0);

  m_textureDescription.mipmapLevelBias     = bias;
  m_textureDescription.minMipmapLevelClamp = minimum;
  m_textureDescription.maxMipmapLevelClamp = maximum;
}

void Texture::setReadMode(bool asInteger)
{
  MY_ASSERT(m_textureObject == 0);

  if (asInteger)
  {
    m_textureDescription.flags |= CU_TRSF_READ_AS_INTEGER;
  }
  else
  {
    m_textureDescription.flags &= ~CU_TRSF_READ_AS_INTEGER;
  }
}

void Texture::setSRGB(bool srgb)
{
  MY_ASSERT(m_textureObject == 0);

  if (srgb) 
  {
    m_textureDescription.flags |= CU_TRSF_SRGB;
  }
  else
  {
    m_textureDescription.flags &= ~CU_TRSF_SRGB;
  }
}

void Texture::setNormalizedCoords(bool normalized)
{
  MY_ASSERT(m_textureObject == 0);

  if (normalized) 
  {
    m_textureDescription.flags |= CU_TRSF_NORMALIZED_COORDINATES; // Default in this app.
  }
  else
  {
    // Note that if the flag CU_TRSF_NORMALIZED_COORDINATES is not set, 
    // the only supported address mode is CU_TR_ADDRESS_MODE_CLAMP!
    m_textureDescription.flags &= ~CU_TRSF_NORMALIZED_COORDINATES;
  }
}


bool Texture::create1D(const Picture* picture, const unsigned int flags)
{
  CUDA_RESOURCE_DESC resourceDescription = {}; // For the final texture object creation.

  // Default initialization for a 1D texture without layers.
  m_descArray3D.Width  = m_width;
  m_descArray3D.Height = 0;
  m_descArray3D.Depth  = 0;
  determineFormatChannels(m_deviceEncoding, m_descArray3D.Format, m_descArray3D.NumChannels);
  m_descArray3D.Flags  = 0;
  
  size_t sizeElements = m_width; // The size for the LOD 0 in elements.

  if (flags & IMAGE_FLAG_LAYER)
  {
    m_descArray3D.Depth = m_depth;              // Mind that the layers are always defined via the depth extent.
    m_descArray3D.Flags = CUDA_ARRAY3D_LAYERED; // Set the array allocation flag.
    sizeElements       *= m_depth;              // The size for the LOD 0 with layers in elements.
  }

  size_t sizeBytes = sizeElements * m_sizeBytesPerElement;

  unsigned char* data = new unsigned char[sizeBytes]; // Allocate enough scratch memory for the conversion to hold the biggest LOD.
  
  // DEBUG I don't know of any image files with 1D layered mipmapped textures, but the Picture class handles that just fine.
  const unsigned int numLevels = picture->getNumberOfLevels(0); // This is the number of mipmap levels including LOD 0.

  if (1 < numLevels && (flags & IMAGE_FLAG_MIPMAP)) // 1D (layered) mipmapped texture.
  {
    // A 1D mipmapped array is allocated if Height and Depth extents are both zero.
    // A 1D layered CUDA mipmapped array is allocated if only Height is zero and the CUDA_ARRAY3D_LAYERED flag is set.
    // Each layer is a 1D array. The number of layers is determined by the Depth extent.
    CU_CHECK( cuMipmappedArrayCreate(&m_d_mipmappedArray, &m_descArray3D, numLevels) );

    for (unsigned int level = 0; level < numLevels; ++level)
    {
      CUarray d_levelArray;

      CU_CHECK( cuMipmappedArrayGetLevel(&d_levelArray, m_d_mipmappedArray, level) );

      const Image* image = picture->getImageLevel(0, level); // Get the image 0 LOD level.
      
      sizeElements = image->m_width * m_depth;
      sizeBytes    = sizeElements * m_sizeBytesPerElement;

      convert(data, m_deviceEncoding, image->m_pixels, m_hostEncoding, sizeElements);

      CUDA_MEMCPY3D params = {};

      params.srcMemoryType = CU_MEMORYTYPE_HOST;
      params.srcHost       = data;
      params.srcPitch      = image->m_width * m_sizeBytesPerElement;
      params.srcHeight     = 1;

      params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
      params.dstArray      = d_levelArray;
      
      params.WidthInBytes  = params.srcPitch;
      params.Height        = 1;
      params.Depth         = m_depth;

      CU_CHECK( cuMemcpy3D(&params) );
    }

    resourceDescription.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    resourceDescription.res.mipmap.hMipmappedArray = m_d_mipmappedArray;
  }
  else // 1D (layered) texture.
  {
    // A 1D array is allocated if the height and depth extents are both zero.
    // A 1D layered CUDA array is allocated if only the height extent is zero and the cudaArrayLayered flag is set.
    // Each layer is a 1D array. The number of layers is determined by the depth extent.
    CU_CHECK( cuArray3DCreate(&m_d_array, &m_descArray3D) );

    const Image* image = picture->getImageLevel(0, 0); // LOD 0 only.

    sizeElements = m_width * m_depth;
    sizeBytes    = sizeElements * m_sizeBytesPerElement;

    convert(data, m_deviceEncoding, image->m_pixels, m_hostEncoding, sizeElements);
    
    CUDA_MEMCPY3D params = {};

    params.srcMemoryType = CU_MEMORYTYPE_HOST;
    params.srcHost       = data;
    params.srcPitch      = m_width * m_sizeBytesPerElement;
    params.srcHeight     = 1;

    params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    params.dstArray      = m_d_array;
      
    params.WidthInBytes  = params.srcPitch;
    params.Height        = 1;
    params.Depth         = m_depth;

    CU_CHECK( cuMemcpy3D(&params) );
    
    resourceDescription.resType = CU_RESOURCE_TYPE_ARRAY;
    resourceDescription.res.array.hArray = m_d_array;
  }
  
  delete[] data;

  m_textureObject = 0;

  CU_CHECK( cuTexObjectCreate(&m_textureObject, &resourceDescription, &m_textureDescription, nullptr) ); 

  return (m_textureObject != 0);
}


bool Texture::create2D(const Picture* picture, const unsigned int flags)
{
  CUDA_RESOURCE_DESC resourceDescription = {}; // For the final texture object creation.

  // Default initialization for a 2D texture without layers.
  m_descArray3D.Width  = m_width;
  m_descArray3D.Height = m_height;
  m_descArray3D.Depth  = 0;
  determineFormatChannels(m_deviceEncoding, m_descArray3D.Format, m_descArray3D.NumChannels);
  m_descArray3D.Flags  = 0;
  
  size_t sizeElements = m_width * m_height; // The size for the LOD 0 in elements.

  if (flags & IMAGE_FLAG_LAYER)
  {
    m_descArray3D.Depth = m_depth;              // Mind that the layers are always defined via the depth extent.
    m_descArray3D.Flags = CUDA_ARRAY3D_LAYERED; // Set the array allocation flag.
    sizeElements       *= m_depth;              // The size for the LOD 0 with layers in elements.
  }

  size_t sizeBytes = sizeElements * m_sizeBytesPerElement;

  unsigned char* data = new unsigned char[sizeBytes]; // Allocate enough scratch memory for the conversion to hold the biggest LOD.
  
  const unsigned int numLevels = picture->getNumberOfLevels(0); // This is the number of mipmap levels including LOD 0.

  if (1 < numLevels && (flags & IMAGE_FLAG_MIPMAP)) // 2D (layered) mipmapped texture
  {
    // A 2D mipmapped array is allocated if only Depth extent is zero.
    // A 2D layered CUDA mipmapped array is allocated if all three extents are non-zero and the ::CUDA_ARRAY3D_LAYERED flag is set.
    // Each layer is a 2D array. The number of layers is determined by the Depth extent.
    CU_CHECK( cuMipmappedArrayCreate(&m_d_mipmappedArray, &m_descArray3D, numLevels) );

    for (unsigned int level = 0; level < numLevels; ++level)
    {
      CUarray d_levelArray;

      CU_CHECK( cuMipmappedArrayGetLevel(&d_levelArray, m_d_mipmappedArray, level) );

      const Image* image = picture->getImageLevel(0, level); // Get the image 0 LOD level.
      
      sizeElements = image->m_width * image->m_height * m_depth;
      sizeBytes    = sizeElements * m_sizeBytesPerElement;
      
      convert(data, m_deviceEncoding, image->m_pixels, m_hostEncoding, sizeElements);

      CUDA_MEMCPY3D params = {};

      params.srcMemoryType = CU_MEMORYTYPE_HOST;
      params.srcHost       = data;
      params.srcPitch      = image->m_width * m_sizeBytesPerElement;
      params.srcHeight     = image->m_height;

      params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
      params.dstArray      = d_levelArray;
      
      params.WidthInBytes  = params.srcPitch;
      params.Height        = image->m_height;
      params.Depth         = m_depth;

      CU_CHECK( cuMemcpy3D(&params) );
    }

    resourceDescription.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    resourceDescription.res.mipmap.hMipmappedArray = m_d_mipmappedArray;
  }
  else // 2D (layered) texture.
  {
    // A 2D array is allocated if only Depth extent is zero.
    // A 2D layered CUDA array is allocated if all three extents are non-zero and the ::CUDA_ARRAY3D_LAYERED flag is set.
    // Each layer is a 2D array. The number of layers is determined by the Depth extent.
    CU_CHECK( cuArray3DCreate(&m_d_array, &m_descArray3D) );

    const Image* image = picture->getImageLevel(0, 0); // LOD 0 only

    sizeElements = m_width * m_height * m_depth;
    sizeBytes    = sizeElements * m_sizeBytesPerElement;

    convert(data, m_deviceEncoding, image->m_pixels, m_hostEncoding, sizeElements);

    CUDA_MEMCPY3D params = {};

    params.srcMemoryType = CU_MEMORYTYPE_HOST;
    params.srcHost       = data;
    params.srcPitch      = m_width * m_sizeBytesPerElement;
    params.srcHeight     = m_height;

    params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    params.dstArray      = m_d_array;
      
    params.WidthInBytes  = params.srcPitch;
    params.Height        = m_height;
    params.Depth         = m_depth;

    CU_CHECK( cuMemcpy3D(&params) );

    resourceDescription.resType = CU_RESOURCE_TYPE_ARRAY;
    resourceDescription.res.array.hArray = m_d_array;
  }
  
  delete[] data;

  m_textureObject = 0;

  CU_CHECK( cuTexObjectCreate(&m_textureObject, &resourceDescription, &m_textureDescription, nullptr) ); 

  return (m_textureObject != 0);
}

bool Texture::create3D(const Picture* picture, const unsigned int flags)
{
  MY_ASSERT((flags & IMAGE_FLAG_LAYER) == 0); // There are no layered 3D textures. The flag is ignored.

  CUDA_RESOURCE_DESC resourceDescription = {}; // For the final texture object creation.

  // Default initialization for a 3D texture. There are no layers.
  m_descArray3D.Width  = m_width;
  m_descArray3D.Height = m_height;
  m_descArray3D.Depth  = m_depth;
  determineFormatChannels(m_deviceEncoding, m_descArray3D.Format, m_descArray3D.NumChannels);
  m_descArray3D.Flags  = 0; // Things like CUDA_ARRAY3D_SURFACE_LDST, ...

  size_t sizeElements = m_width * m_height * m_depth; // The size for the LOD 0 in elements.
  size_t sizeBytes    = sizeElements * m_sizeBytesPerElement;

  unsigned char* data = new unsigned char[sizeBytes]; // Allocate enough scratch memory for the conversion to hold the biggest LOD.
  
  const unsigned int numLevels = picture->getNumberOfLevels(0); // This is the number of mipmap levels including LOD 0.

  if (1 < numLevels && (flags & IMAGE_FLAG_MIPMAP)) // 3D mipmapped texture
  {
    // A 3D mipmapped array is allocated if all three extents are non-zero.
    CU_CHECK( cuMipmappedArrayCreate(&m_d_mipmappedArray, &m_descArray3D, numLevels) );

    for (unsigned int level = 0; level < numLevels; ++level)
    {
      CUarray d_levelArray;

      CU_CHECK( cuMipmappedArrayGetLevel(&d_levelArray, m_d_mipmappedArray, level) );

      const Image* image = picture->getImageLevel(0, level); // Get the image 0 LOD level.
      
      sizeElements = image->m_width * image->m_height * image->m_depth; // Really the image->m_depth here, no layers in 3D textures.
      sizeBytes    = sizeElements * m_sizeBytesPerElement;

      convert(data, m_deviceEncoding, image->m_pixels, m_hostEncoding, sizeElements);

      CUDA_MEMCPY3D params = {};

      params.srcMemoryType = CU_MEMORYTYPE_HOST;
      params.srcHost       = data;
      params.srcPitch      = image->m_width * m_sizeBytesPerElement;
      params.srcHeight     = image->m_height;

      params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
      params.dstArray      = d_levelArray;
      
      params.WidthInBytes  = params.srcPitch;
      params.Height        = image->m_height;
      params.Depth         = image->m_depth; // Really the image->m_depth here, no layers in 3D textures.

      CU_CHECK( cuMemcpy3D(&params) );
    }

    resourceDescription.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    resourceDescription.res.mipmap.hMipmappedArray = m_d_mipmappedArray;
  }
  else // 3D texture.
  {
    // A 3D array is allocated if all three extents are non-zero.
    CU_CHECK( cuArray3DCreate(&m_d_array, &m_descArray3D) );
    
    const Image* image = picture->getImageLevel(0, 0); // LOD 0 only.

    sizeElements = m_width * m_height * m_depth; // Really the image->m_depth here, no layers in 3D textures.
    sizeBytes    = sizeElements * m_sizeBytesPerElement;

    convert(data, m_deviceEncoding, image->m_pixels, m_hostEncoding, sizeElements);

    CUDA_MEMCPY3D params = {};

    params.srcMemoryType = CU_MEMORYTYPE_HOST;
    params.srcHost       = data;
    params.srcPitch      = m_width * m_sizeBytesPerElement;
    params.srcHeight     = m_height;

    params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    params.dstArray      = m_d_array;
      
    params.WidthInBytes  = params.srcPitch;
    params.Height        = m_height;
    params.Depth         = m_depth;

    CU_CHECK( cuMemcpy3D(&params) );

    resourceDescription.resType = CU_RESOURCE_TYPE_ARRAY;
    resourceDescription.res.array.hArray = m_d_array;
  }
  
  delete[] data;

  m_textureObject = 0;

  CU_CHECK( cuTexObjectCreate(&m_textureObject, &resourceDescription, &m_textureDescription, nullptr) ); 

  return (m_textureObject != 0);
}

bool Texture::createCube(const Picture* picture, const unsigned int flags)
{
  if (!picture->isCubemap()) // This implies picture->getNumberOfImages() == 6.
  {
    std::cerr << "ERROR: Texture::createCube() picture is not a cubemap.\n";
    return false;
  }

  if (m_width != m_height || m_depth % 6 != 0)
  {
    std::cerr << "ERROR: Texture::createCube() invalid cubemap image dimensions (" << m_width << ", " << m_height << ", " << m_depth << ")\n";
    return false;
  }

  CUDA_RESOURCE_DESC resourceDescription = {}; // For the final texture object creation.

  // Default initialization for a 1D texture without layers.
  m_descArray3D.Width  = m_width;
  m_descArray3D.Height = m_height;
  m_descArray3D.Depth  = m_depth; // depth == 6 * layers.
  determineFormatChannels(m_deviceEncoding, m_descArray3D.Format, m_descArray3D.NumChannels);
  m_descArray3D.Flags  = CUDA_ARRAY3D_CUBEMAP;

  const unsigned int numLayers = m_depth / 6;

  size_t sizeElements = m_width * m_height * m_depth; // The size for the LOD 0 in elements.

  if (flags & IMAGE_FLAG_LAYER)
  {
    m_descArray3D.Flags |= CUDA_ARRAY3D_LAYERED; // Set the array allocation flag.
  }

  size_t sizeBytes = sizeElements * m_sizeBytesPerElement; // LOD 0 size in bytes.

  unsigned char* data = new unsigned char[sizeBytes]; // Allocate enough scratch memory for the conversion to hold the biggest LOD.
  
  const unsigned int numLevels = picture->getNumberOfLevels(0); // This is the number of mipmap levels including LOD 0.

  if (1 < numLevels && (flags & IMAGE_FLAG_MIPMAP)) // cubemap (layered) mipmapped texture
  {
    // A cubemap CUDA mipmapped array is allocated if all three extents are non-zero and the ::CUDA_ARRAY3D_CUBEMAP flag is set.
    // Width must be equal to \p Height, and Depth must be six.
    // A cubemap is a special type of 2D layered CUDA array, where the six layers represent the six faces of a cube.
    // The order of the six layers in memory is the same as that listed in ::CUarray_cubemap_face. (+x, -x, +y, -y, +z, -z)
    // A cubemap layered CUDA mipmapped array is allocated if all three extents are non-zero, and both, ::CUDA_ARRAY3D_CUBEMAP and ::CUDA_ARRAY3D_LAYERED flags are set.
    // Width must be equal to Height, and Depth must be a multiple of six.
    // A cubemap layered CUDA array is a special type of 2D layered CUDA array that consists of a collection of cubemaps.
    // The first six layers represent the first cubemap, the next six layers form the second cubemap, and so on.
    CU_CHECK( cuMipmappedArrayCreate(&m_d_mipmappedArray, &m_descArray3D, numLevels) );

    for (unsigned int level = 0; level < numLevels; ++level)
    {
      const Image* image; // The last image of each level defines the extent.

      CUarray d_levelArray;

      CU_CHECK( cuMipmappedArrayGetLevel(&d_levelArray, m_d_mipmappedArray, level) );

      for (unsigned int face = 0; face < 6; ++face)
      {
        image = picture->getImageLevel(face, level); // image face, LOD level
      
        const size_t sizeElementsLayer = image->m_width * image->m_height;
        const size_t sizeBytesLayer    = sizeElementsLayer * m_sizeBytesPerElement;

        for (unsigned int layer = 0; layer < numLayers; ++layer)
        {
          // Place the cubemap faces in blocks of 6 times number of layers. Each 6 slices are one cubemap layer.
          void* src = image->m_pixels + sizeBytesLayer *  layer;
          void* dst = data            + sizeBytesLayer * (layer * 6 + face); 
        
          convert(dst, m_deviceEncoding, src, m_hostEncoding, sizeElementsLayer);
        }
      }

      CUDA_MEMCPY3D params = {};

      params.srcMemoryType = CU_MEMORYTYPE_HOST;
      params.srcHost       = data;
      params.srcPitch      = image->m_width * m_sizeBytesPerElement;
      params.srcHeight     = image->m_height;

      params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
      params.dstArray      = d_levelArray;
      
      params.WidthInBytes  = params.srcPitch;
      params.Height        = image->m_height;
      params.Depth         = m_depth;

      CU_CHECK( cuMemcpy3D(&params) );
    }

    resourceDescription.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    resourceDescription.res.mipmap.hMipmappedArray = m_d_mipmappedArray;
  }
  else // cubemap (layered) texture.
  {
    // A cubemap CUDA array is allocated if all three extents are non-zero and the ::CUDA_ARRAY3D_CUBEMAP flag is set.
    // Width must be equal to Height, and Depth must be six.
    // A cubemap is a special type of 2D layered CUDA array, where the six layers represent the six faces of a cube.
    // The order of the six layers in memory is the same as that listed in ::CUarray_cubemap_face. (+x, -x, +y, -y, +z, -z)
    // A cubemap layered CUDA array is allocated if all three extents are non-zero, and both, ::CUDA_ARRAY3D_CUBEMAP and ::CUDA_ARRAY3D_LAYERED flags are set.
    // Width must be equal to Height, and Depth must be a multiple of six.
    // A cubemap layered CUDA array is a special type of 2D layered CUDA array that consists of a collection of cubemaps.
    // The first six layers represent the first cubemap, the next six layers form the second cubemap, and so on.
    CU_CHECK( cuArray3DCreate(&m_d_array, &m_descArray3D) );

    for (unsigned int face = 0; face < 6; ++face)
    {
      const Image* image = picture->getImageLevel(face, 0); // image face, LOD 0
      
      const size_t sizeElementsLayer = m_width * m_height;
      const size_t sizeBytesLayer    = sizeElementsLayer * m_sizeBytesPerElement;

      for (unsigned int layer = 0; layer < numLayers; ++layer)
      {
        // Place the cubemap faces in blocks of 6 times number of layers. Each 6 slices are one cubemap layer.
        void* src = image->m_pixels + sizeBytesLayer *  layer;
        void* dst = data            + sizeBytesLayer * (layer * 6 + face); 
        
        convert(dst, m_deviceEncoding, src, m_hostEncoding, sizeElementsLayer);
      }
    }

    CUDA_MEMCPY3D params = {};

    params.srcMemoryType = CU_MEMORYTYPE_HOST;
    params.srcHost       = data;
    params.srcPitch      = m_width * m_sizeBytesPerElement;
    params.srcHeight     = m_height;

    params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    params.dstArray      = m_d_array;
      
    params.WidthInBytes  = params.srcPitch;
    params.Height        = m_height;
    params.Depth         = m_depth;

    CU_CHECK( cuMemcpy3D(&params) );

    resourceDescription.resType = CU_RESOURCE_TYPE_ARRAY;
    resourceDescription.res.array.hArray = m_d_array;
  }
  
  delete[] data;

  m_textureObject = 0;

  CU_CHECK( cuTexObjectCreate(&m_textureObject, &resourceDescription, &m_textureDescription, nullptr) ); 

  return (m_textureObject != 0);
}

bool Texture::createEnv(const Picture* picture, const unsigned int flags)
{
  CUDA_RESOURCE_DESC resourceDescription = {}; // For the final texture object creation.

  // Default initialization for a 1D texture without layers.
  m_descArray3D.Width  = m_width;
  m_descArray3D.Height = m_height;
  m_descArray3D.Depth  = 0;
  determineFormatChannels(m_deviceEncoding, m_descArray3D.Format, m_descArray3D.NumChannels);
  m_descArray3D.Flags  = 0;
  
  size_t sizeElements = m_width * m_height; // The size for the LOD 0 in elements.
  size_t sizeBytes    = sizeElements * m_sizeBytesPerElement;

  float* data = new float[sizeElements * 4]; // RGBA32F
  
  // A 2D array is allocated if only Depth extent is zero.
  CU_CHECK( cuArray3DCreate(&m_d_array, &m_descArray3D) );

  const Image* image = picture->getImageLevel(0, 0); // LOD 0 only.

  convert(data, m_deviceEncoding, image->m_pixels, m_hostEncoding, sizeElements);

  CUDA_MEMCPY3D params = {};

  params.srcMemoryType = CU_MEMORYTYPE_HOST;
  params.srcHost       = data;
  params.srcPitch      = m_width * m_sizeBytesPerElement;
  params.srcHeight     = m_height;

  params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  params.dstArray      = m_d_array;
      
  params.WidthInBytes  = params.srcPitch;
  params.Height        = m_height;
  params.Depth         = m_depth;

  CU_CHECK( cuMemcpy3D(&params) );

  resourceDescription.resType = CU_RESOURCE_TYPE_ARRAY;
  resourceDescription.res.array.hArray = m_d_array;

  // Generate the CDFs for direct environment lighting and the environment texture sampler itself.
  calculateSphericalCDF(data);
  
  delete[] data;


  // Setup CUDA_TEXTURE_DESC for the spherical environment. 
  // The defaults are set for a bilinear filtered 2D texture already.
  // The spherical environment texture only needs to change the addessmode[1] to clamp.
  //m_textureDescription.addressMode[0] = CU_TR_ADDRESS_MODE_WRAP;
  m_textureDescription.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
  //m_textureDescription.addressMode[2] = CU_TR_ADDRESS_MODE_WRAP;

  //m_textureDescription.filterMode = CU_TR_FILTER_MODE_LINEAR;

  //m_textureDescription.flags = CU_TRSF_NORMALIZED_COORDINATES;

  //m_textureDescription.maxAnisotropy = 1;

  //m_textureDescription.mipmapFilterMode    = CU_TR_FILTER_MODE_POINT; // Bilinear filtering by default.
  //m_textureDescription.mipmapLevelBias     = 0.0f;
  //m_textureDescription.minMipmapLevelClamp = 0.0f;
  //m_textureDescription.maxMipmapLevelClamp = 0.0f; // This should be set to Picture::getNumberOfLevels() when using mipmaps.

  //m_textureDescription.borderColor[0] = 0.0f;
  //m_textureDescription.borderColor[1] = 0.0f;
  //m_textureDescription.borderColor[2] = 0.0f;
  //m_textureDescription.borderColor[3] = 0.0f;

  m_textureObject = 0;

  CU_CHECK( cuTexObjectCreate(&m_textureObject, &resourceDescription, &m_textureDescription, nullptr) ); 

  return (m_textureObject != 0);
}


bool Texture::create(const Picture* picture, const unsigned int flags)
{
  bool success = false;
  
  if (m_textureObject != 0)
  {
    std::cerr << "ERROR: Texture::create() texture object already created.\n";
    return success;
  }

  if (picture == nullptr)
  {
    std::cerr << "ERROR: Texture::create() called with nullptr picture.\n";
    return success;
  }

  // The LOD 0 image of the first face defines the basic settings, including the texture m_width, m_height, m_depth.
  // This returns nullptr when this image doesn't exist. Everything else in this function relies on it.
  const Image* image = picture->getImageLevel(0, 0);

  if (image == nullptr)
  {
    std::cerr << "ERROR: Texture::create() Picture doesn't contain image 0 level 0.\n";
    return success;
  }
  
  // Precalculate some values which are required for all create*() functions.
  m_hostEncoding = determineHostEncoding(image->m_format, image->m_type);
 
  if (flags & IMAGE_FLAG_ENV)
  {
    // Hardcode the device encoding to a floating point HDR image. The input is expected to be an HDR or EXR image. 
    // Fixed point LDR textures will remain unnormalized, e.g. unsigned byte 255 will be converted to float 255.0f.
    // (Just because there is no suitable conversion routine implemented for that.)
    m_deviceEncoding = ENC_RED_0 | ENC_GREEN_1 | ENC_BLUE_2 | ENC_ALPHA_3 | ENC_LUM_NONE | ENC_CHANNELS_4 | ENC_ALPHA_ONE | ENC_TYPE_FLOAT;
  }
  else
  {
    m_deviceEncoding = determineDeviceEncoding(image->m_format, image->m_type);
  }
  
  if ((m_hostEncoding | m_deviceEncoding) & ENC_INVALID) // If either of the encodings is invalid, bail out.
  {
    return success;
  }

  m_sizeBytesPerElement = getElementSize(m_deviceEncoding);

  if (flags & IMAGE_FLAG_1D)
  {
    m_width  = image->m_width;
    m_height = 1;
    m_depth  = (flags & IMAGE_FLAG_LAYER) ? image->m_depth : 1; 
    success = create1D(picture, flags);
  }
  else if ((flags & (IMAGE_FLAG_2D | IMAGE_FLAG_ENV)) == IMAGE_FLAG_2D) // Standard 2D texture. 
  {
    m_width  = image->m_width;
    m_height = image->m_height;
    m_depth  = (flags & IMAGE_FLAG_LAYER) ? image->m_depth : 1;
    success = create2D(picture, flags);
  }
  else if (flags & IMAGE_FLAG_3D)
  {
    m_width  = image->m_width;
    m_height = image->m_height;
    m_depth  = image->m_depth;
    success = create3D(picture, flags);
  }
  else if (flags & IMAGE_FLAG_CUBE)
  {
    m_width  = image->m_width;
    m_height = image->m_height;
    // Note that the Picture class holds the six cubemap faces in six separate mipmap chains!
    // The LOD 0 image depth is the number of layers. The resulting 3D image is six times that depth.
    m_depth = (flags & IMAGE_FLAG_LAYER) ? image->m_depth * 6 : 6; 
    success = createCube(picture, flags);
  }
  else if ((flags & (IMAGE_FLAG_2D | IMAGE_FLAG_ENV)) == (IMAGE_FLAG_2D | IMAGE_FLAG_ENV)) // 2D spherical environment texture. 
  {
    m_width  = image->m_width;
    m_height = image->m_height;
    m_depth  = 1; // No layers for the environment map.
    success = createEnv(picture, flags);
  }
  
  MY_ASSERT(success);
  return success;
}


unsigned int Texture::getWidth() const
{
  return m_width;
}

unsigned int Texture::getHeight() const
{
  return m_height;
}

unsigned int Texture::getDepth() const
{
  return m_depth;
}


cudaTextureObject_t Texture::getTextureObject() const
{
  return m_textureObject;
}


// The following functions are used to build the data needed for an importance sampled spherical HDR environment map. 

// Implement a simple Gaussian 3x3 filter with sigma = 0.5
// Needed for the CDF generation of the importance sampled HDR environment texture light.
static float gaussianFilter(const float* rgba, unsigned int width, unsigned int height, unsigned int x, unsigned int y)
{
  // Lookup is repeated in x and clamped to edge in y.
  unsigned int left   = (0 < x)          ? x - 1 : width - 1; // repeat
  unsigned int right  = (x < width - 1)  ? x + 1 : 0;         // repeat
  unsigned int bottom = (0 < y)          ? y - 1 : y;         // clamp
  unsigned int top    = (y < height - 1) ? y + 1 : y;         // clamp
  
  // Center
  const float *p = rgba + (width * y + x) * 4;
  float intensity = (p[0] + p[1] + p[2]) * 0.619347f;

  // 4-neighbours
  p = rgba + (width * bottom + x) * 4;
  float f = p[0] + p[1] + p[2];
  p = rgba + (width * y + left) * 4;
  f += p[0] + p[1] + p[2];
  p = rgba + (width * y + right) * 4;
  f += p[0] + p[1] + p[2];
  p = rgba + (width * top + x) * 4;
  f += p[0] + p[1] + p[2];
  intensity += f * 0.0838195f;

  // 8-neighbours corners
  p = rgba + (width * bottom + left) * 4;
  f  = p[0] + p[1] + p[2];
  p = rgba + (width * bottom + right) * 4;
  f += p[0] + p[1] + p[2];
  p = rgba + (width * top + left) * 4;
  f += p[0] + p[1] + p[2];
  p = rgba + (width * top + right) * 4;
  f += p[0] + p[1] + p[2];
  intensity += f * 0.0113437f;

  return intensity / 3.0f;
}

// Create cumulative distribution function for importance sampling of spherical environment lights.
// This is a textbook implementation for the CDF generation of a spherical HDR environment.
// See "Physically Based Rendering" v2, chapter 14.6.5 on Infinite Area Lights.
void Texture::calculateSphericalCDF(const float* rgba)
{
  // The original data needs to be retained to calculate the PDF.
  float *funcU = new float[m_width * m_height];
  float *funcV = new float[m_height + 1];

  float sum = 0.0f;
  // First generate the function data.
  for (unsigned int y = 0; y < m_height; ++y)
  {
    // Scale distibution by the sine to get the sampling uniform. (Avoid sampling more values near the poles.)
    // See Physically Based Rendering v2, chapter 14.6.5 on Infinite Area Lights, page 728.
    float sinTheta = float(sin(M_PI * (double(y) + 0.5) / double(m_height))); // Make this as accurate as possible.

    for (unsigned int x = 0; x < m_width; ++x)
    {
      // Filter to keep the piecewise linear function intact for samples with zero value next to non-zero values.
      const float value = gaussianFilter(rgba, m_width, m_height, x, y);
      funcU[y * m_width + x] = value * sinTheta;

      // Compute integral over the actual function.
      const float *p = rgba + (y * m_width + x) * 4;
      const float intensity = (p[0] + p[1] + p[2]) / 3.0f;
      sum += intensity * sinTheta;
    }
  }

  // This integral is used inside the light sampling function (see sysParameter.envIntegral).
  m_integral = sum * 2.0f * M_PIf * M_PIf / float(m_width * m_height);

  // Now generate the CDF data.
  // Normalized 1D distributions in the rows of the 2D buffer, and the marginal CDF in the 1D buffer.
  // Include the starting 0.0f and the ending 1.0f to avoid special cases during the continuous sampling.
  float *cdfU = new float[(m_width + 1) * m_height];
  float *cdfV = new float[m_height + 1];

  for (unsigned int y = 0; y < m_height; ++y)
  {
    unsigned int row = y * (m_width + 1); // Watch the stride!
    cdfU[row + 0] = 0.0f; // CDF starts at 0.0f.

    for (unsigned int x = 1; x <= m_width; ++x)
    {
      unsigned int i = row + x;
      cdfU[i] = cdfU[i - 1] + funcU[y * m_width + x - 1]; // Attention, funcU is only m_width wide! 
    }

    const float integral = cdfU[row + m_width]; // The integral over this row is in the last element.
    funcV[y] = integral;                        // Store this as function values of the marginal CDF.

    if (integral != 0.0f)
    {
      for (unsigned int x = 1; x <= m_width; ++x)
      {
        cdfU[row + x] /= integral;
      }
    }
    else // All texels were black in this row. Generate an equal distribution.
    {
      for (unsigned int x = 1; x <= m_width; ++x)
      {
        cdfU[row + x] = float(x) / float(m_width);
      }
    }
  }

  // Now do the same thing with the marginal CDF.
  cdfV[0] = 0.0f; // CDF starts at 0.0f.
  for (unsigned int y = 1; y <= m_height; ++y)
  {
    cdfV[y] = cdfV[y - 1] + funcV[y - 1];
  }
        
  const float integral = cdfV[m_height]; // The integral over this marginal CDF is in the last element.
  funcV[m_height] = integral;            // For completeness, actually unused.

  if (integral != 0.0f)
  {
    for (unsigned int y = 1; y <= m_height; ++y)
    {
      cdfV[y] /= integral;
    }
  }
  else // All texels were black in the whole image. Seriously? :-) Generate an equal distribution.
  {
    for (unsigned int y = 1; y <= m_height; ++y)
    {
      cdfV[y] = float(y) / float(m_height);
    }
  }

  // Upload the CDFs into CUDA buffers.
  size_t sizeBytes = (m_width + 1) * m_height * sizeof(float);
  CU_CHECK( cuMemAlloc(&m_d_envCDF_U, sizeBytes) );
  CU_CHECK( cuMemcpyHtoD(m_d_envCDF_U, cdfU, sizeBytes) );

  sizeBytes = (m_height + 1) * sizeof(float);
  CU_CHECK( cuMemAlloc(&m_d_envCDF_V, sizeBytes) );
  CU_CHECK( cuMemcpyHtoD(m_d_envCDF_V, cdfV, sizeBytes) );

  delete[] cdfV;
  delete[] cdfU;
        
  delete[] funcV;
  delete[] funcU;
}

CUdeviceptr Texture::getCDF_U() const
{
  return m_d_envCDF_U;
}

CUdeviceptr Texture::getCDF_V() const
{
  return m_d_envCDF_V;
}

float Texture::getIntegral() const
{
  // This is the sum of the piecewise linear function values (roughly the texels' intensity) divided by the number of texels m_width * m_height.
  return m_integral;
}
