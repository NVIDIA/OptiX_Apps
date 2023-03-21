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

#include "inc/Hair.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "inc/MyAssert.h"

#include "shaders/vector_math.h"

// Hair file format and models courtesy of Cem Yuksel: http://www.cemyuksel.com/research/hairmodels/

Hair::Hair()
{
  memset(&m_header, 0, sizeof(m_header));

  m_header.signature[0] = 'H';
  m_header.signature[1] = 'A';
  m_header.signature[2] = 'I';
  m_header.signature[3] = 'R';
}

//Hair::~Hair()
//{
//}

void Hair::setNumStrands(const unsigned int num)
{
  m_header.numStrands = num;
}

unsigned int Hair::getNumStrands() const
{
  return m_header.numStrands;
}

// Used when all strands have the same number of segments.
void Hair::setNumSegments(const unsigned int num)
{
  m_header.numSegments = num;
}

// Used when each hair strand can have a different number of segments.
void Hair::setSegmentsArray(const std::vector<unsigned short>& segments)
{
  // segments arrays can either be empty or must match the number of strands.
  // That means numStrands needs to be set first!
  MY_ASSERT(segments.empty() || m_header.numStrands == static_cast<unsigned int>(segments.size()));
  m_segmentsArray = segments;
}

unsigned int Hair::getNumSegments(const unsigned int idxStrand) const
{
  MY_ASSERT(idxStrand < m_header.numStrands);
  if (!m_segmentsArray.empty())
  {
    return m_segmentsArray[idxStrand];
  }
  return m_header.numSegments;
}

void Hair::setPointsArray(const std::vector<float3>& points)
{
  MY_ASSERT(!points.empty()); // Points array cannot be empty.
  m_pointsArray = points;

  // Changing numPoints potentially breaks the thickness, transparency, or color arrays temporarily.
  // These should always be set after the points array when present.
  m_header.numPoints = static_cast<unsigned int>(m_pointsArray.size());
}

float3 Hair::getPoint(const unsigned int idx) const
{
  MY_ASSERT(idx < m_header.numPoints);
  return m_pointsArray[idx];
}

void Hair::setThickness(const float thickness)
{
  m_header.thickness = thickness;
}

void Hair::setThicknessArray(const std::vector<float>& thickness)
{
  MY_ASSERT(thickness.empty() || m_header.numPoints == static_cast<unsigned int>(thickness.size()));
  m_thicknessArray = thickness;
}

float Hair::getThickness(const unsigned int idx) const
{
  MY_ASSERT(idx < m_header.numPoints);
  if (!m_thicknessArray.empty())
  {
    return m_thicknessArray[idx];
  }
  return m_header.thickness;
}

void Hair::setTransparency(const float transparency)
{
  m_header.transparency = transparency;
}

void Hair::setTransparencyArray(const std::vector<float>& transparency)
{
  MY_ASSERT(transparency.empty() || m_header.numPoints == static_cast<unsigned int>(transparency.size()));
  m_transparencyArray = transparency;
}

float Hair::getTransparency(const unsigned int idx) const
{
  MY_ASSERT(idx < m_header.numPoints);
  if (!m_transparencyArray.empty())
  {
    return m_transparencyArray[idx];
  }
  return m_header.transparency;
}

void Hair::setColor(const float3 color)
{
  m_header.color = color;
}

void Hair::setColorArray(const std::vector<float3>& color)
{
  MY_ASSERT(color.empty() || m_header.numPoints == static_cast<unsigned int>(color.size()));
  m_colorArray = color;
}

float3 Hair::getColor(const unsigned int idx) const
{
  MY_ASSERT(idx < m_header.numPoints);
  if (!m_colorArray.empty())
  {
    return m_colorArray[idx];
  }
  return m_header.color;
}

bool Hair::load(const std::string& filename)
{
  // This is a binary file format for 3D hair models.
  FILE *file = fopen(filename.c_str(), "rb");
  if (!file)
  {
    std::cerr << "ERROR: Hair::load(): fopen(" << filename << ") failed.\n";
    return false;
  }

  // A HAIR file begins with a 128-Byte long header.
  size_t count = fread(&m_header, sizeof(Header), 1, file);
  if (count < 1)
  {
    std::cerr << "ERROR: Hair::load(): Reading the header of " << filename << " failed.\n";
    fclose(file);
    return false;
  }

  if (strncmp(m_header.signature, "HAIR", 4) != 0)
  {
    std::cerr << "ERROR: Hair::load(): Header signature in " << filename << " not matching \"HAIR\"\n";
    fclose(file);
    return false;
  }

  // A HAIR file must have a points array, but all the other arrays are optional.
  // When an array does not exist, corresponding default value from the file header is used instead of the missing array.
  if ((m_header.bits & HAS_POINTS_ARRAY) == 0)
  {
    std::cerr << "ERROR: Hair::load(): Header bits in " << filename << " has no points array\n";
    fclose(file);
    return false;
  }

#if 0 // DEBUG
  std::cout << "Header of " << filename << " :\n";
  std::cout << "Number of hair strands = " << m_header.numStrands << "\n";
  std::cout << "Total number of points = " << m_header.numPoints;
  if (m_header.bits & HAS_POINTS_ARRAY) // Put this first because this must always exist.
  {
    std::cout << "POINTS_ARRAY";
  }
  if (m_header.bits & HAS_SEGMENTS_ARRAY)
  {
    std::cout << " | SEGMENTS_ARRAY";
  }
  if (m_header.bits & HAS_THICKNESS_ARRAY)
  {
    std::cout << " | THICKNESS_ARRAY";
  }
  if (m_header.bits & HAS_TRANSPARENCY_ARRAY)
  {
    std::cout << " | TRANSPARENCY_ARRAY";
  }
  if (m_header.bits & HAS_COLOR_ARRAY)
  {
    std::cout << " | COLOR_ARRAY";
  }
  std::cout << std::endl;

  if ((m_header.bits & HAS_SEGMENTS_ARRAY) == 0)
  {
    std::cout << "Default number of segments = " << m_header.numSegments << '\n';
  }
  if ((m_header.bits & HAS_THICKNESS_ARRAY) == 0)
  {
    std::cout << "Default thickness = " << m_header.thickness << '\n';
  }
  if ((m_header.bits & HAS_TRANSPARENCY_ARRAY) == 0)
  {
    std::cout << "Default transparency = " << m_header.transparency << '\n';
  }
  if ((m_header.bits & HAS_COLOR_ARRAY) == 0)
  {
    std::cout << "Default color = " << m_header.color[0] << ", " << m_header.color[1] << ", " << m_header.color[2] << '\n';
  }
#endif
  
  // The 3D hair model consists of strands, each one of which is represented by a number of line segments.

  // Segments array (unsigned short)
  // This array keeps the number of segments of each hair strand.
  // Each entry is a 16-bit unsigned integer, therefore each hair strand can have up to 65536 segments.
  if (m_header.bits & HAS_SEGMENTS_ARRAY)
  {
    m_segmentsArray.resize(m_header.numStrands);

    size_t count = fread(m_segmentsArray.data(), sizeof(unsigned short), m_header.numStrands, file);
    if (count < m_header.numStrands)
    {
      std::cerr << "ERROR: Hair::load(): Failed to read segments array of " << filename << '\n';
      fclose(file);
      return false;
    }
  }

  // Points array (float)
  // This array keeps the 3D positions each of hair strand point.
  // These points are not shared by different hair strands; each point belongs to a particular hair strand only.
  // Line segments of a hair strand connects consecutive points.
  // The points in this array are ordered by strand and from root to tip;
  // such that it begins with the root point of the first hair strand,
  // continues with the next point of the first hair strand until the tip of the first hair strand,
  // and then comes the points of the next hair strands.
  // Each entry is a 32-bit floating point number, and each point is defined by 3 consecutive numbers
  // that correspond to x, y, and z coordinates.
  if (m_header.bits & HAS_POINTS_ARRAY)
  {
    m_pointsArray.resize(m_header.numPoints);

    size_t count = fread(m_pointsArray.data(), sizeof(float3), m_header.numPoints, file);
    if (count < m_header.numPoints)
    {
      std::cerr << "ERROR: Hair::load(): Failed to read points array of " << filename << '\n';
      fclose(file);
      return false;
    }
  }

  // Thickness array (float)
  // This array keeps the thickness of hair strands at point locations,
  // therefore the size of this array is equal to the number of points.
  // Each entry is a 32-bit floating point number.
  if (m_header.bits & HAS_THICKNESS_ARRAY)
  {
    m_thicknessArray.resize(m_header.numPoints);

    size_t count = fread(m_thicknessArray.data(), sizeof(float), m_header.numPoints, file);
    if (count < m_header.numPoints)
    {
      std::cerr << "ERROR: Hair::load(): Failed to read thickness array of " << filename << '\n';
      fclose(file);
      return false;
    }
  }

  // Transparency array (float)
  // This array keeps the transparency of hair strands at point locations,
  // therefore the size of this array is equal to the number of points.
  // Each entry is a 32-bit floating point number.
  if (m_header.bits & HAS_TRANSPARENCY_ARRAY)
  {
    m_transparencyArray.resize(m_header.numPoints);

    size_t count = fread(m_transparencyArray.data(), sizeof(float), m_header.numPoints, file);
    if (count < m_header.numPoints)
    {
      std::cerr << "ERROR: Hair::load(): Failed to read transparency array of " << filename << '\n';
      fclose(file);
      return false;
    }
  }

  // Color array (float)
  // This array keeps the color of hair strands at point locations,
  // therefore the size of this array is three times the number of points.
  // Each entry is a 32-bit floating point number, and each color is defined by 3 consecutive numbers
  // that correspond to red, green, and blue components.
  if (m_header.bits & HAS_COLOR_ARRAY)
  {
    m_colorArray.resize(m_header.numPoints);

    size_t count = fread(m_colorArray.data(), sizeof(float3), m_header.numPoints, file);
    if (count < m_header.numPoints)
    {
      std::cerr << "ERROR: Hair::load(): Failed to read color array of " << filename << '\n';
      fclose(file);
      return false;
    }
  }

  fclose(file);

  return true;
}

bool Hair::save(const std::string& filename)
{
  // The points array must always be present and defines the sizes for thickness, transparency and color arrays when those are present.
  if (m_pointsArray.empty()) 
  {
    std::cerr << "ERROR: Hair::save(): Points array for " << filename << " is empty.\n";
    return false;
  }
  m_header.bits = HAS_POINTS_ARRAY;

  if (!m_segmentsArray.empty())
  {
    m_header.bits |= HAS_SEGMENTS_ARRAY;

    if (m_header.numStrands != static_cast<unsigned int>(m_segmentsArray.size()))
    {
      std::cerr << "ERROR: Hair::save(): Segments arrays  for " << filename << " has incorrect size " << m_segmentsArray.size() << ", numStrands = "<< m_header.numStrands << '\n';
      return false;
    }
    else if (m_header.numPoints != m_header.numStrands * (m_header.numSegments + 1))
    {
      std::cerr << "ERROR: Hair::save(): Number of points " << m_header.numPoints <<  " for " << filename << " not matching expected count " << m_header.numStrands * (m_header.numSegments + 1) << '\n';
      return false;
    }
  }
  if (!m_thicknessArray.empty())
  {
    m_header.bits |= HAS_THICKNESS_ARRAY;

    if (m_header.numPoints != static_cast<unsigned int>(m_thicknessArray.size()))
    {
      std::cerr << "ERROR: Hair::save(): Thickness array for " << filename << " has incorrect size " << m_thicknessArray.size() << ", numPoints = "<< m_header.numPoints << '\n';
      return false;
    }
  }
  if (!m_transparencyArray.empty())
  {
    m_header.bits |= HAS_TRANSPARENCY_ARRAY;

    if (m_header.numPoints != static_cast<unsigned int>(m_transparencyArray.size()))
    {
      std::cerr << "ERROR: Hair::save(): Transparency array for " << filename << " has incorrect size " << m_transparencyArray.size() << ", numPoints = "<< m_header.numPoints << '\n';
      return false;
    }
  }
  if (!m_colorArray.empty())
  {
    m_header.bits |= HAS_COLOR_ARRAY;

    if (m_header.numPoints != static_cast<unsigned int>(m_colorArray.size()))
    {
      std::cerr << "ERROR: Hair::save(): Color array for " << filename << " has incorrect size " << m_colorArray.size() << ", numPoints = "<< m_header.numPoints << '\n';
      return false;
    }
  }

  // All other fields inside the Hair header are supposed to be filled by the caller.

  FILE *file = fopen(filename.c_str(), "wb");
  if (!file)
  {
    std::cerr << "ERROR: Hair::save(): fopen(" << filename << ") failed.\n";
    return false;
  }

  // A HAIR file begins with a 128-Byte long header.
  size_t count = fwrite(&m_header, sizeof(Header), 1, file);
  if (count < 1)
  {
    std::cerr << "ERROR: Hair::save(): Writing the header of " << filename << " failed.\n";
    fclose(file);
    return false;
  }

  // The 3D hair model consists of strands, each one of which is represented by a number of line segments.

  if (m_header.bits & HAS_SEGMENTS_ARRAY)
  {
    size_t count = fwrite(m_segmentsArray.data(), sizeof(unsigned short), m_header.numStrands, file);
    if (count < m_header.numStrands)
    {
      std::cerr << "ERROR: Hair::save(): Failed to write segments array of " << filename << '\n';
      fclose(file);
      return false;
    }
  }

  if (m_header.bits & HAS_POINTS_ARRAY)
  {
    size_t count = fwrite(m_pointsArray.data(), sizeof(float3), m_header.numPoints, file);
    if (count < m_header.numPoints)
    {
      std::cerr << "ERROR: Hair::save(): Failed to write points array of " << filename << '\n';
      fclose(file);
      return false;
    }
  }

  if (m_header.bits & HAS_THICKNESS_ARRAY)
  {
    size_t count = fread(m_thicknessArray.data(), sizeof(float), m_header.numPoints, file);
    if (count < m_header.numPoints)
    {
      std::cerr << "ERROR: Hair::save(): Failed to write thickness array of " << filename << '\n';
      fclose(file);
      return false;
    }
  }

  if (m_header.bits & HAS_TRANSPARENCY_ARRAY)
  {
    size_t count = fwrite(m_transparencyArray.data(), sizeof(float), m_header.numPoints, file);
    if (count < m_header.numPoints)
    {
      std::cerr << "ERROR: Hair::save(): Failed to write transparency array of " << filename << '\n';
      fclose(file);
      return false;
    }
  }

  if (m_header.bits & HAS_COLOR_ARRAY)
  {
    size_t count = fwrite(m_colorArray.data(), sizeof(float3), m_header.numPoints, file);
    if (count < m_header.numPoints)
    {
      std::cerr << "ERROR: Hair::save(): Failed to write color array of " << filename << '\n';
      fclose(file);
      return false;
    }
  }

  fclose(file);

  return true;
}
