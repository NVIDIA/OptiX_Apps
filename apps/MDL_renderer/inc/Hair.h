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

#ifndef HAIR_H
#define HAIR_H

// Hair file format and models courtesy of Cem Yuksel: http://www.cemyuksel.com/research/hairmodels/

// For the float3
#include <cuda_runtime.h>

#include <string>
#include <vector>

#define HAS_SEGMENTS_ARRAY     (1 << 0)
#define HAS_POINTS_ARRAY       (1 << 1)
#define HAS_THICKNESS_ARRAY    (1 << 2)
#define HAS_TRANSPARENCY_ARRAY (1 << 3)
#define HAS_COLOR_ARRAY        (1 << 4)

// Note that the Hair class setter interfaces are unused in this example but allow generating *.hair files easily.
//
// At minimum that needs an array of float3 points and the matching header values.
// This renderer doesn't care about the transparency and color inside the *.hair model file.
// The material is defined by the assigned MDL hair BSDF material.
// Not setting the color might result in black in other applications.
//
// Example for setting individual cubic B-spline curves with four control points each.
// ... // Build your std::vector<float3> pointsArray here.
//Hair hair;
//hair.setNumStrands(numStrands);   // Number of hair strands inside the file.
//hair.setNumSegments(3);           // Cubic B-Splines with 4 control points each are 3 segments.
//hair.setPointsArray(pointsArray); // This defines the m_header.numPoints as well.
//hair.setThickness(thickness);     // Constant thickness. Use setThicknessArray() if values differ along strands.
//hair.save(filename);

class Hair
{
  // HAIR File Header (128 Bytes)
  struct Header
  {
    char         signature[4];    // Bytes 0-3 Must be "HAIR" in ascii code (48 41 49 52)
    unsigned int numStrands;      // Bytes 4-7 Number of hair strands as unsigned int
    unsigned int numPoints;       // Bytes 8-11 Total number of points of all strands as unsigned int
    unsigned int bits;            // Bytes 12-15 Bit array of data in the file
                                  // Bit-0 is 1 if the file has segments array.
                                  // Bit-1 is 1 if the file has points array (this bit must be 1).
                                  // Bit-2 is 1 if the file has thickness array.
                                  // Bit-3 is 1 if the file has transparency array.
                                  // Bit-4 is 1 if the file has color array.
                                  // Bit-5 to Bit-31 are reserved for future extension (must be 0).
    unsigned int numSegments;     // Bytes 16-19 Default number of segments of hair strands as unsigned int
                                  // If the file does not have a segments array, this default value is used.
    float        thickness;       // Bytes 20-23 Default thickness hair strands as float
                                  // If the file does not have a thickness array, this default value is used.
    float        transparency;    // Bytes 24-27 Default transparency hair strands as float
                                  // If the file does not have a transparency array, this default value is used.
    float3       color;           // Bytes 28-39 Default color hair strands as float array of size 3
                                  // If the file does not have a thickness array, this default value is used.
    char         information[88]; // Bytes 40-127 File information as char array of size 88 in ascii
  };

public:
  Hair();
  //~Hair();
  
  bool load(const std::string& filename);
  bool save(const std::string& filename);

  void         setNumStrands(const unsigned int num);
  unsigned int getNumStrands() const;

  void         setNumSegments(const unsigned int num);
  void         setSegmentsArray(const std::vector<unsigned short>& segments);
  unsigned int getNumSegments(const unsigned int idxStrand) const;

  void   setPointsArray(const std::vector<float3>& points); // This also sets m_header.numPoints!
  float3 getPoint(const unsigned int idx) const;

  void  setThickness(const float thickness);
  void  setThicknessArray(const std::vector<float>& thickness);
  float getThickness(const unsigned int idx) const;

  void  setTransparency(const float transparency);
  void  setTransparencyArray(const std::vector<float>& transparency);
  float getTransparency(const unsigned int idx) const;

  void   setColor(const float3 color);
  void   setColorArray(const std::vector<float3>& color);
  float3 getColor(const unsigned int idx) const;

private:
  Header m_header;

  std::vector<unsigned short> m_segmentsArray;     // Empty or size numStrands.
  std::vector<float3>         m_pointsArray;       // Size numPoints.
  std::vector<float>          m_thicknessArray;    // Empty or size numPoints.
  std::vector<float>          m_transparencyArray; // Empty or size numPoints.
  std::vector<float3>         m_colorArray;        // Empty or size numPoints.
};

#endif // HAIR_H
