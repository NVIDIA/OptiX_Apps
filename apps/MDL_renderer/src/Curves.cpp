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

#include "inc/SceneGraph.h"
#include "inc/MyAssert.h"

#include "inc/Hair.h"

#include "dp/math/math.h"

#include "shaders/vector_math.h"
#include "shaders/shader_common.h"


#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>


namespace sg
{

static float3 cubeProjection(const float3 r)
{
  // Spherical projection was pretty unpredictable with hair models.
  // Try a less distorted cubemap projection instead.
  // Note that each of the faces is the same 2D texture though.

  // See OpenGL 4.6 specs chapter "8.13 Cube Map Texture Selection".
  const float x = fabsf(r.x);
  const float y = fabsf(r.y);
  const float z = fabsf(r.z);

  float ma = 0.0f;
  float sc = 0.0f;
  float tc = 0.0f;

  if (x >= y && x >= z)
  {
    ma = x; // major axis rx
    if (r.x >= 0.0f)
    {
      // major axis +rx
      sc = -r.z;
      tc = -r.y;
    }
    else
    {
      // major axis -rx
      sc =  r.z;
      tc = -r.y;
    }
  }
  else if (y >= z)
  {
    ma = y; // major axis ry
    if (r.y >= 0.0f)
    {
      // major axis +ry
      sc = r.x;
      tc = r.z;
    }
    else
    {
      // major axis -ry
      sc =  r.x;
      tc = -r.z;
    }
  }
  else
  {
    ma = z; // major axis rz
    if (r.z >= 0.0f)
    {
      // major axis +rz
      sc =  r.x;
      tc = -r.y;
    }
    else
    {
      // major axis -rz
      sc = -r.x;
      tc = -r.y;
    }
  }

  const float s = 0.5f * (sc / ma + 1.0f);
  const float t = 0.5f * (tc / ma + 1.0f);

  return make_float3(s, t, 0.0f);
}


bool Curves::createHair(std::string const& filename, const float scale)
{
  // Note that hair files usually have a z-up coordinate system and they seem to be defined in centimeters.
  // That can be adjusted inside the scene description with a scale and rotate transform.
  // The "scale" parameter coming from the "model hair scale material filename" option
  // is modulating the thickness parameter defined inside the the hair file. 
  // Use scale == 1.0 to get the original thickness.

  // push
  // scale 0.01 0.01 0.01 
  // rotate 1 0 0 -90
  // model hair 1.0 bsdf_hair "file.hair"
  // pop

  Hair hair;

  if (!hair.load(filename))
  {
    return false;
  }
  
  // Iterate over all strands and build the curve attributes for the scene graph.
  const unsigned int numStrands = hair.getNumStrands();

  // Calculate a texture coordinate for each strand.
  std::vector<float3> texcoords;
  texcoords.resize(numStrands);

  // Simply do some spherical mapping to the center of the root points.
  // Calculate the center of the root points.
  float3 center = make_float3(0.0f);

  // This variable is always the running index over the segments.
  unsigned int idx = 0; // Set to root index of first strand.

  for (unsigned int strand = 0; strand < numStrands; ++strand)
  {
    const unsigned short numSegments = hair.getNumSegments(strand);

    if (numSegments == 0)
    {
      continue;
    }

    center += hair.getPoint(idx);

    idx += numSegments + 1; // Advance to next strand's root point. 
  }
  
  // Center of mass of the root strand points. 
  center /= float(numStrands);

  idx = 0; // Reset to root index of first strand.

  for (unsigned int strand = 0; strand < numStrands; ++strand)
  {
    const unsigned short numSegments = hair.getNumSegments(strand);

    if (numSegments == 0)
    {
      continue;
    }

    const float3 r = normalize(hair.getPoint(idx) - center);

    texcoords[strand] = cubeProjection(r);

    idx += numSegments + 1; // Advance to next strand's root point. 
  }

  // The idxRoot value is always the root point index of the current strand.
  unsigned int idxRoot = 0; // Set to root point of the first strand.

  for (unsigned int strand = 0; strand < numStrands; ++strand)
  {
    const unsigned short numSegments = hair.getNumSegments(strand);

    // If there is ever a strand defintion with zero segments,
    // just skip that and don't change any of the indices.
    if (numSegments == 0)
    {
      continue;
    }

    // Calculate the length of each strand. 
    // Linear length along the control points, not the actual cubic curve.
    // Needed for uFiber value in state::texture_coordinate(0).x
    float lengthStrand = 0.0f;

    // Calculate some fixed reference vector per strand.
    // Needed for vFiber value inside state::texture_coordinate(0).y
    // It's calculated as a "face normal" of the control points "polygon", which results in something like a 
    // fixed bitangent to the fiber tangent direction which works OK for usual hair definitions.
    float3 reference = make_float3(0.0f, 0.0f, 0.0f);

    idx = idxRoot; // Start local running index over the current strand.

    float3 v0 = hair.getPoint(idx); // Root point of the strand.
    float3 v1;

    for (unsigned short segment = 0; segment < numSegments; ++segment)
    {
      v1 = hair.getPoint(idx + 1);
      
      lengthStrand += length(v1 - v0);

      // Interpret the hair control points as a polygon and calculate a face normal of that.
      // The face normal is proportional to the projected surface of the polygon onto the ortho-normal basis planes.
      reference.x += (v0.y - v1.y) * (v0.z + v1.z);
      reference.y += (v0.z - v1.z) * (v0.x + v1.x);
      reference.z += (v0.x - v1.x) * (v0.y + v1.y);

      // Advance to next segment.
      v0 = v1;
      ++idx;
    }
      
    // v0 contains the endpoint of the last segment here.
    // Close the "polygon" to the root point.
    v1 = hair.getPoint(idxRoot); // First point of the strand.

    reference.x += (v0.y - v1.y) * (v0.z + v1.z);
    reference.y += (v0.z - v1.z) * (v0.x + v1.x);
    reference.z += (v0.x - v1.x) * (v0.y + v1.y);
    
    // If the Control points are not building some polygon face (maybe a straight line with no area),
    // just pick some orthogonal vector to a strand "tangent", the vector from root to tip control point.
    if (reference.x == 0.0f && reference.y == 0.0f && reference.z == 0.0f)
    {
      float3 tangent = hair.getPoint(idxRoot + hair.getNumSegments(strand) + 1) - hair.getPoint(idxRoot);

      // Generate an orthogonal vector to the reference tangent.
      reference = (fabsf(tangent.z) < fabsf(tangent.x))
                ? make_float3(tangent.z, 0.0f, -tangent.x)
                : make_float3(0.0f, tangent.z, -tangent.y);
    }
    reference = normalize(reference);
    
    // Build the cubic B-spline data.
    CurveAttributes attrib;

    // Remember the attribute start index of the B-Spline attributes building this strand.
    unsigned int index = static_cast<unsigned int>(m_attributes.size()); 
    
    idx = idxRoot; // Start local running index over the current strand again.

    // Start point, radius and fiber interpolant.
    float3 p0 = hair.getPoint(idx);                    // Start point of this curve segment.
    float  r0 = hair.getThickness(idx) * 0.5f * scale; // radius = thickness * 0.5f. The scale allows modulating modulate the hair thickness in the file.
    float  u0 = 0.0f;                                  // Interpolant along the hair strand from 0.0f at the root to 1.0 at the tip.

    // Initialize to keep the compiler happy.
    float3 p1 = p0;
    float  r1 = r0;
    float  u1 = u0;

    for (unsigned short segment = 0; segment < numSegments; ++segment)
    {
      // End point, radius and fiber interpolant.
      p1 = hair.getPoint(idx + 1);
      r1 = hair.getThickness(idx + 1) * 0.5f * scale;
      u1 = u0 + length(p1 - p0) / lengthStrand;

      if (segment == 0)
      {
        // Push an additional phantom point before the hair control points
        // to let the cubic B-spline start exactly at the first control point.
        attrib.vertex    = make_float4(p0 + (p0 - p1), std::max(0.0f, r0 + (r0 - r1)));
        attrib.reference = make_float4(reference, 0.0f);
        attrib.texcoord  = make_float4(texcoords[strand], u0 + (u0 - u1));
        m_attributes.push_back(attrib);
      }

      // Push the start point of this segment.
      attrib.vertex    = make_float4(p0, r0);
      attrib.reference = make_float4(reference, 0.0f);
      attrib.texcoord  = make_float4(texcoords[strand], u0);
      m_attributes.push_back(attrib);

      // The last segment will store the strand endpoint and append another phantom control point.
      if (segment + 1 == numSegments)
      {
        // Push the end point of the last segment.
        attrib.vertex    = make_float4(p1, r1);
        attrib.reference = make_float4(reference, 0.0f);
        attrib.texcoord  = make_float4(texcoords[strand], u1);
        m_attributes.push_back(attrib);

        // Push an additional phantom point after the hair control points
        // to let the cubic B-spline end exactly at the last control point.
        attrib.vertex    = make_float4(p1 + (p1 - p0), std::max(0.0f, r1 + (r1 - r0)));
        attrib.reference = make_float4(reference, 0.0f);
        attrib.texcoord  = make_float4(texcoords[strand], u1 + (u1 - u0));
        m_attributes.push_back(attrib);
      }

      // Let the end point become the new start point, except for the last segment
      p0 = p1;
      r0 = r1;
      u0 = u1;

      ++idx; // Advance to the next segment's start point.
    }

    // Generate the indices for this strand.
    const unsigned int indexEnd = static_cast<unsigned int>(m_attributes.size()) - 3; // Cubic B-spline curve uses 4 control points for each primitive.

    while (index < indexEnd)
    {
      m_indices.push_back(index);
      ++index;
    }

    // Done with one strand's control points.
    // (When we reach this numSegments != 0.)

    idxRoot += numSegments + 1; // Skip over all control points of the current strand.
  } 

  return true;
}

} // namespace sg
