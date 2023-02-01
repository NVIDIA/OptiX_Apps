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

#pragma once

#ifndef LOADER_IES_H
#define LOADER_IES_H

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

enum IESFormat
{
  IESNA_86, // LM-63-1986 // Default when there is no format identifier.
  IESNA_91, // LM-63-1991                       
  IESNA_95, // LM-63-1995
  IESNA_02  // LM-63-2002 // DEBUG This implementation was against the LM-63-1995 specs, but contents for LM-63-2002 files looked identical so far.
};

struct IESFile
{
  std::string name; 
  IESFormat   format;
};

enum IESLampOrientation // Lamp-to-luminaire geometry
{
  LO_VERTICAL   = 1, // Lamp vertical base up or down
  LO_HORIZONTAL = 2, // Lamp horizontal
  LO_TILT       = 3  // Lamp tilted
};

struct IESTilt
{
  IESLampOrientation orientation;
  int                numPairs;
  std::vector<float> angles;
  std::vector<float> factors;
};

struct IESLamp
{
  int         numLamps;
  float       lumenPerLamp;
  float       multiplier;
  bool        hasTilt;      // false if TILT=NONE, true for TILT=INCLUDE and TILT=<filename>
  std::string tiltFilename;
  IESTilt     tilt;
};

enum IESUnits
{
  U_FEET   = 1,
  U_METERS = 2
};

struct IESCavityDimension
{
  float width;  // Opening width
  float length; // Opening length
  float height; // Cavity height
};

struct IESElectricalData
{
  float ballastFactor;
  float ballastLampPhotometricFactor; // "future use" in IESNA:LM-63-1995
  float inputWatts;
};

enum IESGoniometerType
{
  TYPE_A = 3,
  TYPE_B = 2,
  TYPE_C = 1
};

struct IESPhotometricData
{
  IESGoniometerType  goniometerType;
  int                numVerticalAngles;
  int                numHorizontalAngles;
  std::vector<float> verticalAngles;
  std::vector<float> horizontalAngles;
  std::vector<float> candela; // numHorizontalAngles * numVerticalAngles values. 2D index [h][v] <==> linear index [h * numVerticalAngles + v]
};

struct IESData
{
  IESFile                  file;
  std::vector<std::string> label;
  IESLamp                  lamp;
  IESUnits                 units;
  IESCavityDimension       dimension;
  IESElectricalData        electrical;
  IESPhotometricData       photometric;
};


class LoaderIES
{
public:
  enum IESTokenType
  {
    ITT_LINE,
    ITT_IDENTIFIER, // Not really used. Only "TILT=" would be an identifier and that is handled with getNextLine(). Everything not a label is an ITT_VALUE.
    ITT_VALUE,
    ITT_EOL,
    ITT_EOF,
    ITT_UNKNOWN
  };

public:
  LoaderIES();
  //~LoaderIES();

  bool load(const std::string& iesFilename);
  bool parse();
  IESData const& getData();
    
private:
  IESTokenType getNextLine(std::string& token);
  IESTokenType getNextToken(std::string& token);

  bool loadTilt(std::string const& tiltFilename);
  
private:
  IESData m_iesData;

  std::string            m_source;        // Parameters file contents.
  std::string::size_type m_index;         // Parser's current character index into m_source.
  std::string::size_type m_lookbackIndex; // To be able to handle filenames with spaces it's necessary to look back at the last token's start index.
                                          // Memorized in getNextToken() used in getFilename();
  unsigned int           m_line;          // Current source code line, one based for error messages.
};

#endif // LOADER_IES_H
