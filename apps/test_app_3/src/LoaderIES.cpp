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

#include "inc/LoaderIES.h"
#include "inc/MyAssert.h"

#include <filesystem>
#include <fstream>
#include <sstream>

LoaderIES::LoaderIES()
{
  m_iesData.file.format = IESNA_86;
  
  m_iesData.lamp.numLamps         = 0;
  m_iesData.lamp.lumenPerLamp     = 0.0f;
  m_iesData.lamp.multiplier       = 1.0f;
  m_iesData.lamp.hasTilt          = false; // Default to TILT=NONE.
  m_iesData.lamp.tilt.orientation = LO_VERTICAL;
  m_iesData.lamp.tilt.numPairs    = 0;

  m_iesData.units = U_METERS;

  m_iesData.dimension.width  = 0.0f;
  m_iesData.dimension.length = 0.0f;
  m_iesData.dimension.height = 0.0f;

  m_iesData.electrical.ballastFactor                = 1.0f;
  m_iesData.electrical.ballastLampPhotometricFactor = 1.0f; 
  m_iesData.electrical.inputWatts                   = 0.0f; 

  m_iesData.photometric.goniometerType      = TYPE_C;
  m_iesData.photometric.numVerticalAngles   = 0;
  m_iesData.photometric.numHorizontalAngles = 0;
}

//LoaderIES::~LoaderIES()
//{
//}


bool LoaderIES::load(const std::string& filename)
{
  std::ifstream inputStream(filename);

  if (!inputStream)
  {
    std::cerr << "ERROR: LoaderIES::load() Failed to open file " << filename << '\n';
    return false;
  }

  std::stringstream data;

  data << inputStream.rdbuf();

  if (inputStream.fail())
  {
    std::cerr << "ERROR: LoaderIES::load() failed to read file " << filename << '\n';
    return false;
  }

  m_source = data.str();
  m_index = 0;
  m_lookbackIndex = m_index;

  m_iesData.file.name = filename; // Store the found file to use the path to search for possible tilt filename.

  return true;
}


bool LoaderIES::loadTilt(const std::string& filename)
{
  // This is a "tilt" filename stored inside the IES file. 
  std::filesystem::path tilt(filename);

  // First check if this exists in the current working directory, which is the module directory in my apps.
  // That's normally not the case if the IES is not loaded from the that directory.
  if (!std::filesystem::exists(tilt))
  {
    // Search for it next to the IES file.
    // Concatenate the path of IES files with the filename and extension of the TILT file
    tilt  = std::filesystem::path(m_iesData.file.name).remove_filename();
    tilt += std::filesystem::path(filename).filename();

    if (!std::filesystem::exists(tilt))
    {
      std::cerr << "ERROR: LoaderIES::load() did not find " << filename << " or " << tilt << '\n';
      return false;
    }
  }

  std::ifstream inputStream(tilt.string());

  if (!inputStream)
  {
    std::cerr << "ERROR: LoaderIES::load() Failed to open file " << filename << '\n';
    return false;
  }

  std::stringstream data;

  data << inputStream.rdbuf();

  if (inputStream.fail())
  {
    std::cerr << "ERROR: LoaderIES::load() failed to read file " << tilt << '\n';
    return false;
  }
  
  // Merge the tilt data string into the source string and handle tilt data as if it was specified with INCLUDE.
  std::string prefix = m_source.substr(0, m_index);
  std::string suffix = m_source.substr(m_index, std::string::npos);

  m_source = prefix + data.str() + suffix;

  return true;
}


LoaderIES::IESTokenType LoaderIES::getNextLine(std::string& token /* , const bool lookback */)
{
  // Line concatenation escape character is handled as whitespace, which is probably not quite right.
  const static std::string whitespace = " \t"; // space, tab
  const static std::string newline    = "\n";

  //token.clear(); // Make sure the returned token starts empty.

  IESTokenType type = ITT_UNKNOWN; // This return value indicates an error.

  std::string::size_type first;
  std::string::size_type last;
  
  m_lookbackIndex = m_index; // Remember the scan index of this line in case this line needs to be re-read.

  // Prune whitespace at the start of the token
  first = m_source.find_first_not_of(whitespace, m_index);
  if (first == std::string::npos)
  {
    token.clear();
    return ITT_EOF;
  }
  m_index = first; // Skip the leading whitespaces.
  
  last = m_source.find_first_of(newline, m_index);
  if (first == std::string::npos)
  {
    token.clear();
    return ITT_EOF;
  }

  m_index = last + 1; // Skip the token and the newline for the next scan.
  m_line++;

  // Prune whitespace at the end of the token.
  while ((first < last) && (m_source[last - 1] == ' '  || 
                            m_source[last - 1] == '\t' || 
                            m_source[last - 1] == '\r' ||
                            m_source[last - 1] == '\n'))
  {
    --last;
  }

  // Empty line! This is actually an error condition in the IES file.
  if (first == last)
  {
    token.clear();
    return ITT_EOL;
  }

  token = m_source.substr(first, last - first); // Get the line.
  return ITT_LINE;
}

LoaderIES::IESTokenType LoaderIES::getNextToken(std::string& token)
{
  const static std::string whitespace = " \t"; // space, tab
  const static std::string newline    = "\n";
  const static std::string value      = "+-0123456789.eE";
  const static std::string delimiter  = " \t\r\n"; // space, tab, carriage return, linefeed

  //token.clear(); // Make sure the returned token starts empty.

  IESTokenType type = ITT_UNKNOWN; // This return value indicates an error.

  std::string::size_type first;
  std::string::size_type last;

  bool done = false;
  while (!done)
  {
    // Find first character which is not a whitespace.
    first = m_source.find_first_not_of(whitespace, m_index);
    if (first == std::string::npos)
    {
      token.clear();
      return ITT_EOF;
    }

    m_lookbackIndex = first; // To be able to handle filenames with spaces it's necessary to look back at the last token's start index.

    // The found character indicates how parsing continues.
    char c = m_source[first];

    // No comments in IES files.
    //if (c == '#') // comment until the next newline
    //{
    //  // m_index = first + 1; // skip '#' // Redundant.
    //  first = m_source.find_first_of(newline, m_index); // Skip everything until the next newline.
    //  if (first == std::string::npos)
    //  {
    //    token.clear();
    //    type = ITT_EOF;
    //    done = true;
    //  }
    //  m_index = first + 1; // skip newline
    //  m_line++;
    //}
    //else
    if (c == '\r') // carriage return 13
    {
      m_index = first + 1;
    }
    else if (c == '\n') // newline (linefeed 10) // When parsing tokens, also skip this like carriage return.
    {
      //token.clear();
      //type = ITT_EOL;
      //done = true;
      m_index = first + 1;
      m_line++;
    }
    else // anything else
    {
      last = m_source.find_first_of(delimiter, first);
      if (last == std::string::npos) 
      { 
        last = m_source.size();
      }
      m_index = last; // Token has been processed from the m_source.

      token = m_source.substr(first, last - first); // Get the current token.
      // Check if token is only built of characters used for numbers. 
      // (Not perfectly parsing a floating point number but good enough for most filenames.)
      if (isdigit(c) || c == '-' || c == '+' || c == '.') // Legal start characters for a floating point number.
      {
        last = token.find_first_not_of(value, 0);
        if (last == std::string::npos) 
        { 
          type = ITT_VALUE;
        }
      }
      if (type == ITT_UNKNOWN) // Not a valid number, could still be an option keyword.
      {
        //std::map<std::string, IESTokenType>::const_iterator it = m_mapMtlKeywords.find(token);
        //if (it != m_mapMtlKeywords.end())
        //{
        //  type = it->second; // Known keyword.
        //}
        //else
        //{
          type = ITT_IDENTIFIER;
        //}
      }
      done = true;
    }
  }

  return type;
}

bool LoaderIES::parse()
{
  std::cout << "LoaderIES::parse() begin.\n";

  std::string token;
  
  IESTokenType type = getNextLine(token); // First line can be the format, a label or TILT=
  if (type == ITT_LINE)
  {
    if (token == std::string("IESNA:LM-63-2002"))
    {
      m_iesData.file.format = IESNA_02;
    }
    else if (token == std::string("IESNA:LM-63-1995"))
    {
      m_iesData.file.format = IESNA_95;
    }
    else if (token == std::string("IESNA91"))
    {
      m_iesData.file.format = IESNA_91;
    }
    else // First line is a label or "TILT=". Need to reread.
    {
      m_iesData.file.format = IESNA_86;
      m_index = m_lookbackIndex; // Rewind.
    }
  }
  else
  {
    std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading format.\n";
    return false;
  }

  // Label or "TILT="
  while (type == ITT_LINE)
  {
    type = getNextLine(token);
    if (type == ITT_LINE)
    {
      if (token.substr(0, 5) == "TILT=")
      {
        break; // Labels done. No rewind, just keep the line in token.
      }
      else
      {
        m_iesData.label.push_back(token); // Everything is a label until the TILT token.
      }
    }
    else
    {
      std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading labels.\n";
      return false;
    }
  }

  // token == "TILT=<type>"
  if (type == ITT_LINE)
  {
    std::string tilt = token.substr(5, std::string::npos); // Get the token behind "TILT=".
    
    m_iesData.lamp.tiltFilename = tilt; // This is the tilt filename in case TILT is neither NONE nor INCLUDE.

    if (tilt != std::string("NONE"))
    {
      m_iesData.lamp.hasTilt = true; // Tilt data already included or specified in a separate file.

      if (tilt != std::string("INCLUDE"))
      {
        // Simplest method is to insert the file contents at the current m_index into the source,
        // that is directly behind the TILT line, and keep on parsing as if it was "INCLUDE".
        if (!loadTilt(tilt))
        {
          return false;
        }
      }
    }
  }

  if (m_iesData.lamp.hasTilt) // Read tilt data.
  {
    type = getNextToken(token);
    if (ITT_VALUE)
    {
      const int ori = atoi(token.c_str());
      MY_ASSERT(1 <= ori && ori <= 3);
      m_iesData.lamp.tilt.orientation = (IESLampOrientation) ori;
    }
    else
    {
      std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading tilt.orientation.\n";
      return false;
    }

    type = getNextToken(token);
    if (ITT_VALUE)
    {
      m_iesData.lamp.tilt.numPairs = atoi(token.c_str());
      
      m_iesData.lamp.tilt.angles.reserve(m_iesData.lamp.tilt.numPairs);

      for (int i = 0; i < m_iesData.lamp.tilt.numPairs; ++i)
      {
        type = getNextToken(token);
        if (ITT_VALUE)
        {
          m_iesData.lamp.tilt.angles.push_back((float) atof(token.c_str()));
        }
        else
        {
          std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading tilt.angles.\n";
          return false;
        }
      }

      m_iesData.lamp.tilt.factors.reserve(m_iesData.lamp.tilt.numPairs);

      for (int i = 0; i < m_iesData.lamp.tilt.numPairs; ++i)
      {
        type = getNextToken(token);
        if (ITT_VALUE)
        {
          m_iesData.lamp.tilt.factors.push_back((float) atof(token.c_str()));
        }
        else
        {
          std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading tilt.factors.\n";
          return false;
        }
      }
    }
    else
    {
      std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading tilt.numPairs.\n";
      return false;
    }
  }
  
  type = getNextToken(token);
  if (type == ITT_VALUE) 
  {
    m_iesData.lamp.numLamps = atoi(token.c_str());
  }
  else
  {
    std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading lamp.numLamps.\n";
    return false;
  }

  type = getNextToken(token);
  if (type == ITT_VALUE) 
  {
    m_iesData.lamp.lumenPerLamp = (float) atof(token.c_str());
  }
  else
  {
    std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading lamp.lumenPerLamp.\n";
    return false;
  }

  type = getNextToken(token);
  if (type == ITT_VALUE) 
  {
    m_iesData.lamp.multiplier = (float) atof(token.c_str());
  }
  else
  {
    std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading lamp.multiplier.\n";
    return false;
  }

  type = getNextToken(token);
  if (type == ITT_VALUE) 
  {
    m_iesData.photometric.numVerticalAngles = atoi(token.c_str());
  }
  else
  {
    std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading photometric.numVerticalAngles.\n";
    return false;
  }

  type = getNextToken(token);
  if (type == ITT_VALUE) 
  {
    m_iesData.photometric.numHorizontalAngles = atoi(token.c_str());
  }
  else
  {
    std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading photometric.numHorizontalAngles.\n";
    return false;
  }

  type = getNextToken(token);
  if (type == ITT_VALUE) 
  {
    int gonio = atoi(token.c_str());
    MY_ASSERT(1 <= gonio && gonio <= 3);
    m_iesData.photometric.goniometerType = (IESGoniometerType) gonio;
  }
  else
  {
    std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading photometric.goniometerType.\n";
    return false;
  }

  type = getNextToken(token);
  if (type == ITT_VALUE) 
  {
    int units = atoi(token.c_str());
    MY_ASSERT(1 <= units && units <= 2);
    m_iesData.units = (IESUnits) units;
  }
  else
  {
    std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading units.\n";
    return false;
  }

  type = getNextToken(token);
  if (type == ITT_VALUE) 
  {
    m_iesData.dimension.width = (float) atof(token.c_str());
  }
  else
  {
    std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading dimension.width.\n";
    return false;
  }

  type = getNextToken(token);
  if (type == ITT_VALUE) 
  {
    m_iesData.dimension.length = (float) atof(token.c_str());
  }
  else
  {
    std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading dimension.length.\n";
    return false;
  }

  type = getNextToken(token);
  if (type == ITT_VALUE) 
  {
    m_iesData.dimension.height = (float) atof(token.c_str());
  }
  else
  {
    std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading dimension.height.\n";
    return false;
  }

  type = getNextToken(token);
  if (type == ITT_VALUE) 
  {
    m_iesData.electrical.ballastFactor = (float) atof(token.c_str());
  }
  else
  {
    std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading electrical.ballastFactor.\n";
    return false;
  }

  type = getNextToken(token);
  if (type == ITT_VALUE) 
  {
    m_iesData.electrical.ballastLampPhotometricFactor = (float) atof(token.c_str()); // Exception: In IESNA:LM-63-1995 this float value is meant for "future use".
  }
  else
  {
    std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading electrical.ballastLampPhotometricFactor.\n";
    return false;
  }

  type = getNextToken(token);
  if (type == ITT_VALUE) 
  {
    m_iesData.electrical.inputWatts = (float) atof(token.c_str());
  }
  else
  {
    std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading electrical.inputWatts.\n";
    return false;
  }

  m_iesData.photometric.verticalAngles.reserve(m_iesData.photometric.numVerticalAngles);
  
  for (int i = 0; i < m_iesData.photometric.numVerticalAngles; ++i)
  {
    type = getNextToken(token);
    if (type == ITT_VALUE) 
    {
      m_iesData.photometric.verticalAngles.push_back((float) atof(token.c_str()));
    }
    else
    {
      std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading photometric.verticalAngles.\n";
      return false;
    }
  }
  MY_ASSERT((size_t) m_iesData.photometric.numVerticalAngles == m_iesData.photometric.verticalAngles.size());

  m_iesData.photometric.horizontalAngles.reserve(m_iesData.photometric.numHorizontalAngles);
  
  for (int i = 0; i < m_iesData.photometric.numHorizontalAngles; ++i)
  {
    type = getNextToken(token);
    if (type == ITT_VALUE) 
    {
      m_iesData.photometric.horizontalAngles.push_back((float) atof(token.c_str()));
    }
    else
    {
      std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading photometric.horizontalAngles.\n";
      return false;
    }
  }
  MY_ASSERT((size_t) m_iesData.photometric.numHorizontalAngles == m_iesData.photometric.horizontalAngles.size());

  // numHorizontalAngles * numVerticalAngles candela values.
  // 2D index [h][v] <==> linear index [h * numVerticalAngles + v]
  m_iesData.photometric.candela.reserve(m_iesData.photometric.numHorizontalAngles * m_iesData.photometric.numVerticalAngles);

  for (int i = 0; i < m_iesData.photometric.numHorizontalAngles * m_iesData.photometric.numVerticalAngles; ++i)
  {
    type = getNextToken(token);
    if (type == ITT_VALUE) 
    {
      m_iesData.photometric.candela.push_back((float) atof(token.c_str()));
    }
    else
    {
      std::cerr << "ERROR: LoaderIES::parse(): Unexpected token \'" << token << "\' when reading photometric.candela.\n";
      return false;
    }
  }
  MY_ASSERT((size_t) (m_iesData.photometric.numVerticalAngles * m_iesData.photometric.numHorizontalAngles) == m_iesData.photometric.candela.size());
  
  m_source.clear(); // The input data isn't needed anymore.

  std::cout << "LoaderIES::parse() done.\n";

  return true;
}


const IESData& LoaderIES::getData()
{
  return m_iesData;
}