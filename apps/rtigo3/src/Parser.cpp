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


#include "inc/Parser.h"

#include <iostream>
#include <fstream>
#include <sstream>


Parser::Parser()
: m_index(0)
, m_line(1)
{
}

//Parser::~Parser()
//{
//}

bool Parser::load(std::string const& filename)
{
  m_source.clear();

  std::ifstream inputStream(filename);
  if (!inputStream)
  {
    std::cerr << "ERROR: Parser::load() failed to open file " << filename << '\n';
    return false;
  }

  std::stringstream data;

  data << inputStream.rdbuf();

  if (inputStream.fail())
  {
    std::cerr << "ERROR: loadString() Failed to read file " << filename << '\n';
    return false;
  }

  m_source = data.str();
  return true;
}

ParserTokenType Parser::getNextToken(std::string& token)
{
  const static std::string whitespace = " \t"; // space, tab
  const static std::string value      = "+-0123456789.eE";
  const static std::string delimiter  = " \t\r\n"; // space, tab, carriage return, linefeed
  const static std::string newline    = "\n";
  const static std::string quotation  = "\"";

  token.clear(); // Make sure the returned token starts empty.

  ParserTokenType type = PTT_UNKNOWN; // This return value indicates an error.

  std::string::size_type first;
  std::string::size_type last;

  bool done = false;
  while (!done)
  {
    // Find first character which is not a whitespace.
    first = m_source.find_first_not_of(whitespace, m_index);
    if (first == std::string::npos)
    {
      token = std::string();
      type = PTT_EOF;
      done = true;
      continue;
    }

    // The found character indicates how parsing continues.
    char c = m_source[first];

    if (c == '#') // comment until the next newline
    {
      // m_index = first + 1; // skip '#' // Redundant.
      first = m_source.find_first_of(newline, m_index); // Skip everything until the next newline.
      if (first == std::string::npos)
      {
        type = PTT_EOF;
        done = true;
      }
      m_index = first + 1; // skip newline
      m_line++;
    }
    else if (c == '\r') // carriage return 13
    {
      m_index = first + 1;
    }
    else if (c == '\n') // newline (linefeed 10)
    {
      m_index = first + 1;
      m_line++;
    }
    else if (c == '\"') // Quotation mark delimits strings (filenames or material names with spaces.)
    {
      ++first; // Skip beginning quotation mark.
      last = m_source.find_first_of(quotation, first); // Find the ending quotation mark. Should be in the same line!
      if (last == std::string::npos) // Error, no matching end quotation mark found.
      { 
        m_index = first; // Keep scanning behind the quotation mark.
      }
      else
      {
        m_index = last + 1; // Skip the ending quotation mark.
        token = m_source.substr(first, last - first);
        type = PTT_STRING;
        done = true;
      }
    }
    else // anything else
    {
      last = m_source.find_first_of(delimiter, first);
      if (last == std::string::npos) 
      { 
        last = m_source.size();
      }
      m_index = last;
      token = m_source.substr(first, last - first);
      type = PTT_ID; // Default to general identifier.
      // Check if token is only built of characters used for numbers. 
      // (Not perfectly parsing a floating point number but good enough for most filenames.)
      if (isdigit(c) || c == '-' || c == '+' || c == '.') // Legal start characters for a floating point number.
      {
        last = token.find_first_not_of(value, 0);
        if (last == std::string::npos) 
        { 
          type = PTT_VAL;
        }
      }
      done = true;
    }
  }

  return type;
}

std::string::size_type Parser::getSize() const
{
  return m_source.size();
}

std::string::size_type Parser::getIndex() const
{
  return m_index;
}

unsigned int Parser::getLine() const
{
  return m_line;
}



