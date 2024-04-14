/* 
 * Copyright (c) 2013-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef PARSER_H
#define PARSER_H

#include <string>

enum ParserTokenType
{
  PTT_UNKNOWN, // Unknown, normally indicates an error.
  PTT_ID,      // Keywords and identifiers (not a number).
  PTT_VAL,     // Immediate floating point value.
  PTT_STRING,  // Filenames and any other identifier in quotation marks.
  PTT_EOL,     // End of line.
  PTT_EOF      // End of file.
};


// System and scene file parsing information.
class Parser
{
public:
  Parser();
  //~Parser();
  
  bool load(const std::string& filename);

  ParserTokenType getNextToken(std::string& token);

  size_t                 getSize() const;
  std::string::size_type getIndex() const;
  unsigned int           getLine() const;
    
private:
  std::string            m_source; // System or scene description file contents.
  std::string::size_type m_index;  // Parser's current character index into m_source.
  unsigned int           m_line;   // Current source code line, one-based for error messages.
};

#endif // PARSER_H
