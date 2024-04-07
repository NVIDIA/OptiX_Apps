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

#ifndef LOGGER_H
#define LOGGER_H

#include <mutex>
#include <ostream>

// Adapts an std::ostream to the log callback interface used by the OptiX 7 API.
//
// It forwards all log messages to the ostream irrespective of their log level.
// To make use of this class, pass OptixLogBuffer::callback as log callback and 
// the address of your instance as log callback data.

class Logger
{
public:
  Logger(std::ostream& s)
  : m_stream(s)
  {
  }

  static void callback(unsigned int level, const char* tag, const char* message, void* cbdata)
  {
    Logger* self = static_cast<Logger*>(cbdata);
    self->callback(level, tag, message);
  }
  
  // Need this detour because m_mutex is not static.
  void callback( unsigned int /*level*/, const char* tag, const char* message )
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_stream << tag << ":" << ((message) ? message : "(no message)") << "\n";
  }

private:
  std::mutex    m_mutex;  // Mutex that protects m_stream.
  std::ostream& m_stream; // Needs m_mutex.
};

#endif // LOGGER_H
