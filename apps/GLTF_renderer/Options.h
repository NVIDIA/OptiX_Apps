/* 
 * Copyright (c) 2013-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef OPTIONS_H
#define OPTIONS_H

#include <string>
#include <filesystem>

// The maximum number of launches per render() call.
#define MAX_LAUNCHES 1000

class Options
{
public:
  Options();
  //~Options();

  bool parseCommandLine(int argc, char *argv[]);

  std::filesystem::path getFilename() const;
  int                   getWidthClient() const;
  int                   getHeightClient() const;
  int                   getWidthResolution() const;
  int                   getHeightResolution() const;
  int                   getLaunches() const;
  int                   getInterop() const;
  bool                  getPunctual() const;
  int                   getMiss() const;
  std::string           getEnvironment() const;
  // Radius of all the spheres (for glTF points) as fraction of the scene diameter.
  float                 getSphereRadiusFraction() const;

private:
  void printUsage(std::string const& argv);

   /// To size windows and render sizes for different DPI-s (HD, 4K,...)
   float getViewportScale() const;

private:
  std::filesystem::path m_filename;
  int                   m_widthClient;
  int                   m_heightClient;
  int                   m_widthResolution;
  int                   m_heightResolution;
  int                   m_launches;
  int                   m_interop;
  bool                  m_punctual;
  int                   m_miss;
  std::string           m_environment;
  float                 m_sphereRadiusFraction;
};

#endif // OPTIONS_H
