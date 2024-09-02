/* 
 * Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "inc/Options.h"

#include <iostream>

Options::Options()
: m_width(512)
, m_height(512)
, m_mode(0)
, m_optimize(false)
{
}

//Options::~Options()
//{
//}

bool Options::parseCommandLine(int argc, char *argv[])
{
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg(argv[i]);

    if (arg == "?" || arg == "help" || arg == "--help")
    {
      printUsage(std::string(argv[0])); // Application name.
      return false;
    }
    else if (arg == "-w" || arg == "--width")
    {
      if (i == argc - 1)
      { 
        std::cerr << "Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_width = atoi(argv[++i]);
    }
    else if (arg == "-h" || arg == "--height")
    {
      if (i == argc - 1)
      { 
        std::cerr << "Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_height = atoi(argv[++i]);
    }
    else if (arg == "-m" || arg == "--mode")
    {
      if (i == argc - 1)
      { 
        std::cerr << "Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_mode = atoi(argv[++i]);
    }
    else if (arg == "-o" || arg == "--optimize")
    {
      m_optimize = true;
    }
    else if (arg == "-s" || arg == "--system")
    {
      if (i == argc - 1)
      { 
        std::cerr << "Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_filenameSystem = std::string(argv[++i]);
    }
    else if (arg == "-d" || arg == "--desc")
    {
      if (i == argc - 1)
      { 
        std::cerr << "Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_filenameScene = std::string(argv[++i]);
    }
    else
    {
      std::cerr << "Unknown option '" << arg << "'\n";
      printUsage(argv[0]);
      return false;
    }
  }

  if (m_filenameSystem.empty())
  {
    std::cerr << "ERROR: Options::parseCommandLine() System description filename is empty.\n";
    printUsage(argv[0]);
    return false;
  }

  if (m_filenameScene.empty())
  {
    std::cerr << "ERROR: Options::parseCommandLine() Scene description filename is empty.\n";
    printUsage(argv[0]);
    return false;
  }

  return true;
}

int Options::getWidth() const
{
  return m_width;
}

int Options::getHeight() const
{
  return m_height;
}

int Options::getMode() const
{
  return m_mode;
}

bool Options::getOptimize() const
{
  return m_optimize;
}

std::string Options::getSystem() const
{
  return m_filenameSystem;
}

std::string Options::getScene() const
{
  return m_filenameScene;
}


void Options::printUsage(const std::string& argv0)
{
  std::cerr << "\nUsage: " << argv0 << " [options]\n";
  std::cerr <<
    "App Options:\n"
    "   ? | help | --help       Print this usage message and exit.\n"
    "  -w | --width <int>       Width of the client window  (512) \n"
    "  -h | --height <int>      Height of the client window (512)\n"
    "  -m | --mode <int>        0 = interactive, 1 == benchmark (0)\n"
    "  -o | --optimize          Optimize the assimp scene graph (false)\n"
    "  -s | --system <filename> Filename for system options (empty).\n"
    "  -d | --desc   <filename> Filename for scene description (empty).\n"
  "App Keystrokes:\n"
  "  SPACE  Toggles GUI display.\n";
}
