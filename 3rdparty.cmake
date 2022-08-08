# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This whole CMake script is only required for Windows.

cmake_minimum_required(VERSION 3.17)

# Determine Visual Studio compiler version by analyzing the cl.exe output.
execute_process(COMMAND "cl.exe" OUTPUT_VARIABLE dummy ERROR_VARIABLE cl_info_string)

message("cl_info_string  = " "${cl_info_string}")

string(REGEX REPLACE ".*Version (..).(..).*" "\\1.\\2" cl_version ${cl_info_string})
string(REGEX MATCH "x64|x86" cl_architecture ${cl_info_string})

message("cl_version      = " "${cl_version}")
message("cl_architecture = " "${cl_architecture}")

# Only 64-bit is supported.
if ("${cl_architecture}" STREQUAL "x64")
  set(BUILD_ARCH x64)
else()
  message(FATAL_ERROR "Unsupported CPU architecture ${cl_architecture}")
endif()

# MSVC_VERSION in CMake identifies compiler versions by using the cl.exe majorminor string:
# 1200      = VS  6.0
# 1300      = VS  7.0
# 1310      = VS  7.1
# 1400      = VS  8.0 (v80 toolset)
# 1500      = VS  9.0 (v90 toolset)
# 1600      = VS 10.0 (v100 toolset)
# 1700      = VS 11.0 (v110 toolset)
# 1800      = VS 12.0 (v120 toolset)
# 1900      = VS 14.0 (v140 toolset)
# 1910-1919 = VS 15.0 (v141 toolset)
# 1920-1929 = VS 16.0 (v142 toolset)
# 1930-1939 = VS 17.0 (v143 toolset)

if(${cl_version} VERSION_EQUAL "19.00")
  # MSVS 2015 with VC 14.0
  set(GENERATOR "Visual Studio 14 2015")
  set(MSVC_TOOLSET "msvc-14.0")
elseif((${cl_version} VERSION_GREATER_EQUAL "19.10") AND (${cl_version} VERSION_LESS_EQUAL "19.19"))
  # MSVS 2017 with VC toolset 14.1
  set(GENERATOR "Visual Studio 15 2017")
  set(MSVC_TOOLSET "msvc-14.1")
elseif((${cl_version} VERSION_GREATER_EQUAL "19.20") AND (${cl_version} VERSION_LESS_EQUAL "19.29"))
  # MSVS 2019 with VC toolset 14.2
  set(GENERATOR "Visual Studio 16 2019")
  set(MSVC_TOOLSET "msvc-14.2")
elseif((${cl_version} VERSION_GREATER_EQUAL "19.30") AND (${cl_version} VERSION_LESS_EQUAL "19.39"))
  # MSVS 2022 with VC toolset 14.3
  set(GENERATOR "Visual Studio 17 2022")
  set(MSVC_TOOLSET "msvc-14.3")
  # Newer MSVS versions are not supported by available CUDA toolkits at this time (2022-08-08). 
endif()

#message("CMAKE_COMMAND = " "${CMAKE_COMMAND}")
#message("GENERATOR     = " "${GENERATOR}")
#message("MSVC_TOOLSET  = " "${MSVC_TOOLSET}")

if (NOT GENERATOR)
  message("Please check if you're running the 3rdparty.cmd inside the correct x64 Native Tools Command Prompt for VS2017, VS2019 or 2022")
  message("If yes, then check if the cl_info_string in line 34 is matching the expected regular expression in line 36.")
  message("This can fail on localized language systems where cl.exe is not reporting the expected 'Version' string.")
  message("In that case you can adjust the regular expression or hardcode the GENERATOR and MSVC_TOOLSET.")
  # For example like this for MSVS 2019:
  # set(GENERATOR "Visual Studio 16 2019")
  # set(MSVC_TOOLSET "msvc-14.2")
  message(FATAL_ERROR, "No suitable Visual Studio GENERATOR found.")
endif()

message("Creating 3rdparty library folder for ${GENERATOR} ${BUILD_ARCH}")

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty" CACHE PATH "default install path" FORCE)

set(DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/downloads")
set(PATCH_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/patches")

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/temp/sources")
set(BUILD_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/temp/build/${MSVC_TOOLSET}")

message("Install prefix: ${CMAKE_INSTALL_PREFIX} ${ARGC} ${ARGV}")

file(MAKE_DIRECTORY ${SOURCE_DIR})
file(MAKE_DIRECTORY ${BUILD_DIR})

macro(glew_sourceforge)
    message("GLEW")
    set(FILENAME "glew-2.1.0-win32.zip")
    if (NOT EXISTS "${DOWNLOAD_DIR}/${FILENAME}")
        message("  downloading")
        file(DOWNLOAD "https://sourceforge.net/projects/glew/files/glew/2.1.0/${FILENAME}" "${DOWNLOAD_DIR}/${FILENAME}" STATUS downloaded)
    endif()
    if (EXISTS "${CMAKE_INSTALL_PREFIX}/glew")
      message("  removing ${CMAKE_INSTALL_PREFIX}/glew")
      execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_INSTALL_PREFIX}/glew")
    endif()
    message("  extracting")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_DIR}/${FILENAME}" WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}")
    message("  renaming")
    file(RENAME "${CMAKE_INSTALL_PREFIX}/glew-2.1.0" "${CMAKE_INSTALL_PREFIX}/glew")
endmacro()

macro(glfw_sourceforge)
    message("GLFW")
    set(FILENAME "glfw-3.3.8.bin.WIN64.zip")
    if (NOT EXISTS "${DOWNLOAD_DIR}/${FILENAME}")
        message("  downloading")
        file(DOWNLOAD "https://sourceforge.net/projects/glfw/files/glfw/3.3.8/${FILENAME}" "${DOWNLOAD_DIR}/${FILENAME}" STATUS downloaded)
    endif()
    if (EXISTS "${CMAKE_INSTALL_PREFIX}/glfw")
      message("  removing ${CMAKE_INSTALL_PREFIX}/glfw")
      execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_INSTALL_PREFIX}/glfw")
    endif()
    message("  extracting")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_DIR}/${FILENAME}" WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}")
    message("  renaming")
    file(RENAME "${CMAKE_INSTALL_PREFIX}/glfw-3.3.8.bin.WIN64" "${CMAKE_INSTALL_PREFIX}/glfw")
endmacro()

macro(glfw_github)
    message("GLFW")
    set(FILENAME "glfw-master.zip")
    if (NOT EXISTS "${DOWNLOAD_DIR}/${FILENAME}")
        message("  downloading")
        file(DOWNLOAD "https://github.com/glfw/glfw/archive/master.zip" "${DOWNLOAD_DIR}/${FILENAME}" STATUS downloaded)
    endif()
    if (EXISTS "${CMAKE_INSTALL_PREFIX}/glfw")
      message("  removing ${CMAKE_INSTALL_PREFIX}/glfw")
      execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_INSTALL_PREFIX}/glfw")
    endif()
    message("  extracting")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_DIR}/${FILENAME}" WORKING_DIRECTORY "${SOURCE_DIR}")
    if (NOT EXISTS "${BUILD_DIR}/glfw")
      message("  creating ${BUILD_DIR}/glfw")
      file(MAKE_DIRECTORY "${BUILD_DIR}/glfw")
    endif()
    message("  generating")
    execute_process(COMMAND ${CMAKE_COMMAND} "-G${GENERATOR}" "-A${BUILD_ARCH}" "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/glfw" "${SOURCE_DIR}/glfw-master" WORKING_DIRECTORY "${BUILD_DIR}/glfw")
    message("  compiling")
    execute_process(COMMAND devenv.exe "${BUILD_DIR}/glfw/glfw.sln" /Build "Release|${BUILD_ARCH}" WORKING_DIRECTORY "${BUILD_DIR}/glfw")
    message("  installing")
    execute_process(COMMAND devenv.exe "${BUILD_DIR}/glfw/glfw.sln" /Build "Release|${BUILD_ARCH}" /Project INSTALL WORKING_DIRECTORY "${BUILD_DIR}/glfw")
endmacro()

macro(assimp_github)
    message("ASSIMP")
    set(FILENAME "assimp-master.zip")
    if (NOT EXISTS "${DOWNLOAD_DIR}/${FILENAME}")
        message("  downloading... (~46 MB)")
        file(DOWNLOAD "https://github.com/assimp/assimp/archive/master.zip" "${DOWNLOAD_DIR}/${FILENAME}" STATUS downloaded)
    endif()
    if (EXISTS "${CMAKE_INSTALL_PREFIX}/assimp")
      message("  removing ${CMAKE_INSTALL_PREFIX}/assimp")
      execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_INSTALL_PREFIX}/assimp")
    endif()
    message("  extracting")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_DIR}/${FILENAME}" WORKING_DIRECTORY "${SOURCE_DIR}")
    if (NOT EXISTS "${BUILD_DIR}/assimp")
      message("  creating ${BUILD_DIR}/assimp")
      file(MAKE_DIRECTORY "${BUILD_DIR}/assimp")
    endif()
    message("  generating")
    execute_process(COMMAND ${CMAKE_COMMAND} "-G${GENERATOR}" "-A${BUILD_ARCH}" "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/assimp" "${SOURCE_DIR}/assimp-master" WORKING_DIRECTORY "${BUILD_DIR}/assimp")
    message("  compiling")
    execute_process(COMMAND devenv.exe "${BUILD_DIR}/assimp/assimp.sln" /Build "Release|${BUILD_ARCH}" WORKING_DIRECTORY "${BUILD_DIR}/assimp")
    message("  installing")
    execute_process(COMMAND devenv.exe "${BUILD_DIR}/assimp/assimp.sln" /Build "Release|${BUILD_ARCH}" /Project INSTALL WORKING_DIRECTORY "${BUILD_DIR}/assimp")
endmacro()

glew_sourceforge()
glfw_sourceforge()
assimp_github()

# If the 3rdparty tools should be updated with additional libraries, commenting out these two lines avoids expensive recompilation of existing tools again.
message("deleting temp folder")
execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_SOURCE_DIR}/temp")

