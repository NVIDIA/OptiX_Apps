# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
cmake_minimum_required(VERSION 3.17)

if(UNIX)
  # Avoid a wrong CUDA version
  if( (NOT DEFINED ENV{CUDACXX}) OR ("$ENV{CUDACXX}" STREQUAL "") )
    message(FATAL_ERROR "Environment variable CUDACXX is undefined, please set to e.g. /usr/local/cuda-13.0/bin/nvcc")
  else()
    message(STATUS "Using env. var. CUDACXX: $ENV{CUDACXX}")
  endif()
  set(CMAKE_BUILD_TYPE Release)
endif()

project( optix_apps )

set(LOCAL_3RDPARTY "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")
message("LOCAL_3RDPARTY = " "${LOCAL_3RDPARTY}")

set(CMAKE_MODULE_PATH "${LOCAL_3RDPARTY}/CMake")
message("CMAKE_MODULE_PATH = " "${CMAKE_MODULE_PATH}")

find_package(CUDAToolkit 10.0 REQUIRED)

# Set sm_xx for all the apps
if( ${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "13.0" )
  # Drop architectures older than sm_75
  set(SM_COMPUTE compute_75)
else()
  set(SM_COMPUTE compute_50)
endif()

message(STATUS "SM_COMPUTE set to ${SM_COMPUTE}")

# This contains a macro which generates custom build rules for compiling *.cu input files to either *.ptx or *.optixir.
include("nvcuda_compile_module")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(UNIX) 
  set(OS "linux")
  add_definitions("-DLINUX")
  add_definitions("-Wno-unused-local-typedefs -Wno-delete-non-virtual-dtor")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
else(UNIX)
  if(APPLE)
  else(APPLE)
    if(WIN32)
      set(OS "win")
      add_definitions("-DNOMINMAX")
    endif(WIN32)
  endif(APPLE)
endif(UNIX)


# C++17 is required for std::filesystem
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

if(WIN32 AND "${CMAKE_GENERATOR}" MATCHES "^(Visual Studio).*")
  # Set the base folder where the per-project "core" folders with the *.ptx or *.optixir files get created.
  set(MODULE_TARGET_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(ConfigurationName)")
  # Enable multi-processor build on all Visual Studio versions.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
  # DAR This should be independent of ${CMAKE_BUILD_TYPE} because that single-configuration generator will not create subfolders, will it?
  # Otherwise add something with if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(MODULE_TARGET_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()


# Some useful macros
macro(ADD_TARGET_PROPERTIES _target _name)
  set(_properties)
  foreach(_prop ${ARGN})
    set(_properties "${_properties} ${_prop}")
  endforeach(_prop)
  get_target_property(_old_properties ${_target} ${_name})
  if(NOT _old_properties)
    # In case it's NOTFOUND
    set(_old_properties)
  endif(NOT _old_properties)
  set_target_properties(${_target} PROPERTIES ${_name} "${_old_properties} ${_properties}")
endmacro(ADD_TARGET_PROPERTIES)

macro(TARGET_INCLUDE_SYMBOL target symbol)
  if (WIN32)
    if ( LOCAL_ARCH STREQUAL "amd64" )
      add_target_properties( ${target} LINK_FLAGS /include:${symbol} )
    endif()
  endif()
  if(UNIX)
    add_target_properties( ${target} LINK_FLAGS "-Wl,--undefined=${symbol}" )
  endif()
endmacro()

# Search for all shipping OptiX 7 versions before adding subdirectories.
# The resp. FindOptiX7*.cmake looks for OPTIX7*_PATH environment variables.
# Doing this here allows excluding examples from the build which require a specific OptiX 7 minor version.
# If multiple OptiX SDK versions are found, the newest is picked by the individual examples.
find_package(OptiX70)
find_package(OptiX71)
# The intro_motion_blur example requires OptiX 7.2.0 or newer.
find_package(OptiX72)
find_package(OptiX73)
find_package(OptiX74)
find_package(OptiX75)
find_package(OptiX76)
find_package(OptiX77)
find_package(OptiX80)
find_package(OptiX81)
find_package(OptiX90)
find_package(OptiX91)
# The rtigo9_omm requires the CUDA Opacity Micromap Baking tool inside the OptiX Toolkit.
# The OptiX Toolkit is only built for OptiX SDK 7.6.0 and newer.
find_package(OptiXToolkit)

# The MDL_renderer requires the MDL SDK to generate shader code and load images.
find_package(MDL_SDK)

add_subdirectory( apps )
