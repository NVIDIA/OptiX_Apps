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

cmake_minimum_required(VERSION 3.31 FATAL_ERROR)

if(UNIX)
  # Avoid a wrong CUDA version
  if( (NOT DEFINED ENV{CUDACXX}) OR ("$ENV{CUDACXX}" STREQUAL "") )
    message(FATAL_ERROR "Environment variable CUDACXX is undefined, please set to e.g. /usr/local/cuda-13.0/bin/nvcc")
  else()
    message(STATUS "Using env. var. CUDACXX: $ENV{CUDACXX}")
  endif()
else(UNIX)
  set(CMAKE_CUDA_COMPILER_FORCED TRUE) # gives error <math.h> not found on Linux
endif()

# Set to true to speedup kernel compilation during debugging.
set(DEBUG_ARCH FALSE)

# Do NOT run the C:\Program Files\CMake\share\cmake-<version>\Modules\CMakeTestCUDACompiler.cmake test
# which fails when using older CUDA Toolkits which are incompatible with the current MSVS 2022 17.10.x MSC 19.40 version. 
# NVCC checks the MSC version compatibility inside its host_config.h and requires the --allow-unsupported-compiler 
# command line option to work nonetheless and I don't know how to set that for this CMake built-in compiler test project.

project(GLTF_renderer LANGUAGES CXX CUDA)
message(STATUS "PROJECT_NAME = " "${PROJECT_NAME}")

# Find the CUDA Toolkit 12.0 or newer.
# This legacy script is used to explicitly set the CUDA::cudart and CUDA::cuda_driver libraries inside target_link_libraries()
# See https://cmake.org/cmake/help/v3.31/module/FindCUDAToolkit.html
# TODO depends on the architectures to support
find_package(CUDAToolkit 12.0 REQUIRED)

# ====================================================================================
# Set CMAKE_CUDA_ARCHITECTURES and LIB_CUDA_ARCHITECTURE.
# CMAKE_CUDA_ARCHITECTURES is the default value for CUDA_ARCHITECTURES property of target,
# which in turn is the list of architectures to generate device code for.
#
# See https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
# ====================================================================================

# CMake sets CMAKE_CUDA_ARCHITECTURES to the environment CUDAARCHS or oldest or default architecture chosen by the resp. compiler.
# "Users are encouraged to override this, as the default varies across compilers and compiler versions."
# Unconditionally setting this to all SM architectures the native CUDA kernels should support.
# (The native CUDA kernels are only used with config.h #define USE_GPU_SKINNING 1.)
# See https://cmake.org/cmake/help/latest/envvar/CUDAARCHS.html#envvar:CUDAARCHS

# 50 52    Maxwell
# 60 61 62 Pascal
# 70       Volta
# 75       Turing
# 80 86    Ampere
# 89       Ada
# 90       Hopper
# 100      GB200 and B200
# 120      Blackwell
# ???      Rubin

# CMAKE_CUDA_ARCHITECTURES "native" should only compile for the GPU version installed inside the system. 
# TODO What is that selecting when there are multiple GPU installed with different SM versions? The GPU device ordinal zero?
# set(CMAKE_CUDA_ARCHITECTURES native)
# NOTE: CMAKE_CUDA_ARCHITECTURES is not the CUDA_ARCHITECTURES property!

if( ${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "13.0" )
  # Drop architectures older than sm_75
  # It results in these arguments on the nvcc.exe command line:
  # --generate-code=arch=compute_75,code=[compute_75,sm_75]
  # --generate-code=arch=compute_80,code=[compute_80,sm_80]
  # --generate-code=arch=compute_86,code=[compute_86,sm_86]
  # --generate-code=arch=compute_89,code=[compute_89,sm_89]
  # ...
  set(LIB_CUDA_ARCHITECTURE 75)
  set(CMAKE_CUDA_ARCHITECTURES 75)
  if(NOT DEBUG_ARCH)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 120)
  endif()
else()
  # The above plus:
  # --generate-code=arch=compute_60,code=[compute_60,sm_60]
  # --generate-code=arch=compute_70,code=[compute_70,sm_70]
  set(LIB_CUDA_ARCHITECTURE 60)
  set(CMAKE_CUDA_ARCHITECTURES 60)
  if (NOT DEBUG_ARCH)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90 120)
  endif()
endif()

# message(STATUS "CMAKE_CUDA_ARCHITECTURES set to ${CMAKE_CUDA_ARCHITECTURES}")
# message(STATUS "LIB_CUDA_ARCHITECTURES set to ${LIB_CUDA_ARCHITECTURE}")

# ${CMAKE_CURRENT_SOURCE_DIR}       is <root>/apps/GLTF_renderer
# ${CMAKE_CURRENT_SOURCE_DIR}/..    is <root>/apps
# ${CMAKE_CURRENT_SOURCE_DIR}/../.. is <root>
#
# If the CMake build folder of this standalone solution is set to the same folder used for 
# all other OptiX_Apps examples, the executable and the OptiX device modules inside the
# GLTF_renderer_core folder will automatically appear in the same destination.
# Mind that the GLTF_renderer executable also expects the sheen_lut.hdr file next to the executable.

set(LOCAL_3RDPARTY "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty")
message(STATUS "LOCAL_3RDPARTY = " "${LOCAL_3RDPARTY}")

list(APPEND CMAKE_MODULE_PATH "${LOCAL_3RDPARTY}/CMake")
message(STATUS "CMAKE_MODULE_PATH = " "${CMAKE_MODULE_PATH}")

# Platform ####################################################################

# message(STATUS "System Processor:      ${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "Host System Processor: ${CMAKE_HOST_SYSTEM_PROCESSOR}")

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
      add_definitions("-DNOMINMAX" "-D_CRT_SECURE_NO_WARNINGS") # HACK Disable warnings about strcat and strcpy.
    endif(WIN32)
  endif(APPLE)
endif(UNIX)

# C++17 is required for std::filesystem
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

message(STATUS "PROJECT_BINARY_DIR = " "${PROJECT_BINARY_DIR}")

if(WIN32 AND "${CMAKE_GENERATOR}" MATCHES "^(Visual Studio).*")
  # Set the base folder where the per-project "core" folders with the *.ptx or *.optixir files get created.
  set(MODULE_TARGET_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(ConfigurationName)")

  # Enable multi-processor build on all Visual Studio versions.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")

  # Optionally link against the static VC runtime libraries.
  #add_compile_options(
  #  $<$<CONFIG:>:/MT> #---------|
  #  $<$<CONFIG:Debug>:/MTd> #---|-- Statically link the runtime libraries
  #  $<$<CONFIG:Release>:/MT> #--|
  #)
else()
  # DAR This should be independent of ${CMAKE_BUILD_TYPE} because that single-configuration generator will not create subfolders, will it?
  # Otherwise add something with if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(MODULE_TARGET_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()

find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLEW REQUIRED)
find_package(FastGLTF REQUIRED)
find_package(STB REQUIRED)
find_package(GLM REQUIRED)
find_package(DevIL_1_8_0 REQUIRED)

# Find the newest OptiX SDK

find_package(OptiX91)
find_package(OptiX90)
find_package(OptiX81)
find_package(OptiX80)
find_package(OptiX77)

# Make the build work with released OptiX SDK 7.7.0 and later versions.
if(OptiX91_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX91_INCLUDE_DIR}")
elseif(OptiX90_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX90_INCLUDE_DIR}")
elseif(OptiX81_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX81_INCLUDE_DIR}")
elseif(OptiX80_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX80_INCLUDE_DIR}")
elseif(OptiX77_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX77_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "No OptiX SDK 9.*.0 or 8.*.0 or 7.7.0 found.")
endif()
message(STATUS "OPTIX_INCLUDE_DIR = " "${OPTIX_INCLUDE_DIR}")

# OptiX SDK 7.5.0 and CUDA 11.7 added support for a new OptiX IR target, which is a binary intermediate format for the module input.
# The default module build target is PTX.
set(USE_OPTIX_IR FALSE)

if(OptiX91_FOUND OR OptiX90_FOUND OR OptiX81_FOUND OR OptiX80_FOUND OR OptiX77_FOUND)
  # Define USE_OPTIX_IR and change the target to OptiX-IR if the combination of OptiX SDK and CUDA Toolkit versions supports this mode.
  if((${CUDAToolkit_VERSION_MAJOR} GREATER 11) OR ((${CUDAToolkit_VERSION_MAJOR} EQUAL 11) AND (${CUDAToolkit_VERSION_MINOR} GREATER_EQUAL 7)))
    set(USE_OPTIX_IR TRUE)
  endif()
endif()
message(STATUS "USE_OPTIX_IR = " "${USE_OPTIX_IR}")

set(MODULE_EXT "ptx")
if(USE_OPTIX_IR)
  set(MODULE_EXT "optixir")
  # This define switches the OptiX program module filenames to either *.ptx or *.optixir extensions at compile time.
  add_definitions("-DUSE_OPTIX_IR")
endif()

# GLTF Renderer: imgui

set(IMGUI
  imgui/imconfig.h
  imgui/imgui.cpp
  imgui/imgui.h
  imgui/imgui_demo.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_impl_glfw.cpp
  imgui/imgui_impl_glfw.h
  imgui/imgui_impl_opengl3.cpp
  imgui/imgui_impl_opengl3.h
  imgui/imgui_impl_opengl3_loader.h
  imgui/imgui_internal.h
  imgui/imgui_tables.cpp
  imgui/imgui_widgets.cpp
  imgui/imstb_rectpack.h
  imgui/imstb_textedit.h
  imgui/imstb_truetype.h
)

# GLTF Renderer: headers

set(HEADERS_HOST
  Animation.h
  Application.h
  Arena.h
  Camera.h
  CheckMacros.h
  ConversionArguments.h
  DeviceBuffer.h
  HostBuffer.h
  HostKernels.h
  Logger.h
  Light.h
  Mesh.h
  MyAssert.h
  Node.h
  Options.h
  Picture.h
  Record.h
  SceneExtent.h
  Skin.h
  Texture.h
  Trackball.h
  Utils.h
  Utils.cpp
)

# GLTF Renderer: source

set(SOURCES_HOST
  Animation.cpp
  Application.cpp
  Arena.cpp
  Camera.cpp
  main.cpp
  Mesh.cpp
  Options.cpp
  Picture.cpp
  Texture.cpp
  Trackball.cpp
)

# GLTF Renderer: CUDA headers

set(HEADERS_CUDA
  ${CMAKE_CURRENT_SOURCE_DIR}/DeviceBuffer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Mesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MyAssert.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/vector_math.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/geometry_data.h
)

# GLTF Renderer: CUDA kernels

set(SOURCES_CUDA
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/kernel_morphing.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/kernel_skinning.cu
)

# GLTF Renderer: OptiX specific headers

set(HEADERS_OPTIX
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/bxdf_common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/config.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/geometry_data.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/hit_group_data.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/launch_parameters.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/light_definition.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/function_indices.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/material_data.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/per_ray_data.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/random.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/shader_common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/transform.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/vector_math.h
)

# GLTF Renderer: OptiX specific kernels

set(SOURCES_OPTIX
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/exception.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/hit.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/miss.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/raygen.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/light_sample.cu
)

source_group("imgui" FILES ${IMGUI})
source_group("host"  FILES ${HEADERS_HOST} ${SOURCES_HOST})
source_group("cuda"  FILES ${HEADERS_CUDA} ${SOURCES_CUDA})
source_group("optix" FILES ${HEADERS_OPTIX} ${SOURCES_OPTIX})

# ====================================================================================
# This section builds OptiX device program code with the CMake LANGUAGES CUDA feature!
# ====================================================================================

# Set library name to compile all OptiX device programs.
set(OPTIX_LIB "${PROJECT_NAME}_optix_lib")
message(STATUS "OPTIX_LIB = " "${OPTIX_LIB}")

# Create a CMake object library for OptiX compilation.
# This will just be a list of files which are not compiled as native CUDA kernels but as *.ptx or *.optixir OptiX module input code.
add_library(${OPTIX_LIB} OBJECT
  ${HEADERS_OPTIX}
  ${SOURCES_OPTIX}
)

# Set CUDA_OPTIX_COMPILATION property on the object library.
if(USE_OPTIX_IR)
  set_property(TARGET ${OPTIX_LIB} PROPERTY CUDA_OPTIX_COMPILATION ON) # -optix-ir
else()
  set_property(TARGET ${OPTIX_LIB} PROPERTY CUDA_PTX_COMPILATION ON)   # -ptx
endif()

# OptiX callable programs strictly require the --relocatable-device-code option, 
# otherwise the compiler will eliminate them as dead code because they aren't called from device code.

set_property(TARGET ${OPTIX_LIB} PROPERTY CUDA_SEPARABLE_COMPILATION ON) # -rdc=true (--relocatable-device-code=true)

set_property(TARGET ${OPTIX_LIB} PROPERTY CUDA_ARCHITECTURES ${LIB_CUDA_ARCHITECTURE}) 
# set_property(TARGET ${OPTIX_LIB} PROPERTY CUDA_ARCHITECTURES native)

#
# CUDA default compile options for the OptiX device programs.
#

target_compile_options(${OPTIX_LIB} PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:
    # --machine=64                   # => Implicit for x64 build targets.
    # --gpu-architecture=compute_60  # => set_property(TARGET ${OPTIX_LIB} PROPERTY CUDA_ARCHITECTURES 60)
    # --relocatable-device-code=true # => set_property(TARGET ${OPTIX_LIB} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    --use_fast_math
    --generate-line-info
    -Wno-deprecated-gpu-targets
    --allow-unsupported-compiler
  >
  # You can also set different nvcc command line options per configuration:
  # -lineinfo in CUDA release builds, -G in debug.
  # $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo> # This is not affecting performance and can always be set.
  # $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>          # WARNING! Debug device code is abysmally slow.
)

target_include_directories(${OPTIX_LIB} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${OPTIX_INCLUDE_DIR}"
)

# ================================================
# End of the OptiX device programs object library.
# ================================================

# Create an executable for the main project.
add_executable(${PROJECT_NAME} 
  ${IMGUI}
  ${HEADERS_HOST}
  ${SOURCES_HOST}
  ${HEADERS_CUDA}
  ${SOURCES_CUDA}
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON) # -rdc=true (--relocatable-device-code=true)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}) 


if(MSVC)
target_compile_options(${PROJECT_NAME} PRIVATE 
  $<$<COMPILE_LANGUAGE:CXX>:
    /W3 # Warning level only for the C++ language code.
  >
  $<$<COMPILE_LANGUAGE:CUDA>:
    # --machine=64                                        # => Implicit for x64 build targets.
    # --gpu-architecture=${CMAKE_CUDA_ARCHITECTURES}      # => set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 60;75;...)
    # --relocatable-device-code=true                      # => set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    --use_fast_math
    --generate-line-info
    -Wno-deprecated-gpu-targets
    --allow-unsupported-compiler
  >
)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OPTIX_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${GLEW_INCLUDE_DIRS}
  ${GLFW_INCLUDE_DIR}
  ${FASTGLTF_INCLUDE_DIR}
  ${STB_INCLUDE_DIR}
  ${GLM_INCLUDE_DIR}
  ${IL_INCLUDE_DIR}
)

# Disable the automatic addition of the cudadevrt.lib and cudart_static.lib.
set(CMAKE_CUDA_RUNTIME_LIBRARY None)

# Link the object library with the executable
target_link_libraries(${PROJECT_NAME} PRIVATE
  # Link against the shared CUDA runtime library explicitly.
  CUDA::cudart      # CUDA runtime link library. DLL ships with the CUDA toolkit.
  CUDA::cuda_driver # CUDA driver  link library. DLL ships with the display driver.
  OpenGL::GL
  ${GLEW_LIBRARIES}
  ${GLFW_LIBRARIES}
  ${FASTGLTF_LIBRARY}
  ${IL_LIBRARIES}
  ${ILU_LIBRARIES}
  ${ILUT_LIBRARIES}
  
  # Optionally link against different libraries for release and debug configurations.
  # optimized ${LIBRARIES_OPTIMIZED}
  # debug ${LIBRARIES_DEBUG}
  # Add additional link libraries for the application here.
  # ${PLATFORM_LIBRARIES}
)

# Making sure the OptiX object library is always compiled before the main project.
add_dependencies(${PROJECT_NAME} ${OPTIX_LIB})

set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "application")
set_property(TARGET ${OPTIX_LIB}    PROPERTY FOLDER "application")

# This will copy the OptiX modules to the destination folder when building the INSTALL target.
# If the INSTALL target is enabled inside the MSVS Build -> Configuration Manager
# (which must be re-enabled every time this CMakeLists.txt is changed)
# this will copy only the changed OptiX module files into the desired folder next to the executable.

install(FILES $<TARGET_OBJECTS:${OPTIX_LIB}> DESTINATION "$<TARGET_FILE_DIR:${PROJECT_NAME}>/${PROJECT_NAME}_core")

# Print CUDA info

message(STATUS "CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "LIB_CUDA_ARCHITECTURE ${LIB_CUDA_ARCHITECTURE}")
message(STATUS "CMAKE_CUDA_COMPILE_FEATURES ${CMAKE_CUDA_COMPILE_FEATURES}")
message(STATUS "CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "CMAKE_CUDA_KNOWN_FEATURES ${CMAKE_CUDA_KNOWN_FEATURES}" )
message(STATUS "CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ${CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS}")
message(STATUS "CMAKE_CUDA_RUNTIME_LIBRARY ${CMAKE_CUDA_RUNTIME_LIBRARY}")
message(STATUS "CMAKE_CUDA_SEPARABLE_COMPILATION ${CMAKE_CUDA_SEPARABLE_COMPILATION}")
message(STATUS "CMAKE_CUDA_STANDARD ${CMAKE_CUDA_STANDARD}")
message(STATUS "CMAKE_CUDA_STANDARD_REQUIRED ${CMAKE_CUDA_STANDARD_REQUIRED}")
message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
