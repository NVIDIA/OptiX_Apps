# Copyright (c) 2013-2022, NVIDIA CORPORATION. All rights reserved.
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

# FindCUDA.cmake is deprecated since CMake 3.10.
# Use FindCUDAToolkit.cmake added in CMake 3.17 instead.
cmake_minimum_required(VERSION 3.17)

project( rtigo9 )
message("\nPROJECT_NAME = " "${PROJECT_NAME}")

find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDAToolkit 10.0 REQUIRED)
find_package(DevIL_1_8_0 REQUIRED)
find_package(ASSIMP REQUIRED)

# OptiX SDK 7.x versions are searched inside the top-level CMakeLists.txt.
# Make the build work with all currently released OptiX SDK 7.x versions.
if(OptiX77_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX77_INCLUDE_DIR}")
elseif(OptiX76_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX76_INCLUDE_DIR}")
elseif(OptiX75_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX75_INCLUDE_DIR}")
elseif(OptiX74_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX74_INCLUDE_DIR}")
elseif(OptiX73_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX73_INCLUDE_DIR}")
elseif(OptiX72_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX72_INCLUDE_DIR}")
elseif(OptiX71_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX71_INCLUDE_DIR}")
elseif(OptiX70_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX70_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "No OptiX SDK 7.x found.")
endif()
#message("OPTIX_INCLUDE_DIR = " "${OPTIX_INCLUDE_DIR}")

# OptiX SDK 7.5.0 and CUDA 11.7 added support for a new OptiX IR target, which is a binary intermediate format for the module input.
# The default module build target is PTX.
set(USE_OPTIX_IR FALSE)
set(OPTIX_MODULE_EXTENSION ".ptx")
set(OPTIX_PROGRAM_TARGET "--ptx")

if (OptiX76_FOUND OR OptiX75_FOUND)
  # Define USE_OPTIX_IR and change the target to OptiX IR if the combination of OptiX SDK and CUDA Toolkit versions supports this mode.
  if ((${CUDAToolkit_VERSION_MAJOR} GREATER 11) OR ((${CUDAToolkit_VERSION_MAJOR} EQUAL 11) AND (${CUDAToolkit_VERSION_MINOR} GREATER_EQUAL 7)))
    set(USE_OPTIX_IR TRUE)
    set(OPTIX_MODULE_EXTENSION ".optixir")
    set(OPTIX_PROGRAM_TARGET "--optix-ir")
  endif()
endif()

set( IMGUI
  imgui/imconfig.h
  imgui/imgui.h
  imgui/imgui_impl_glfw_gl3.h
  imgui/imgui_internal.h
  imgui/stb_rect_pack.h
  imgui/stb_textedit.h
  imgui/stb_truetype.h
  imgui/imgui.cpp
  imgui/imgui_demo.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_impl_glfw_gl3.cpp
)

# Reusing some routines from the NVIDIA nvpro-pipeline https://github.com/nvpro-pipeline/pipeline
# Not built as a library, just using classes and functions directly.
# Asserts replaced with my own versions.
set( NVPRO_MATH
  # Math functions:
  dp/math/Config.h
  dp/math/math.h
  dp/math/Matmnt.h
  dp/math/Quatt.h
  dp/math/Trafo.h
  dp/math/Vecnt.h
  dp/math/src/Math.cpp
  dp/math/src/Matmnt.cpp
  dp/math/src/Quatt.cpp
  dp/math/src/Trafo.cpp
)

set( HEADERS
  inc/Application.h
  inc/Arena.h
  inc/Camera.h
  inc/CheckMacros.h
  inc/Device.h
  inc/LightGUI.h
  inc/LoaderIES.h
  inc/MaterialGUI.h
  inc/MyAssert.h
  inc/NVMLImpl.h
  inc/Options.h
  inc/Parser.h
  inc/Picture.h
  inc/Rasterizer.h
  inc/Raytracer.h
  inc/SceneGraph.h
  inc/Texture.h
  inc/Timer.h
  inc/TonemapperGUI.h
)

set( SOURCES
  src/Application.cpp
  src/Arena.cpp
  src/Assimp.cpp
  src/Box.cpp
  src/Camera.cpp
  src/Device.cpp
  src/LoaderIES.cpp
  src/main.cpp
  src/NVMLImpl.cpp
  src/Options.cpp
  src/Parser.cpp
  src/Picture.cpp
  src/Plane.cpp
  src/Rasterizer.cpp
  src/Raytracer.cpp
  src/Rect.cpp
  src/SceneGraph.cpp
  src/Sphere.cpp
  src/Texture.cpp
  src/Timer.cpp
  src/Torus.cpp
)

# Prefix the shaders with the full path name to allow stepping through errors with F8.
set( SHADERS
  # Core shaders.
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/anyhit.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/closesthit.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/exception.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/miss.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/raygeneration.cu

  # Direct callables
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lens_shader.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/light_sample.cu
  # BxDFs (BRDF, BTDF, BSDF implementations)
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bxdf.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bxdf_diffuse.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bxdf_ggx_smith.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bxdf_specular.cu
)

set( KERNELS
  # Native CUDA kernels
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compositor.cu
)

set( SHADERS_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/camera_definition.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compositor_data.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/config.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/function_indices.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/light_definition.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/material_definition.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/per_ray_data.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/random_number_generators.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader_common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/system_data.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/transform.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vector_math.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vertex_attributes.h
)

# When using OptiX SDK 7.5.0 and CUDA 11.7 or higher, the modules can either be built from OptiX IR input or from PTX input.
# OPTIX_PROGRAM_TARGET and OPTIX_MODULE_EXTENSION switch the NVCC compilation between the two options.
NVCUDA_COMPILE_MODULE(
  SOURCES ${SHADERS}
  DEPENDENCIES ${SHADERS_HEADERS}
  TARGET_PATH "${MODULE_TARGET_DIR}/rtigo9_core"
  EXTENSION "${OPTIX_MODULE_EXTENSION}"
  GENERATED_FILES PROGRAM_MODULES
  NVCC_OPTIONS "${OPTIX_PROGRAM_TARGET}" "--machine=64" "--gpu-architecture=compute_50" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "-Wno-deprecated-gpu-targets" "-I${OPTIX_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

# The native CUDA Kernels will be translated to *.ptx unconditionally.
NVCUDA_COMPILE_MODULE(
  SOURCES ${KERNELS}
  DEPENDENCIES ${SHADERS_HEADERS}
  TARGET_PATH "${MODULE_TARGET_DIR}/rtigo9_core"
  EXTENSION ".ptx"
  GENERATED_FILES KERNEL_MODULES
  NVCC_OPTIONS "--ptx" "--machine=64" "--gpu-architecture=compute_50" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "-Wno-deprecated-gpu-targets" "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

source_group( "imgui"           FILES ${IMGUI} )
source_group( "nvpro_math"      FILES ${NVPRO_MATH} )
source_group( "headers"         FILES ${HEADERS} )
source_group( "sources"         FILES ${SOURCES} )
source_group( "shaders"         FILES ${SHADERS} )
source_group( "shaders_headers" FILES ${SHADERS_HEADERS} )
source_group( "prg"             FILES ${PROGRAM_MODULES} )
source_group( "ptx"             FILES ${KERNEL_MODULES} )

include_directories(
  "."
  "inc"
  "imgui"
  ${GLEW_INCLUDE_DIRS}
  ${GLFW_INCLUDE_DIR}
  ${OPTIX_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${IL_INCLUDE_DIR}
  ${ASSIMP_INCLUDE_DIRS}
)

add_definitions(
  # Disable warnings for file operations fopen etc.
  "-D_CRT_SECURE_NO_WARNINGS"
)

if(USE_OPTIX_IR)
add_definitions(
  # This define switches the OptiX program module filenames to either *.optixir or *.ptx extensions at compile time.
  "-DUSE_OPTIX_IR"
)
endif()

add_executable( rtigo9
  ${IMGUI}
  ${NVPRO_MATH}
  ${HEADERS}
  ${SOURCES}
  ${SHADERS_HEADERS}
  ${SHADERS}
  ${PROGRAM_MODULES}
  ${KERNEL_MODULES}
)

target_link_libraries( rtigo9
  OpenGL::GL
  ${GLEW_LIBRARIES}
  ${GLFW_LIBRARIES}
  CUDA::cuda_driver
  ${IL_LIBRARIES}
  ${ILU_LIBRARIES}
  ${ILUT_LIBRARIES}
  ${ASSIMP_LIBRARIES}
)

if (UNIX)
  target_link_libraries( rtigo9 dl )
endif()

set_target_properties( rtigo9 PROPERTIES FOLDER "apps")

