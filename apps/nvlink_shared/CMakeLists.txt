# Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

cmake_minimum_required(VERSION 2.8)

project( nvlink_shared )
message("\nPROJECT_NAME = " "${PROJECT_NAME}")

find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLEW REQUIRED)
find_package(OptiX7 REQUIRED)
find_package(CUDA 10.0 REQUIRED)
find_package(DevIL_1_8_0 REQUIRED)
find_package(ASSIMP REQUIRED)

# Make the build work with either OptiX 7.0.0 and 7.1.0 releases
if (OptiX7_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX7_INCLUDE_DIR}")
elseif(OptiX71_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX71_INCLUDE_DIR}")
endif()
#message("OPTIX_INCLUDE_DIR = " "${OPTIX_INCLUDE_DIR}")

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
  inc/MaterialGUI.h
  inc/MyAssert.h
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
  src/main.cpp
  src/Options.cpp
  src/Parallelogram.cpp
  src/Parser.cpp
  src/Picture.cpp
  src/Plane.cpp
  src/Rasterizer.cpp
  src/Raytracer.cpp
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
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bxdf_diffuse.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bxdf_ggx_smith.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bxdf_specular.cu

  # Application kernels
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
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vector_math.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vertex_attributes.h
)

# Generate code for the lowest Maxwell SM 5.0 target.
NVCUDA_COMPILE_PTX( SOURCES ${SHADERS}
                    DEPENDENCIES ${SHADERS_HEADERS}
                    TARGET_PATH "${PTX_TARGET_DIR}/nvlink_shared_core"
                    GENERATED_FILES PTX_SOURCES
                    NVCC_OPTIONS "--gpu-architecture=compute_50" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "-I${OPTIX_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                  )

source_group( "imgui"           FILES ${IMGUI} )
source_group( "nvpro_math"      FILES ${NVPRO_MATH} )
source_group( "headers"         FILES ${HEADERS} )
source_group( "sources"         FILES ${SOURCES} )
source_group( "shaders"         FILES ${SHADERS} )
source_group( "shaders_headers" FILES ${SHADERS_HEADERS} )
source_group( "ptx"             FILES ${PTX_SOURCES})

#message("GLEW_INCLUDE_DIRS    = " "${GLEW_INCLUDE_DIRS}")
#message("GLFW_INCLUDE_DIR     = " "${GLFW_INCLUDE_DIR}")
#message("CUDA_TOOLKIT_INCLUDE = " "${CUDA_TOOLKIT_INCLUDE}")
#message("IL_INCLUDE_DIR       = " "${IL_INCLUDE_DIR}")
#message("ASSIMP_INCLUDE_DIRS  = " "${ASSIMP_INCLUDE_DIRS}")

include_directories(
  "."
  "inc"
  "imgui"
  ${GLEW_INCLUDE_DIRS}
  ${GLFW_INCLUDE_DIR}
  ${OPTIX_INCLUDE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${IL_INCLUDE_DIR}
  ${ASSIMP_INCLUDE_DIRS}
)

add_definitions(
  # Disable warnings for file operations fopen etc.
  "-D_CRT_SECURE_NO_WARNINGS"
)

add_executable( nvlink_shared
  ${IMGUI}
  ${NVPRO_MATH}
  ${HEADERS}
  ${SOURCES}
  ${SHADERS_HEADERS}
  ${SHADERS}
  ${PTX_SOURCES}
)

#message("OPENGL_gl_LIBRARY = " "${OPENGL_gl_LIBRARY}")
#message("GLEW_LIBRARIES    = " "${GLEW_LIBRARIES}")
#message("GLFW_LIBRARIES    = " "${GLFW_LIBRARIES}")
#message("CUDA_CUDA_LIBRARY = " "${CUDA_CUDA_LIBRARY}")
#message("IL_LIBRARIES      = " "${IL_LIBRARIES}")
#message("ILU_LIBRARIES     = " "${ILU_LIBRARIES}")
#message("ILUT_LIBRARIES    = " "${ILUT_LIBRARIES}")
#message("ASSIMP_LIBRARIES  = " "${ASSIMP_LIBRARIES}")

target_link_libraries( nvlink_shared
  ${OPENGL_gl_LIBRARY}
  ${GLEW_LIBRARIES}
  ${GLFW_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  ${IL_LIBRARIES}
  ${ILU_LIBRARIES}
  ${ILUT_LIBRARIES}
  ${ASSIMP_LIBRARIES}
)

if (UNIX)
  target_link_libraries( nvlink_shared dl )
endif()

set_target_properties( nvlink_shared PROPERTIES FOLDER "apps")

