# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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

project(GLTF_renderer)
message("\nPROJECT_NAME = " "${PROJECT_NAME}")

find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDAToolkit 10.0 REQUIRED)
find_package(FastGLTF REQUIRED)
find_package(STB REQUIRED)
find_package(GLM REQUIRED)
find_package(DevIL_1_8_0 REQUIRED)

# OptiX SDK 7.x and 8.x versions are searched inside the top-level CMakeLists.txt.
# Make the build work with all currently released OptiX SDK 7.x and 8.x versions.
if(OptiX80_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX80_INCLUDE_DIR}")
elseif(OptiX77_FOUND)
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
  message(FATAL_ERROR "No OptiX SDK 7.x or 8.x found.")
endif()
#message("OPTIX_INCLUDE_DIR = " "${OPTIX_INCLUDE_DIR}")

# OptiX SDK 7.5.0 and CUDA 11.7 added support for a new OptiX IR target, which is a binary intermediate format for the module input.
# The default module build target is PTX.
set(USE_OPTIX_IR FALSE)
set(OPTIX_MODULE_EXTENSION ".ptx")
set(OPTIX_PROGRAM_TARGET "--ptx")

if(OptiX80_FOUND OR OptiX77_FOUND OR OptiX76_FOUND OR OptiX75_FOUND)
  # Define USE_OPTIX_IR and change the target to OptiX IR if the combination of OptiX SDK and CUDA Toolkit versions supports this mode.
  if((${CUDAToolkit_VERSION_MAJOR} GREATER 11) OR ((${CUDAToolkit_VERSION_MAJOR} EQUAL 11) AND (${CUDAToolkit_VERSION_MINOR} GREATER_EQUAL 7)))
    set(USE_OPTIX_IR TRUE)
    set(OPTIX_MODULE_EXTENSION ".optixir")
    set(OPTIX_PROGRAM_TARGET "--optix-ir")
  endif()
endif()

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

set(HEADERS
  Animation.h
  Application.h
  Arena.h
  Camera.h
  CheckMacros.h
  DeviceBuffer.h
  Logger.h
  Light.h
  Mesh.h
  MyAssert.h
  Node.h
  Options.h
  Picture.h
  Record.h
  Texture.h
  Trackball.h
)

set(SOURCES
  Animation.cpp
  Application.cpp
  Arena.cpp
  Camera.cpp
  main.cpp
  Options.cpp
  Picture.cpp
  Texture.cpp
  Trackball.cpp
)

# Prefix the shaders with the full path name to allow stepping through errors with F8.
set(SHADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/exception.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/hit.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/miss.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/raygen.cu
  
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/light_sample.cu
)

set(SHADERS_HEADERS
  # Rename these. I don't use CamelCase for CUDA code.
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

# When using OptiX SDK 7.5.0 and CUDA 11.7 or higher, the modules can either be built from OptiX IR input or from PTX input.
# OPTIX_PROGRAM_TARGET and OPTIX_MODULE_EXTENSION switch the NVCC compilation between the two options.
NVCUDA_COMPILE_MODULE(
  SOURCES ${SHADERS}
  DEPENDENCIES ${SHADERS_HEADERS}
  TARGET_PATH "${MODULE_TARGET_DIR}/${PROJECT_NAME}_core"
  EXTENSION "${OPTIX_MODULE_EXTENSION}"
  GENERATED_FILES PROGRAM_MODULES
  NVCC_OPTIONS "${OPTIX_PROGRAM_TARGET}" "--machine=64" "--gpu-architecture=compute_50" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "-Wno-deprecated-gpu-targets" "-I${OPTIX_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

source_group("imgui"           FILES ${IMGUI})
source_group("headers"         FILES ${HEADERS})
source_group("sources"         FILES ${SOURCES})
source_group("shaders"         FILES ${SHADERS})
source_group("shaders_headers" FILES ${SHADERS_HEADERS})
# The ${PROGRAM_MODULES} are in add_executable() and will appear inside the solution.
# Hide them away inside a source group, also because *.optixir cannot be opened inside the editor.
source_group("program_modules" FILES ${PROGRAM_MODULES})

include_directories(
  "."
  ${GLEW_INCLUDE_DIRS}
  ${GLFW_INCLUDE_DIR}
  ${OPTIX_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${FASTGLTF_INCLUDE_DIR}
  ${STB_INCLUDE_DIR}
  ${GLM_INCLUDE_DIR}
  ${IL_INCLUDE_DIR}
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

add_executable(${PROJECT_NAME}
  ${IMGUI}
  ${HEADERS}
  ${SOURCES}
  ${SHADERS_HEADERS}
  ${SHADERS}
  ${PROGRAM_MODULES}
)

target_link_libraries(${PROJECT_NAME}
  OpenGL::GL
  ${GLEW_LIBRARIES}
  ${GLFW_LIBRARIES}
  CUDA::cudart
  CUDA::cuda_driver
  ${FASTGLTF_LIBRARY}
  ${IL_LIBRARIES}
  ${ILU_LIBRARIES}
  ${ILUT_LIBRARIES}
)

if (UNIX)
  target_link_libraries(${PROJECT_NAME} dl)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "apps")
