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

# Slightly advanced OptiX 7 introductory examples.
add_subdirectory( intro_runtime )  # Port of the advanced sample optixIntro_07 using the CUDA Runtime API.
add_subdirectory( intro_driver )   # Same as intro_runtime but using the CUDA Driver API.
add_subdirectory( intro_denoiser ) # Same as intro_driver plus built-in OptiX 7.x HDR denoiser.

if (OptiX91_FOUND OR OptiX90_FOUND OR OptiX81_FOUND OR OptiX80_FOUND OR OptiX77_FOUND OR OptiX76_FOUND OR OptiX75_FOUND OR OptiX74_FOUND OR OptiX73_FOUND OR OptiX72_FOUND)
  add_subdirectory( intro_motion_blur ) # Simple transform based object and camera motion blur.
else()
  message("WARNING: intro_motion_blur requires OptiX SDK 7.2.0 or higher. Example excluded from build.")
endif()

# More advanced test application specifically designed to compare different multi-GPU and OpenGL interop strategies.
# This is meant as multi-GPU rendering distribution testbed in a simple viewer-like application foundation.
add_subdirectory( rtigo3 )

# Multi-GPU NVLINK texture and geometry acceleration structure peer-to-peer sharing example.
# Rendering is derived from the rtigo3 multi-GPU strategy RS_INTERACTIVE_MULTI_GPU_LOCAL_COPY.
# Also implements a simple arena allocator to reduce CUDA device memory fragmentation.
add_subdirectory( nvlink_shared )

# Similar to nvlink_shared, but showing how to implement more light types.
# Implements singular point lights, spot lights, and IES light profiles (all with and without additional colored projection texture),
# rectangle area lights with or without importance sampled emission texture and support for cutout opacity,
# arbitrary triangle mesh area lights with and without emission texture and support for cutout opacity,
# constant and importance sampled spherical HDR environment lights.
# All lights can be placed and oriented with transform matrices and the spherical HDR environment can be freely oriented with a rotation matrix.
# The scene description file format has been overhauled to allow overrides of the camera and tonemapper settings,
# definition of emissive materials and lights, and surface materials and geometry (as before).
# Both rtigo9 and rtigo10 allow toggling the direct lighting at runtime which is usful when testing direct lighting implementations.
# Singular lights won't have any effect without direct lighting because they cannot be hit implicitly
# because they don't exist as geometry inside the scene. (Infinitely small lights do not exist in the physical world.)
add_subdirectory( rtigo9 )

# Same as rtigo9 but using Opacity Micromaps (OMM) for cutout textures which require OptiX SDK 7.6.0 or higher.
# OMMs are hardware accelerated on RTX Ada generation GPUs.
# The OMM generation is done with the OptiX Toolkit CUDA OmmBaking tool.
# The main renderer differences are the use of the OMMs for materials with cutout opacity.
# That required a change of the GAS instancing since not all GAS can be used with all materials anymore.
# Additionally the shadow/visibility ray implementation must use a different, faster implementation because
# fully transparent or fully opaque micro triangle intersections on cutout opacity materials do not invoke 
# the anyhit program anymore, which is the main purpose and performance benefit of OMMs.
if ((OptiX91_FOUND OR OptiX90_FOUND OR OptiX81_FOUND OR OptiX80_FOUND OR OptiX77_FOUND OR OptiX76_FOUND) AND OptiXToolkit_FOUND)
  add_subdirectory( rtigo9_omm )
else()
  message("WARNING: rtigo9_omm requires OptiX SDK 7.6.0 or higher and the OptiX Toolkit OMM Baking Library. Example excluded from build.")
endif()

# Similar to rtigo9, but showing the maximum performance implementation with the smallest possible Shader Binding Table
# with one hit record entry per material shader and no hit records for the shadow ray!
# For that all BxDF sampling and evaluation callable programs have been replaced with individual closesthit programs.
# The cutoutout opacity feature has been removed to be able to use the fastest possible implementation for the shadow/visibility ray type.
# That can then be implemented without anyhit program by just using a miss shader instead.
# It's using the same scene description as rtigo9, but cutoput opacity textures are ignored.
add_subdirectory( rtigo10 )

# Based on rtigo10 but changed integrator to work like the MDL_renderer for throughput, pdf, and lights.
# Note that mesh and rect lights are now defined with radiant exitance instead of radiant intensity,
# so with the diffuse EDF these are 1/PI darker than in rtigo10 but match the MDL_renderer.
# Replaced the GGX-Smith implementation with excerpts from the MDL SDK libbsdf to support direct lighting of glossy transparent materials.
# That means singular light types will now show proper reflections on glossy transparent objects 
# and even caustics when the roughness is not too smooth, because hitting backfaces will be directly lit from lights
# on the transmission side which adds radiance.
# Added support for Specular and GGX_Smith BTDF materials.
# Also homogeneous volume scattering is implemented in this example the same way as inside the MDL_renderer. (See scene_rtigo12_*.txt files for examples.)
add_subdirectory( rtigo12 )

if (MDL_SDK_FOUND)
  # Based on rtigo9, but replacing all material handling with MDL materials.
  # The MDL SDK is used to generate the minimal amount of direct callable programs per MDL material shader.
  # The closesthit and anyhit programs inside hit.cu call the valid direct callable programs to handle the input values
  # from MDL expressions and the sampling and evaluation of the material's distribution functions.
  add_subdirectory( MDL_renderer )
  # Based on MDL_renderer, but adding support for a custom Signed-Distance-Field (SDF) primitive
  # which is defined by a box inside which a SDF as 3D texture is sampled. 
  # The 3D texture data can either be single-component fp16 (*.bin) or fp32 (any other extension than *.bin) formats.
  # Supports any MDL material without emission or cutout-opacity.
  add_subdirectory( MDL_sdf )
else()
  message("WARNING: MDL_renderer and MDL_sdf require the MDL_SDK. Set the MDL_SDK_PATH variable. Examples excluded from build.") 
endif()

# Note that the GLTF_renderer/CMakeLists.txt is now a standalone solution!
# That's because it is an example for using the native CMake LANGUAGE CUDA feature and 
# a CMake "Object Library" for the OptiX device code translation.
# This allows running native CUDA kernels with the CUDA Runtime API chevron <<<>>> operator
# which is used for the expensive skinning animations so far.
# Unfortunately that CMake LANGUAGE CUDA feature affects all projects inside a solution and 
# would break the build for all examples using the custom build rules for the OptiX device code *.cu files.
message("The GLTF_renderer example is a standalone solution now! Details inside the README.md files.")


# Derived from nvlink_shared but adjusted to allow dynamic control about the resource sharing via the new system option peerToPeer bitfield.
# Allows selective sharing of resources via NVLINK or PCI-E (new), texture data, GAS and vertex attribute data, HDR environment CDFs.
add_subdirectory( bench_shared )

# Same as bench_shared without any window or OpenGL handling.
add_subdirectory( bench_shared_offscreen )
