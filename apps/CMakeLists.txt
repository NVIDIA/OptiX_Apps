cmake_minimum_required(VERSION 3.17)

# Slightly advanced OptiX 7 introductory examples.
add_subdirectory( intro_runtime )  # Port of the advanced sample optixIntro_07 using the CUDA Runtime API.
add_subdirectory( intro_driver )   # Same as intro_runtime but using the CUDA Driver API.
add_subdirectory( intro_denoiser ) # Same as intro_driver plus built-in OptiX 7.x HDR denoiser.

if (OptiX72_FOUND OR OptiX73_FOUND OR OptiX74_FOUND OR OptiX75_FOUND OR OptiX76_FOUND)
  add_subdirectory( intro_motion_blur ) # Simple transform based object and camera motion blur.
else()
  message("WARNING: intro_motion_blur requires OptiX SDK 7.2.0 or higher. Example excluded from built.")
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

# Similar to rtigo9, but showing the maximum performance implementation with the smallest possible Shader Binding Table
# with one hit record entry per material shader and no hit records for the shadow ray!
# For that all BxDF sampling and evaluation callable programs have been replaced with individual closesthit programs.
# The cutoutout opacity feature has been removed to be able to use the fastest possible implementation for the shadow/visibility ray type.
# That can then be implemented without anyhit program by just using a miss shader instead.
# It's using the same scene description as rtigo9, but cutoput opacity textures are ignored.
add_subdirectory( rtigo10 )

