cmake_minimum_required(VERSION 2.8)

# Slightly advanced OptiX 7 introductory examples.
add_subdirectory( intro_runtime )  # Port of the advanced sample optixIntro_07 using the CUDA Runtime API.
add_subdirectory( intro_driver )   # Same as intro_runtime but using the CUDA Driver API.
add_subdirectory( intro_denoiser ) # Same as intro_driver plus built-in OptiX 7 AI denoiser.

# More advanced test application specifically designed to compare different multi-GPU and OpenGL interop strategies.
# This is meant as multi-GPU rendering distribution testbed in a simple viewer-like application foundation.
add_subdirectory( rtigo3 )

# Multi-GPU NVLINK texture and geometry acceleration structure peer-to-peer sharing example.
# Rendering is derived from the rtigo3 multi-GPU strategy RS_INTERACTIVE_MULTI_GPU_LOCAL_COPY.
# Also implements a simple arena allocator to reduce CUDA device memory fragmentation.
add_subdirectory( nvlink_shared )
