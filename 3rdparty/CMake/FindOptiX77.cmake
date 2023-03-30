# Looks for the environment variable:
# OPTIX77_PATH

# Sets the variables :
# OPTIX77_INCLUDE_DIR

# OptiX77_FOUND

set(OPTIX77_PATH $ENV{OPTIX77_PATH})

if ("${OPTIX77_PATH}" STREQUAL "")
  if (WIN32)
    # Try finding it inside the default installation directory under Windows first.
    set(OPTIX77_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0")
  else()
    # Adjust this if the OptiX SDK 7.7.0 installation is in a different location.
    set(OPTIX77_PATH "~/NVIDIA-OptiX-SDK-7.7.0-linux64")
  endif()
endif()

find_path(OPTIX77_INCLUDE_DIR optix_host.h ${OPTIX77_PATH}/include)

# message("OPTIX77_INCLUDE_DIR = " "${OPTIX77_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX77 DEFAULT_MSG OPTIX77_INCLUDE_DIR)

mark_as_advanced(OPTIX77_INCLUDE_DIR)

# message("OptiX77_FOUND = " "${OptiX77_FOUND}")
