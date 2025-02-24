# Looks for the environment variable:
# OPTIX90_PATH

# Sets the variables :
# OPTIX90_INCLUDE_DIR

# OptiX90_FOUND

set(OPTIX90_PATH $ENV{OPTIX90_PATH})

if ("${OPTIX90_PATH}" STREQUAL "")
  if (WIN32)
    # Try finding it inside the default installation directory under Windows first.
    set(OPTIX90_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0")
  else()
    # Adjust this if the OptiX SDK 9.0.0 installation is in a different location.
    set(OPTIX90_PATH "~/NVIDIA-OptiX-SDK-9.0.0-linux64")
  endif()
endif()

find_path(OPTIX90_INCLUDE_DIR optix_host.h ${OPTIX90_PATH}/include)

# message("OPTIX90_INCLUDE_DIR = " "${OPTIX90_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX90 DEFAULT_MSG OPTIX90_INCLUDE_DIR)

mark_as_advanced(OPTIX90_INCLUDE_DIR)

# message("OptiX90_FOUND = " "${OptiX90_FOUND}")
