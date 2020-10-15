# Looks for the environment variable:
# OPTIX71_PATH

# Sets the variables :
# OPTIX71_INCLUDE_DIR

# OptiX71_FOUND

set(OPTIX71_PATH $ENV{OPTIX71_PATH})

if ("${OPTIX71_PATH}" STREQUAL "")
  if (WIN32)
    # Try finding it inside the default installation directory under Windows first.
    set(OPTIX71_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.1.0")
  else()
    # Adjust this if the OptiX SDK 7.1.0 installation is in a different location.
    set(OPTIX71_PATH "~/NVIDIA-OptiX-SDK-7.1.0-linux64")
  endif()
endif()

find_path(OPTIX71_INCLUDE_DIR optix_7_host.h ${OPTIX71_PATH}/include)

# message("OPTIX71_INCLUDE_DIR = " "${OPTIX71_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX71 DEFAULT_MSG OPTIX71_INCLUDE_DIR)

mark_as_advanced(OPTIX71_INCLUDE_DIR)

# message("OptiX71_FOUND = " "${OptiX71_FOUND}")
