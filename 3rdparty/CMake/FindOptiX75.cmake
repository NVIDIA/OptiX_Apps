# Looks for the environment variable:
# OPTIX75_PATH

# Sets the variables :
# OPTIX75_INCLUDE_DIR

# OptiX75_FOUND

set(OPTIX75_PATH $ENV{OPTIX75_PATH})

if ("${OPTIX75_PATH}" STREQUAL "")
  if (WIN32)
    # Try finding it inside the default installation directory under Windows first.
    set(OPTIX75_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.5.0")
  else()
    # Adjust this if the OptiX SDK 7.5.0 installation is in a different location.
    set(OPTIX75_PATH "~/NVIDIA-OptiX-SDK-7.5.0-linux64")
  endif()
endif()

find_path(OPTIX75_INCLUDE_DIR optix_7_host.h ${OPTIX75_PATH}/include)

# message("OPTIX75_INCLUDE_DIR = " "${OPTIX75_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX75 DEFAULT_MSG OPTIX75_INCLUDE_DIR)

mark_as_advanced(OPTIX75_INCLUDE_DIR)

# message("OptiX75_FOUND = " "${OptiX75_FOUND}")
