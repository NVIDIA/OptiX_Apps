# Looks for the environment variable:
# OPTIX72_PATH

# Sets the variables :
# OPTIX72_INCLUDE_DIR

# OptiX72_FOUND

set(OPTIX72_PATH $ENV{OPTIX72_PATH})

if ("${OPTIX72_PATH}" STREQUAL "")
  if (WIN32)
    # Try finding it inside the default installation directory under Windows first.
    set(OPTIX72_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.2.0")
  else()
    # Adjust this if the OptiX SDK 7.2.0 installation is in a different location.
    set(OPTIX72_PATH "~/NVIDIA-OptiX-SDK-7.2.0-linux64")
  endif()
endif()

find_path(OPTIX72_INCLUDE_DIR optix_7_host.h ${OPTIX72_PATH}/include)

# message("OPTIX72_INCLUDE_DIR = " "${OPTIX72_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX72 DEFAULT_MSG OPTIX72_INCLUDE_DIR)

mark_as_advanced(OPTIX72_INCLUDE_DIR)

# message("OptiX72_FOUND = " "${OptiX72_FOUND}")
