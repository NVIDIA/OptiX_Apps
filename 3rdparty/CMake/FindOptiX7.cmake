# Looks for the environment variable:
# OPTIX7_PATH

# Sets the variables :
# OPTIX7_INCLUDE_DIR

# OptiX7_FOUND

set(OPTIX7_PATH $ENV{OPTIX7_PATH})

if ("${OPTIX7_PATH}" STREQUAL "")
  if (WIN32)
    # Try finding it inside the default installation directory under Windows first.
    set(OPTIX7_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0")
  else()
    # Adjust this if the OptiX SDK 7.0.0 installation is in a different location.
    set(OPTIX7_PATH "~/NVIDIA-OptiX-SDK-7.0.0-linux64")
  endif()
endif()

find_path(OPTIX7_INCLUDE_DIR optix_7_host.h ${OPTIX7_PATH}/include)

# message("OPTIX7_INCLUDE_DIR = " "${OPTIX7_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX7 DEFAULT_MSG OPTIX7_INCLUDE_DIR)

mark_as_advanced(OPTIX7_INCLUDE_DIR)

# message("OptiX7_FOUND = " "${OptiX7_FOUND}")
