# Looks for the environment variable:
# OPTIX81_PATH

# Sets the variables :
# OPTIX81_INCLUDE_DIR

# OptiX81_FOUND

set(OPTIX81_PATH $ENV{OPTIX81_PATH})

if ("${OPTIX81_PATH}" STREQUAL "")
  if (WIN32)
    # Try finding it inside the default installation directory under Windows first.
    set(OPTIX81_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0")
  else()
    # Adjust this if the OptiX SDK 8.1.0 installation is in a different location.
    set(OPTIX81_PATH "~/NVIDIA-OptiX-SDK-8.1.0-linux64")
  endif()
endif()

find_path(OPTIX81_INCLUDE_DIR optix_host.h ${OPTIX81_PATH}/include)

# message("OPTIX81_INCLUDE_DIR = " "${OPTIX81_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX81 DEFAULT_MSG OPTIX81_INCLUDE_DIR)

mark_as_advanced(OPTIX81_INCLUDE_DIR)

# message("OptiX81_FOUND = " "${OptiX81_FOUND}")
