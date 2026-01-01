# Looks for the environment variable:
# OPTIX91_PATH

# Sets the variables :
# OPTIX91_INCLUDE_DIR

# OptiX91_FOUND

set(OPTIX91_PATH $ENV{OPTIX91_PATH})

if ("${OPTIX91_PATH}" STREQUAL "")
  if (WIN32)
    # Try finding it inside the default installation directory under Windows first.
    set(OPTIX91_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0")
  else()
    # Adjust this if the OptiX SDK 9.1.0 installation is in a different location.
    set(OPTIX91_PATH "~/NVIDIA-OptiX-SDK-9.1.0-linux64")
  endif()
endif()

find_path(OPTIX91_INCLUDE_DIR optix_host.h ${OPTIX91_PATH}/include)

# message("OPTIX91_INCLUDE_DIR = " "${OPTIX91_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX91 DEFAULT_MSG OPTIX91_INCLUDE_DIR)

mark_as_advanced(OPTIX91_INCLUDE_DIR)

# message("OptiX91_FOUND = " "${OptiX91_FOUND}")
