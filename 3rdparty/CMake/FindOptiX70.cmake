# Looks for the environment variable:
# OPTIX70_PATH

# Sets the variables :
# OPTIX70_INCLUDE_DIR

# OptiX70_FOUND

set(OPTIX70_PATH $ENV{OPTIX70_PATH})

if ("${OPTIX70_PATH}" STREQUAL "")
  if (WIN32)
    # Try finding it inside the default installation directory under Windows first.
    set(OPTIX70_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0")
  else()
    # Adjust this if the OptiX SDK 7.0.0 installation is in a different location.
    set(OPTIX70_PATH "~/NVIDIA-OptiX-SDK-7.0.0-linux64")
  endif()
endif()

find_path(OPTIX70_INCLUDE_DIR optix_7_host.h ${OPTIX70_PATH}/include)

# message("OPTIX70_INCLUDE_DIR = " "${OPTIX70_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX70 DEFAULT_MSG OPTIX70_INCLUDE_DIR)

mark_as_advanced(OPTIX70_INCLUDE_DIR)

# message("OptiX70_FOUND = " "${OptiX70_FOUND}")
