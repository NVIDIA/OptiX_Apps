# Looks for the environment variable:
# OPTIX74_PATH

# Sets the variables :
# OPTIX74_INCLUDE_DIR

# OptiX74_FOUND

set(OPTIX74_PATH $ENV{OPTIX74_PATH})

if ("${OPTIX74_PATH}" STREQUAL "")
  if (WIN32)
    # Try finding it inside the default installation directory under Windows first.
    set(OPTIX74_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0")
  else()
    # Adjust this if the OptiX SDK 7.4.0 installation is in a different location.
    set(OPTIX74_PATH "~/NVIDIA-OptiX-SDK-7.4.0-linux64")
  endif()
endif()

find_path(OPTIX74_INCLUDE_DIR optix_7_host.h ${OPTIX74_PATH}/include)

# message("OPTIX74_INCLUDE_DIR = " "${OPTIX74_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX74 DEFAULT_MSG OPTIX74_INCLUDE_DIR)

mark_as_advanced(OPTIX74_INCLUDE_DIR)

# message("OptiX74_FOUND = " "${OptiX74_FOUND}")
