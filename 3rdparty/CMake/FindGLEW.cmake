# Looks for environment variable:
# GLEW_PATH 

# Sets the variables:
# GLEW_INCLUDE_DIRS
# GLEW_LIBRARIES
# GLEW_FOUND

set(GLEW_PATH $ENV{GLEW_PATH})

# If there was no environment variable override for the GLEW_PATH
# try finding it inside the local 3rdparty path.
if ("${GLEW_PATH}" STREQUAL "")
  set(GLEW_PATH "${LOCAL_3RDPARTY}/glew")
endif()

# message("GLEW_PATH = " "${GLEW_PATH}")

find_path( GLEW_INCLUDE_DIRS "GL/glew.h"
  PATHS /usr/include ${GLEW_PATH}/include )

# message("GLEW_INCLUDE_DIRS = " "${GLEW_INCLUDE_DIRS}")

if (WIN32)
  set(GLEW_LIBRARY_DIR ${GLEW_PATH}/lib/Release/x64)
else()
  set(GLEW_LIBRARY_DIR ${GLEW_PATH}/lib)
endif()

# message("GLEW_LIBRARY_DIR = " "${GLEW_LIBRARY_DIR}")

find_library(GLEW_LIBRARIES
  NAMES glew32 GLEW
  PATHS ${GLEW_LIBRARY_DIR} )

# message("GLEW_LIBRARIES = " "${GLEW_LIBRARIES}")

# DAR Not using the static GLEW libraries. What's the name under Linux?
#find_library(GLEW_STATIC_LIBRARIES
# NAMES glew32s
#  PATHS ${GLEW_LIBRARY_DIR} )

# message("GLEW_STATIC_LIBRARIES = " "${GLEW_STATIC_LIBRARIES}")

include(FindPackageHandleStandardArgs)

#find_package_handle_standard_args(GLEW DEFAULT_MSG GLEW_INCLUDE_DIRS GLEW_LIBRARIES GLEW_STATIC_LIBRARIES)
find_package_handle_standard_args(GLEW DEFAULT_MSG GLEW_INCLUDE_DIRS GLEW_LIBRARIES)

#mark_as_advanced(GLEW_INCLUDE_DIRS GLEW_LIBRARIES GLEW_STATIC_LIBRARIES)
mark_as_advanced(GLEW_INCLUDE_DIRS GLEW_LIBRARIES)

# message("GLEW_FOUND = " "${GLEW_FOUND}")
