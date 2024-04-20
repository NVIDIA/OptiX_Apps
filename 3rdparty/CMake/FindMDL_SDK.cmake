# Looks for environment variable:
# MDL_SDK_PATH 

# Sets the variables:
# MDL_SDK_INCLUDE_DIRS
# MDL_SDK_LIBRARIES
# MDL_SDK_FOUND

set(MDL_SDK_PATH $ENV{MDL_SDK_PATH})

# If there was no environment variable override for the MDL_SDK_PATH
# try finding it inside the local 3rdparty path.
if ("${MDL_SDK_PATH}" STREQUAL "")
  set(MDL_SDK_PATH "${LOCAL_3RDPARTY}/MDL_SDK")
endif()

message("MDL_SDK_PATH = " "${MDL_SDK_PATH}")

find_path( MDL_SDK_INCLUDE_DIRS "mi/mdl_sdk.h"
  PATHS /usr/include ${MDL_SDK_PATH}/include )

message("MDL_SDK_INCLUDE_DIRS = " "${MDL_SDK_INCLUDE_DIRS}")

# There are no link libraries inside the (pre-built) MDL SDK. DLLs are loaded manually.
#if (WIN32)
#  set(MDL_SDK_LIBRARY_DIR ${MDL_SDK_PATH}/nt-x86-x64/lib)
#else()
#  set(MDL_SDK_LIBRARY_DIR ${MDL_SDK_PATH}/lib)
#endif()

# message("MDL_SDK_LIBRARY_DIR = " "${MDL_SDK_LIBRARY_DIR}")

#find_library(MDL_SDK_LIBRARIES
#  NAMES MDL_SDK libmdl_sdk
#  PATHS ${MDL_SDK_LIBRARY_DIR} )

#message("MDL_SDK_LIBRARIES = " "${MDL_SDK_LIBRARIES}")

include(FindPackageHandleStandardArgs)

#find_package_handle_standard_args(MDL_SDK DEFAULT_MSG MDL_SDK_INCLUDE_DIRS MDL_SDK_LIBRARIES)
find_package_handle_standard_args(MDL_SDK DEFAULT_MSG MDL_SDK_INCLUDE_DIRS)

#mark_as_advanced(MDL_SDK_INCLUDE_DIRS MDL_SDK_LIBRARIES)
mark_as_advanced(MDL_SDK_INCLUDE_DIRS)

 message("MDL_SDK_FOUND = " "${MDL_SDK_FOUND}")
