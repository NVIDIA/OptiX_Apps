# Looks for the environment variable:
# FASTGLTF_PATH

# Sets the variables :
# FASTGLTF_INCLUDE_DIR
# FASTGLTF_LIBRARY

# FastGLTF_FOUND

set(FASTGLTF_PATH $ENV{FASTGLTF_PATH})

if ("${FASTGLTF_PATH}" STREQUAL "")
  set(FASTGLTF_PATH "${LOCAL_3RDPARTY}/fastgltf")
endif()

message("FASTGLTF_PATH = " "${FASTGLTF_PATH}")

# After the fastgltf INSTALL project ran, the release library resides in 3rdparty/fastgltf/lib
# Unfortunately that is always only one build target.
# The 3rdparty.cmake script builds both Debug and Release targets under Windows but installs only the the Release target.
# If you need to run the GLTF_renderer as Debug target, copy the files from 3rdparty\fastgltf\build\<msc-toolset>\Debug to 3rdparty\fastgltf\lib

set(FASTGLTF_INCLUDE_DIR "${FASTGLTF_PATH}/include")
message("FASTGLTF_INCLUDE_DIR = " "${FASTGLTF_INCLUDE_DIR}")

find_library(FASTGLTF_LIBRARY NAMES fastgltf PATHS "${FASTGLTF_PATH}/lib")
message("FASTGLTF_LIBRARY = " "${FASTGLTF_LIBRARY}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(FastGLTF DEFAULT_MSG FASTGLTF_INCLUDE_DIR FASTGLTF_LIBRARY)

mark_as_advanced(FASTGLTF_INCLUDE_DIR FASTGLTF_LIBRARY)

message("FastGLTF_FOUND = " "${FastGLTF_FOUND}")
 