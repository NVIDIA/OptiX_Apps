# Looks for the environment variable:
# GLM_PATH

# Sets the variables :
# GLM_INCLUDE_DIR

# GLM_FOUND

set(GLM_PATH $ENV{GLM_PATH})

if ("${GLM_PATH}" STREQUAL "")
  set(GLM_PATH "${LOCAL_3RDPARTY}/glm")
endif()

message("GLM_PATH = " "${GLM_PATH}")

set(GLM_INCLUDE_DIR "${GLM_PATH}")
message("GLM_INCLUDE_DIR = " "${GLM_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(GLM DEFAULT_MSG GLM_INCLUDE_DIR)

mark_as_advanced(GLM_INCLUDE_DIR)

message("GLM_FOUND = " "${GLM_FOUND}")
