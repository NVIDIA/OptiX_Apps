# Looks for the environment variable:
# STB_PATH

# Sets the variables :
# STB_INCLUDE_DIR

# STB_FOUND

set(STB_PATH $ENV{STB_PATH})

if ("${STB_PATH}" STREQUAL "")
  set(STB_PATH "${LOCAL_3RDPARTY}/stb")
endif()

message("STB_PATH = " "${STB_PATH}")

set(STB_INCLUDE_DIR "${STB_PATH}")
message("STB_INCLUDE_DIR = " "${STB_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(STB DEFAULT_MSG STB_INCLUDE_DIR)

mark_as_advanced(STB_INCLUDE_DIR)

message("STB_FOUND = " "${STB_FOUND}")
