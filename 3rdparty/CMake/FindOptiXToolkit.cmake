# Looks for the environment variable:
# OPTIX_TOOLKIT_PATH

# Sets the variables:
# OTK_INCLUDE_DIR

# DAR Currently only need the OMM baking headers and library.
# OTK_CUOMMBAKING_LIBRARY

# OptiXToolkit_FOUND

set(OPTIX_TOOLKIT_PATH $ENV{OPTIX_TOOLKIT_PATH})

if ("${OPTIX_TOOLKIT_PATH}" STREQUAL "")
  set(OPTIX_TOOLKIT_PATH "${LOCAL_3RDPARTY}/optix-toolkit")
endif()

set(OTK_INCLUDE_DIR "${OPTIX_TOOLKIT_PATH}/include")

message("OTK_INCLUDE_DIR = " "${OTK_INCLUDE_DIR}")

find_library(OTK_CUOMMBAKING_LIBRARY NAMES CuOmmBaking PATHS ${OPTIX_TOOLKIT_PATH}/lib/OptiXToolkit)

message("OTK_CUOMMBAKING_LIBRARY = " "${OTK_CUOMMBAKING_LIBRARY}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiXToolkit DEFAULT_MSG OTK_INCLUDE_DIR OTK_CUOMMBAKING_LIBRARY)

mark_as_advanced(OTK_INCLUDE_DIR OTK_CUOMMBAKING_LIBRARY)

message("OptiXToolkit_FOUND = " "${OptiXToolkit_FOUND}")
