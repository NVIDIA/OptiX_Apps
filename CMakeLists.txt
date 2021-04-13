cmake_minimum_required(VERSION 2.8)

PROJECT( optix_apps )

set(LOCAL_3RDPARTY "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")
message("LOCAL_3RDPARTY = " "${LOCAL_3RDPARTY}")

set(CMAKE_MODULE_PATH "${LOCAL_3RDPARTY}/CMake")
message("CMAKE_MODULE_PATH = " "${CMAKE_MODULE_PATH}")

# Use the NVCUDA_COMPILE_PTX function to produce the desired custom rule and output filenames when compiling OptiX programs from *.cu to *.ptx.
include("nvcuda_compile_ptx")

# amd64 is the only supported platform here. # DAR FIXME Remove all other remaining cases.
set(LOCAL_ARCH "amd64")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(UNIX) 
  set(OS "linux")
  add_definitions("-DLINUX")
  add_definitions("-Wno-unused-local-typedefs -Wno-delete-non-virtual-dtor")
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
else(UNIX)
  if(APPLE)
  else(APPLE)
    if(WIN32)
      set(OS "win")
      add_definitions("-DNOMINMAX")
    endif(WIN32)
  endif(APPLE)
endif(UNIX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

if (WIN32 AND "${CMAKE_GENERATOR}" MATCHES "^(Visual Studio).*")
  # Set the base folder where the per-project ptx folders get created.
  set (PTX_TARGET_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(ConfigurationName)")
  # Enable multi-processor build on all Visual Studio versions.
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
  # DAR This should be independent of ${CMAKE_BUILD_TYPE} because that single-configuration generator will not create subfolders, will it?
  # Otherwise add something with if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set (PTX_TARGET_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()


# Some useful macros
MACRO(ADD_TARGET_PROPERTIES _target _name)
  SET(_properties)
  FOREACH(_prop ${ARGN})
    SET(_properties "${_properties} ${_prop}")
  ENDFOREACH(_prop)
  GET_TARGET_PROPERTY(_old_properties ${_target} ${_name})
  IF(NOT _old_properties)
    # In case it's NOTFOUND
    SET(_old_properties)
  ENDIF(NOT _old_properties)
  SET_TARGET_PROPERTIES(${_target} PROPERTIES ${_name} "${_old_properties} ${_properties}")
ENDMACRO(ADD_TARGET_PROPERTIES)

MACRO(TARGET_INCLUDE_SYMBOL target symbol)
  if (WIN32)
    if ( LOCAL_ARCH STREQUAL "amd64" )
      add_target_properties( ${target} LINK_FLAGS /include:${symbol} )
    endif()
  endif()
  if(UNIX)
    add_target_properties( ${target} LINK_FLAGS "-Wl,--undefined=${symbol}" )
  endif()
ENDMACRO()

# Search for all shipping OptiX 7 versions before adding subdirectories.
# The resp. FindOptiX*.cmake looks for OPTIX7_PATH, OPTIX71_PATH, OPTIX72_PATH, OPTIX73_PATH environment variables.
# Doing this here allows excluding examples from the build which require a specific OptiX 7 minor version.
# If multiple OptiX SDK versions are found, the newest is picked by the individual examples.
find_package(OptiX73)
find_package(OptiX72)
find_package(OptiX71)
find_package(OptiX7)

add_subdirectory( apps )
