# Looks for the environment variable:
# ASSIMP_PATH

# Sets the variables :
# ASSIMP_INCLUDE_DIRS
# ASSIMP_LIBRARIES
# ASSIMP_FOUND

set(ASSIMP_PATH $ENV{ASSIMP_PATH})

# If there was no environment variable override for the ASSIMP_PATH
# try finding it inside the local 3rdparty path.
if ("${ASSIMP_PATH}" STREQUAL "")
  set(ASSIMP_PATH "${LOCAL_3RDPARTY}/assimp")
endif()

# message("ASSIMP_PATH = " "${ASSIMP_PATH}")

set(ASSIMP_INCLUDE_DIRS "${ASSIMP_PATH}/include")

# message("ASSIMP_INCLUDE_DIRS = " "${ASSIMP_INCLUDE_DIRS}")

# The MSVC toolset version defines the library filename under Windows.
if (WIN32)
  if("${MSVC_VERSION}" VERSION_EQUAL "1900") # MSVS 2015
    set(MSVC_TOOLSET vc140)
  elseif(("${MSVC_VERSION}" VERSION_GREATER_EQUAL "1910") AND ("${MSVC_VERSION}" VERSION_LESS_EQUAL "1919")) # MSVS 2017
    set(MSVC_TOOLSET vc141)
  elseif(("${MSVC_VERSION}" VERSION_GREATER_EQUAL "1920") AND ("${MSVC_VERSION}" VERSION_LESS_EQUAL "1929")) # MSVS 2019
    set(MSVC_TOOLSET vc142)
  elseif(("${MSVC_VERSION}" VERSION_GREATER_EQUAL "1930") AND ("${MSVC_VERSION}" VERSION_LESS_EQUAL "1939")) # MSVS 2022
    set(MSVC_TOOLSET vc143)
  endif()
  set(ASSIMP_LIBRARIES ${ASSIMP_PATH}/lib/assimp-${MSVC_TOOLSET}-mt.lib)
else()
  find_library( ASSIMP_LIBRARIES
    NAMES assimp
  )
endif()

# message("ASSIMP_LIBRARIES = " "${ASSIMP_LIBRARIES}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ASSIMP DEFAULT_MSG ASSIMP_LIBRARIES ASSIMP_INCLUDE_DIRS)

mark_as_advanced(ASSIMP_INCLUDE_DIRS ASSIMP_LIBRARIES)

# message("ASSIMP_FOUND = " "${ASSIMP_FOUND}")
