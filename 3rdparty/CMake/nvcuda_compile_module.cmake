# Generate a custom build rule to translate *.cu files to *.ptx or *.optixir files.
# NVCUDA_COMPILE_MODULE(
#   SOURCES file1.cu file2.cu
#   DEPENDENCIES header1.h header2.h
#   TARGET_PATH <path where output files should be stored>
#   EXTENSION ".ptx" | ".optixir"
#   GENERATED_FILES program_modules
#   NVCC_OPTIONS -arch=sm_50
# )

# Generates *.ptx or *.optixir files for the given source files.
# The program_modules argument will receive the list of generated files.
# DAR Using this because I do not want filenames like "cuda_compile_ptx_generated_raygeneration.cu.ptx" but just "raygeneration.ptx".

FUNCTION(NVCUDA_COMPILE_MODULE)
  if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "ERROR: Only 64-bit programs supported.")
  endif()

  set(options "")
  set(oneValueArgs TARGET_PATH GENERATED_FILES EXTENSION)
  set(multiValueArgs NVCC_OPTIONS SOURCES DEPENDENCIES)

  CMAKE_PARSE_ARGUMENTS(NVCUDA_COMPILE_MODULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  if (NOT WIN32) # Do not create a folder with the name ${ConfigurationName} under Windows.
    # Under Linux make sure the target directory exists. 
    FILE(MAKE_DIRECTORY ${NVCUDA_COMPILE_MODULE_TARGET_PATH})
  endif()
  
  # Custom build rule to generate either *.ptx or *.optixir files from *.cu files.
  FOREACH(input ${NVCUDA_COMPILE_MODULE_SOURCES})
    get_filename_component(input_we "${input}" NAME_WE)
    get_filename_component(ABS_PATH "${input}" ABSOLUTE)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" REL_PATH "${ABS_PATH}")

    # Generate the output *.ptx or *.optixir files directly into the executable's selected target directory.
    set(output "${NVCUDA_COMPILE_MODULE_TARGET_PATH}/${input_we}${NVCUDA_COMPILE_MODULE_EXTENSION}")
    # message("output = ${output}")

    LIST(APPEND OUTPUT_FILES "${output}")
    
    # This prints the standalone NVCC command line for each CUDA file.
    # CUDAToolkit_NVCC_EXECUTABLE has been set with FindCUDAToolkit.cmake in CMake 3.17 and newer.
    # message("${CUDAToolkit_NVCC_EXECUTABLE} " "${NVCUDA_COMPILE_MODULE_NVCC_OPTIONS} " "${input} " "-o " "${output}")

    add_custom_command(
      OUTPUT  "${output}"
      DEPENDS "${input}" ${NVCUDA_COMPILE_MODULE_DEPENDENCIES}
      COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} ${NVCUDA_COMPILE_MODULE_NVCC_OPTIONS} "${input}" -o "${output}"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  ENDFOREACH( )

  set(${NVCUDA_COMPILE_MODULE_GENERATED_FILES} ${OUTPUT_FILES} PARENT_SCOPE)
ENDFUNCTION()
