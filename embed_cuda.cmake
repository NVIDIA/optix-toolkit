## Adapted from OWL: https://github.com/owl-project/owl/blob/master/owl/cmake/embed_ptx.cmake 
## Copyright 2021 Jefferson Amstutz
## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)

# NOTE(jda) - CMake 3.17 defines CMAKE_CURRENT_FUNCTION_LIST_DIR, but alas can't
#             use it yet.
set(EMBED_CUDA_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "")

# embed_cuda
#
# Compile CUDA sources to OptiXIR (default) or PTX and use bin2c from the CUDA SDK 
# to create data arrays containing the resulting OptiXIR or PTX output.
#
# Keyword arguments:
# CONST             Pass --const to bin2c to generate constant data arrays.
# RELOCATABLE       Pass -rdc=true to nvcc to generate relocatable PTX/OptiXIR.
# GEN_PTX           Generate and embed PTX instead of OptiXIR (default is OptiXIR).
# GEN_DEBUG         Generate OptiX debug symbols (OptiX IR required). 
#
# Single value arguments:
# OUTPUT_TARGET     Name of the target that contains the generated C file.
#                   Required.
# CUDA_TARGET       Name of the target that compiles CUDA to OptiXIR/PTX.
#                   Default: ${OUTPUT_TARGET}_optixir or ${OUTPUT_TARGET}_ptx
# FOLDER            IDE folder property for generated targets, if any.
# HEADER            Generate a header file with the given name to contain
#                   declarations for the generated data arrays.
#
# Multiple value arguments:
# CUDA_INCLUDE_DIRECTORIES   List of directories to search when compiling to OptiXIR/PTX
# CUDA_LINK_LIBRARIES        List of libraries to link against when compiling OptiXIR/PTX.
# SOURCES                   List of CUDA source files to compile to OptiXIR/PTX.
# EMBEDDED_SYMBOL_NAMES     List of names for embedded data arrays, one per source file.
#
function(embed_cuda)
  set(noArgs CONST RELOCATABLE GEN_PTX GEN_DEBUG)
  set(oneArgs OUTPUT_TARGET CUDA_TARGET FOLDER HEADER)
  set(multiArgs CUDA_INCLUDE_DIRECTORIES CUDA_LINK_LIBRARIES SOURCES EMBEDDED_SYMBOL_NAMES)
  cmake_parse_arguments(EMBED_CUDA "${noArgs}" "${oneArgs}" "${multiArgs}" ${ARGN})

  if(NOT EMBED_CUDA_OUTPUT_TARGET)
    message(FATAL_ERROR "Missing required OUTPUT_TARGET argument")
  endif()

  if(EMBED_CUDA_EMBEDDED_SYMBOL_NAMES)
    list(LENGTH EMBED_CUDA_EMBEDDED_SYMBOL_NAMES NUM_NAMES)
    list(LENGTH EMBED_CUDA_SOURCES NUM_SOURCES)
    if (NOT ${NUM_SOURCES} EQUAL ${NUM_NAMES})
      message(FATAL_ERROR
        "embed_CUDA(): the number of names passed as EMBEDDED_SYMBOL_NAMES must \
        match the number of files in SOURCES."
      )
    endif()
  else()
    unset(EMBED_CUDA_EMBEDDED_SYMBOL_NAMES)
    foreach(source ${EMBED_CUDA_SOURCES})
      get_filename_component(name ${source} NAME_WE)
      if(EMBED_CUDA_GEN_PTX)
        list(APPEND EMBED_CUDA_EMBEDDED_SYMBOL_NAMES ${name}_ptx)
      else()
        list(APPEND EMBED_CUDA_EMBEDDED_SYMBOL_NAMES ${name}_optixir)
      endif()
    endforeach()
  endif()

  ## Find bin2c and CMake script to feed it ##

  # We need to wrap bin2c with a script for multiple reasons:
  #   1. bin2c only converts a single file at a time
  #   2. bin2c has only standard out support, so we have to manually redirect to
  #      a cmake buffer
  #   3. We want to pack everything into a single output file, so we need to use
  #      the --name option

  get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  find_program(BIN_TO_C NAMES bin2c PATHS ${CUDA_COMPILER_BIN})
  if(NOT BIN_TO_C)
    message(FATAL_ERROR
      "bin2c not found:\n"
      "  CMAKE_CUDA_COMPILER='${CMAKE_CUDA_COMPILER}'\n"
      "  CUDA_COMPILER_BIN='${CUDA_COMPILER_BIN}'\n"
      )
  endif()

  set(EMBED_CUDA_RUN ${EMBED_CUDA_DIR}/run_bin2c.cmake)

  ## Create PTX/OPTIXIR object target ##

  if (NOT EMBED_CUDA_CUDA_TARGET)
    if(EMBED_CUDA_GEN_PTX)
      set(CUDA_TARGET ${EMBED_CUDA_OUTPUT_TARGET}_ptx)
    else()
      set(CUDA_TARGET ${EMBED_CUDA_OUTPUT_TARGET}_optixir)
    endif()
  else()
    set(CUDA_TARGET ${EMBED_CUDA_CUDA_TARGET})
  endif()

  add_library(${CUDA_TARGET} OBJECT)
  target_sources(${CUDA_TARGET} PRIVATE ${EMBED_CUDA_SOURCES})
  target_include_directories(${CUDA_TARGET} PRIVATE ${EMBED_CUDA_CUDA_INCLUDE_DIRECTORIES})
  target_link_libraries(${CUDA_TARGET} PRIVATE ${EMBED_CUDA_CUDA_LINK_LIBRARIES})

  if(EMBED_CUDA_GEN_PTX)
    message(STATUS "embed_cuda - generate PTX: " ${CUDA_TARGET})
    set_property(TARGET ${CUDA_TARGET} PROPERTY CUDA_PTX_COMPILATION ON)
  else()
    message(STATUS "embed_cuda - generate OptiX IR: " ${CUDA_TARGET})
    set_property(TARGET ${CUDA_TARGET} PROPERTY CUDA_OPTIX_COMPILATION ON)
  endif()
  
  set(DEBUG_FLAGS "-lineinfo")
  if(EMBED_CUDA_GEN_DEBUG)
    if(EMBED_CUDA_GEN_PTX)
      message(WARNING "Requested OptiX symbols but GEN_PTX is set, fall back to -lineinfo")
    else()
      message(STATUS "Generating debug symbols.")
      set(DEBUG_FLAGS "-G") # lineinfo is assumed with -G, not necessary to set
    endif()
  endif()
  target_compile_options(${CUDA_TARGET} PRIVATE ${DEBUG_FLAGS})

  if(EMBED_CUDA_RELOCATABLE)
    target_compile_options(${CUDA_TARGET} PRIVATE "-rdc=true")
  endif()
  if(EMBED_CUDA_FOLDER)
    set_property(TARGET ${CUDA_TARGET} PROPERTY FOLDER ${EMBED_CUDA_FOLDER})
  endif()

  ## Create command to run the bin2c via the CMake script ##

  set(EMBED_CUDA_C_FILE ${CMAKE_CURRENT_BINARY_DIR}/${EMBED_CUDA_OUTPUT_TARGET}/${EMBED_CUDA_OUTPUT_TARGET}.c)
  if(EMBED_CUDA_HEADER)
    set(EMBED_CUDA_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${EMBED_CUDA_OUTPUT_TARGET}/${EMBED_CUDA_HEADER})
  endif()
  get_filename_component(OUTPUT_FILE_NAME ${EMBED_CUDA_C_FILE} NAME)
  if(EMBED_CUDA_HEADER)
    add_custom_command(
      OUTPUT ${EMBED_CUDA_C_FILE} ${EMBED_CUDA_HEADER}
      COMMAND ${CMAKE_COMMAND}
        "-DBIN_TO_C_COMMAND=${BIN_TO_C}"
        "-DOBJECTS=$<TARGET_OBJECTS:${CUDA_TARGET}>"
        "-DSYMBOL_NAMES=${EMBED_CUDA_EMBEDDED_SYMBOL_NAMES}"
        "-DOUTPUT=${EMBED_CUDA_C_FILE}"
        "-DCONST=${EMBED_CUDA_CONST}"
        "-DHEADER=${EMBED_CUDA_HEADER}"
        -P ${EMBED_CUDA_RUN}
      VERBATIM
      DEPENDS $<TARGET_OBJECTS:${CUDA_TARGET}> ${CUDA_TARGET}
      COMMENT "Generating embedded PTX/OPTIXIR file: ${OUTPUT_FILE_NAME}"
    )
  else()
    add_custom_command(
      OUTPUT ${EMBED_CUDA_C_FILE}
      COMMAND ${CMAKE_COMMAND}
        "-DBIN_TO_C_COMMAND=${BIN_TO_C}"
        "-DOBJECTS=$<TARGET_OBJECTS:${CUDA_TARGET}>"
        "-DSYMBOL_NAMES=${EMBED_CUDA_EMBEDDED_SYMBOL_NAMES}"
        "-DOUTPUT=${EMBED_CUDA_C_FILE}"
        "-DCONST=${EMBED_CUDA_CONST}"
        "-DHEADER=${EMBED_CUDA_HEADER}"
        -P ${EMBED_CUDA_RUN}
      VERBATIM
      DEPENDS $<TARGET_OBJECTS:${CUDA_TARGET}> ${CUDA_TARGET}
      COMMENT "Generating embedded PTX/OPTIXIR file: ${OUTPUT_FILE_NAME}"
    )
  endif()

  add_library(${EMBED_CUDA_OUTPUT_TARGET} OBJECT)
  target_sources(${EMBED_CUDA_OUTPUT_TARGET} PRIVATE ${EMBED_CUDA_C_FILE})
  if(EMBED_CUDA_HEADER)
    target_sources(${EMBED_CUDA_OUTPUT_TARGET} PRIVATE ${EMBED_CUDA_HEADER})
    target_include_directories(${EMBED_CUDA_OUTPUT_TARGET} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/${EMBED_CUDA_OUTPUT_TARGET})
  endif()
  if(EMBED_CUDA_FOLDER)
    set_property(TARGET ${EMBED_CUDA_OUTPUT_TARGET} PROPERTY FOLDER ${EMBED_CUDA_FOLDER})
  endif()
endfunction()
