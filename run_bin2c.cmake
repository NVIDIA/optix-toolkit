## Adapted from OWL: https://github.com/owl-project/owl/blob/master/owl/cmake/run_bin2c.cmake 
## Copyright 2021 Jefferson Amstutz
## SPDX-License-Identifier: Apache-2.0

# NOTE: This script is only to be invoked by the embed_cuda() function.

set(file_contents "#include <stddef.h>\n\n")
unset(header_contents)

if(CONST)
  set(const_decl "const ")
else()
  set(const_decl "")
endif()

function(set_size_contents obj_name)
  string(CONCAT text
      "#ifdef __cplusplus\nextern \"C\" {\n"
      "#endif\n"
      "\n"
      "${const_decl}size_t ${obj_name}Size = sizeof(${obj_name})/sizeof(${obj_name}[0]);\n"
      "\n"
      "#ifdef __cplusplus\n"
      "}\n"
      "#endif\n")
  set(size_contents "${text}" PARENT_SCOPE)
endfunction()

function(set_header_decl obj_name)
  if(HEADER)
    string(CONCAT text
        "#ifdef __cplusplus\n"
        "extern \"C\" {\n"
        "#endif\n"
        "\n"
        "extern ${const_decl}unsigned char ${obj_name}[];\n"
        "extern ${const_decl}size_t ${obj_name}Size;\n"
        "\n"
        "#ifdef __cplusplus\n"
        "}\n"
        "\n"
        "inline ${const_decl}char* ${obj_name}Text() { return reinterpret_cast<${const_decl}char*>( ${obj_name} ); }\n"
        "\n"
        "#endif\n")
    set(header_decl "${text}" PARENT_SCOPE)
  endif()
endfunction()

foreach(obj ${OBJECTS})
  get_filename_component(obj_ext ${obj} EXT)
  get_filename_component(obj_dir ${obj} DIRECTORY)

  list(POP_FRONT SYMBOL_NAMES obj_name)
  if(NOT EXISTS ${obj})
    message(FATAL_ERORR "${obj} does not exist.")
    set(file_contents "${file_contents}\n#error ${obj} does not exist.\n")
  elseif(obj_ext MATCHES ".ptx" OR obj_ext MATCHES ".optixir")
    set(args --name ${obj_name} ${obj})
    if(obj_ext MATCHES ".ptx")
      list(APPEND(args --padd 0,0))
    endif()
    if(CONST)
        list(APPEND args --const)
    endif()
    execute_process(
      COMMAND "${BIN_TO_C_COMMAND}" ${args}
      WORKING_DIRECTORY ${obj_dir}
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error_var
    )
    if(result)
      string(REPLACE ";" " " printArgs "${args}")
      message(FATAL_ERROR "bin2c failed:\nCommand: ${BIN_TO_C_COMMAND} ${printArgs}\nResult: ${result}\nOutput: ${output}\nError: ${error_var}\n")
    endif()
    set_size_contents(${obj_name})
    set_header_decl(${obj_name})
    set(file_contents "${file_contents}\n${output}\n${size_contents}\n")
    if(HEADER)
      set(header_contents "${header_contents}\n${header_decl}\n")
    endif()
  endif()
endforeach()

file(WRITE "${OUTPUT}" "${file_contents}")

if(HEADER)
  get_filename_component(include_guard ${HEADER} NAME_WE)
  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" include_guard "${include_guard}")
  while(include_guard MATCHES ".*__.*")
    # Identifiers with double underscores are reserved for the implementation.
    string(REPLACE "__" "_" include_guard "${include_guard}")
  endwhile()
  string(TOUPPER "${include_guard}" include_guard)
  set(include_guard "CUDA_${include_guard}")
  string(CONCAT header_contents
    "#ifndef ${include_guard}\n"
    "#define ${include_guard}\n"
    "\n"
    "#include <stddef.h>\n"
    "\n"
    "${header_contents}\n"
    "#endif\n")
  file(WRITE "${HEADER}" "${header_contents}")
endif()
