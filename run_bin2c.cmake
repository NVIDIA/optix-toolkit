## Adapted from OWL: https://github.com/owl-project/owl/blob/master/owl/cmake/run_bin2c.cmake 
## Copyright 2021 Jefferson Amstutz
## SPDX-License-Identifier: Apache-2.0

# NOTE: This script is only to be invoked by the EmbedPTX() function.

set(file_contents "#include <stddef.h>\n\n")
unset(header_contents)

foreach(obj ${OBJECTS})
  get_filename_component(obj_ext ${obj} EXT)
  get_filename_component(obj_dir ${obj} DIRECTORY)

  list(POP_FRONT SYMBOL_NAMES obj_name)
  if(obj_ext MATCHES ".ptx")
    set(args --name ${obj_name} ${obj} --padd 0,0)
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
    if(CONST)
        set(size_contents 
          "#ifdef __cpluplus\nextern \"C\" {\n#endif\n\nconst size_t ${obj_name}_size = sizeof(${obj_name})/sizeof(${obj_name}[0]);\n\n#ifdef __cplusplus\n}\n#endif\n")
        if(HEADER)
          set(header_decl
            "#ifdef __cpluplus\nextern \"C\" {\n#endif\n\nextern const unsigned char ${obj_name}[];\nextern const size_t ${obj_name}_size;\n\n#ifdef __cplusplus\n}\n#endif\n")
        endif()
    else()
        set(size_contents 
          "#ifdef __cpluplus\nextern \"C\" {\n#endif\n\nsize_t ${obj_name}_size = sizeof(${obj_name})/sizeof(${obj_name}[0]);\n\n#ifdef __cplusplus\n}\n#endif\n")
        if(HEADER)
          set(header_decl
            "#ifdef __cpluplus\nextern \"C\" {\n#endif\n\nextern unsigned char ${obj_name}[];\nextern size_t ${obj_name}_size;\n\n#ifdef __cplusplus\n}\n#endif\n")
        endif()
    endif()
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
  set(include_guard "PTX_${include_guard}")
  file(WRITE "${HEADER}" "#ifndef ${include_guard}\n#define ${include_guard}\n\n${header_contents}\n#endif\n")
endif()
