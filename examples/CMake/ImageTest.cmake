# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

# image_test_get_resource_dir
#
# Get the image test resource directory location for a particular target.
#
function(image_test_get_resource_dir var target)
    set(${var} "${CMAKE_BINARY_DIR}/tests/resources/${target}" PARENT_SCOPE)
endfunction()

# add_image_test <target> <name>
#
# Adds an image comparison test named <target>-<name>.  The test
# will have the labels <target> and 'image'.  <target> will be invoked
# with --file <output-image> ${ARGS} to save an output image which is then
# compared against a 'gold' image representing the correct image output.
# The gold image is in the current list directory with the filename
# 'gold-<name>.png'.  Test output is placed in ${CMAKE_CURRENT_BINARY_DIR},
# in files prefixed with 'test-<name>'.
#
# The command-line, captured stdout output, captured stderr output, gold,
# output and difference images are uploaded to the CDash dashboard on failure.
#
# When the test fails locally, run ctest with --output-on-failure to see
# the command-line, stdout and stderr for the test case.
#
# All path arguments to the ARGS parameter should be given as absolute paths.
#
# Parameters:
#
# ARGS arg0...argN          The arguments to <target>, not including --file ${OUTPUT_IMAGE}.  Optional.
# DIFF_THRESHOLD arg        The difference threshold above which pixels are considered different.  The default is 1.
# ALLOWED_PERCENTAGE arg    The percentage of pixels required for the images to be considered different.  The default is 3.
# DISABLED                  Mark the test as DISABLED with CTest
# RESOURCES arg0...argN     List of additional resource files to be copied to target's resource directory.
# RESOURCE_TARGET           The name of the target that ensures resources are copied.  Defaults to ${target}ImageTestResources
# RESOURCE_DIR_NAME         The name of the subdirectory containing the resources.  Defaults to ${target}.
#
function(add_image_test target name)
    cmake_parse_arguments(IMGTEST "DISABLED" "DIFF_THRESHOLD;ALLOWED_PERCENTAGE;FOLDER;RESOURCE_TARGET;RESOURCE_DIR_NAME" "RESOURCES;ARGS" ${ARGN})
    if(IMGTEST_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown arguments to add_image_test: ${IMGTEST_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT IMGTEST_DIFF_THRESHOLD)
        set(IMGTEST_DIFF_THRESHOLD 1)
    endif()
    if(NOT IMGTEST_ALLOWED_PERCENTAGE)
        set(IMGTEST_ALLOWED_PERCENTAGE 3)
    endif()

    set(test_name ${target}.${name})
    set(cmd_file "${CMAKE_CURRENT_BINARY_DIR}/test-${name}-cmd.txt")
    set(stdout_file "${CMAKE_CURRENT_BINARY_DIR}/test-${name}-stdout.txt")
    set(stderr_file "${CMAKE_CURRENT_BINARY_DIR}/test-${name}-stderr.txt")
    set(gold_dir "${CMAKE_BINARY_DIR}/tests/gold/${target}")
    set(gold_name "gold-${name}.png")
    set(gold_image "${gold_dir}/${gold_name}")
    set(output_image "${CMAKE_CURRENT_BINARY_DIR}/test-${name}-out.png")
    set(diff_image "${CMAKE_CURRENT_BINARY_DIR}/test-${name}-diff.png")
    add_test(NAME ${test_name}
        COMMAND ${CMAKE_COMMAND}
            "-DPROGRAM=$<TARGET_FILE:${target}>"
            "-DARGS=${IMGTEST_ARGS}"
            "-DIMGTOOL=$<TARGET_FILE:imgtool>"
            "-DCMD_FILE=${cmd_file}"
            "-DSTDOUT_FILE=${stdout_file}"
            "-DSTDERR_FILE=${stderr_file}"
            "-DGOLD_IMAGE=${gold_image}"
            "-DOUTPUT_IMAGE=${output_image}"
            "-DDIFF_IMAGE=${diff_image}"
            -DDIFF_THRESHOLD=${IMGTEST_DIFF_THRESHOLD}
            -DALLOWED_PERCENTAGE=${IMGTEST_ALLOWED_PERCENTAGE}
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/RunImageTest.cmake")
    set_tests_properties(${test_name} PROPERTIES
        LABELS "${target};image"
        ATTACHED_FILES_ON_FAIL "${cmd_file};${stdout_file};${stderr_file};${gold_image};${output_image};${diff_image}"
    )
    if(IMGTEST_DISABLED)
        set_tests_properties(${test_name} PROPERTIES DISABLED ON)
    endif()

    # Set up an image test resources target to ensure that gold images (and other
    # test resources) are always copied to the build directory when changed.
    if(NOT IMGTEST_RESOURCE_TARGET)
        set(IMGTEST_RESOURCE_TARGET "${target}ImageTestResources")
    endif()
    if(NOT TARGET ${IMGTEST_RESOURCE_TARGET})
        add_custom_target(${IMGTEST_RESOURCE_TARGET})
        if(IMGTEST_FOLDER)
            set_property(TARGET ${IMGTEST_RESOURCE_TARGET} PROPERTY FOLDER ${IMGTEST_FOLDER})
        endif()
        set_property(TARGET ${IMGTEST_RESOURCE_TARGET} PROPERTY EXCLUDE_FROM_ALL FALSE)
        source_group("Gold Images" REGULAR_EXPRESSION "gold-.*\\.png")
    elseif(IMGTEST_FOLDER)
        get_property(existing_folder TARGET ${IMGTEST_RESOURCE_TARGET} PROPERTY FOLDER)
        if(NOT "${existing_folder}" STREQUAL "${IMGTEST_FOLDER}")
            message(FATAL "Conflicting FOLDER set for different image tests on target ${target}: '${existing_folder}' and '${IMGTEST_FOLDER}'")
        endif()
    endif()
    set_property(TARGET ${IMGTEST_RESOURCE_TARGET} APPEND PROPERTY SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${gold_name}")
    add_custom_command(
        OUTPUT "${gold_dir}/${gold_name}"
        MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/${gold_name}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${gold_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/${gold_name}" "${gold_dir}"
    )
    if(NOT IMGTEST_RESOURCE_DIR_NAME)
        set(IMGTEST_RESOURCE_DIR_NAME ${target})
    endif()
    image_test_get_resource_dir(resource_dir ${IMGTEST_RESOURCE_DIR_NAME})
    foreach(resource ${IMGTEST_RESOURCES})
        set_property(TARGET ${IMGTEST_RESOURCE_TARGET} APPEND PROPERTY SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${resource}")
        get_filename_component(dir ${resource} DIRECTORY)
        add_custom_command(
            OUTPUT "${resource_dir}/${resource}"
            MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/${resource}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${resource_dir}/${dir}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/${resource}" "${resource_dir}/${dir}"
        )
    endforeach()
endfunction()
