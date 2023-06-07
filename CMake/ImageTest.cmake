#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# add_image_test <target> <name>
#
# Adds an image comparison test named <target>-<name>.  The test
# will have the labels <target> and 'image'.  <target> will be invoked
# with -f <output-image> ${ARGS} to save an output image which is then
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
# All path arguments should be given as absolute paths.
#
# Arguments:
#
# ARGS                  The arguments to <target>, not including -f ${OUTPUT_IMAGE}.  Optional.
# DIFF_THRESHOLD        The difference threshold above which pixels are considered different.  The default is 0.
# ALLOWED_PERCENTAGE    The percentage of pixels required for the images to be considered different.  The default is 0.
#
function(add_image_test target name)
    cmake_parse_arguments(IMGTEST "" "DIFF_THRESHOLD;ALLOWED_PERCENTAGE" "ARGS" ${ARGN})
    if(NOT IMGTEST_DIFF_THRESHOLD)
        set(IMGTEST_DIFF_THRESHOLD 0)
    endif()
    if(NOT IMGTEST_ALLOWED_PERCENTAGE)
        set(IMGTEST_ALLOWED_PERCENTAGE 0)
    endif()

    set(test_name ${target}.${name})
    set(cmd_file "${CMAKE_CURRENT_BINARY_DIR}/test-${name}-cmd.txt")
    set(stdout_file "${CMAKE_CURRENT_BINARY_DIR}/test-${name}-stdout.txt")
    set(stderr_file "${CMAKE_CURRENT_BINARY_DIR}/test-${name}-stderr.txt")
    set(gold_image "${CMAKE_CURRENT_SOURCE_DIR}/gold-${name}.png")
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
        LABELS "${target}"
        ATTACHED_FILES_ON_FAIL "${cmd_file};${stdout_file};${stderr_file};${gold_image};${output_image};${diff_image}"
    )
endfunction()
