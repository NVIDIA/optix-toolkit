# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

# Script to run an image-based test by invoking an executable to get an
# output image and then running imgtool to compare the output image to
# a gold image.

# Variables that should be set when invoking this script:
#
# All file values are given as absolute paths.
#
# PROGRAM               The program to run to generate the output image.
# ARGS                  The arguments to ${PROGRAM}, not including -f ${OUTPUT_IMAGE}.
# IMGTOOL               The imgtool image comparison program.
# CMD_FILE              The file into which the command-lines are logged.
# STDOUT_FILE           The file into which standard output is logged.
# STDERR_FILE           The file into which standard error is logged.
# GOLD_IMAGE            The reference gold image.
# OUTPUT_IMAGE          The output image.
# DIFF_IMAGE            The diff image.
# DIFF_THRESHOLD        The difference threshold above which pixels are considered different.
# ALLOWED_PERCENTAGE    The percentage of pixels required for the images to be considered different.
#
function(runProgram exe)
    cmake_parse_arguments(RUNPROG "" "" "ARGS" ${ARGN})
    string(JOIN " " cmd_line ${exe} ${RUNPROG_ARGS})
    file(APPEND ${CMD_FILE} "${cmd_line}\n")
    execute_process(COMMAND ${exe} ${RUNPROG_ARGS}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE stderr ERROR_STRIP_TRAILING_WHITESPACE)
    file(APPEND ${STDOUT_FILE} "${stdout}\n")
    file(APPEND ${STDERR_FILE} "${stderr}\n")
    if(result)
        execute_process(COMMAND ${CMAKE_COMMAND} -E cat ${CMD_FILE} ${STDOUT_FILE} ${STDERR_FILE})
        string(REPLACE ";" " " printArgs "${RUNPROG_ARGS}")
        message(FATAL_ERROR "Error running ${exe} ${printArgs}, exit code ${result}")
    endif()
endfunction()

macro(logvar var)
    message(STATUS "${var}=${${var}}")
endmacro()

# Only used when debugging this script.
set(DEBUG OFF)
if(DEBUG)
    logvar(PROGRAM)
    logvar(ARGS)
    logvar(IMGTOOL)
    logvar(CMD_FILE)
    logvar(STDOUT_FILE)
    logvar(STDERR_FILE)
    logvar(GOLD_IMAGE)
    logvar(OUTPUT_IMAGE)
    logvar(DIFF_IMAGE)
    logvar(DIFF_THRESHOLD)
    logvar(ALLOWED_PERCENTAGE)
endif()

file(WRITE ${CMD_FILE} "\nCommand line:\n")
file(WRITE ${STDOUT_FILE} "\nStandard output:\n")
file(WRITE ${STDERR_FILE} "\nStandard error:\n")
runProgram(${PROGRAM} ARGS ${ARGS} --file ${OUTPUT_IMAGE})
runProgram(${IMGTOOL} ARGS pngcompare ${GOLD_IMAGE} ${OUTPUT_IMAGE} ${DIFF_IMAGE} ${DIFF_THRESHOLD} ${ALLOWED_PERCENTAGE})
