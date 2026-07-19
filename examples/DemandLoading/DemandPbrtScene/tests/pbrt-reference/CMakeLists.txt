# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(ImageTest)

set(referenceMaterials
    matte
    matte-imagemap
    matte-checkerboard
    plastic
    uber
    mirror
    glass
    metal
    substrate
    translucent
    mix
)

set(pbrtReferenceComparableMaterials
    matte
    mirror
    glass
)

add_custom_target(DemandPbrtScenePbrtReferenceResources)
set_property(TARGET DemandPbrtScenePbrtReferenceResources PROPERTY FOLDER Examples/DemandLoading/Tests)
source_group("Scene Files" REGULAR_EXPRESSION ".*\\.pbrt")
source_group("PBRT Reference Images" REGULAR_EXPRESSION "gold-.*\\.png")
source_group("Texture Images" REGULAR_EXPRESSION "matte-imagemap\\.png")

foreach(material ${referenceMaterials})
    set_property(TARGET DemandPbrtScenePbrtReferenceResources APPEND PROPERTY SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/${material}.pbrt"
        "${CMAKE_CURRENT_SOURCE_DIR}/gold-${material}.png"
    )
endforeach()
set_property(TARGET DemandPbrtScenePbrtReferenceResources APPEND PROPERTY SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/matte-imagemap.png"
)

define_property(TEST PROPERTY PBRT_REFERENCE_MATERIAL
    BRIEF_DOCS "PBRT material family covered by this reference comparison"
    FULL_DOCS "PBRT material family rendered by DemandPbrtScene and compared with a PBRT-rendered baseline.")
define_property(TEST PROPERTY PBRT_REFERENCE_TOLERANCE_REASON
    BRIEF_DOCS "Reason for this PBRT reference comparison tolerance"
    FULL_DOCS "Short explanation of the chosen image-diff threshold and allowed percentage.")

image_test_get_resource_dir(resourceDir DemandPbrtScene)
if(OTK_USE_MDL)
    function(add_pbrt_reference_image_test name)
        set(oneValueArgs MATERIAL SCENE WARMUP DIFF_THRESHOLD ALLOWED_PERCENTAGE TOLERANCE_REASON)
        cmake_parse_arguments(PBRT_REF "" "${oneValueArgs}" "OPTIONS;RESOURCES" ${ARGN})
        if(PBRT_REF_UNPARSED_ARGUMENTS)
            message(FATAL_ERROR "Unknown arguments to add_pbrt_reference_image_test: ${PBRT_REF_UNPARSED_ARGUMENTS}")
        endif()
        if(NOT PBRT_REF_MATERIAL)
            message(FATAL_ERROR "PBRT reference test ${name} is missing MATERIAL")
        endif()
        list(FIND referenceMaterials "${PBRT_REF_MATERIAL}" materialIndex)
        if(materialIndex EQUAL -1)
            message(FATAL_ERROR "PBRT reference test ${name} uses unknown material family ${PBRT_REF_MATERIAL}")
        endif()
        list(FIND pbrtReferenceComparableMaterials "${PBRT_REF_MATERIAL}" comparableMaterialIndex)
        if(comparableMaterialIndex EQUAL -1)
            message(FATAL_ERROR
                "PBRT reference test ${name} enables ${PBRT_REF_MATERIAL}, which is not marked comparable to PBRT yet")
        endif()
        if(NOT PBRT_REF_SCENE)
            message(FATAL_ERROR "PBRT reference test ${name} is missing SCENE")
        endif()
        if("${PBRT_REF_WARMUP}" STREQUAL "")
            message(FATAL_ERROR "PBRT reference test ${name} is missing WARMUP")
        endif()
        if(NOT PBRT_REF_TOLERANCE_REASON)
            message(FATAL_ERROR "PBRT reference test ${name} is missing TOLERANCE_REASON")
        endif()
        if(NOT PBRT_REF_DIFF_THRESHOLD)
            set(PBRT_REF_DIFF_THRESHOLD 1)
        endif()
        if(NOT PBRT_REF_ALLOWED_PERCENTAGE)
            set(PBRT_REF_ALLOWED_PERCENTAGE 3)
        endif()

        add_image_test(DemandPbrtScene ${name}
            FOLDER Examples/DemandLoading/Tests
            RESOURCE_TARGET DemandPbrtScenePbrtReferenceImageTestResources
            RESOURCES ${PBRT_REF_SCENE} ${PBRT_REF_RESOURCES}
            DIFF_THRESHOLD ${PBRT_REF_DIFF_THRESHOLD}
            ALLOWED_PERCENTAGE ${PBRT_REF_ALLOWED_PERCENTAGE}
            ARGS
                --bg=0/0/0
                --dim=32x32
                --warmup=${PBRT_REF_WARMUP}
                ${resourceDir}/${PBRT_REF_SCENE}
                --use-mdl-materials
                --mdl-synchronous-compilation
                ${PBRT_REF_OPTIONS})
        set(toleranceReason "${PBRT_REF_TOLERANCE_REASON}")
        string(APPEND toleranceReason " (diff threshold ${PBRT_REF_DIFF_THRESHOLD}")
        string(APPEND toleranceReason ", allowed percentage ${PBRT_REF_ALLOWED_PERCENTAGE})")
        set_tests_properties(DemandPbrtScene.${name} PROPERTIES
            PBRT_REFERENCE_MATERIAL "${PBRT_REF_MATERIAL}"
            PBRT_REFERENCE_TOLERANCE_REASON "${toleranceReason}"
        )
    endfunction()

    add_pbrt_reference_image_test(matte
        MATERIAL matte
        SCENE matte.pbrt
        WARMUP 2
        TOLERANCE_REASON "Default tolerance, front-lit diffuse trianglemesh fixture matches the generated MDL matte path")
    add_pbrt_reference_image_test(matte-checkerboard
        MATERIAL matte
        SCENE matte-checkerboard.pbrt
        WARMUP 4
        TOLERANCE_REASON "Default tolerance, CPU checkerboard demand texture feeds the generated MDL matte path")
    add_pbrt_reference_image_test(mirror
        MATERIAL mirror
        SCENE mirror.pbrt
        WARMUP 4
        OPTIONS --render-mode=path
        ALLOWED_PERCENTAGE 8
        TOLERANCE_REASON "Silhouette tolerance, generated MDL mirror uses specular BSDF sampling")
    add_pbrt_reference_image_test(glass
        MATERIAL glass
        SCENE glass.pbrt
        WARMUP 4
        OPTIONS --render-mode=path
        ALLOWED_PERCENTAGE 8
        TOLERANCE_REASON "Silhouette tolerance, generated MDL glass uses specular transmission sampling")
    if(OTK_USE_OIIO)
        add_pbrt_reference_image_test(matte-imagemap
            MATERIAL matte
            SCENE matte-imagemap.pbrt
            RESOURCES matte-imagemap.png
            WARMUP 4
            TOLERANCE_REASON "Default tolerance, direct imagemap Kd uses the existing sparse demand texture path")
    endif()
endif()
