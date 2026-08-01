# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(ImageTest)

set(referenceMaterials
    matte
    matte-imagemap
    matte-mix-imagemap
    matte-checkerboard
    plastic
    plastic-textured-specular
    uber
    uber-transmission
    uber-textured-transmission
    mirror
    glass
    metal
    substrate
    substrate-textured-roughness
    translucent
    mix
    mix-rgb-amount
    fourier
    landscape-uber-textured
    landscape-uber-alpha
    landscape-uber-specular-textures
    landscape-uber-bump
    landscape-mix-bump
    landscape-translucent-textured
    landscape-mix-vegetation
)

set(pbrtReferenceComparableMaterials
    matte
    plastic
    uber
    mirror
    glass
    metal
    substrate
    translucent
    mix
    fourier
)

add_custom_target(DemandPbrtScenePbrtReferenceResources)
set_property(TARGET DemandPbrtScenePbrtReferenceResources PROPERTY FOLDER Examples/DemandLoading/Tests)
source_group("Scene Files" REGULAR_EXPRESSION ".*\\.pbrt")
source_group("PBRT Reference Images" REGULAR_EXPRESSION "gold-.*\\.png")
source_group("Texture Images" REGULAR_EXPRESSION
    "(matte-imagemap\\.png|matte-infinite\\.exr|plastic-specular-imagemap\\.png|landscape-fixture-.*\\.png)")
source_group("BSDF Tables" REGULAR_EXPRESSION ".*\\.bsdf")

foreach(material ${referenceMaterials})
    set_property(TARGET DemandPbrtScenePbrtReferenceResources APPEND PROPERTY SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/${material}.pbrt"
        "${CMAKE_CURRENT_SOURCE_DIR}/gold-${material}.png"
    )
endforeach()
set_property(TARGET DemandPbrtScenePbrtReferenceResources APPEND PROPERTY SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/matte-imagemap.png"
    "${CMAKE_CURRENT_SOURCE_DIR}/matte-infinite.exr"
    "${CMAKE_CURRENT_SOURCE_DIR}/matte-infinite.pbrt"
    "${CMAKE_CURRENT_SOURCE_DIR}/gold-matte-infinite.png"
    "${CMAKE_CURRENT_SOURCE_DIR}/matte-distant-infinite.pbrt"
    "${CMAKE_CURRENT_SOURCE_DIR}/gold-matte-distant-infinite.png"
    "${CMAKE_CURRENT_SOURCE_DIR}/plastic-specular-imagemap.png"
    "${CMAKE_CURRENT_SOURCE_DIR}/bsdfs/roughgold_alpha_0.2.bsdf"
    "${CMAKE_CURRENT_SOURCE_DIR}/landscape-fixture-albedo.png"
    "${CMAKE_CURRENT_SOURCE_DIR}/landscape-fixture-alpha.png"
    "${CMAKE_CURRENT_SOURCE_DIR}/landscape-fixture-bump.png"
    "${CMAKE_CURRENT_SOURCE_DIR}/landscape-fixture-reflectance.png"
    "${CMAKE_CURRENT_SOURCE_DIR}/landscape-fixture-specular.png"
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
    add_pbrt_reference_image_test(plastic
        MATERIAL plastic
        SCENE plastic.pbrt
        WARMUP 4
        TOLERANCE_REASON "Default tolerance, generated MDL plastic uses the diffuse/glossy BSDF callable path")
    add_pbrt_reference_image_test(uber
        MATERIAL uber
        SCENE uber.pbrt
        WARMUP 4
        TOLERANCE_REASON "Default tolerance, generated MDL uber uses the diffuse/glossy BSDF callable path")
    add_pbrt_reference_image_test(uber-transmission
        MATERIAL uber
        SCENE uber-transmission.pbrt
        WARMUP 4
        OPTIONS --render-mode=path
        ALLOWED_PERCENTAGE 8
        TOLERANCE_REASON "Silhouette tolerance, generated MDL uber uses Kt specular transmission sampling")
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
    add_pbrt_reference_image_test(metal
        MATERIAL metal
        SCENE metal.pbrt
        WARMUP 4
        OPTIONS --render-mode=path
        DIFF_THRESHOLD 3
        ALLOWED_PERCENTAGE 8
        TOLERANCE_REASON "Silhouette and PBRT sampling tolerance, generated MDL metal uses GGX conductor sampling")
    add_pbrt_reference_image_test(substrate
        MATERIAL substrate
        SCENE substrate.pbrt
        WARMUP 4
        DIFF_THRESHOLD 9
        TOLERANCE_REASON "PBRT FresnelBlend diffuse tolerance, generated MDL substrate uses a layered diffuse/glossy BSDF")
    add_pbrt_reference_image_test(translucent
        MATERIAL translucent
        SCENE translucent.pbrt
        WARMUP 4
        TOLERANCE_REASON "Default tolerance, generated MDL translucent uses diffuse reflection/transmission BSDF callables")
    add_pbrt_reference_image_test(mix
        MATERIAL mix
        SCENE mix.pbrt
        WARMUP 4
        TOLERANCE_REASON "Default tolerance, generated MDL mix composes named-material BSDF callables")
    add_pbrt_reference_image_test(mix-rgb-amount
        MATERIAL mix
        SCENE mix-rgb-amount.pbrt
        WARMUP 4
        TOLERANCE_REASON "Default tolerance, generated MDL mix folds equal-RGB amount to the scalar material parameter")
    add_pbrt_reference_image_test(fourier
        MATERIAL fourier
        SCENE fourier.pbrt
        RESOURCES bsdfs/roughgold_alpha_0.2.bsdf
        WARMUP 4
        DIFF_THRESHOLD 3
        TOLERANCE_REASON "PBRT Fourier table tolerance, dedicated GPU evaluator samples the material-indexed roughgold table")
    if(OTK_USE_OIIO)
        add_pbrt_reference_image_test(matte-imagemap
            MATERIAL matte
            SCENE matte-imagemap.pbrt
            RESOURCES matte-imagemap.png
            WARMUP 4
            TOLERANCE_REASON "Default tolerance, direct imagemap Kd uses the existing sparse demand texture path")
        add_pbrt_reference_image_test(matte-mix-imagemap
            MATERIAL matte
            SCENE matte-mix-imagemap.pbrt
            RESOURCES matte-imagemap.png
            WARMUP 4
            TOLERANCE_REASON "Default tolerance, one-leaf mix imagemap Kd uses the affine sparse demand texture path")
        add_pbrt_reference_image_test(matte-infinite
            MATERIAL matte
            SCENE matte-infinite.pbrt
            RESOURCES matte-infinite.exr
            WARMUP 16
            DIFF_THRESHOLD 18
            TOLERANCE_REASON "Single-sample environment tolerance, generated MDL matte evaluates a mapped infinite light through its BSDF callable")
        add_pbrt_reference_image_test(matte-distant-infinite
            MATERIAL matte
            SCENE matte-distant-infinite.pbrt
            RESOURCES matte-infinite.exr
            WARMUP 16
            DIFF_THRESHOLD 18
            TOLERANCE_REASON "Single-sample environment tolerance, generated MDL matte adds distant and mapped infinite light contributions")
        add_pbrt_reference_image_test(plastic-textured-specular
            MATERIAL plastic
            SCENE plastic-textured-specular.pbrt
            RESOURCES plastic-specular-imagemap.png
            WARMUP 4
            DIFF_THRESHOLD 24
            ALLOWED_PERCENTAGE 8
            TOLERANCE_REASON "Glossy lobe tolerance, generated MDL plastic applies a runtime Ks demand texture to BSDF callable output")
        add_pbrt_reference_image_test(substrate-textured-roughness
            MATERIAL substrate
            SCENE substrate-textured-roughness.pbrt
            RESOURCES landscape-fixture-bump.png
            WARMUP 4
            DIFF_THRESHOLD 32
            ALLOWED_PERCENTAGE 8
            TOLERANCE_REASON "Roughness and FresnelBlend tolerance, generated MDL substrate patches a per-hit runtime roughness demand texture into the BSDF argument block")
        add_pbrt_reference_image_test(uber-textured-transmission
            MATERIAL uber
            SCENE uber-textured-transmission.pbrt
            RESOURCES matte-imagemap.png
            WARMUP 4
            OPTIONS --render-mode=path
            DIFF_THRESHOLD 80
            ALLOWED_PERCENTAGE 8
            TOLERANCE_REASON "Path-output encoding tolerance, generated MDL uber applies a runtime Kt demand texture to specular transmission sampling")
        add_pbrt_reference_image_test(landscape-uber-specular-textures
            MATERIAL uber
            SCENE landscape-uber-specular-textures.pbrt
            RESOURCES landscape-fixture-albedo.png landscape-fixture-specular.png landscape-fixture-reflectance.png
            WARMUP 16
            DIFF_THRESHOLD 64
            ALLOWED_PERCENTAGE 3
            TOLERANCE_REASON "Landscape fixture tolerance, direct uber Kd/Ks/Kr imagemaps use generated MDL BSDF callables with runtime texture-slot scaling and approximate MDL mix normalization")
        add_pbrt_reference_image_test(landscape-uber-bump
            MATERIAL uber
            SCENE landscape-uber-bump.pbrt
            RESOURCES landscape-fixture-albedo.png landscape-fixture-bump.png
            WARMUP 16
            DIFF_THRESHOLD 64
            ALLOWED_PERCENTAGE 3
            TOLERANCE_REASON "Landscape fixture tolerance, direct uber Kd/bump imagemaps use generated MDL BSDF callables with runtime single-sample tangent bump approximation")
        add_pbrt_reference_image_test(landscape-mix-bump
            MATERIAL mix
            SCENE landscape-mix-bump.pbrt
            RESOURCES landscape-fixture-albedo.png landscape-fixture-bump.png
            WARMUP 16
            DIFF_THRESHOLD 64
            ALLOWED_PERCENTAGE 3
            TOLERANCE_REASON "Landscape fixture tolerance, generated MDL mix composes named-material BSDF callables while runtime branch bumpmap slots use the same single-sample tangent bump approximation as direct materials")
        add_pbrt_reference_image_test(landscape-translucent-textured
            MATERIAL translucent
            SCENE landscape-translucent-textured.pbrt
            RESOURCES landscape-fixture-albedo.png
            WARMUP 16
            DIFF_THRESHOLD 64
            ALLOWED_PERCENTAGE 3
            TOLERANCE_REASON "Landscape fixture tolerance, direct translucent Kd imagemap uses generated MDL diffuse reflection/transmission BSDF callables with runtime texture-slot scaling")
    endif()
endif()
