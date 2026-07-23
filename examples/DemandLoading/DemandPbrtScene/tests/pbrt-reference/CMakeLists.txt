# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

set(referenceMaterials
    matte
    plastic
    uber
    mirror
    glass
    metal
    substrate
    translucent
    mix
)

add_custom_target(DemandPbrtScenePbrtReferenceResources)
set_property(TARGET DemandPbrtScenePbrtReferenceResources PROPERTY FOLDER Examples/DemandLoading/Tests)
source_group("Scene Files" REGULAR_EXPRESSION ".*\\.pbrt")
source_group("PBRT Reference Images" REGULAR_EXPRESSION "gold-.*\\.png")

foreach(material ${referenceMaterials})
    set_property(TARGET DemandPbrtScenePbrtReferenceResources APPEND PROPERTY SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/${material}.pbrt"
        "${CMAKE_CURRENT_SOURCE_DIR}/gold-${material}.png"
    )
endforeach()
