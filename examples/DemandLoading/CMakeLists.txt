# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

add_subdirectory(CdfInversion)
add_subdirectory(CompressedTextureCache)
add_subdirectory(DemandGeometryViewer)
add_subdirectory(DemandPbrtScene)
add_subdirectory(DemandTextureViewer)
if(TARGET OptiXToolkit::NeuralTextures)
    add_subdirectory(NeuralTextureViewer)
endif()
add_subdirectory(ImageSourceInfo)
add_subdirectory(RayCones)
add_subdirectory(Simple)
add_subdirectory(StochasticTextureFiltering)
add_subdirectory(Texture)
add_subdirectory(TexturePainting)
add_subdirectory(UdimTextureViewer)
