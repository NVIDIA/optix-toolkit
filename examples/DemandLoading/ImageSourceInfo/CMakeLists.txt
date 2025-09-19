# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(BuildConfig)

add_executable(ImageSourceInfo main.cpp)
target_link_libraries(ImageSourceInfo PUBLIC OptiXToolkit::ImageSource)
set_property(TARGET ImageSourceInfo PROPERTY FOLDER Examples/DemandLoading)
