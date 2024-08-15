# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(ImageTest)

add_image_test(demandLoadTexture basic FOLDER Examples/DemandLoading/Tests ALLOWED_PERCENTAGE 0.1)
