# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(ImageTest)

add_image_test(neuralTextureViewer basic FOLDER Examples/DemandLoading/Tests ARGS --dim=512x512 --no-gl-interop ALLOWED_PERCENTAGE 1.0)
