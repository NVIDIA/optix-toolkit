# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(ImageTest)

# Render proxy geometry
add_image_test(DemandGeometryViewer frame0 FOLDER Examples/DemandLoading/Tests ARGS --bg=1,0,1 --warmup=0)

# Render realized geometry with proxy material
add_image_test(DemandGeometryViewer frame1 ARGS --bg=1,0,1 --warmup=1)

# Render realized geometry with realized material
add_image_test(DemandGeometryViewer frame2 ARGS --bg=1,0,1 --warmup=2)
