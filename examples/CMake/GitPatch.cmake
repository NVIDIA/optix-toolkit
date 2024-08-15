# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

# Tries to robustly patch glog by first doing a git restore, then doing
# a git apply on the patch.

execute_process(COMMAND ${GITCOMMAND} restore .)
execute_process(COMMAND ${GITCOMMAND} apply ${PATCHFILE})
