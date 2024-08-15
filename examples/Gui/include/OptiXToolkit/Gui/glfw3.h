// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

// glfw3.h defines APIENTRY, causing an error when it is redefined by minwindef.h
#ifdef APIENTRY
#include <GLFW/glfw3.h>
#else
#include <GLFW/glfw3.h>
#undef APIENTRY
#endif
