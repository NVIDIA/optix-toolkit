// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

// glad.h defines APIENTRY, causing an error when it is redefined by minwindef.h
#ifndef __gl_h_
#ifdef APIENTRY
#include <glad/glad.h>
#else
#include <glad/glad.h>
#undef APIENTRY
#endif
#endif
