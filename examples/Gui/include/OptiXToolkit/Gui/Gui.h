// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

struct GLFWwindow;

namespace otk {

/// Initialize ImGui using window obtained from initGLFW (see Gui/Window.h)
void initImGui( GLFWwindow* window );

GLFWwindow* initUI( const char* window_title, int width, int height );

    void cleanupUI( GLFWwindow* window );

void beginFrameImGui();

void endFrameImGui();


void displayText( const char* text, float x, float y );

void displayFPS( unsigned int frame_count );

} // namespace otk
