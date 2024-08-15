// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/Gui/glad.h>  // Glad insists on being included first.

#include <OptiXToolkit/Gui/Window.h>

#include <cstdint>
#include <string>

namespace otk {

class GLDisplay
{
public:
    GLDisplay(BufferImageFormat format = otk::BufferImageFormat::UNSIGNED_BYTE4);
    ~GLDisplay();

    void display( GLint screen_res_x, GLint screen_res_y, GLint framebuf_res_x, GLint framebuf_res_y, GLuint pbo ) const;

private:
    GLuint   m_render_tex = 0u;
    GLuint   m_program = 0u;
    GLint    m_render_tex_uniform_loc = -1;
    GLuint   m_quad_vertex_buffer = 0;
    GLuint   m_vertex_array{};

    otk::BufferImageFormat m_image_format;

    static const std::string s_vert_source;
    static const std::string s_frag_source;
};

} // end namespace otk
