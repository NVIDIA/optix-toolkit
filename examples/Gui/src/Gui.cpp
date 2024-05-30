//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <OptiXToolkit/Gui/Gui.h>
#include <OptiXToolkit/Gui/Window.h>
#include <OptiXToolkit/Gui/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <chrono>
#include <cstdio>

namespace otk {

void initImGui( GLFWwindow* window )
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    ImGui_ImplGlfw_InitForOpenGL( window, false );
    ImGui_ImplOpenGL3_Init();
    ImGui::StyleColorsDark();
    io.Fonts->AddFontDefault();

    ImGui::GetStyle().WindowBorderSize = 0.0f;
}

GLFWwindow* initUI( const char* window_title, int width, int height )
{
    GLFWwindow* window = initGLFW( window_title, width, height );
    initGL();
    initImGui( window );
    return window;
}


void cleanupUI( GLFWwindow* window )
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow( window );
    glfwTerminate();
}

void beginFrameImGui()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}


void endFrameImGui()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );
}


void displayText( const char* text, float x, float y )
{
    ImGui::SetNextWindowBgAlpha( 0.0f );
    ImGui::SetNextWindowPos( ImVec2( x, y ) );
    ImGui::Begin( "TextOverlayFG", nullptr,
                  ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
                      | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs );
    ImGui::TextColored( ImColor( 0.7f, 0.7f, 0.7f, 1.0f ), "%s", text );
    ImGui::End();
}

void displayFPS( unsigned int frame_count )
{
    constexpr std::chrono::duration<double> display_update_min_interval_time( 0.5 );
    static double                           fps              = -1.0;
    static unsigned                         last_frame_count = 0;
    static auto                             last_update_time = std::chrono::steady_clock::now();
    auto                                    cur_time         = std::chrono::steady_clock::now();

    if( cur_time - last_update_time > display_update_min_interval_time )
    {
        fps = ( frame_count - last_frame_count ) / std::chrono::duration<double>( cur_time - last_update_time ).count();
        last_frame_count = frame_count;
        last_update_time = cur_time;
    }
    if( frame_count > 0 && fps >= 0.0 )
    {
        static char fps_text[32];
        sprintf( fps_text, "fps: %7.2f", fps );
        displayText( fps_text, 10.0f, 10.0f );
    }
}

} // namespace otk
