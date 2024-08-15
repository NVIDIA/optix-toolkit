# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

if(TARGET imgui)
  return()
endif()

if(OTK_USE_VCPKG)
    find_package(imgui CONFIG REQUIRED)
    return()
endif()

include(FetchContent)

message(VERBOSE "Finding imgui...")
FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.89.5
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(imgui)
# Let find_package know we have it
set(imgui_FOUND ON PARENT_SCOPE)

add_library( imgui STATIC
    ${imgui_SOURCE_DIR}/imconfig.h
    ${imgui_SOURCE_DIR}/imgui.h
    ${imgui_SOURCE_DIR}/imgui_internal.h
    ${imgui_SOURCE_DIR}/imstb_rectpack.h
    ${imgui_SOURCE_DIR}/imstb_textedit.h
    ${imgui_SOURCE_DIR}/imstb_truetype.h
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.h
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.h
)
target_include_directories( imgui PUBLIC ${imgui_SOURCE_DIR} )
target_include_directories( imgui INTERFACE ${imgui_SOURCE_DIR}/backends )
target_compile_definitions( imgui PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD )
target_link_libraries( imgui PUBLIC glfw glad::glad ${OPENGL_gl_LIBRARY} ${CMAKE_DL_LIBS})
set_target_properties( imgui PROPERTIES FOLDER ThirdParty )

add_library(imgui::imgui ALIAS imgui)
