# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(BuildConfig)
include(FetchGlfw)
include(FetchGlad)
include(FetchImgui)

otk_add_library(Gui STATIC
  src/Camera.cpp
  src/GLCheck.cpp
  src/GLDisplay.cpp
  src/Gui.cpp
  src/Trackball.cpp
  src/TrackballCamera.cpp
  src/Window.cpp
  include/OptiXToolkit/Gui/BufferMapper.h
  include/OptiXToolkit/Gui/CUDAOutputBuffer.h
  include/OptiXToolkit/Gui/Camera.h
  include/OptiXToolkit/Gui/GLCheck.h
  include/OptiXToolkit/Gui/GLDisplay.h
  include/OptiXToolkit/Gui/Gui.h
  include/OptiXToolkit/Gui/Trackball.h
  include/OptiXToolkit/Gui/TrackballCamera.h
  include/OptiXToolkit/Gui/Window.h
  include/OptiXToolkit/Gui/glad.h
  include/OptiXToolkit/Gui/glfw3.h
  )

add_library(OptiXToolkit::Gui ALIAS Gui)

target_include_directories(Gui PUBLIC include)

target_link_libraries(Gui
  PUBLIC
    OptiXToolkit::ShaderUtil
    OptiXToolkit::Util
    imgui::imgui
    glfw
    glad::glad
    Stb::Image
)

set_target_properties(Gui PROPERTIES FOLDER Examples/Libraries)
