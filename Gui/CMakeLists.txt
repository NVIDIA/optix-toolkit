#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

include(BuildConfig)
include(FetchGlfw)
include(FetchGlad)
include(FetchImgui)

add_library(Gui STATIC
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
  )

add_library(OptiXToolkit::Gui ALIAS Gui)

target_include_directories(Gui PUBLIC include)

target_link_libraries(Gui
  PUBLIC
    OptiXToolkit::ShaderUtil
    OptiXToolkit::Util
    imgui
    glfw
    glad::glad
    Stb::Image
)

set_target_properties(Gui PROPERTIES FOLDER Examples)
