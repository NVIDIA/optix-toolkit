#
#  Copyright (c) 2023 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

include(Fetchrply)

find_package(OptiX 7.5)

add_library( PbrtSceneLoader STATIC
    include/OptiXToolkit/PbrtSceneLoader/GoogleLogger.h
    include/OptiXToolkit/PbrtSceneLoader/Logger.h
    include/OptiXToolkit/PbrtSceneLoader/MeshReader.h
    include/OptiXToolkit/PbrtSceneLoader/PbrtApiImpl.h
    include/OptiXToolkit/PbrtSceneLoader/PlyReader.h
    include/OptiXToolkit/PbrtSceneLoader/SceneDescription.h
    GoogleLogger.cpp
    PbrtApiImpl.cpp
    PlyReader.cpp
)
target_link_libraries( PbrtSceneLoader PUBLIC pbrtParser rply::rply )
target_link_libraries( PbrtSceneLoader PRIVATE OptiX::OptiX ShaderUtil Util )
target_include_directories( PbrtSceneLoader PUBLIC include )
set_property( TARGET PbrtSceneLoader PROPERTY FOLDER "Examples" )

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
