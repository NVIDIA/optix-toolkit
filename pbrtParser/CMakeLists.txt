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

include( CheckCXXSourceCompiles )
include( CheckIncludeFiles )
include( Fetchglog )
include(FetchContent)

# The pbrt repository does not have any tags, so to avoid the expense of cloning
# the entire repository we use a shallow clone of a fork with a recent tag.
string(TIMESTAMP t1 "%s")
message(VERBOSE "Finding pbrt...")
FetchContent_Declare(
  pbrt
  GIT_REPOSITORY https://github.com/MarkLeone/pbrt-v3.git # forked from mmp/pbrt-v3
  GIT_TAG v2023-09-03 # 13d871faae88233b327d04cda24022b8bb0093ee
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
FetchContent_Populate(pbrt)

string(TIMESTAMP t2 "%s")
math(EXPR elapsed "${t2} - ${t1}")
message(VERBOSE "FetchContent(pbrt) took ${elapsed} seconds.")

set(pbrt_api_dir ${pbrt_SOURCE_DIR}/src/core)

add_library( pbrtParser STATIC
  ${pbrt_api_dir}/api.h
  ${pbrt_api_dir}/error.h
  ${pbrt_api_dir}/fileutil.cpp
  ${pbrt_api_dir}/fileutil.h
  ${pbrt_api_dir}/floatfile.cpp
  ${pbrt_api_dir}/floatfile.h
  ${pbrt_api_dir}/geometry.h
  ${pbrt_api_dir}/interaction.h
  ${pbrt_api_dir}/material.h
  ${pbrt_api_dir}/medium.h
  ${pbrt_api_dir}/memory.cpp
  ${pbrt_api_dir}/memory.h
  ${pbrt_api_dir}/parallel.cpp
  ${pbrt_api_dir}/parallel.h
  ${pbrt_api_dir}/paramset.cpp
  ${pbrt_api_dir}/paramset.h
  ${pbrt_api_dir}/parser.cpp
  ${pbrt_api_dir}/parser.h
  ${pbrt_api_dir}/pbrt.h
  ${pbrt_api_dir}/quaternion.cpp
  ${pbrt_api_dir}/quaternion.h
  ${pbrt_api_dir}/spectrum.cpp
  ${pbrt_api_dir}/spectrum.h
  ${pbrt_api_dir}/stats.cpp
  ${pbrt_api_dir}/stats.h
  ${pbrt_api_dir}/stringprint.h
  ${pbrt_api_dir}/texture.h
  ${pbrt_api_dir}/transform.cpp
  ${pbrt_api_dir}/transform.h
  ${pbrt_SOURCE_DIR}/src/textures/constant.h
  # These source files provide the missing pbrt functions that bridge
  # the API to an interface.
  PbrtApi.cpp
  include/OptiXToolkit/PbrtApi/PbrtApi.h
  )
source_group("Source Files" REGULAR_EXPRESSION "[^/]*\\.cpp")
source_group("Header Files" REGULAR_EXPRESSION "[^/]*\\.h")
source_group("Header Files/core" REGULAR_EXPRESSION "core/[^/]*\\.h")
source_group("Header Files/textures" REGULAR_EXPRESSION "textures/[^/]*\\.h")
source_group("Interface/Source Files" PbrtApi.cpp)
source_group("Interface/Header Files" include/OptiXToolkit/PbrtApi/PbrtApi.h)

set_property(TARGET pbrtParser PROPERTY FOLDER "ThirdParty")

if( WIN32 )
  string( REPLACE "/WX" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" )
else()
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w" )
endif()

target_include_directories( pbrtParser PUBLIC ${pbrt_SOURCE_DIR}/src include)
target_include_directories( pbrtParser PRIVATE ${pbrt_api_dir} )
target_compile_definitions( pbrtParser PRIVATE $<$<CONFIG:Release>:NDEBUG> )
target_link_libraries( pbrtParser PUBLIC glog )

###########################################################################
# Annoying compiler-specific details

if(CMAKE_COMPILER_IS_GNUCXX)
  target_compile_options( pbrtParser PRIVATE "-std=gnu++11" )
  target_compile_options( pbrtParser PRIVATE "-Wno-conversion-null")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options( pbrtParser PRIVATE "-Wno-deprecated-register")
endif()

if(MSVC)
  target_compile_definitions( pbrtParser PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

check_include_files( alloca.h HAVE_ALLOCA_H )
if( HAVE_ALLOCA_H )
  target_compile_definitions( pbrtParser PRIVATE PBRT_HAVE_ALLOCA_H )
endif()

check_include_files( memory.h HAVE_MEMORY_H )
if( HAVE_MEMORY_H )
  target_compile_definitions( pbrtParser PRIVATE PBRT_HAVE_MEMORY_H )
endif()

########################################
# thread-local variables

check_cxx_source_compiles( "
#ifdef __CYGWIN__
// Hack to work around https://gcc.gnu.org/bugzilla/show_bug.cgi?id=64697
#error \"No thread_local on cygwin\"
#endif  // __CYGWIN__
thread_local int x; int main() { }
" HAVE_THREAD_LOCAL )

check_cxx_source_compiles( "
__declspec(thread) int x; int main() { }
" HAVE_DECLSPEC_THREAD )

check_cxx_source_compiles( "
__thread int x; int main() { }
" HAVE___THREAD )

if( HAVE_THREAD_LOCAL )
  target_compile_definitions( pbrtParser PRIVATE PBRT_THREAD_LOCAL=thread_local )
elseif( HAVE___THREAD )
  target_compile_definitions( pbrtParser PRIVATE PBRT_THREAD_LOCAL=__thread )
elseif( HAVE_DECLSPEC_THREAD )
  target_compile_definitions( pbrtParser PRIVATE "PBRT_THREAD_LOCAL=__declspec(thread)" )
else()
  message( SEND_ERROR "Unable to find a way to declare a thread-local variable" )
endif()

########################################
# Aligned memory allocation

check_cxx_source_compiles( "
#include <malloc.h>
int main() { void * ptr = _aligned_malloc(1024, 32); }
" HAVE__ALIGNED_MALLOC )

check_cxx_source_compiles( "
#include <stdlib.h>
int main() {
  void *ptr;
  posix_memalign(&ptr, 32, 1024);
} " HAVE_POSIX_MEMALIGN )

check_cxx_source_compiles( "
#include <malloc.h>
int main() {
    void *ptr = memalign(32, 1024);
} " HAVE_MEMALIGN )

if( HAVE__ALIGNED_MALLOC )
  target_compile_definitions( pbrtParser PRIVATE PBRT_HAVE__ALIGNED_MALLOC )
elseif( HAVE_POSIX_MEMALIGN )
  target_compile_definitions( pbrtParser PRIVATE PBRT_HAVE_POSIX_MEMALIGN )
elseif( HAVE_MEMALIGN )
  target_compile_definitions( pbrtParser PRIVATE PBRTHAVE_MEMALIGN )
else()
  message( SEND_ERROR "Unable to find a way to allocate aligned memory" )
endif()

###########################################################################
# Check for various C++11 features and set preprocessor variables or
# define workarounds.

check_cxx_source_compiles(
  "int main() { constexpr int x = 0; }"
  HAVE_CONSTEXPR )
if( HAVE_CONSTEXPR )
  target_compile_definitions( pbrtParser PUBLIC PBRT_CONSTEXPR=constexpr )
else()
  target_compile_definitions( pbrtParser PUBLIC PBRT_CONSTEXPR=const )
endif()
