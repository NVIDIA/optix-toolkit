cmake_minimum_required(VERSION 3.2)

project(tinyddsloader)

set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
include_directories(${GLFW_INCLUDE_DIR})
include_directories(3rdparty/gl3w/include)

add_executable(example example.cpp)

add_executable(example_gl tinyddsloader.h example_gl.cpp 3rdparty/gl3w/src/gl3w.c)
target_link_libraries(example_gl
    ${OPENGL_LIBRARY}
    ${GLFW_LIBRARIES}
)
