cmake_minimum_required(VERSION 3.0.0)
project(single-file-vulkan-pathtracing VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

file(GLOB SHADERS shaders/*)
add_executable(single-file-vulkan-pathtracing main.cpp ${SHADERS})

source_group("Shader Files" FILES ${SHADERS})

target_link_libraries(${PROJECT_NAME} PUBLIC glfw)
target_include_directories(${PROJECT_NAME} PUBLIC 
    "$ENV{VULKAN_SDK}/Include"
    "${PROJECT_SOURCE_DIR}/external/tinyobjloader"
)
