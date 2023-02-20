cmake_minimum_required(VERSION 3.20)

project(CNN)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

add_executable(${PROJECT_NAME} main.cpp)

add_definitions(-Wall)

add_subdirectory(vendor/stb_image)
add_subdirectory(random)

target_link_directories(${PROJECT_NAME} PUBLIC vendor/stb_image)
target_link_directories(${PROJECT_NAME} PUBLIC random)

add_library(conv_net image.cpp feature_map.cpp conv_kernel.cpp conv_layer.cpp pool_layer.cpp neuron_layer.cpp
	conv_net.cpp)

target_link_libraries(${PROJECT_NAME} conv_net)
target_link_libraries(${PROJECT_NAME} stb_image)
target_link_libraries(${PROJECT_NAME} random)
