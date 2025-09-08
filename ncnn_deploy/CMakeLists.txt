cmake_minimum_required(VERSION 3.10)
project(unet_ncnn)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

# Set NCNN path to where ncnnConfig.cmake is located
set(ncnn_DIR "/home/nick/Documents/code/lane-detection-unet-ncnn/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

# Add NCNN include directory
include_directories("/home/nick/Documents/code/lane-detection-unet-ncnn/ncnn/build/install/include")

add_executable(unet_ncnn main.cpp)
target_link_libraries(unet_ncnn ncnn ${OpenCV_LIBS})