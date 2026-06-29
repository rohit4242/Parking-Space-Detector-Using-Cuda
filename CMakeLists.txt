cmake_minimum_required(VERSION 3.18)
project(parking_space_detector_cuda LANGUAGES CUDA CXX)

# GT 1030 (Pascal) — deggs206pc05 lab PC
set(CMAKE_CUDA_ARCHITECTURES 61)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# System OpenCV 4.10 (Debian)
set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4)
find_package(OpenCV REQUIRED)

# OpenCV is built with GCC 14; nvcc links with g++-13 — need GCC 14 libstdc++
execute_process(
    COMMAND /usr/bin/g++-14 -print-file-name=libstdc++.so
    OUTPUT_VARIABLE GCC_LIBSTDCXX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
get_filename_component(GCC_LIBDIR ${GCC_LIBSTDCXX} DIRECTORY)

# --- parking_detector ---
add_executable(parking_detector src/main.cu src/kernels.cu)

target_include_directories(parking_detector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_directories(parking_detector PRIVATE ${GCC_LIBDIR})
target_link_libraries(parking_detector PRIVATE ${OpenCV_LIBS} stdc++)
set_target_properties(parking_detector PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- image_filter ---
add_executable(image_filter
    src/image_filter/main.cu
    src/image_filter/kernels.cu
)

target_include_directories(image_filter PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image_filter
)

target_link_directories(image_filter PRIVATE ${GCC_LIBDIR})
target_link_libraries(image_filter PRIVATE ${OpenCV_LIBS} stdc++)
set_target_properties(image_filter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
