cmake_minimum_required(VERSION 3.18)
project(parking_space_detector_cuda LANGUAGES CUDA CXX)

# ---------------------------------------------------------------------------
# CUDA architecture — RTX 3050 is Ampere sm_86
# Change this if you use a different GPU (e.g. 75 for RTX 20xx/GTX 16xx)
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(parking_detector
    src/main.cu
    src/kernels.cu
)

target_include_directories(parking_detector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(parking_detector PRIVATE
    ${OpenCV_LIBS}
)

# /Zc:preprocessor: CUDA 13.x CCCL + MSVC 18 require conforming preprocessor for some headers.
target_compile_options(parking_detector PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
)

set_target_properties(parking_detector PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ---------------------------------------------------------------------------
# Standalone image filter (grayscale / Sobel / effect demo)
# ---------------------------------------------------------------------------
add_executable(image_filter
    src/image_filter/main.cu
    src/image_filter/kernels.cu
)

target_include_directories(image_filter PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image_filter
)

target_link_libraries(image_filter PRIVATE
    ${OpenCV_LIBS}
)

target_compile_options(image_filter PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
)

set_target_properties(image_filter PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
