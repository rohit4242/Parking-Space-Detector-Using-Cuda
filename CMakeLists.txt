cmake_minimum_required(VERSION 3.18)
project(parking_space_detector_cuda LANGUAGES CUDA CXX)

# =============================================================================
# USER SETTINGS — edit these for your GPU
# =============================================================================
#
# Set your GPU compute capability (sm_XX). Examples:
#   61  — GTX 10xx (e.g. GT 1030, GTX 1050)
#   75  — GTX 16xx / RTX 20xx (Turing)
#   86  — RTX 30xx (Ampere, e.g. RTX 3050)
#   89  — RTX 40xx (Ada Lovelace)
#
# Find yours: nvidia-smi --query-gpu=compute_cap --format=csv
# Or: https://developer.nvidia.com/cuda-gpus
#
# Override at configure time without editing this file:
#   cmake -B build -DCUDA_ARCH=61
#
set(CUDA_ARCH "86" CACHE STRING "CUDA compute capability without the 'sm_' prefix (e.g. 61, 75, 86)")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# OpenCV
# =============================================================================
# If find_package fails, re-run cmake with:
#   -DOpenCV_DIR=/path/to/folder/containing/OpenCVConfig.cmake
#
# Linux (Debian/Ubuntu) example:
#   -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
#
# Windows (manual build / vcpkg) example:
#   -DOpenCV_DIR=C:/path/to/opencv/build
#   -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
#
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# =============================================================================
# Optional: older host compiler on Linux (CUDA may not support GCC 14 yet)
# =============================================================================
# If nvcc errors with "unsupported GNU version", re-configure with e.g.:
#   cmake -B build -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
#
# Or uncomment and set a default host compiler for CUDA:
# set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-13)

# =============================================================================
# Shared CUDA compile options (platform-specific)
# =============================================================================
function(apply_cuda_target_options target_name)
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    # MSVC only: CUDA 13.x + MSVC need conforming preprocessor for some headers.
    # Linux GCC/Clang must NOT get this flag.
    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        )
    endif()

    # Fallback if only a very new GCC is installed and nvcc rejects it.
    # Prefer CMAKE_CUDA_HOST_COMPILER=g++-13 instead when possible.
    # if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT MSVC)
    #     target_compile_options(${target_name} PRIVATE
    #         $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
    #     )
    # endif()
endfunction()

# =============================================================================
# parking_detector — main video + parking spot demo
# =============================================================================
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

apply_cuda_target_options(parking_detector)

# =============================================================================
# image_filter — standalone grayscale / Sobel / effect demo (single image)
# =============================================================================
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

apply_cuda_target_options(image_filter)
