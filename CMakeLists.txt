cmake_minimum_required(VERSION 3.18)
project(parking_space_detector_cuda LANGUAGES CUDA CXX)

# GPU compute capability — override at configure: cmake -B build -DCUDA_ARCH=61
#   61 = GT 1030 / GTX 10xx    75 = Turing    86 = RTX 30xx    89 = RTX 40xx
set(CUDA_ARCH "86" CACHE STRING "CUDA compute capability (e.g. 61, 86)")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)

function(apply_cuda_target_options target_name)
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        )
    endif()

    # Linux: link libstdc++ from CMAKE_CXX_COMPILER (fixes OpenCV/GCC ABI mismatch)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT MSVC)
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
            OUTPUT_VARIABLE _gcc_libstdcxx
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_gcc_libstdcxx AND EXISTS "${_gcc_libstdcxx}")
            get_filename_component(_gcc_libdir "${_gcc_libstdcxx}" DIRECTORY)
            target_link_directories(${target_name} PRIVATE ${_gcc_libdir})
        endif()
        target_link_libraries(${target_name} PRIVATE stdc++)
    endif()
endfunction()

# --- parking_detector ---
add_executable(parking_detector src/main.cu src/kernels.cu)

target_include_directories(parking_detector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(parking_detector PRIVATE ${OpenCV_LIBS})
apply_cuda_target_options(parking_detector)

# --- image_filter ---
add_executable(image_filter
    src/image_filter/main.cu
    src/image_filter/kernels.cu
)

target_include_directories(image_filter PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image_filter
)

target_link_libraries(image_filter PRIVATE ${OpenCV_LIBS})
apply_cuda_target_options(image_filter)
