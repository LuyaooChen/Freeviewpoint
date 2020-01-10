cmake_minimum_required(VERSION 2.8)

#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(OpenCV_DIR /usr/local/opencv4/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(  CUDA_NVCC_FLAGS
     ${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61)

cuda_include_directories(${OpenCV_INCLUDE_DIRS})
cuda_include_directories(${CUDA_INCLUDE_DIRS})
cuda_include_directories("kernel.h")

file(GLOB_RECURSE CURRENT_HEADERS *.h)
file(GLOB CURRENT_SOURCES *.cu)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

cuda_add_library(   cuda_lib STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES} )
#                    "kernel.cu"
#                    "kernel.h")

target_link_libraries(  cuda_lib
                        ${CUDA_LIBRARIES}
                        ${OpenCV_LIBS}
                       )
