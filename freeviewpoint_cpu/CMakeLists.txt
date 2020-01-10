cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(freeviewpoint_cpu)

set(OpenCV_DIR /usr/local/opencv4/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
#find_package(PkgConfig REQUIRED)
#find_package(Boost COMPONENTS program_options REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(${Boost_INCLUDE_DIRS})

#set (EXTRA_LIBS ${EXTRA_LIBS} cuda_lib)

add_executable(${PROJECT_NAME}
                    "main.cpp"
                    "calib_deal.cpp"
                    "kernel.cpp"
                    )

target_link_libraries(freeviewpoint_cpu
                      ${OpenCV_LIBS}
#                      ${Boost_LIBRARIES}
                        )
