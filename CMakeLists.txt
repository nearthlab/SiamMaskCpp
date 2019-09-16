cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SiamMaskCpp)

find_package(Torch REQUIRED)
find_package(OpenCV 3.4 REQUIRED)

message(STATUS "CMAKE_PREFIX_PATH: " ${CMAKE_PREFIX_PATH})
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

include_directories(
    "."
    "${CMAKE_PREFIX_PATH}/include"
)

add_executable(demo demo.cpp)
target_link_libraries(demo "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" dlib)
set_property(TARGET demo PROPERTY CXX_STANDARD 11)
