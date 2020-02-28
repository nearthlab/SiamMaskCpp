cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SiamMaskCpp)

if (NOT DEFINED TORCH_PATH)
  execute_process(COMMAND python3 -c "import torch; print(torch.__path__[0], end='')" OUTPUT_VARIABLE TORCH_PATH RESULT_VARIABLE TORCH_FOUND)
  if (TORCH_FOUND EQUAL 0)
    message(STATUS "Configured TORCH_PATH: ${TORCH_PATH}")
  else()
    message(FATAL_ERROR "Failed to configure torch path automatically. Please specify option -DTORCH_PATH='/path/to/lib/python-3.x/site-packages/torch'")
  endif()
endif()

find_package(Torch 1.3 REQUIRED PATHS ${TORCH_PATH})
find_package(OpenCV REQUIRED
  COMPONENTS
  opencv_core
  opencv_imgproc
  opencv_highgui
)

include_directories(
    "."
    "./third_party/argparse"
    "./third_party/json/include"
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

add_executable(demo demo.cpp)
target_link_libraries(demo ${OpenCV_LIBRARIES} ${TORCH_LIBRARIES})
set_property(TARGET demo PROPERTY CXX_STANDARD 11)
