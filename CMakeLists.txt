cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(tensorrt_test)

add_definitions("-Wall -g")
find_package(CUDA REQUIRED)
include_directories("/home/w/TensorRT-8.0.0.3/include" ${CUDA_INCLUDE_DIRS})
LINK_DIRECTORIES("/home/w/TensorRT-8.0.0.3/lib" ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
add_executable(main main.cpp)

message(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

target_link_libraries(main ${CUDA_LIBRARYS} nvinfer nvonnxparser cudart)
