cmake_minimum_required(VERSION 3.24)

project(yolov8_obb_dfl_tensorcore LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 5060 is Blackwell Compute Capability 12.0. Override this when moving
# the project to another GPU, for example: -DCMAKE_CUDA_ARCHITECTURES=89.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(dfl_conv_benchmark_v0
  src/dfl_conv_kernels_v0.cu
)

add_executable(dfl_conv_benchmark_v1
  src/dfl_conv_kernels_v1.cu
)

target_include_directories(dfl_conv_benchmark_v0 PRIVATE include)
target_link_libraries(dfl_conv_benchmark_v0 PRIVATE CUDA::cudart)

# Keep source locations in Nsight Compute and make resource/spill information
# visible while learning. Remove the ptxas diagnostics only after recording it.
target_compile_options(dfl_conv_benchmark_v0 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-warn-spills>
)

target_include_directories(dfl_conv_benchmark_v1 PRIVATE include)
target_link_libraries(dfl_conv_benchmark_v1 PRIVATE CUDA::cudart)

# Keep source locations in Nsight Compute and make resource/spill information
# visible while learning. Remove the ptxas diagnostics only after recording it.
target_compile_options(dfl_conv_benchmark_v1 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-warn-spills>
)
