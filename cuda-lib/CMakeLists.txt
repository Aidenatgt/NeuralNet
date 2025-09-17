cmake_minimum_required(VERSION 3.24)
project(cuda_lib LANGUAGES CUDA CXX)

set(CUDA_SOURCES
  add.cu
  cuda_init.cu
  emul.cu
  mmul.cu
  sub.cu
  transpose.cu
  unary.cu
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(cuda_lib SHARED ${CUDA_SOURCES})

target_include_directories(cuda_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

set_target_properties(cuda_lib PROPERTIES
  CUDA_ARCHITECTURES 120
  POSITION_INDEPENDENT_CODE ON
  CUDA_STANDARD 17
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(cuda_lib
  PUBLIC CUDA::cudart CUDA::cuda_driver)

# Optionally install
# install(TARGETS cuda_lib LIBRARY DESTINATION lib)
# install(DIRECTORY include/ DESTINATION include)

if (NOT DEFINED CMAKE_CUDA_COMPILER_ID OR CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda" CACHE PATH "")

  target_compile_options(cuda_lib PRIVATE
    "--cuda-path=${CUDA_TOOLKIT_ROOT_DIR}"
  )
endif()

message(STATUS "Archive out dir: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
