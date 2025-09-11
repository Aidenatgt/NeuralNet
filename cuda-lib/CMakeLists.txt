cmake_minimum_required(VERSION 3.18)
project(cuda_lib LANGUAGES CXX CUDA)

add_library(cuda_lib STATIC cuda_lib.cu)
#target_include_directories(cuda_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set_target_properties(cuda_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 17
  CUDA_STANDARD 17
)

if(NOT DEFINED CUDA_ARCH_LIST)
  set(CUDA_ARCH_LIST "120")
endif()
set_target_properties(cuda_lib PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
  set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda")
endif()
#target_include_directories(cuda_lib PUBLIC "${CUDA_TOOLKIT_ROOT_DIR}/include")

install(TARGETS cuda_lib
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
#         DESTINATION include
#         FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
