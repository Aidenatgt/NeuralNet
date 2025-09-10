cmake_minimum_required(VERSION 3.18)
project(unary LANGUAGES CXX CUDA)

# A library we can link from Rust
add_library(unary STATIC unary_wrapper.cu)
target_include_directories(unary PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# let clangd work properly
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set_target_properties(unary PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 17
  CUDA_STANDARD 17
)

# Allow passing SM list from build.rs; default e.g. to 89
if(NOT DEFINED CUDA_ARCH_LIST)
  set(CUDA_ARCH_LIST "120") # set "120" for compute 12.0, etc.
endif()
set_target_properties(unary PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

# If you want to force Clang as the CUDA compiler from build.rs:
#   -DCMAKE_CUDA_COMPILER=/usr/bin/clang++
# Clang needs the CUDA path for headers/libs:
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
  set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda")
endif()
target_include_directories(unary PUBLIC "${CUDA_TOOLKIT_ROOT_DIR}/include")

# Where to put installed artifacts under CMAKE_INSTALL_PREFIX
install(TARGETS unary
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

# (optional) install headers so Rust users can include them
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
