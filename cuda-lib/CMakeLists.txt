cmake_minimum_required(VERSION 3.24)
project(cuda_lib LANGUAGES CUDA CXX)

add_library(cuda_lib STATIC
  add.cu
  emul.cu
  mmul.cu
  sub.cu
  transpose.cu
  unary.cu
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

target_include_directories(cuda_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Choose your real architecture; 86 for Ampere (RTX 30xx), 89 for Ada (RTX 40xx)
# You can pass this from build.rs too.
set_target_properties(cuda_lib PROPERTIES
  CUDA_ARCHITECTURES 120
  POSITION_INDEPENDENT_CODE ON
  CUDA_STANDARD 17
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# If using clang as CUDA compiler, CMake will pass --cuda-gpu-arch automatically
# based on CUDA_ARCHITECTURES. We still need the CUDA toolkit path:
if (NOT DEFINED CMAKE_CUDA_COMPILER_ID OR CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  # Tell clang where the CUDA toolkit lives
  # Adjust path if your CUDA is elsewhere
  set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda" CACHE PATH "")

  # Add explicit flags for clang’s CUDA frontend
  target_compile_options(cuda_lib PRIVATE
    "--cuda-path=${CUDA_TOOLKIT_ROOT_DIR}"
    # If you do require cross-TU device code later, also add: "-fgpu-rdc"
  )
  # cudart link will be handled in Rust build.rs, but you can also do:
  # target_link_directories(cuda_lib PUBLIC "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
  # target_link_libraries(cuda_lib PUBLIC cudart)
endif()

# If you compile with default hidden visibility elsewhere, ensure we export these:
# target_compile_options(cuda_lib PRIVATE -fvisibility=default)
message(STATUS "Archive out dir: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
