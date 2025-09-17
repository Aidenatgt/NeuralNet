use std::{env, path::PathBuf};

fn main() {
    // ---- Config knobs (via env) --------------------------------------------
    // GPU arch (SM) to compile for: e.g. 86=Ampere, 89=Ada, 75=Turing
    let sm = env::var("CUDA_ARCH")
        .ok()
        .or_else(|| env::var("CMAKE_CUDA_ARCHITECTURES").ok())
        .unwrap_or_else(|| "120".to_string());

    // If you need a specific host compiler for nvcc (common on Arch):
    //   export CUDA_HOST_COMPILER=/usr/bin/g++-12
    let host_cxx = env::var("CUDA_HOST_COMPILER").ok();

    // If you want to force a CUDA compiler:
    //   export CMAKE_CUDA_COMPILER=/usr/bin/nvcc
    // or clang++ if you’re using clang CUDA
    let cuda_compiler = env::var("CMAKE_CUDA_COMPILER").ok();

    // Link libstdc++ too? (only if your .cu uses C++ stdlib)
    let link_stdcxx = env::var("CUDA_LINK_STDCXX")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // ---- Drive CMake --------------------------------------------------------
    let mut cfg = cmake::Config::new("cuda-lib");
    cfg.define("CMAKE_CUDA_ARCHITECTURES", &sm)
        .define("CMAKE_BUILD_TYPE", "Debug") // or "Release"
        // <- this line is the switch: use clang++ as the CUDA compiler
        .define("CMAKE_VERBOSE_MAKEFILE", "ON")
        .build_target("cuda_lib"); // must match add_library(cuda_lib ...)

    if let Some(h) = host_cxx {
        // nvcc will use this as the host compiler
        cfg.define("CMAKE_CUDA_HOST_COMPILER", h);
    }
    if let Some(cc) = cuda_compiler {
        cfg.define("CMAKE_CUDA_COMPILER", cc);
    }

    let dst = cfg.build();

    // ---- Tell rustc where to find and what to link --------------------------
    // cmake crate places artifacts under .../out/{lib,build}
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/build", dst.display());

    // Link the static library built by CMake (target name = cuda_lib)
    println!("cargo:rustc-link-lib=dylib=cuda_lib");

    // CUDA runtime (required for cudaMalloc/cudaMemcpy/kernel launches)
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Rebuild when these change
    println!("cargo:rerun-if-changed=cuda-lib/CMakeLists.txt");
    println!("cargo:rerun-if-changed=cuda-lib/include/cuda_lib.h");
    println!("cargo:rerun-if-changed=cuda-lib/add.cu");
    println!("cargo:rerun-if-changed=cuda-lib/emul.cu");
    println!("cargo:rerun-if-changed=cuda-lib/mmul.cu");
    println!("cargo:rerun-if-changed=cuda-lib/sub.cu");
    println!("cargo:rerun-if-changed=cuda-lib/transpose.cu");
    println!("cargo:rerun-if-changed=cuda-lib/unary.cu");
}
