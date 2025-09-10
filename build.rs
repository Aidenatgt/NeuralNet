use std::env;

fn main() {
    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/opt/cuda".into());
    let sm = env::var("CUDA_SM").unwrap_or_else(|_| "89".into()); // e.g. "120"

    let mut cfg = cmake::Config::new("cuda-lib");
    // Tell CMake to use clang++ as CUDA compiler:
    cfg.define("CMAKE_CUDA_COMPILER", "/usr/bin/clang++");
    cfg.define("CUDA_TOOLKIT_ROOT_DIR", &cuda_home);
    cfg.define("CUDA_ARCH_LIST", &sm);

    // Build the subproject; returns an install-ish dir with lib/ and include/
    let dst = cfg.build();

    // Link the produced static lib
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=unary");

    // And CUDA runtime + libstdc++
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");

    // Rebuild triggers
    println!("cargo:rerun-if-changed=cuda-lib/CMakeLists.txt");
    println!("cargo:rerun-if-changed=cuda-lib/unary_wrapper.cu");
    println!("cargo:rerun-if-changed=cuda-lib/unary.cuh");
    println!("cargo:rerun-if-changed=cuda-lib/include/unary.h");
}
