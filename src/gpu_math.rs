use std::{
    fmt::Display,
    os::raw::{c_int, c_ulong, c_void},
};

#[allow(non_camel_case_types)]
type cudaError_t = c_int;

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: c_ulong) -> cudaError_t;
    fn cudaFree(ptr: *mut c_void) -> cudaError_t;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: c_ulong, kind: c_int) -> cudaError_t;
}

// cudaMemcpyKind enum
pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

pub struct CudaMatrix<const R: usize, const C: usize> {
    pub ptr: *mut f32, // device memory
}

impl<const R: usize, const C: usize> CudaMatrix<R, C> {
    pub fn new() -> Self {
        let mut raw: *mut c_void = std::ptr::null_mut();
        let bytes = R * C * std::mem::size_of::<f32>();
        unsafe {
            let err = cudaMalloc(&mut raw as *mut _, bytes as _);
            if err != 0 {
                panic!("cudaMalloc failed with error code {}", err);
            }
        }
        Self {
            ptr: raw as *mut f32,
        }
    }

    pub fn from_host(data: &[f32]) -> Self {
        let mut m = Self::new();
        let bytes = Self::rows() * Self::cols() * std::mem::size_of::<f32>();
        unsafe {
            let err = cudaMemcpy(
                m.ptr as *mut _,
                data.as_ptr() as *const _,
                bytes as _,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );
            if err != 0 {
                panic!("cudaMemcpy H2D failed with error code {}", err);
            }
        }
        m
    }

    pub fn to_host_dense(&self) -> Vec<f32> {
        let mut data = vec![0.0f32; Self::rows() * Self::cols()];
        let bytes = Self::rows() * Self::cols() * std::mem::size_of::<f32>();
        unsafe {
            let err = cudaMemcpy(
                data.as_mut_ptr() as *mut _,
                self.ptr as *const _,
                bytes as _,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
            if err != 0 {
                panic!("cudaMemcpy D2H failed with error code {}", err);
            }
        }
        data
    }

    pub fn to_host_2d(&self) -> Vec<f32> {
        todo!("give it shape")
    }

    pub fn rows() -> usize {
        R
    }

    pub fn cols() -> usize {
        C
    }
}

impl<const R: usize, const C: usize> Drop for CudaMatrix<R, C> {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                let _ = cudaFree(self.ptr as *mut _);
                self.ptr = std::ptr::null_mut();
            }
        }
    }
}

impl<const R: usize, const C: usize> Display for CudaMatrix<R, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}
