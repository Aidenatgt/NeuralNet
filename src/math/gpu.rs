use std::{
    ffi::{c_int, c_ulong},
    ptr::{NonNull, null_mut},
    sync::mpsc,
    thread,
};

use lazy_static::lazy_static;

use crate::math::UnaryOp;

#[allow(non_camel_case_types)]
type cudaError_t = c_int;

#[link(name = "cudart")]
unsafe extern "C" {
    unsafe fn cudaMalloc(ptr: *mut *mut f32, size: c_ulong) -> cudaError_t;
    unsafe fn cudaFree(ptr: *mut f32) -> cudaError_t;
    unsafe fn cudaMemcpy(dst: *mut f32, src: *const f32, size: c_ulong, kind: c_int)
    -> cudaError_t;
}

#[link(name = "cuda_lib", kind = "static")]
unsafe extern "C" {
    unsafe fn t_f32(a_ptr: *mut f32, r: c_int, c: c_int) -> *mut f32;
    unsafe fn mmul_f32(a_ptr: *mut f32, b_ptr: *mut f32, r: c_int, c: c_int, n: c_int) -> *mut f32;
    unsafe fn emul_f32(a_ptr: *mut f32, b_ptr: *mut f32, d_ptr: *mut f32, n: c_int) -> *mut f32;
    unsafe fn add_f32(a_ptr: *const f32, b_ptr: *const f32, d_ptr: *mut f32, n: c_int) -> *mut f32;
    unsafe fn sub_f32(a_ptr: *mut f32, b_ptr: *mut f32, d_ptr: *mut f32, n: c_int) -> *mut f32;
    unsafe fn unary_op_f32(a_ptr: *mut f32, d_ptr: *mut f32, op: c_int, n: c_int) -> *mut f32;
    unsafe fn unary_op_grad_f32(a_ptr: *mut f32, d_ptr: *mut f32, op: c_int, n: c_int) -> *mut f32;
    unsafe fn cuda_init_runtime_on_primary();
    unsafe fn bind_primary_ctx(device_ordinal: c_int) -> c_int;
}

lazy_static! {
    pub static ref GPU_THREAD: GpuThread = GpuThread::start();
}

// cudaMemcpyKind enum
const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
struct DevPtr(NonNull<f32>);

// We promise to only touch the pointer on the GPU thread:
unsafe impl Send for DevPtr {}

impl DevPtr {
    pub fn from_raw(p: *mut f32) -> Option<Self> {
        NonNull::new(p).map(Self)
    }
    pub fn as_mut_ptr(self) -> *mut f32 {
        self.0.as_ptr()
    }
}

#[derive(Debug)]
enum GpuJob {
    Alloc {
        bytes: usize,
        resp: mpsc::Sender<DevPtr>,
    },
    Free {
        p: DevPtr,
    },
    DtoD {
        src: DevPtr,
        dst: DevPtr,
        n: usize,
        resp: mpsc::Sender<()>,
    },
    HtoD {
        dst: DevPtr,
        src: Vec<f32>,
        resp: mpsc::Sender<()>,
    },
    DtoH {
        src: DevPtr,
        len: usize,
        resp: mpsc::Sender<Vec<f32>>,
    },
    T {
        a: DevPtr,
        r: c_int,
        c: c_int,
        resp: mpsc::Sender<DevPtr>,
    },
    Emul {
        a: DevPtr,
        b: DevPtr,
        c: DevPtr,
        n: c_int,
        resp: mpsc::Sender<DevPtr>,
    },
    Mmul {
        a: DevPtr,
        b: DevPtr,
        m: i32,
        k: i32,
        n: i32,
        resp: mpsc::Sender<DevPtr>,
    },
    Add {
        a: DevPtr,
        b: DevPtr,
        c: DevPtr,
        n: i32,
        resp: mpsc::Sender<DevPtr>,
    },
    Sub {
        a: DevPtr,
        b: DevPtr,
        c: DevPtr,
        n: i32,
        resp: mpsc::Sender<DevPtr>,
    },
    Unary {
        a: DevPtr,
        b: DevPtr,
        n: i32,
        op: UnaryOp,
        resp: mpsc::Sender<DevPtr>,
    },
    UnaryGrad {
        a: DevPtr,
        b: DevPtr,
        n: i32,
        op: UnaryOp,
        resp: mpsc::Sender<DevPtr>,
    },
    Shutdown,
}

pub struct GpuThread {
    tx: mpsc::Sender<GpuJob>,
    handle: Option<thread::JoinHandle<()>>,
}

impl GpuThread {
    pub fn start() -> Self {
        let (tx, rx) = mpsc::channel::<GpuJob>();
        let handle = thread::spawn(move || {
            unsafe {
                bind_primary_ctx(0);
            } // runtime bound to this thread

            while let Ok(job) = rx.recv() {
                match job {
                    GpuJob::Mmul {
                        a,
                        b,
                        m,
                        k,
                        n,
                        resp,
                    } => {
                        let c = unsafe { mmul_f32(a.as_mut_ptr(), b.as_mut_ptr(), m, k, n) }; // your FFI
                        let _ = resp.send(DevPtr::from_raw(c).expect("null c"));
                    }
                    GpuJob::Add { a, b, c, n, resp } => {
                        let c =
                            unsafe { add_f32(a.as_mut_ptr(), b.as_mut_ptr(), c.as_mut_ptr(), n) };
                        let _ = resp.send(DevPtr::from_raw(c).expect("null c"));
                    }
                    GpuJob::Shutdown => break,
                    GpuJob::T { a, r, c, resp } => {
                        let c = unsafe { t_f32(a.as_mut_ptr(), r, c) };
                        let _ = resp.send(DevPtr::from_raw(c).expect("null c"));
                    }
                    GpuJob::Emul { a, b, c, n, resp } => {
                        let c =
                            unsafe { emul_f32(a.as_mut_ptr(), b.as_mut_ptr(), c.as_mut_ptr(), n) };
                        let _ = resp.send(DevPtr::from_raw(c).expect("null c"));
                    }
                    GpuJob::Sub { a, b, c, n, resp } => {
                        let c =
                            unsafe { sub_f32(a.as_mut_ptr(), b.as_mut_ptr(), c.as_mut_ptr(), n) };
                        let _ = resp.send(DevPtr::from_raw(c).expect("null c"));
                    }
                    GpuJob::Unary { a, b, n, op, resp } => {
                        let c =
                            unsafe { unary_op_f32(a.as_mut_ptr(), b.as_mut_ptr(), op as i32, n) };
                        let _ = resp.send(DevPtr::from_raw(c).expect("null c"));
                    }
                    GpuJob::UnaryGrad { a, b, n, op, resp } => {
                        let c = unsafe {
                            unary_op_grad_f32(a.as_mut_ptr(), b.as_mut_ptr(), op as i32, n)
                        };
                        let _ = resp.send(DevPtr::from_raw(c).expect("null c"));
                    }
                    GpuJob::Alloc { bytes, resp } => {
                        let mut p: *mut f32 = null_mut();
                        unsafe {
                            let err = cudaMalloc(&mut p as *mut _, bytes as c_ulong);
                            assert_eq!(err, 0, "cudaMalloc failed: code {}", err);
                        }
                        let _ = resp.send(DevPtr::from_raw(p).expect("null c"));
                    }
                    GpuJob::Free { p } => unsafe {
                        if !p.as_mut_ptr().is_null() {
                            let _ = cudaFree(p.as_mut_ptr());
                        }
                    },
                    GpuJob::HtoD { dst, src, resp } => {
                        let bytes = src.len() * std::mem::size_of::<f32>();
                        unsafe {
                            let err = cudaMemcpy(
                                dst.as_mut_ptr(),
                                (&src).as_ptr(),
                                bytes as _,
                                CUDA_MEMCPY_HOST_TO_DEVICE,
                            );
                            if err != 0 {
                                panic!("cudaMemcpy D2H failed with error code {}", err);
                            }
                        }
                        let _ = resp.send(());
                    }
                    GpuJob::DtoH { src, len, resp } => {
                        let mut data = vec![0.0f32; len];
                        let bytes = len * std::mem::size_of::<f32>();
                        unsafe {
                            let err = cudaMemcpy(
                                (&mut data).as_mut_ptr(),
                                src.as_mut_ptr(),
                                bytes as _,
                                CUDA_MEMCPY_DEVICE_TO_HOST,
                            );
                            if err != 0 {
                                panic!("cudaMemcpy D2H failed with error code {}", err);
                            }
                        }
                        let _ = resp.send(data);
                    }
                    GpuJob::DtoD { src, dst, n, resp } => unsafe {
                        cudaMemcpy(
                            dst.as_mut_ptr(),
                            src.as_mut_ptr(),
                            n as _,
                            CUDA_MEMCPY_DEVICE_TO_DEVICE,
                        );
                        let _ = resp.send(());
                    },
                }
            }
        });
        Self {
            tx,
            handle: Some(handle),
        }
    }

    pub fn mmul(&self, a: *mut f32, b: *mut f32, m: i32, k: i32, n: i32) -> *mut f32 {
        let (rtx, rrx) = mpsc::channel();
        self.tx
            .send(GpuJob::Mmul {
                a: DevPtr::from_raw(a).expect("null c"),
                b: DevPtr::from_raw(b).expect("null c"),
                m,
                k,
                n,
                resp: rtx,
            })
            .unwrap();
        rrx.recv().unwrap().as_mut_ptr()
    }

    pub fn add(&self, a: *mut f32, b: *mut f32, c: *mut f32, n: i32) -> *mut f32 {
        let (rtx, rrx) = mpsc::channel();
        self.tx
            .send(GpuJob::Add {
                a: DevPtr::from_raw(a).expect("null c"),
                b: DevPtr::from_raw(b).expect("null c"),
                c: DevPtr::from_raw(c).expect("null c"),
                n,
                resp: rtx,
            })
            .unwrap();
        rrx.recv().unwrap().as_mut_ptr()
    }

    pub fn sub(&self, a: *mut f32, b: *mut f32, c: *mut f32, n: i32) -> *mut f32 {
        let (rtx, rrx) = mpsc::channel();
        self.tx
            .send(GpuJob::Sub {
                a: DevPtr::from_raw(a).expect("null c"),
                b: DevPtr::from_raw(b).expect("null c"),
                c: DevPtr::from_raw(c).expect("null c"),
                n,
                resp: rtx,
            })
            .unwrap();
        rrx.recv().unwrap().as_mut_ptr()
    }

    pub fn emul(&self, a: *mut f32, b: *mut f32, c: *mut f32, n: i32) -> *mut f32 {
        let (rtx, rrx) = mpsc::channel();
        self.tx
            .send(GpuJob::Emul {
                a: DevPtr::from_raw(a).expect("null c"),
                b: DevPtr::from_raw(b).expect("null c"),
                c: DevPtr::from_raw(c).expect("null c"),
                n,
                resp: rtx,
            })
            .unwrap();
        rrx.recv().unwrap().as_mut_ptr()
    }

    pub fn t(&self, a: *mut f32, r: i32, c: i32) -> *mut f32 {
        let (rtx, rrx) = mpsc::channel();
        self.tx
            .send(GpuJob::T {
                a: DevPtr::from_raw(a).expect("null c"),
                r,
                c,
                resp: rtx,
            })
            .unwrap();
        rrx.recv().unwrap().as_mut_ptr()
    }

    pub fn unary_op(&self, a: *mut f32, b: *mut f32, n: i32, op: UnaryOp) -> *mut f32 {
        let (rtx, rrx) = mpsc::channel();
        self.tx
            .send(GpuJob::Unary {
                a: DevPtr::from_raw(a).expect("null c"),
                b: DevPtr::from_raw(b).expect("null c"),
                n,
                op,
                resp: rtx,
            })
            .unwrap();
        rrx.recv().unwrap().as_mut_ptr()
    }

    pub fn unary_op_grad(&self, a: *mut f32, b: *mut f32, n: i32, op: UnaryOp) -> *mut f32 {
        let (rtx, rrx) = mpsc::channel();
        self.tx
            .send(GpuJob::UnaryGrad {
                a: DevPtr::from_raw(a).expect("null c"),
                b: DevPtr::from_raw(b).expect("null c"),
                n,
                op,
                resp: rtx,
            })
            .unwrap();
        rrx.recv().unwrap().as_mut_ptr()
    }

    pub fn malloc(&self, n: usize) -> *mut f32 {
        let (rtx, rrx) = mpsc::channel();
        self.tx
            .send(GpuJob::Alloc {
                bytes: n * size_of::<f32>(),
                resp: rtx,
            })
            .unwrap();
        rrx.recv().unwrap().as_mut_ptr()
    }

    pub fn free(&self, a: *mut f32) {
        self.tx
            .send(GpuJob::Free {
                p: DevPtr::from_raw(a).expect("null c"),
            })
            .unwrap();
    }

    pub fn h2d(&self, dst: *mut f32, src: Vec<f32>) {
        let (rtx, rrx) = mpsc::channel();
        self.tx
            .send(GpuJob::HtoD {
                dst: DevPtr::from_raw(dst).expect("null c"),
                src,
                resp: rtx,
            })
            .unwrap();
        let _ = rrx.recv().unwrap();
    }

    pub fn d2h(&self, src: *mut f32, n: usize) -> Vec<f32> {
        let (rtx, rrx) = mpsc::channel();
        self.tx
            .send(GpuJob::DtoH {
                src: DevPtr::from_raw(src).expect("null c"),
                len: n,
                resp: rtx,
            })
            .unwrap();
        rrx.recv().unwrap()
    }

    pub fn d2d(&self, src: *mut f32, dst: *mut f32, n: usize) {
        let (rtx, rrx) = mpsc::channel();
        self.tx
            .send(GpuJob::DtoD {
                src: DevPtr::from_raw(src).expect("null c"),
                dst: DevPtr::from_raw(dst).expect("null c"),
                n: n * size_of::<f32>(),
                resp: rtx,
            })
            .unwrap();
        let _ = rrx.recv();
    }

    pub fn shutdown(&self) {
        self.tx.send(GpuJob::Shutdown {}).unwrap();
    }
}

impl Drop for GpuThread {
    fn drop(&mut self) {
        let _ = self.tx.send(GpuJob::Shutdown);

        if let Some(handle) = self.handle.take() {
            if handle.thread().id() != thread::current().id() {
                let _ = handle.join();
            }
        }
    }
}
