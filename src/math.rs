use std::{any::TypeId, marker::PhantomData};

use custos_math::{
    Gemm, Matrix, TransposeOp,
    custos::{Alloc, CUDA, Device, RawConv, Read, cuda::AsCudaCvoidPtr},
};

pub trait MatDevice<'d>:
    Device + RawConv + Alloc<'d, f32> + TransposeOp<f32> + Gemm<f32> + Read<f32>
{
}

impl<'d, T> MatDevice<'d> for T where
    T: Device + RawConv + Alloc<'d, f32> + TransposeOp<f32> + Gemm<f32> + Read<f32>
{
}

pub struct Mat<'m, 'd: 'm, D: MatDevice<'d>, const M: usize, const N: usize> {
    data: Matrix<'m, f32, D>,
    dev: &'d D,
    _marker: PhantomData<&'d D>,
}

pub type VecCol<'v, 'd: 'v, D: MatDevice<'d>, const N: usize> = Mat<'v, 'd, D, N, 1>;
pub type VecRow<'v, 'd: 'v, D: MatDevice<'d>, const N: usize> = Mat<'v, 'd, D, 1, N>;
pub type SquareMat<'m, 'd, D: MatDevice<'d>, const N: usize> = Mat<'m, 'd, D, N, N>;

impl<'m, 'd: 'm, D: MatDevice<'d>, const M: usize, const N: usize> Mat<'m, 'd, D, M, N> {
    pub fn from_host(dev: &'d D, host: &[f32]) -> anyhow::Result<Self> {
        assert_eq!(host.len(), M * N, "host slice len != M*N");
        Ok(Self {
            data: Matrix::from((dev, (M, N), host)),
            dev,
            _marker: PhantomData,
        })
    }
    pub fn zeros(dev: &'d D) -> anyhow::Result<Self> {
        Ok(Self {
            data: Matrix::from((dev, (M, N), &vec![0.0_f32; M * N])),
            dev,
            _marker: PhantomData,
        })
    }
    pub fn as_inner(&self) -> &Matrix<f32, D> {
        &self.data
    }
    pub fn T(&self) -> Mat<'m, 'd, D, N, M> {
        Mat::<'m, 'd, D, N, M> {
            data: self.data.T(),
            dev: self.dev,
            _marker: PhantomData,
        }
    }
    pub fn take_vec(&self) -> Vec<f32>
    where
        for<'a> <D as Read<f32>>::Read<'a>: ToOwned<Owned = Vec<f32>>,
    {
        self.as_inner().read().to_owned()
    }
    pub fn device(&self) -> &'d D {
        self.dev
    }
}

pub trait MatMul<
    'm,
    'r,
    'd: 'm + 'm,
    D: MatDevice<'d>,
    const M: usize,
    const K: usize,
    const N: usize,
>
{
    fn matmul<'x>(&'x self, rhs: &Mat<'x, 'd, D, K, N>) -> Mat<'x, 'd, D, M, N>;
}
impl<'m, 'r, 'd: 'm + 'r, D: MatDevice<'d>, const M: usize, const K: usize, const N: usize>
    MatMul<'m, 'r, 'd, D, M, K, N> for Mat<'m, 'd, D, M, K>
{
    fn matmul<'x>(&'x self, rhs: &Mat<'x, 'd, D, K, N>) -> Mat<'x, 'd, D, M, N> {
        let c: Matrix<'x, f32, D> = self.data.gemm(&rhs.data);
        Mat {
            data: c,
            dev: self.dev,
            _marker: PhantomData,
        }
    }
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
pub enum UnaryOp {
    Relu = 0,
    LeakyRelu = 1,
    Silu = 2,
    Gelu = 3,
    Tanh = 4,
    Sigmoid = 5,
    Softplus = 6,
}

#[link(name = "unary", kind = "static")] // found via build.rs link-search hints
unsafe extern "C" {
    fn map_unary_f32(i_ptr: *mut f32, o_ptr: *mut f32, n: usize, op: i32, p0: f32);
    fn map_unary_grad_f32(i_ptr: *mut f32, o_ptr: *mut f32, n: usize, op: i32, p0: f32);
}

pub fn apply_unary_vec<'v, 'd, const N: usize>(
    vector: &mut VecCol<'v, 'd, CUDA, N>,
    dest: &mut VecCol<'v, 'd, CUDA, N>,
    op: UnaryOp,
    p0: f32,
) {
    unsafe {
        let dptr = vector.as_inner().as_buf().ptr.ptr as *mut f32;
        let o_dptr = dest.as_inner().as_buf().ptr.ptr as *mut f32;
        map_unary_f32(dptr, o_dptr, vector.as_inner().size(), op as i32, p0);
    }
}
pub fn apply_unary_grad_vec<'v, 'd, const N: usize>(
    vector: &mut VecCol<'v, 'd, CUDA, N>,
    dest: &mut VecCol<'v, 'd, CUDA, N>,
    op: UnaryOp,
    p0: f32,
) {
    unsafe {
        let dptr = vector.as_inner().as_buf().ptr.ptr as *mut f32;
        let o_dptr = dest.as_inner().as_buf().ptr.ptr as *mut f32;
        map_unary_grad_f32(dptr, o_dptr, vector.as_inner().size(), op as i32, p0);
    }
}

pub fn map_unary_vec<'v, 'd, const N: usize>(
    vector: &mut VecCol<'v, 'd, CUDA, N>,
    op: UnaryOp,
    p0: f32,
) {
    unsafe {
        let dptr = vector.as_inner().as_buf().ptr.ptr as *mut f32;
        map_unary_f32(dptr, dptr, vector.as_inner().size(), op as i32, p0);
    }
}
pub fn map_unary_grad_vec<'v, 'd, const N: usize>(
    vector: &mut VecCol<'v, 'd, CUDA, N>,
    op: UnaryOp,
    p0: f32,
) {
    unsafe {
        let dptr = vector.as_inner().as_buf().ptr.ptr as *mut f32;
        map_unary_grad_f32(dptr, dptr, vector.as_inner().size(), op as i32, p0);
    }
}
