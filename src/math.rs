use std::marker::PhantomData;

use custos_math::{
    Gemm, Matrix, TransposeOp,
    custos::{Alloc, Device, RawConv, Read},
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
            _marker: PhantomData,
        })
    }
    pub fn zeros(dev: &'d D) -> anyhow::Result<Self> {
        Ok(Self {
            data: Matrix::from((dev, (M, N), &vec![0.0_f32; M * N])),
            _marker: PhantomData,
        })
    }
    pub fn as_inner(&self) -> &Matrix<f32, D> {
        &self.data
    }
    pub fn T(&self) -> Mat<'m, 'd, D, N, M> {
        Mat::<'m, 'd, D, N, M> {
            data: self.data.T(),
            _marker: PhantomData,
        }
    }
    pub fn take_vec(&self) -> Vec<f32>
    where
        for<'a> <D as Read<f32>>::Read<'a>: ToOwned<Owned = Vec<f32>>,
    {
        self.as_inner().read().to_owned()
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
            _marker: PhantomData,
        }
    }
}

pub trait Function {
    fn calculate(x: f32) -> f32;
    fn derivative(x: f32) -> f32;
}

pub struct Relu {}

impl Function for Relu {
    fn calculate(x: f32) -> f32 {
        if x > 0.0 { x } else { 0.0 }
    }

    fn derivative(x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}
