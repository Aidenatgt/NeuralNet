use custos::CUDA;
use custos_math::Matrix;
use std::marker::PhantomData;

// Column-major/row-major doesn't matter here; custos-math stores flat.
pub struct Mat<const M: usize, const N: usize> {
    data: Matrix<f32>, // lives on the GPU
    _marker: PhantomData<[(); M * N]>,
}

impl<const M: usize, const N: usize> Mat<M, N> {
    pub fn from_host(dev: &CUDA, host: &[f32]) -> anyhow::Result<Self> {
        assert_eq!(host.len(), M * N, "host slice len != M*N");
        Ok(Self {
            data: Matrix::from((dev, (M, N), host)),
            _marker: PhantomData,
        })
    }
    pub fn zeros(dev: &CUDA) -> anyhow::Result<Self> {
        Ok(Self {
            data: Matrix::zeros::<f32>(dev, (M, N)),
            _marker: PhantomData,
        })
    }
    pub fn as_inner(&self) -> &Matrix<f32> {
        &self.data
    }
}

// MatMul with compile-time shape check: (M×K) @ (K×N) -> (M×N)
pub trait MatMul<Rhs> {
    type Out;
    fn matmul(&self, rhs: &Rhs, dev: &CUDA) -> Self::Out;
}
impl<const M: usize, const K: usize, const N: usize> MatMul<Mat<K, N>> for Mat<M, K> {
    type Out = Mat<M, N>;
    fn matmul(&self, rhs: &Mat<K, N>, dev: &CUDA) -> Self::Out {
        use custos_math::Gemm;
        let c = self.data.gemm(&rhs.data);
        Mat {
            data: c,
            _marker: PhantomData,
        }
    }
}

// Vector as skinny matrix
pub type VecCol<const N: usize> = Mat<N, 1>;
pub type VecRow<const N: usize> = Mat<1, N>;
