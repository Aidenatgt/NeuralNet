use custos_math::{Matrix, custos::CUDA};

use crate::{
    math::{Mat, MatMul, UnaryOp, VecCol},
    structure::layers::{DenseLayer, Layer},
};

mod gpu_math;
mod math;
mod structure;

fn main() -> anyhow::Result<()> {
    let dev = CUDA::new(0).unwrap();
    let a: Mat<'_, '_, _, 3, 2> = Mat::from_host(&dev, &[1., 2., 3., 4., 5., 6.])?;
    let b: Mat<'_, '_, _, 2, 4> = Mat::from_host(&dev, &[2., 3., 4., 5., 6., 7., 8., 9.])?;

    let c = a.matmul(&b);

    println!("{:?}", c.as_inner());

    let weights: Mat<'_, '_, _, 3, 2> = Mat::from_host(&dev, &[2., 3., 2., -3., 2., 1.])?;
    let biases: VecCol<'_, '_, _, 3> = VecCol::from_host(&dev, &[1., 1., 1.])?;
    let layer: DenseLayer<'_, '_, CUDA, 2, 3> = DenseLayer::new(weights, biases, UnaryOp::Relu);

    let input: VecCol<'_, '_, _, 2> = VecCol::from_host(&dev, &[1., 2.])?;
    println!("{:?}", layer.calculate(&input).as_inner());

    Ok(())
}
