use crate::{
    math::{CudaFam, CudaMatrix, HostMatrix, Matrix},
    structure::layers::{DenseLayer, Layer},
};

mod math;
mod structure;

fn main() -> anyhow::Result<()> {
    let a: HostMatrix<3, 2> = HostMatrix::from_slice(&[1., 2., 3., 4., 5., -6.])?;
    let b: HostMatrix<3, 1> = HostMatrix::from_slice(&[3., 2., 1.])?;

    let layer: DenseLayer<CudaFam, 2, 3> = DenseLayer::new(
        CudaMatrix::from(&a),
        CudaMatrix::from(&b),
        math::UnaryOp::Relu,
    );

    let input: CudaMatrix<2, 1> = CudaMatrix::from(&HostMatrix::<2, 1>::from_slice(&[1., 2.])?);
    let output = layer.calc(&input);

    println!("{}", output);

    Ok(())
}
