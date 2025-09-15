use std::time::{Duration, SystemTime};

use crate::{
    math::{CudaFam, CudaMatrix, HostMatrix, Matrix},
    structure::{
        layers::{DenseLayer, Layer},
        models::{Link, Node},
    },
};

mod math;
mod structure;

fn benchmark<S: AsRef<str>, R, F: Fn() -> R>(name: S, func: F) -> anyhow::Result<R> {
    let start = SystemTime::now();
    let duration: Duration;
    let result = func();
    duration = start.elapsed()?;

    println!("Benchmark {}: {}ns", name.as_ref(), duration.as_nanos());

    Ok(result)
}

fn main() -> anyhow::Result<()> {
    let w1: CudaMatrix<3, 2> = CudaMatrix::from_slice(&[1., 2., 3., 4., 5., 6.])?;
    let b1: CudaMatrix<3, 1> = CudaMatrix::from_slice(&[3., 2., 1.])?;

    let w2: CudaMatrix<1, 3> = CudaMatrix::from_slice(&[1., 2., 3.])?;
    let b2: CudaMatrix<1, 1> = CudaMatrix::from_slice(&[1.])?;

    let layer1: DenseLayer<CudaFam, 2, 3> = DenseLayer::new(w1, b1, math::UnaryOp::Relu);
    let layer2: DenseLayer<CudaFam, 3, 1> = DenseLayer::new(w2, b2, math::UnaryOp::Relu);

    let model: Link<DenseLayer<CudaFam, 2, 3>, DenseLayer<CudaFam, 3, 1>, 3> = Link(layer1, layer2);

    let input: CudaMatrix<2, 1> = CudaMatrix::from_slice(&[1., 2.])?;
    let output = benchmark("GPU Runtime", || model.forward(&input))?;

    println!("{}", output);

    Ok(())
}
