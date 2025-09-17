use std::time::{Duration, SystemTime};

use crate::math::CudaMatrix;
use crate::{
    math::{CudaFam, Matrix},
    structure::{layers::DenseLayer, models::Node},
};

mod math;
mod structure;

fn benchmark<S: AsRef<str>, R, F: Fn() -> R>(name: S, func: F) -> anyhow::Result<R> {
    let start = SystemTime::now();
    let duration: Duration;
    let result = func();
    duration = start.elapsed()?;

    println!(
        "Benchmark {}: {}ms",
        name.as_ref(),
        duration.as_nanos() as f32 / 1000.0
    );

    Ok(result)
}

fn main() -> anyhow::Result<()> {
    let w1: CudaMatrix<3, 2> = CudaMatrix::from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])?;
    let b1: CudaMatrix<3, 1> = CudaMatrix::from_slice(&[0.2, 0.3, 0.4])?;

    let layer1: DenseLayer<CudaFam, 2, 3> = DenseLayer::new(w1, b1, math::UnaryOp::Tanh);

    let input: CudaMatrix<2, 1> = CudaMatrix::from_slice(&[0.9, 1.1])?;
    let output = benchmark("GPU Runtime", || layer1.forward(&input))?;

    println!("{}", output);

    Ok(())
}
