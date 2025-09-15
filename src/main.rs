use crate::math::{CudaMatrix, HostMatrix, Matrix};

mod math;
mod structure;

fn main() -> anyhow::Result<()> {
    let a: HostMatrix<2, 2> = HostMatrix::from_slice(&[1., 2., 3., 4.])?;
    let b: HostMatrix<2, 2> = HostMatrix::from_slice(&[4., 3., 2., 1.])?;

    let A = CudaMatrix::from(a);
    let B = CudaMatrix::from(b);

    let C = A.emul(&B);

    println!("{}", C);
    Ok(())
}
