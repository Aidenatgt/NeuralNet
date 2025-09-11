use crate::math::{HostMatrix, Matrix};

mod math;
mod structure;

fn main() -> anyhow::Result<()> {
    let a: HostMatrix<3, 4> =
        HostMatrix::from_slice(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])?;

    let b: HostMatrix<4, 2> = HostMatrix::from_slice(&[1., 2., 3., 4., 5., 6., 7., 8.])?;

    println!("{}", a);
    println!("{}", b);
    println!("{}", a.mmul(&b));
    Ok(())
}
