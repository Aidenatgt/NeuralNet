use crate::math::HostMatrix;

mod math;
mod structure;

fn main() -> anyhow::Result<()> {
    let a: HostMatrix<3, 4> =
        HostMatrix::from(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);

    println!("{}", a);
    Ok(())
}
