use custos_math::{Matrix, custos::CUDA};

mod math;
mod structure;

fn main() {
    let device = CUDA::new(0).unwrap();
    let a = Matrix::from((&device, (2, 3), [1., 2., 3., 4., 5., 6.]));
    let b = Matrix::from((&device, (3, 2), [6., 5., 4., 3., 2., 1.]));

    let c = a.gemm(&b);

    println!("{:?}", c)
}
