use custos_math::custos::CUDA;

use crate::math::{Mat, MatDevice, MatMul, UnaryOp, VecCol, map_unary_vec};

pub trait Layer<'a, 'd: 'a, D: MatDevice<'d>, const I: usize, const O: usize> {
    fn new(
        weights: Mat<'a, 'd, D, O, I>,
        biases: VecCol<'a, 'd, D, O>,
        activation: UnaryOp,
    ) -> Self;
    fn calculate<'b>(&'b self, input: &'b VecCol<'b, 'd, D, I>) -> VecCol<'b, 'd, D, O>;
    fn weights(&self) -> &Mat<'a, 'd, D, I, O>;
    fn biases(&self) -> &VecCol<'a, 'd, D, O>;
}

pub struct DenseLayer<'a, 'd: 'a, D: MatDevice<'d>, const I: usize, const O: usize> {
    weights: Mat<'a, 'd, D, O, I>,
    biases: VecCol<'a, 'd, D, O>,
    op: UnaryOp,
}

impl<'a, 'd: 'a, const I: usize, const O: usize> Layer<'a, 'd, CUDA, I, O>
    for DenseLayer<'a, 'd, CUDA, I, O>
{
    fn new(weights: Mat<'a, 'd, CUDA, O, I>, biases: VecCol<'a, 'd, CUDA, O>, op: UnaryOp) -> Self {
        Self {
            weights,
            biases,
            op,
        }
    }

    fn calculate<'b>(&'b self, input: &'b VecCol<'b, 'd, CUDA, I>) -> VecCol<'b, 'd, CUDA, O> {
        let mut output: VecCol<'b, 'd, CUDA, O> = self.weights.matmul(input);

        map_unary_vec(&mut output, self.op, 0.0);

        output
    }

    fn weights(&self) -> &Mat<'a, 'd, CUDA, I, O> {
        todo!()
    }

    fn biases(&self) -> &VecCol<'a, 'd, CUDA, O> {
        todo!()
    }
}
