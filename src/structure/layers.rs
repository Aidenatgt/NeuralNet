use crate::{
    math::{MatFamily, Matrix, UnaryOp},
    structure::layers::sealed::Sealed,
};

pub struct Record<F: MatFamily, const I: usize, const O: usize> {
    input: F::Mat<I, 1>,
    output: F::Mat<O, 1>,
    activation: F::Mat<O, 1>,
    gradient: F::Mat<O, 1>,
}

mod sealed {
    pub trait Sealed {}
}

pub trait Layer<F: MatFamily, const I: usize, const O: usize>: Sealed {
    fn calc(&self, input: &F::Mat<I, 1>) -> F::Mat<O, 1>;
    fn record_calc(&self, input: &F::Mat<I, 1>) -> Record<F, I, O>;
}

pub struct DenseLayer<F: MatFamily, const I: usize, const O: usize> {
    weights: F::Mat<O, I>,
    biases: F::Mat<O, 1>,
    activation: UnaryOp,
}

impl<F: MatFamily, const I: usize, const O: usize> DenseLayer<F, I, O> {
    pub fn new(weights: F::Mat<O, I>, biases: F::Mat<O, 1>, activation: UnaryOp) -> Self {
        Self {
            weights,
            biases,
            activation,
        }
    }
}

impl<F: MatFamily, const I: usize, const O: usize> Sealed for DenseLayer<F, I, O> {}

impl<F: MatFamily, const I: usize, const O: usize> Layer<F, I, O> for DenseLayer<F, I, O> {
    fn calc(&self, input: &F::Mat<I, 1>) -> F::Mat<O, 1> {
        let weighted = self.weights.mmul(input);
        let activated = weighted.unary_op(self.activation);
        self.biases.add(&activated)
    }

    fn record_calc(&self, input: &F::Mat<I, 1>) -> Record<F, I, O> {
        let weighted = self.weights.mmul(input);
        let activated = weighted.unary_op(self.activation);
        let output = self.biases.add(&activated);

        Record::<F, I, O> {
            input: input.clone(),
            output,
            activation: activated.clone(),
            gradient: activated.unary_op_grad(self.activation),
        }
    }
}
