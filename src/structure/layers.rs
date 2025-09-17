use crate::{
    math::{MatFamily, Matrix, UnaryOp},
    structure::layers::sealed::Sealed,
};

pub struct Record<F: MatFamily, const I: usize, const O: usize> {
    input: F::Mat<I, 1>,
    pre_act: F::Mat<O, 1>,
    post_act: F::Mat<O, 1>,
}

mod sealed {
    pub trait Sealed {}
}

pub trait Layer<F: MatFamily, const I: usize, const O: usize>: Sealed {
    fn calc(&self, input: &F::Mat<I, 1>) -> F::Mat<O, 1>;
    fn record_calc(&self, input: &F::Mat<I, 1>) -> Record<F, I, O>;
    fn backward(
        &self,
        rec: &Record<F, I, O>,
        upstream: &F::Mat<O, 1>,
    ) -> (F::Mat<I, 1>, F::Mat<O, I>, F::Mat<O, 1>);
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
        let biased = weighted.add(&self.biases);
        biased.unary_op(self.activation)
    }

    fn record_calc(&self, input: &F::Mat<I, 1>) -> Record<F, I, O> {
        let mut pre_act = self.weights.mmul(input);
        pre_act.add_assign(&self.biases);
        let post_act = pre_act.unary_op(self.activation);

        Record::<F, I, O> {
            input: input.clone(),
            pre_act,
            post_act,
        }
    }

    fn backward(
        &self,
        rec: &Record<F, I, O>,
        upstream: &<F as MatFamily>::Mat<O, 1>,
    ) -> (
        <F as MatFamily>::Mat<I, 1>,
        <F as MatFamily>::Mat<O, I>,
        <F as MatFamily>::Mat<O, 1>,
    ) {
        todo!()
    }
}
