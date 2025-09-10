use std::marker::PhantomData;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::math::{Function, Mat, MatDevice, MatMul, VecCol};

pub trait Layer<'a, 'd: 'a, D: MatDevice<'d>, F: Function, const I: usize, const O: usize> {
    fn new(weights: Mat<'a, 'd, D, O, I>, biases: VecCol<'a, 'd, D, O>) -> Self;
    fn calculate<'b>(&'b self, input: &'b VecCol<'b, 'd, D, I>) -> VecCol<'b, 'd, D, O>;
    fn weights(&self) -> &Mat<'a, 'd, D, I, O>;
    fn biases(&self) -> &VecCol<'a, 'd, D, O>;
}

pub struct DenseLayer<'a, 'd: 'a, D: MatDevice<'d>, F: Function, const I: usize, const O: usize> {
    weights: Mat<'a, 'd, D, O, I>,
    biases: VecCol<'a, 'd, D, O>,
    _marker: PhantomData<F>,
}

impl<'a, 'd: 'a, D: MatDevice<'d>, F: Function, const I: usize, const O: usize>
    Layer<'a, 'd, D, F, I, O> for DenseLayer<'a, 'd, D, F, I, O>
{
    fn new(weights: Mat<'a, 'd, D, O, I>, biases: VecCol<'a, 'd, D, O>) -> Self {
        Self {
            weights,
            biases,
            _marker: PhantomData,
        }
    }

    fn calculate<'b>(&'b self, input: &'b VecCol<'b, 'd, D, I>) -> VecCol<'b, 'd, D, O> {
        let mut output: VecCol<'b, 'd, D, O> = self.weights.matmul(input);

        println!("{}", output.as_inner().read());
        output
    }

    fn weights(&self) -> &Mat<'a, 'd, D, I, O> {
        todo!()
    }

    fn biases(&self) -> &VecCol<'a, 'd, D, O> {
        todo!()
    }
}
