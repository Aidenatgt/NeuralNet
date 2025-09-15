use crate::math::{MatFamily, Matrix, UnaryOp};

pub struct Record<F: MatFamily, const I: usize, const O: usize> {
    input: <F as MatFamily>::Mat<I, 1>,
    output: <F as MatFamily>::Mat<O, 1>,
    activation: <F as MatFamily>::Mat<O, 1>,
    gradient: <F as MatFamily>::Mat<O, 1>,
}
pub trait Layer<F: MatFamily, const I: usize, const O: usize> {
    type Fam: MatFamily;

    fn new(
        weights: <Self::Fam as MatFamily>::Mat<O, I>,
        biases: <Self::Fam as MatFamily>::Mat<O, 1>,
        activation: UnaryOp,
    ) -> Self;
    fn calc(&self, input: &<Self::Fam as MatFamily>::Mat<I, 1>) -> F::Mat<O, 1>;
    fn record_calc(&self, input: &<Self::Fam as MatFamily>::Mat<I, 1>) -> Record<F, I, O>;
}

pub struct DenseLayer<F: MatFamily, const I: usize, const O: usize> {
    weights: <F as MatFamily>::Mat<O, I>,
    biases: <F as MatFamily>::Mat<O, 1>,
    activation: UnaryOp,
}

impl<F: MatFamily, const I: usize, const O: usize> Layer<F, I, O> for DenseLayer<F, I, O> {
    type Fam = F;
    fn new(
        weights: <Self::Fam as MatFamily>::Mat<O, I>,
        biases: <Self::Fam as MatFamily>::Mat<O, 1>,
        activation: UnaryOp,
    ) -> Self {
        Self {
            weights,
            biases,
            activation,
        }
    }

    fn calc(
        &self,
        input: &<Self::Fam as MatFamily>::Mat<I, 1>,
    ) -> <Self::Fam as MatFamily>::Mat<O, 1> {
        let weighted = self.weights.mmul(input);
        let activated = weighted.unary_op(self.activation);
        self.biases.add(&activated)
    }

    fn record_calc(&self, input: &<Self::Fam as MatFamily>::Mat<I, 1>) -> Record<F, I, O> {
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
