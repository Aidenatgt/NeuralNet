use crate::{math::MatFamily, structure::layers::Layer};

pub struct Assert<const B: bool>;
pub trait True {}
impl True for Assert<true> {}

pub trait Node<F: MatFamily, const I: usize, const O: usize> {
    fn forward(&self, x: &F::Mat<I, 1>) -> F::Mat<O, 1>;
}

impl<F: MatFamily, const I: usize, const O: usize, L: Layer<F, I, O>> Node<F, I, O> for L {
    fn forward(&self, x: &F::Mat<I, 1>) -> F::Mat<O, 1> {
        self.calc(x)
    }
}

pub struct Link<A, B, const M: usize>(pub A, pub B);

impl<F, A, B, const I: usize, const M: usize, const O: usize> Node<F, I, O> for Link<A, B, M>
where
    F: MatFamily,
    A: Node<F, I, M>,
    B: Node<F, M, O>,
{
    fn forward(&self, x: &<F as MatFamily>::Mat<I, 1>) -> <F as MatFamily>::Mat<O, 1> {
        let mid = self.0.forward(x);
        self.1.forward(&mid)
    }
}
