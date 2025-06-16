use qudit_core::ComplexScalar;
use qudit_core::BitWidthConvertible;

pub trait InitialGuessGenerator<C: ComplexScalar> {
    fn generate(&self, num_params: usize) -> Vec<C::R>;
}

pub struct Zeros<C: ComplexScalar> {
    _phantom: std::marker::PhantomData<C>,
}

impl<C: ComplexScalar> InitialGuessGenerator<C> for Zeros<C> {
    fn generate(&self, num_params: usize) -> Vec<C::R> {
        vec![C::R::from64(0.0); num_params]
    }
}

// UniformRandom
