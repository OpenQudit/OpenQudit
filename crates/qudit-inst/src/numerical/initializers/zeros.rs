use super::super::InitialGuessGenerator;
use qudit_core::RealScalar;

#[derive(Default, Clone)]
pub struct Zeros<R: RealScalar> {
    _phantom: std::marker::PhantomData<R>,
}

impl<R: RealScalar> InitialGuessGenerator<R> for Zeros<R> {
    fn generate(&self, num_params: usize) -> Vec<R> {
        vec![R::from64(0.0); num_params]
    }
}
