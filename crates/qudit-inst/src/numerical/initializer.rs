use qudit_core::RealScalar;

pub trait InitialGuessGenerator<R: RealScalar>: Clone {
    fn generate(&self, num_params: usize) -> Vec<R>;
}
