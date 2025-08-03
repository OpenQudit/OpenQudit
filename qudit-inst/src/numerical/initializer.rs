use qudit_core::RealScalar;

pub trait InitialGuessGenerator<R: RealScalar> {
    fn generate(&self, num_params: usize) -> Vec<R>;
}

