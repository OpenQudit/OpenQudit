use qudit_core::RealScalar;
use super::super::InitialGuessGenerator;
use rand::Rng;
use rand::distr::Uniform as RandUniform;

#[derive(Clone)]
pub struct Uniform<R: RealScalar> {
    lower_bound: R,
    upper_bound: R,
    _phantom: std::marker::PhantomData<R>,
}

impl<R: RealScalar> Uniform<R> {
    pub fn new(mut lower_bound: R, mut upper_bound: R) -> Self {
        if lower_bound > upper_bound {
            panic!("Lower bound cannot be larger than upper bound.");
        }
        Self {
            lower_bound,
            upper_bound,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<R: RealScalar> Default for Uniform<R> {
    fn default() -> Self {
        Self::new(
            R::from64(-std::f64::consts::PI),
            R::from64(std::f64::consts::PI),
        )
    }
}

impl<R: RealScalar> InitialGuessGenerator<R> for Uniform<R> {
    fn generate(&self, num_params: usize) -> Vec<R> {
        let mut rng = rand::thread_rng();
        let distribution = RandUniform::new(
            self.lower_bound.to64(),
            self.upper_bound.to64(),
        ).unwrap();

        (0..num_params).map(|_| R::from64(rng.sample(distribution))).collect()
    }
}
