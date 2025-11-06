use super::MinimizationResult;
use super::Problem;
use qudit_core::RealScalar;

pub trait Runner<R: RealScalar, P: Problem>: Clone {
    fn run(&self, problem: P) -> MinimizationResult<R>;
}
