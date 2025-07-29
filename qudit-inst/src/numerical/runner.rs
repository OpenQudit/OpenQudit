use qudit_core::RealScalar;
use super::Problem;
use super::MinimizationResult;

pub trait Runner<R: RealScalar, P: Problem> {
    fn run(&self, problem: P) -> MinimizationResult<R>;
}

