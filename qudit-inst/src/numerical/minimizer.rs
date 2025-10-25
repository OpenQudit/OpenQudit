use qudit_core::RealScalar;

use super::Problem;
use super::Function;
use super::MinimizationResult;

pub trait MinimizationAlgorithm<R: RealScalar, P: Problem>: Clone {
    type Func: Function;

    fn initialize(&self, problem: &P) -> Self::Func;

    fn minimize(&self, objective: &mut Self::Func, x0: &[R]) -> MinimizationResult<R>;
}

