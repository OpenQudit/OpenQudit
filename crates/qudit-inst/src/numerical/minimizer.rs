use qudit_core::RealScalar;

use super::Function;
use super::MinimizationResult;
use super::Problem;

pub trait MinimizationAlgorithm<R: RealScalar, P: Problem>: Clone {
    type Func: Function;

    fn initialize(&self, problem: &P) -> Self::Func;

    fn minimize(&self, objective: &mut Self::Func, x0: &[R]) -> MinimizationResult<R>;
}
