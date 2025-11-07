use super::super::MinimizationAlgorithm;
use super::super::MinimizationResult;
use super::super::ProvidesGradient;
use qudit_core::RealScalar;

#[derive(Clone)]
pub struct LBFGS;

impl<R, P> MinimizationAlgorithm<R, P> for LBFGS
where
    R: RealScalar,
    P: ProvidesGradient<R>,
{
    type Func = P::Gradient;

    fn initialize(&self, problem: &P) -> Self::Func {
        problem.build_gradient()
    }

    fn minimize(&self, _objective: &mut Self::Func, _x0: &[R]) -> MinimizationResult<R> {
        todo!()
    }
}
