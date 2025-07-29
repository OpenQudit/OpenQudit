use qudit_core::RealScalar;
use super::super::MinimizationAlgorithm;
use super::super::MinimizationResult;
use super::super::ProvidesGradient;

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

    fn minimize(&self, objective: &mut Self::Func, x0: &[R]) -> MinimizationResult<R> {    
        todo!()
    }
}
