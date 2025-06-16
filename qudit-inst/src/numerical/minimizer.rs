use qudit_core::unitary::UnitaryMatrix;
use qudit_core::ComplexScalar;

use crate::numerical::problem::Problem;
use crate::numerical::problem::ProvidesCostFunction;
use crate::numerical::problem::ProvidesHessian;
use crate::numerical::problem::ProvidesResidualFunction;
use crate::numerical::problem::ProvidesJacobian;
use crate::numerical::problem::ProvidesGradient;
use crate::numerical::function::CostFunction;
use crate::numerical::function::Function;
use crate::numerical::function::Gradient;
use crate::numerical::function::Hessian;
use crate::numerical::function::Jacobian;
use crate::numerical::function::ResidualFunction;
use crate::numerical::x0::InitialGuessGenerator;
use crate::numerical::x0::Zeros;
use crate::InstantiationResult;
use crate::InstantiationTarget;
use crate::Instantiater;

pub struct MinimizationResult<C: ComplexScalar> {
    pub params: Vec<C::R>,
    pub fun: C::R,
    pub status: usize,
    pub message: Option<String>,
}

pub trait MinimizationAlgorithm<C: ComplexScalar, P: Problem, F: Function<C>> {
    fn prepare(&self, problem: &P) -> F;

    fn execute(&self, objective: &F, x0: &[C::R]) -> MinimizationResult<C>;
}

pub struct LBFGS;
impl<C, P> MinimizationAlgorithm<C, P, P::Gradient> for LBFGS
where
    C: ComplexScalar,
    P: ProvidesGradient<C>,
{
    fn prepare(&self, problem: &P) -> P::Gradient 
    {
        problem.build_gradient()
    }

    fn execute(&self, objective: &<P as ProvidesGradient<C>>::Gradient, x0: &[C::R]) -> MinimizationResult<C> {
        // Stub: real L‑BFGS would mutate x; here we just echo inputs.
        MinimizationResult { params: x0.to_vec(), fun: objective.cost(x0), status: 0, message: None }
    }
}


pub trait Runner<C: ComplexScalar, P: Problem, F: Function<C>> {
    fn run(&self, problem: P) -> MinimizationResult<C>;
}

pub struct MultiStartRunner<M, G, C: ComplexScalar> {
    pub minimizer: M,
    pub guess_generator: G,
    pub num_starts: usize,
    pub _phantom: std::marker::PhantomData<C>,
}


impl<C, P, M, F, G> Runner<C, P, F> for MultiStartRunner<M, G, C>
where
    C: ComplexScalar,
    P: Problem,
    M: MinimizationAlgorithm<C, P, F>,
    F: Function<C>,
    G: InitialGuessGenerator<C>,
{
    fn run(&self, problem: P) -> MinimizationResult<C> {
        let mut best: Option<MinimizationResult<C>> = None;
        let func = self.minimizer.prepare(&problem);
        for _ in 0..self.num_starts {
            let x0 = self.guess_generator.generate(problem.num_params());
            let res = self.minimizer.execute(&func, &x0);
            match &mut best {
                None => best = Some(res),
                Some(b) if res.fun < b.fun => *b = res,
                _ => {}
            }
        }
        best.expect("MultiStart: num_starts==0")
    }
}

// impl<C, P, M, F, G> Runner<C, P, F> for BatchedMultiStartRunner<M, G, C>
// where
//     C: ComplexScalar,
//     P: Problem + ProvidesBulk,
//     M: MinimizationAlgorithm<C, P, F>,
//     F: Function<C>,
//     G: InitialGuessGenerator<C>,
// {
//     fn run(&self, problem: P) -> MinimizationResult<C> {
//         let mut best: Option<MinimizationResult<C>> = None;
//         let func = self.minimizer.prepare_bulk(&problem, num_starts);
//         let x0 = self.guess_generator.generate_bulk(problem.num_params(), num_starts);
//         let res = self.minimizer.execute_bulk(&func, &x0);
//             match &mut best {
//                 None => best = Some(res),
//                 Some(b) if res.fun < b.fun => *b = res,
//                 _ => {}
//             }
//         }
//         best.expect("MultiStart: num_starts==0")
//     }
// }

// ParallelStartRunner; EnsembleRunner; IterativeRefinementRunner/ChainedRunner;
// HyperTunningRunner; BenchmarkingRunner; BatchedMultiStart

