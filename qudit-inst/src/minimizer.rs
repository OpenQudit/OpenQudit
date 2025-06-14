use qudit_core::ComplexScalar;

use crate::function::CostFunction;
use crate::function::Function;
use crate::function::Gradient;
use crate::function::Hessian;
use crate::function::Jacobian;
use crate::function::ResidualFunction;
use crate::InstantiationResult;
use crate::InstantiationTarget;
use crate::Instantiater;
use qudit_core::BitWidthConvertible;


pub struct MinimizationResult<C: ComplexScalar> {
    pub params: Vec<C::R>,
    pub fun: C::R,
    pub status: usize,
    pub message: Option<String>,
}

/// A trait for numerical optimization algorithms (e.g., L-BFGS, Nelder-Mead).
pub trait Minimizer<C: ComplexScalar> {
    fn minimize<F: Function<C>>(
        &self,
        function: &F,
        x0: &[C::R]
    ) -> MinimizationResult<C>;
}

/// A trait for generating initial parameter guesses.
pub trait InitialGuessGenerator<C: ComplexScalar> {
    fn generate(&self, num_params: usize) -> Vec<C::R>;
}

/// A simple generator that creates a vector of zeros.
pub struct Zeros<C: ComplexScalar> {
    _marker: std::marker::PhantomData<C>,
}

impl<C: ComplexScalar> InitialGuessGenerator<C> for Zeros<C> {
    fn generate(&self, num_params: usize) -> Vec<C::R> {
        vec![C::R::from64(0.0); num_params]
    }
}

// UniformRandom

/// A trait for defining an optimization strategy (e.g., single start, multi-start).
pub trait Runner<C: ComplexScalar> {
    fn run<F: Function<C>>(&self, function: &F) -> MinimizationResult<C>;
}

pub struct MultiStartRunner<M, G, C: ComplexScalar> {
    minimizer: M,
    guess_generator: G,
    num_starts: usize,
    _phantom: std::marker::PhantomData<C>,
}


impl<M: Minimizer<C>, G: InitialGuessGenerator<C>, C: ComplexScalar> Runner<C> for MultiStartRunner<M, G, C> {
    fn run<F: Function<C>>(&self, _function: &F) -> MinimizationResult<C> {
        todo!("Implement multi-start logic here")
    }
}

// ParallelStartRunner; EnsembleRunner; IterativeRefinementRunner/ChainedRunner;
// HyperTunningRunner; BenchmarkingRunner;

/// An instantiater that uses a `Runner` to solve a minimization problem.
pub struct MinimizingInstantiater<R, C: ComplexScalar> {
    pub runner: R,
    _phantom: std::marker::PhantomData<C>,
}

impl<R: Runner<C>, C: ComplexScalar> Instantiater<C> for MinimizingInstantiater<R, C> {
    fn instantiate(
        &self,
        _circuit: &qudit_circuit::QuditCircuit<C>,
        _target: &InstantiationTarget<C>,
        _data: &std::collections::HashMap<String, Box<dyn std::any::Any>>,
    ) -> InstantiationResult<C> {
        // 1. Create a struct that holds the circuit and target.
        // 2. Implement the `CostFunction` (and `Gradient`, etc.) traits for that struct.
        // 3. Pass that function object to the runner.
        // 4. Convert the `MinimizationResult` to an `InstantiationResult`.
        // let func = target.to_function(circuit);
        // self.runner.run(func)
        todo!()
    }
}

