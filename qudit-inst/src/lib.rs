
use qudit_core::ComplexScalar;
use qudit_circuit::QuditCircuit;

mod result;
mod target;
mod instantiater;

pub use result::InstantiationResult;
pub use target::InstantiationTarget;
pub use instantiater::Instantiater;

// Open Questions:
//  - How do I expose intitial input generation; even through wrapper instantiaters
//
//  Answered Questions:
//  - Multi Start will be a wrapper instantiation; will not be part of API
//  - Should batched instantiation be part of instantiate? Yes.
//  - Sample will be a wrapper instantiation if possible, otherwise its own thing
//  - What is an Instantiation Result? (params, and termination status, with optional fun eval and
//          message)
//      - More importantly, should I allow for callers to teach instantiaters about true success
//              No
//
//  Requires Experimentation:
//  - How do I differentiate between Residual Optimizers and normal ones
//      Explicitly for now; need to determine if sum of square residuals works just fine
//      In theory, LM minimizes sum of square, the residual function is just API design by CERES



// pub trait Minimizer<C: ComplexScalar> {
//     fn minimize(
//         &self,
//         cost: BoxedCostFunction,
//         x0: Vec<C::R>,
//     ) -> InstantiationResult<Vec<C::R>>;

//     fn gen_x0(&self, cost: &BoxedCostFunction) -> Vec<C::R>;
// }

// pub trait LeastSquaresSolver<C: ComplexScalar> {
//     fn minimize(&self, cost, BoxedResidualFunction, x0: Vec<C::R>) -> InstantiationResult<Vec<C::R>>;
// }

// impl<C: ComplexScalar, M: Minimizer> Instantiater<C> for M {
//     fn instantiate(
//         &self,
//         circuit: &QuditCircuit<C>,
//         target: &InstantiationTarget<C>,
//     ) -> InstantiationResult<Vec<<C as ComplexScalar>::R>> {
//         let cost = target.gen_cost_fn(circuit);
//         let x0 = self.gen_x0(&cost);
//         self.minimize(cost, x0)
//     }
// }

// pub struct MultiStartMinimizer<C: ComplexScalar, M: Minimizer<C>> {
//     inner: M,
//     num_starts: usize,
// }

// impl<C: ComplexScalar, M: Minimizer<C>> MultiStartMinimizer<C, M> {
//     pub fn new(minimizer: M, num_starts: usize) -> Self {
//         MultiStartMinimizer { inner: minimizer, num_starts }
//     }
// }

// impl<C: ComplexScalar, M: Minimizer<C>> Minimizer<C: ComplexScalar> for MultiStartMinimizer<C, M> {
//     fn minimize(
//         &self,
//         cost: BoxedCostFunction,
//         x0: Vec<C::R>,
//     ) -> InstantiationResult<Vec<C::R>> {
        
//     }
// }

// Desired Features:
//  - Enable instantiation experimentation for better results
//      - expose API so that users can swap out components and test end to end results
//          - instantiation algorithms, optimizers, settings, x0 generation
//
//  - Provide a suite of instantiaters that work well across the spectrum of standard problems 

// Use Cases
// let synthesizer = SomeSynthesizer(..., BoxedInstantiater<C>, ...);
// synthesizer.synthesize(unitary.into())
//
// fn synthesize(self, input: InstantiationTarget):
//      let mut circuit = self.initial_circuit();
//      while not synthesized:
//          self.extend(circuit)
//          let result = self.intantiater.instantiate_in_place(circuit, input)
//          if result.fail() -> report to user depending on fail
//
// CPUParallelInstantiater(num_cores)
// CPUSampleInstantiater
//
// Multi-Start:
// let ms_instantiater = MultiStartCPUWrapper(BoxedInstantiater<C>, SelectionFN, 16)
//
// GPU
// let gpu_instantiater = GPU_Optimized_Instantiater(SelectionFN, 16)
// fn gpu_instantiate:
//      let bytecode = circuit.compile_to_bytecode()
//      let gpu_qvm = GPUQVM::new(bytecode)
//      let gpu_optimizer = WhateverLibrary::Optimizer::new(gpu_qvm::fn_eval, starting_x)
//      let result = gpu_optimizer.optimize()
//
// GPUSampleInstantiater
//
// QFACTOR
// let qfactor_instantiater = QFactorInstantiater::new(all_the_params)
// fn qfactor_instantiate:
//      let mut unitary_builder = self.initialize_circuit_tensor(circuit, &target);
//      while should keep going:
//          sweep_circuit(unitary_builder, circuit)
//          if should rebuild: rebuild_unitary_tensor
//
//
// let qfactor_sample_instantiater = QfactorSample... (same thing but more complicated)
//
// QFactorSampleGPU = we will figure it out somehow, most likely same jax python implementation
