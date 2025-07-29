mod instantiater;
mod numerical;
mod qfactor;
mod result;
mod target;

pub use instantiater::DataMap;
pub use instantiater::Instantiater;
pub use result::InstantiationResult;
pub use target::InstantiationTarget;

// Open Questions:
//
//  Answered Questions:
//  - How do I expose intitial input generation; even through wrapper instantiaters
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




// pub trait LeastSquaresSolver<C: ComplexScalar> {
//     fn minimize(&self, cost, BoxedResidualFunction, x0: Vec<C::R>) -> InstantiationResult<Vec<C::R>>;
// }


// Desired Features:
//  - Enable instantiation experimentation for better results
//      - expose API so that users can swap out components and test end to end results
//          - instantiation algorithms, optimizers, settings, x0 generation
//
//  - Provide a suite of instantiaters that work well across the spectrum of standard problems 

// pub trait Synthesizer {
//     fn synthesize<C: ComplexScalar>(&self, target: InstantiationTarget<C>) -> QuditCircuit<C>;
// }

// pub struct QSearchSynthesizer<C: ComplexScalar, I: Instantiater<C>> {
//     // layer gen
//     instantiater: I,
// }

// Use Cases
// let synthesizer = SomeSynthesizer(..., BoxedInstantiater<C>, ...);
// synthesizer.synthesize(unitary.into())


// impl<C: ComplexScalar, I: Instantiater<C>> Synthesizer for QSearchSynthesizer<C, I> {
//     fn synthesize(&self, target: InstantiationTarget<C>) -> QuditCircuit<C> {
//         // let mut circuit = self.initial_circuit();
//         //
//         // while not synthesized:
//         //      self.extend(&circuit)
//         //      let result = self.instantiater.instantiate(circuit, target)
//         //      if self.success(circuit, result) {
//         //          return self.finalize(circuit, result)
//         //      }
//         todo!()
//     }
// }

//
// CPUParallelInstantiater(num_cores)
// CPUSampleInstantiater
// GPUSampleInstantiater
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
