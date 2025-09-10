mod instantiater;
pub mod numerical;
mod qfactor;
mod result;
mod target;

pub use instantiater::DataMap;
pub use instantiater::Instantiater;
use qudit_expr::ExpressionGenerator;
use qudit_expr::UnitaryExpression;
pub use result::InstantiationResult;
pub use target::InstantiationTarget;


#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;
    use qudit_core::radices;
    use qudit_core::c32;
    use qudit_core::c64;
    use qudit_core::unitary::UnitaryMatrix;
    use qudit_core::QuditRadices;
    use qudit_circuit::QuditCircuit;
    use qudit_circuit::CircuitLocation;
    use qudit_expr::UnitaryExpression;
    use qudit_gates::Gate;
    use qudit_tensor::TNVM;
    use qudit_expr::FUNCTION;
    use qudit_expr::GRADIENT;
    use crate::numerical::functions::HSProblem;
    use crate::numerical::initializers::GreedyFurthestPoint;
    use crate::numerical::runners::MultiStartRunner;
    use crate::numerical::minimizers::LM;
    use crate::numerical::initializers::Zeros;
    use crate::numerical::initializers::Uniform;
    use crate::numerical::MinimizingInstantiater;
    use qudit_core::QuditSystem;
    use qudit_expr::ExpressionGenerator;
    use qudit_circuit::Operation;
    use qudit_circuit::ControlState;

    pub fn build_qsearch_thin_step_circuit(n: usize) -> QuditCircuit {
        let block_expr = Gate::U3().generate_expression().otimes(&Gate::U3().generate_expression()).dot(&Gate::CX().generate_expression());
        let mut circ = QuditCircuit::pure(vec![2; n]);
        for i in 0..n {
            circ.append_parameterized(Gate::U3(), [i]);
        }
        for _ in 0..2 {
            for i in 0..(n - 1) {
                circ.append_parameterized(block_expr.clone(), [i, i+1]);
            }
        }
        circ
    } 

    struct CSUM {}
    impl ExpressionGenerator for CSUM {
        type ExpressionType = UnitaryExpression;
        fn generate_expression(&self) -> Self::ExpressionType {    
            UnitaryExpression::new("CSUM() {
                [
                    [ 1, 0, 0, 0, 0, 0, 0, 0, 0 ],
                    [ 0, 1, 0, 0, 0, 0, 0, 0, 0 ],
                    [ 0, 0, 1, 0, 0, 0, 0, 0, 0 ],
                    [ 0, 0, 0, 0, 0, 1, 0, 0, 0 ],
                    [ 0, 0, 0, 1, 0, 0, 0, 0, 0 ],
                    [ 0, 0, 0, 0, 1, 0, 0, 0, 0 ],
                    [ 0, 0, 0, 0, 0, 0, 0, 1, 0 ],
                    [ 0, 0, 0, 0, 0, 0, 0, 0, 1 ],
                    [ 0, 0, 0, 0, 0, 0, 1, 0, 0 ],
                ]
            }")
        }
    }

    pub fn build_qutrit_thin_step_circuit(n: usize) -> QuditCircuit {
        let block_expr = Gate::P(3).generate_expression().otimes(&Gate::P(3).generate_expression()).dot(CSUM{}.generate_expression());
        let mut circ = QuditCircuit::pure(vec![3; n]);
        for i in 0..n {
            circ.append_parameterized(Gate::P(3), [i]);
        }
        for _ in 0..2 {
            for i in 0..(n - 1) {
                circ.append_parameterized(Gate::Expression(block_expr.clone()), [i, i+1]);
            }
        }
        circ
    }

    // pub fn build_dynamic_circuit() -> QuditCircuit<c64> {
    //     let mut circ: QuditCircuit<c64> = QuditCircuit::new([2, 2, 2, 2], [2, 2]);

    //     circ.zero_initialize([1, 2]);

    //     for i in 0..4 {
    //         circ.append_uninit_gate(Gate::U3(), [i]);
    //     }

    //     // TODO: add otimes, dot, and friends to gate methods.
    //     let block_expr = Gate::U3().gen_expr().otimes(Gate::U3().gen_expr()).dot(Gate::CX().gen_expr());
    //     let block_gate = Gate::Expression(block_expr);
    //     circ.append_uninit_gate(block_gate.clone(), [0, 1]);
    //     circ.append_uninit_gate(block_gate.clone(), [2, 3]);
    //     circ.append_uninit_gate(block_gate.clone(), [1, 2]);

    //     let two_qubit_basis_measurement = StateSystemExpression::new("TwoQMeasure() {
    //         [
    //             [[ 1, 0, 0, 0 ]],
    //             [[ 0, 1, 0, 0 ]],
    //             [[ 0, 0, 1, 0 ]],
    //             [[ 0, 0, 0, 1 ]],
    //         ]
    //     }");
    //     circ.append_instruction(Instruction::new(Operation::TerminatingMeasurement(two_qubit_basis_measurement), ([1, 2], [0,1]), vec![]));
    //     // circ.z_basis_measure([1, 2], [0, 1]);
    //     // circ.classically_multiplex(Gate::U3().otimes(Gate::U3()), [0, 3], [0, 1]);

    //     let cs1 = ControlState::from_binary_state([0,0]);
    //     circ.uninit_classically_control(Gate::U3(), cs1.clone(), ([0], [0, 1]));
    //     circ.uninit_classically_control(Gate::U3(), cs1.clone(), ([3], [0, 1]));

    //     let cs2 = ControlState::from_binary_state([0,1]);
    //     circ.uninit_classically_control(Gate::U3(), cs2.clone(), ([0], [0, 1]));
    //     circ.uninit_classically_control(Gate::U3(), cs2.clone(), ([3], [0, 1]));

    //     let cs3 = ControlState::from_binary_state([1,0]);
    //     circ.uninit_classically_control(Gate::U3(), cs3.clone(), ([0], [0, 1]));
    //     circ.uninit_classically_control(Gate::U3(), cs3.clone(), ([3], [0, 1]));

    //     let cs4 = ControlState::from_binary_state([1,1]);
    //     circ.uninit_classically_control(Gate::U3(), cs4.clone(), ([0], [0, 1]));
    //     circ.uninit_classically_control(Gate::U3(), cs4.clone(), ([3], [0, 1]));

    //     circ
    // }


    #[test]
    fn test_lm_minimization_simple() {
        // create simple circuit
        // let circ = build_dynamic_circuit();
        let circ = build_qsearch_thin_step_circuit(3);

        // sample target
        let network = circ.to_tensor_network();
        let code = qudit_tensor::compile_network(network);
        let mut tnvm = qudit_tensor::TNVM::<c32, GRADIENT>::new(&code);
        let result = tnvm.evaluate::<FUNCTION>(&vec![1.7; circ.num_params()]).get_fn_result().unpack_matrix();
        let target_utry = UnitaryMatrix::new(circ.radices(), result.to_owned());
        // let target_utry = UnitaryMatrix::new([2, 2], mat![
        //     [c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0)],
        //     [c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0)],
        //     [c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(1.0, 0.0)],
        //     [c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(1.0, 0.0), c64::new(0.0, 0.0)],
        // ]);
        let target = InstantiationTarget::UnitaryMatrix(target_utry);

        // build instantiater
        let minimizer = LM::default();
        // let initializer = Zeros::default();
        let initializer = Uniform::default();
        // let initializer = GreedyFurthestPoint::default();
        let runner = MultiStartRunner { minimizer, guess_generator: initializer, num_starts: 16 };
        let instantiater = MinimizingInstantiater::<_, HSProblem<f32>>::new(runner);
        let data = std::collections::HashMap::new();

        // call instantiater
        let result = instantiater.instantiate(&circ, &target, &data);
        let mut success_times = vec![];
        let mut failure_times = vec![];
        for _ in 0..1000 {
            let now = std::time::Instant::now();
            let result = instantiater.instantiate(&circ, &target, &data);
            let elapsed = now.elapsed();
            if let Some(f) = result.fun {
                if f < 1e-4 {
                    success_times.push(elapsed);
                } else {
                    failure_times.push(elapsed);
                }
            }
        }
        println!("Number of successes: {:?}", success_times.len());
        println!("Number of failures: {:?}", failure_times.len());
        let average_success_time = success_times.iter().cloned().sum::<std::time::Duration>()/(success_times.len() as u32);
        if failure_times.len() != 0 {
            let average_failure_time = failure_times.iter().cloned().sum::<std::time::Duration>()/(failure_times.len() as u32);
            println!("Average failure time: {:?}", average_failure_time); 
        }
        let average_time = success_times.iter().chain(failure_times.iter()).cloned().sum::<std::time::Duration>()/1000;
        println!("Average success time: {:?}", average_success_time); 
        println!("Average overall time: {:?}", average_time); 
        // println!("Instantiation took: {:?}", elapsed/100);
    }
}

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
