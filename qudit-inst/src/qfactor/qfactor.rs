use qudit_circuit::QuditCircuit; 
use qudit_core::{c64, ComplexScalar, QuditSystem, unitary::UnitaryMatrix, HasParams};
use crate::{Instantiater, InstantiationTarget, DataMap, InstantiationResult};
use crate::qfactor::unitary_builder::UnitaryBuilder;
use qudit_tensor::{compile_network, TNVM};
use qudit_expr::{FUNCTION, GRADIENT, UnitaryExpressionGenerator};
use std::ops::Deref;
use faer_traits::num_traits::ToPrimitive;

//------------------------------------------------------------------------------------------------------
//-START------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------

fn circ_to_utry<C: ComplexScalar>(circuit: &QuditCircuit<C>) -> UnitaryMatrix<C> {
    let circ_code = compile_network(circuit.to_tensor_network());
    let mut circ_tnvm = TNVM::<C, FUNCTION>::new(&circ_code);
    let circ_result = circ_tnvm.evaluate::<FUNCTION>(circuit.params());
    let circ_mat = circ_result.get_fn_result().unpack_matrix();
    let circ_utry = UnitaryMatrix::<C>::new(circuit.radices(), circ_mat.to_owned());
    return circ_utry;
}

//------------------------------------------------------------------------------------------------------
//-END------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct QFactorInstantiator {
    diff_tol_a: f64,
    diff_tol_r: f64,
    dist_tol: f64,
    max_iters: usize,
    min_iters: usize,
    slowdown_factor: f64,
    reinit_delay: usize,
}

impl Default for QFactorInstantiator {
    fn default() -> Self {
        QFactorInstantiator {
            diff_tol_a: 1e-12,
            diff_tol_r: 1e-6,
            dist_tol: 1e-16,
            max_iters: 100000,
            min_iters: 1000,
            slowdown_factor: 0.0,
            reinit_delay: 40,
        }
    }
}

impl QFactorInstantiator {
    pub fn new(
        diff_tol_a: Option<f64>,
        diff_tol_r: Option<f64>,
        dist_tol: Option<f64>,
        max_iters: Option<usize>,
        min_iters: Option<usize>,
        slowdown_factor: Option<f64>,
        reinit_delay: Option<usize>,
    ) -> Self {
        QFactorInstantiator {
            diff_tol_a: diff_tol_a.unwrap_or(1e-12),
            diff_tol_r: diff_tol_r.unwrap_or(1e-6),
            dist_tol: dist_tol.unwrap_or(1e-16),
            max_iters: max_iters.unwrap_or(100000),
            min_iters: min_iters.unwrap_or(1000),
            slowdown_factor: slowdown_factor.unwrap_or(0.0),
            reinit_delay: reinit_delay.unwrap_or(40),
        }
    }

    pub fn initialize_circuit_tensor<C: ComplexScalar>(
        &self,
        circuit: &QuditCircuit<C>,
        target: &UnitaryMatrix<C>,
    ) -> UnitaryBuilder<C> {
        let radixes: Vec<usize> = circuit.radices().deref().iter().map(|&x| x as usize).collect();
        let entire_circ_location: Vec<usize> = (0..circuit.num_qudits()).collect();

        let mut unitary_builder = UnitaryBuilder::new(circuit.num_qudits(), radixes);
        
        unitary_builder.apply_right(target, &entire_circ_location, true);

        let circ_utry = circ_to_utry(circuit);

        unitary_builder.apply_right(&circ_utry, &entire_circ_location, false,);

        return unitary_builder;
    }

    pub fn sweep_circuit<C: ComplexScalar>(
        &self,
        unitary_builder: &mut UnitaryBuilder<C>,
        circuit: &mut QuditCircuit<C>
    ) {
        // Start by looping backwards
        for inst_index in (0..circuit.get_num_insts()).rev() {
            let mut op_utry = circuit.get_op_utry(inst_index);
            let op_location = circuit.get_op_location(inst_index);
            unitary_builder.apply_right(&op_utry, &op_location, true);

            if circuit.get_op_num_params(inst_index) != 0 {
                let env = unitary_builder.calc_env_matrix(&op_location);
                let opt_params = circuit.get_op_opt_params(inst_index, &env, self.slowdown_factor);
                circuit.update_inst_params(inst_index, &opt_params);
                op_utry = circuit.get_op_utry(inst_index);
            }

            unitary_builder.apply_left(&op_utry, &op_location, false);
        }

        // Reset for new loop through all the gates the opposite order
        for inst_index in 0..circuit.get_num_insts() {
            let mut op_utry = circuit.get_op_utry(inst_index);
            let op_location = circuit.get_op_location(inst_index);
            unitary_builder.apply_left(&op_utry, &op_location, true);

            if circuit.get_op_num_params(inst_index) != 0 {
                let env = unitary_builder.calc_env_matrix(&op_location);
                let opt_params = circuit.get_op_opt_params(inst_index, &env, self.slowdown_factor);
                circuit.update_inst_params(inst_index, &opt_params);
                op_utry = circuit.get_op_utry(inst_index);
            }

            unitary_builder.apply_right(&op_utry, &op_location, false);
        }
    }
}

impl<'a, C: ComplexScalar> Instantiater<'a, C> for QFactorInstantiator {
    fn instantiate(
        &self,
        circuit: &'a QuditCircuit<C>,
        target: &'a InstantiationTarget<C>,
        data: &'a DataMap,
    ) -> InstantiationResult<C> {
        let circuit_copy_mut_ref = &mut circuit.clone();
        let target_utry_ref: &UnitaryMatrix<C>;
        match target {
            InstantiationTarget::UnitaryMatrix(utry) => {
                target_utry_ref = utry;
            }
            _ => {todo!()}
        }

        // if data.len() != circuit.num_params() {
        //     return InstantiationResult::<C>::new(
        //         Some(circuit.params().to_vec()),
        //         None,
        //         1,
        //         Some(format!("Incorrect number of parameters in data for the QFactor instantiator, expected {}, got {}",
        //         circuit.num_params(), data.len()))
        //     );
        // }
        // circuit_copy_mut_ref.set_params(data);

        let mut unitary_builder = self.initialize_circuit_tensor(circuit_copy_mut_ref, target_utry_ref);
        let mut dist1 = 0.0f64;
        let mut dist2 = 0.0f64;
        let mut it = 0usize;
        println!("Panic Test - Circuit is I: {:?}", circ_to_utry(circuit_copy_mut_ref));
        println!("Panic Test - Target is I: {:?}", target_utry_ref);
        println!("Panic Test - Init Builder is I: {:?}", unitary_builder.get_utry());
        loop {
            if it > self.min_iters {
                let diff_tol = self.diff_tol_a + self.diff_tol_r * dist1.abs();
                if (dist1 - dist2).abs() <= diff_tol {
                    break;
                }

                if it > self.max_iters {
                    break;
                }
            }

            it += 1;

            self.sweep_circuit(&mut unitary_builder, circuit_copy_mut_ref);
            println!("Panic Test - Builder Rep is Utry: {:?}", unitary_builder.get_utry());

            dist2 = dist1;
            dist1 = unitary_builder.get_utry().trace().abs().to_f64().unwrap();
            dist1 = 1. - (dist1 / circuit_copy_mut_ref.dimension() as f64);

            if dist1 < self.dist_tol {
                return InstantiationResult::<C>::new(
                    Some(circuit_copy_mut_ref.params().to_vec()),
                    None,
                    0,
                    Some("Instantiation complete with `dist < dist_tol`".to_string())
                );
            }

            if it % self.reinit_delay == 0 {
                unitary_builder = self.initialize_circuit_tensor(circuit_copy_mut_ref, target_utry_ref)
            }
        }

        return InstantiationResult::<C>::new(
            Some(circuit_copy_mut_ref.params().to_vec()),
            None,
            0,
            Some("Instantiation complete with `dist >= dist_tol`".to_string())
        );
    }
}










#[cfg(test)]
mod tests {
    use crate::qfactor::qfactor::QFactorInstantiator;
    use crate::{Instantiater, InstantiationTarget, DataMap};
    use qudit_circuit::QuditCircuit;
    use qudit_core::{c64, ComplexScalar, QuditSystem, unitary::UnitaryMatrix, matrix::Mat};
    use qudit_gates::{Gate, VariableUnitary};
    use std::f64::consts::PI;
    use num_complex::ComplexFloat;
    use faer_traits::num_traits::ToPrimitive;
    use qudit_core::HasParams;
    use qudit_expr::UnitaryExpressionGenerator;
    use std::ops::Deref;

    #[test]
    fn test_instantiator_creation() {
        let instantiator = QFactorInstantiator::default();
        
        assert_eq!(instantiator.diff_tol_a, 1e-12);
        assert_eq!(instantiator.max_iters, 100000);
        
        let custom_instantiator = QFactorInstantiator::new(
            Some(1e-3), Some(1e-2), Some(1e-4), Some(20), Some(5), Some(0.5), Some(5)
        );
        
        assert_eq!(custom_instantiator.diff_tol_a, 1e-3);
        assert_eq!(custom_instantiator.max_iters, 20);
    }

    #[test]
    fn test_initialization_basic() {
        let circuit: QuditCircuit<c64> = QuditCircuit::pure([2]);
        let target = UnitaryMatrix::identity([2]);
        
        let instantiator = QFactorInstantiator::default();
        let unitary_builder = instantiator.initialize_circuit_tensor(&circuit, &target);
    }

    #[test]
    fn test_empty_circuit_only() {
        let circuit: QuditCircuit<c64> = QuditCircuit::pure([2]);
        let target = UnitaryMatrix::identity([2]);
        
        let instantiator = QFactorInstantiator::new(
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(2),
            Some(1),
            Some(0.0),
            Some(100)
        );
        
        let result = instantiator.instantiate(
            &circuit, 
            &InstantiationTarget::UnitaryMatrix(target), 
            &DataMap::new()
        );
        
        println!("Status: {}", result.status);
        println!("Message: {:?}", result.message);
        
        assert_eq!(result.params.unwrap().len(), 0);
    }

}