mod instantiater;
pub mod numerical;
mod qfactor;
mod result;
mod target;

pub use instantiater::DataMap;
pub use instantiater::Instantiater;
pub use result::InstantiationResult;
pub use target::InstantiationTarget;

#[cfg(feature = "python")]
pub use instantiater::python::PyInstantiater;

////////////////////////////////////////////////////////////////////////
/// Python Module.
////////////////////////////////////////////////////////////////////////
#[cfg(feature = "python")]
pub(crate) mod python {
    use pyo3::prelude::{Bound, PyAnyMethods, PyModule, PyModuleMethods, PyResult};

    /// A trait for objects that can register importables with a PyO3 module.
    pub struct PyInstantiationRegistrar {
        /// The registration function
        pub func: fn(parent_module: &Bound<'_, PyModule>) -> PyResult<()>,
    }

    inventory::collect!(PyInstantiationRegistrar);

    /// Registers the Circuit submodule with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        let submodule = PyModule::new(parent_module.py(), "instantiation")?;

        for registrar in inventory::iter::<PyInstantiationRegistrar> {
            (registrar.func)(&submodule)?;
        }

        parent_module.add_submodule(&submodule)?;
        parent_module
            .py()
            .import("sys")?
            .getattr("modules")?
            .set_item("openqudit.instantiation", submodule)?;

        Ok(())
    }

    inventory::submit!(qudit_core::PyRegistrar { func: register });
}
////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::MinimizingInstantiater;
    use crate::numerical::functions::HSProblem;
    // use crate::numerical::initializers::GreedyFurthestPoint;
    use crate::numerical::initializers::Uniform;
    // use crate::numerical::initializers::Zeros;
    use crate::numerical::minimizers::LM;
    use crate::numerical::runners::MultiStartRunner;
    use faer::mat;
    use qudit_circuit::ArgumentList;
    use qudit_circuit::QuditCircuit;
    use qudit_core::UnitaryMatrix;
    // use qudit_core::c32;
    use qudit_core::c64;
    use qudit_expr::BraSystemExpression;
    // use qudit_expr::FUNCTION;
    // use qudit_expr::GRADIENT;
    use qudit_expr::UnitaryExpression;
    use qudit_expr::library::Controlled;
    use qudit_expr::library::PGate;
    use qudit_expr::library::U3Gate;
    use qudit_expr::library::XGate;
    // use qudit_tensor::TNVM;
    use std::sync::Arc;

    #[allow(dead_code)]
    pub fn build_qsearch_thin_step_circuit(n: usize) -> QuditCircuit {
        let block_expr = U3Gate()
            .otimes(U3Gate())
            .dot(Controlled(XGate(2), [2].into(), None));
        let mut circ = QuditCircuit::pure(vec![2; n]);
        for i in 0..n {
            circ.append(U3Gate(), [i], None);
        }
        for _ in 0..2 {
            for i in 0..(n - 1) {
                circ.append(block_expr.clone(), [i, i + 1], None);
            }
        }
        circ
    }

    #[allow(dead_code)]
    pub fn build_qutrit_thin_step_circuit(n: usize) -> QuditCircuit {
        let csum = UnitaryExpression::new(
            "CSUM() {
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
        }",
        );

        let block_expr = PGate(3).otimes(PGate(3)).dot(csum);
        let mut circ = QuditCircuit::pure(vec![3; n]);
        for i in 0..n {
            circ.append(PGate(3), [i], None);
        }
        for _ in 0..2 {
            for i in 0..(n - 1) {
                circ.append(block_expr.clone(), [i, i + 1], None);
            }
        }
        circ
    }

    #[allow(dead_code)]
    pub fn build_dynamic_circuit() -> QuditCircuit {
        let mut circ: QuditCircuit = QuditCircuit::new([2, 2, 2, 2], [2, 2]);

        circ.zero_initialize([1, 2]);

        for i in 0..4 {
            circ.append(qudit_expr::library::U3Gate(), [i], None);
        }

        let block_expr = U3Gate()
            .otimes(U3Gate())
            .dot(Controlled(XGate(2), [2].into(), None));
        circ.append(block_expr.clone(), [1, 2], None);
        circ.append(
            block_expr.clone(),
            [0, 1],
            ArgumentList::new(vec![None::<f64>.try_into().unwrap(); 6]),
        );
        circ.append(block_expr.clone(), [2, 3], None);

        let one_qubit_basis_measurement = BraSystemExpression::new(
            "OneQMeasure() {
            [
                [[ 1, 0, ]],
                [[ 0, 1, ]],
            ]
        }",
        );

        circ.append(one_qubit_basis_measurement.clone(), ([1], [0]), None);
        circ.append(one_qubit_basis_measurement, ([2], [1]), None);

        let u3_u3 = U3Gate().otimes(U3Gate());
        circ.append(
            UnitaryExpression::classically_multiplex(&[&u3_u3, &u3_u3, &u3_u3, &u3_u3], &[2, 2]),
            ([0, 3], [0, 1]),
            None,
        );

        //////// START CNOT TELEPORTATION
        // circ.zero_initialize([1, 2]);
        // circ.append(Gate::H(), [1], None);
        // circ.append(Gate::CX(), [1, 2], None);
        // circ.append(Gate::CX(), [0, 1], None);
        // circ.append(Gate::CX(), [2, 3], None);
        // circ.append(Gate::H(), [2], None);

        // let one_qubit_basis_measurement = BraSystemExpression::new("OneQMeasure() {
        //     [
        //         [[ 1, 0, ]],
        //         [[ 0, 1, ]],
        //     ]
        // }");

        // circ.append_parameterized(Operation::TerminatingMeasurement(one_qubit_basis_measurement.clone()), ([1], [0]));
        // circ.append_parameterized(Operation::TerminatingMeasurement(one_qubit_basis_measurement), ([2], [1]));

        // circ.append(Gate::X(2).generate_expression().classically_control(&[1], &[2]), ([3], [0]), None);
        // circ.append(Gate::Z(2).generate_expression().classically_control(&[1], &[2]), ([0], [1]), None);
        //////// END CNOT TELEPORTATION
        circ
    }

    #[test]
    fn test_lm_minimization_simple() {
        // create simple circuit
        let circ = build_dynamic_circuit();
        // let circ = build_qsearch_thin_step_circuit(3);

        // sample target
        // let network = circ.to_tensor_network();
        // let code = qudit_tensor::compile_network(network);
        // let mut tnvm = qudit_tensor::TNVM::<c32, GRADIENT>::new(&code);
        // let result = tnvm.evaluate::<FUNCTION>(&vec![1.7; circ.num_params()]).get_fn_result().unpack_matrix();
        // let target_utry = UnitaryMatrix::new(circ.radices(), result.to_owned());
        // let target_utry = UnitaryMatrix::new([2], mat![
        //     [c64::new(1.0, 0.0), c64::new(0.0, 0.0)],
        //     [c64::new(0.0, 0.0), c64::new(1.0, 0.0)],
        // ]);
        let target_utry = UnitaryMatrix::new(
            [2, 2],
            mat![
                [
                    c64::new(1.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0)
                ],
                [
                    c64::new(0.0, 0.0),
                    c64::new(1.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0)
                ],
                [
                    c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(1.0, 0.0)
                ],
                [
                    c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(1.0, 0.0),
                    c64::new(0.0, 0.0)
                ],
            ],
        );
        let target = InstantiationTarget::UnitaryMatrix(target_utry);

        // build instantiater
        let minimizer = LM::default();
        // let initializer = Zeros::default();
        let initializer = Uniform::default();
        // let initializer = GreedyFurthestPoint::default();
        let runner = MultiStartRunner {
            minimizer,
            guess_generator: initializer,
            num_starts: 16,
        };
        let instantiater = MinimizingInstantiater::<_, HSProblem<f64>>::new(runner);
        let data = std::collections::HashMap::new();

        // call instantiater
        let circ = Arc::new(circ);
        let target = Arc::new(target);
        let data = Arc::new(data);
        let _result = instantiater.instantiate(circ.clone(), target.clone(), data.clone());
        let mut success_times = vec![];
        let mut failure_times = vec![];
        let n = 1000;
        for _ in 0..n {
            let now = std::time::Instant::now();
            let result = instantiater.instantiate(circ.clone(), target.clone(), data.clone());
            let elapsed = now.elapsed();
            if let Some(f) = result.fun {
                // dbg!(&f);
                // dbg!(result.message);
                if f < 1e-4 {
                    success_times.push(elapsed);
                } else {
                    failure_times.push(elapsed);
                }
            }
        }
        println!("Number of successes: {:?}", success_times.len());
        println!("Number of failures: {:?}", failure_times.len());
        if !success_times.is_empty() {
            let average_success_time = success_times.iter().cloned().sum::<std::time::Duration>()
                / (success_times.len() as u32);
            println!("Average success time: {:?}", average_success_time);
        }
        if !failure_times.is_empty() {
            let average_failure_time = failure_times.iter().cloned().sum::<std::time::Duration>()
                / (failure_times.len() as u32);
            println!("Average failure time: {:?}", average_failure_time);
        }
        let average_time = success_times
            .iter()
            .chain(failure_times.iter())
            .cloned()
            .sum::<std::time::Duration>()
            / n;
        println!("Average overall time: {:?}", average_time);
    }
}
