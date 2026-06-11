use crate::cycle::CycleList;
use crate::cycle::CycleId;
use crate::instruction::InstructionId;
use crate::operation::OperationSet;
use crate::Result;
use crate::param::ParameterVector;
use crate::param::IntoArgumentList;
use crate::wire::Wire;
use crate::wire::WireList;
use qudit_core::Radices;
use qudit_core::HybridSystem;
use rustc_hash::FxHashMap;

/// A quantum circuit that can be defined with qudits and classical bits.
///
/// The circuit is internally represented as a list of cycles, where each
/// cycles represents an abstract moment in time. In each cycle, instructions
/// can direct operations to be applied to quantum and/or classical wires.
/// This data structure holds invariant that there never will be an empty
/// cycle. As a result, cycles can be removed automatically from anywhere in
/// the circuit, when operations are removed. However, cycles are identified
/// by a persistent identifier, which never changes and enables fast O(1) lookup.
/// This additionally enables instruction identifiers that will always point
/// to their correct instruction, no matter how the circuit changes underneath.
#[derive(Clone)]
pub struct QuditCircuit {
    /// The QuditRadices object that describes the quantum dimension of the circuit.
    qudit_radices: Radices,

    /// The QuditRadices object that describes the classical dimension of the circuit.
    dit_radices: Radices,

    /// All instructions in the circuit stored in cycles.
    cycles: CycleList,

    /// The set of cached operations in the circuit.
    operations: OperationSet,

    /// The stored parameters of the circuit.
    params: ParameterVector,

    /// A pointer to the first operation on each wire.
    front: FxHashMap<Wire, CycleId>,

    /// A pointer to the last operation on each wire.
    rear: FxHashMap<Wire, CycleId>,
}

mod append;
mod collection;
mod construct;
mod dag;
mod evaluation;
mod grouping;
mod iteration;
mod properties;
mod trait_impls;

#[cfg(feature = "python")]
mod python;

pub use append::InternableOperation;

#[cfg(feature = "python")]
pub use python::PyQuditCircuit;

#[cfg(test)]
mod tests {
    use super::*;
    use qudit_core::c32;
    use qudit_expr::GRADIENT;
    use qudit_expr::library::Controlled;
    use qudit_expr::library::U3Gate;
    use qudit_expr::library::XGate;

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

    #[test]
    fn build_qsearch_thin_step_circuit_test() {
        build_qsearch_thin_step_circuit(3);
        build_qsearch_thin_step_circuit(4);
        build_qsearch_thin_step_circuit(5);
        build_qsearch_thin_step_circuit(6);
        build_qsearch_thin_step_circuit(7);
    }

    #[test]
    fn build_qsearch_thin_step_circuit_to_tensor_test() {
        const N: usize = 3;
        let circ = build_qsearch_thin_step_circuit(N);
        let network = circ.to_tensor_network();
        let code = qudit_tensor::compile_network(network);
        let mut tnvm =
            qudit_tensor::TNVM::<c32, GRADIENT>::new(&code, Some(&circ.params.const_map()));
        let result = tnvm.evaluate::<GRADIENT>(&[1.7; (3 * N) + (6 * (N - 1) * 2)]);
        let _unitary = result.get_fn_result().unpack_matrix();
    }
}

