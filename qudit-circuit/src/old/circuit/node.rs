use std::vec;

use super::operation::Operation;
use crate::{math::ComplexScalar, QuditSystem};

/// OperationNode encapsulates an active operation inside a circuit.
///
/// This is an internal data structure and should never be handled externally.
#[derive(PartialEq, Clone, Debug)]
pub(super) struct OperationNode<C: ComplexScalar> {
    /// The instruction that this operation represents.
    pub(super) op: Operation<C>,

    /// The physical cycle index this operation is in.
    pub(super) cycle: usize,

    /// The cycle indices of the next operations for each qudit.
    pub(super) qnext: Vec<Option<usize>>,
    pub(super) qprev: Vec<Option<usize>>,
    pub(super) cnext: Vec<Option<usize>>,
    pub(super) cprev: Vec<Option<usize>>,
}

impl<C: ComplexScalar> OperationNode<C> {
    pub(super) fn new(op: Operation<C>, cycle: usize) -> OperationNode<C> {
        OperationNode {
            qnext: vec![None; op.get_num_qudits()],
            qprev: vec![None; op.get_num_qudits()],
            cnext: vec![None; op.get_num_clbits()],
            cprev: vec![None; op.get_num_clbits()],
            op,
            cycle,
        }
    }
}

// impl QuditSystem for OperationNode {
//     fn get_radices(&self) -> QuditRadices {
//         self.inst.get_gate().get_radices()
//     }
// }

// impl Function for OperationNode {
//     fn get_num_params(&self) -> usize {
//         match self.inst.as_ref() {
//             Operation::Quantum(gate) => gate.get_num_params(),
//             Operation::Classical(_) => 0,
//             Operation::CircuitOp(circuit) => circuit.get_num_params(),
//         }
//     }
// }

// impl UnitaryFn for OperationNode {
//     fn get_unitary(&self, params: &[f64]) -> UnitaryMatrix {
//         match self.inst.as_ref() {
//             Operation::Quantum(gate) => gate.get_unitary(&params),
//             Operation::Classical(_) => {
//                 panic!("Cannot calculate unitary of a non-quantum
// operation.")             }
//             Operation::CircuitOp(circuit) => circuit.get_unitary(&params),
//         }
//     }
// }

// impl DifferentiableUnitaryFn for OperationNode {
//     fn get_gradient(&self, params: &[f64]) -> Array3<c64> {
//         match self.inst.as_ref() {
//             Operation::Quantum(gate) => gate.get_gradient(&params),
//             Operation::Classical(_) => {
//                 panic!("Cannot calculate gradient of a non-quantum
// operation.")             }
//             Operation::CircuitOp(circuit) => circuit.get_gradient(&params),
//         }
//     }

//     fn get_unitary_and_gradient(&self, params: &[f64]) -> (UnitaryMatrix,
// Array3<c64>) {         match self.inst.as_ref() {
//             Operation::Quantum(gate) =>
// gate.get_unitary_and_gradient(&params),             Operation::Classical(_)
// => {                 panic!("Cannot calculate unitary of a non-quantum
// operation.")             }
//             Operation::CircuitOp(circuit) =>
// circuit.get_unitary_and_gradient(&params),         }
//     }
// }
