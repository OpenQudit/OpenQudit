use qudit_core::{ComplexScalar, HasParams, RealScalar};
use qudit_expr::UnitaryExpressionGenerator;
use qudit_expr::{StateExpression, StateSystemExpression, TensorExpression};
use qudit_gates::Gate;
use qudit_core::state::StateVector;
use bit_set::BitSet;
// use qudit_tree::ExpressionTree;

use crate::circuit::QuditCircuit;

pub enum OperationType {
    Gate,
    Subcircuit,
    Control,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Operation {
    Gate(Gate),
    ProjectiveMeasurement(TensorExpression, BitSet), // TODO: Switch to kraus operator/andor POVM
    TerminatingMeasurement(StateSystemExpression, BitSet),
    ClassicallyControlled(Gate, BitSet),
    Initialization(StateExpression),
    // TODO: Delay
    // Subcircuit(ImmutableCircuit),
    Reset,
    Barrier,
}

impl Operation {
    pub fn name(&self) -> String {
        match self {
            Operation::Gate(gate) => gate.name().to_string(),
            Operation::ProjectiveMeasurement(t, _) => format!("ProjectiveMeasurement({})", t.name()),
            Operation::TerminatingMeasurement(s, _) => format!("TerminatingMeasurement({})", s.name()),
            Operation::ClassicallyControlled(g, _) => format!("ClassicallyControlled({})", g.name()),
            Operation::Initialization(s) => format!("Initialization({})", s.name()),
            Operation::Reset => "Reset".to_string(),
            Operation::Barrier => "Barrier".to_string(),
        }
    }

    pub fn discriminant(&self) -> usize {
        match self {
            Operation::Gate(_) => 0,
            Operation::ProjectiveMeasurement(_, _) => 1,
            Operation::TerminatingMeasurement(_, _) => 2,
            Operation::ClassicallyControlled(_, _) => 3,
            Operation::Initialization(_) => 4,
            Operation::Reset => 5,
            Operation::Barrier => 6,
        }
    }
}


#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy, PartialOrd, Ord)]
pub struct OperationReference(u64);

impl OperationReference {
    #[inline(always)]
    pub(super) fn new(id: u64) -> Self {
        OperationReference(id)
    }
    // pub fn new<C: ComplexScalar>(circuit: &mut QuditCircuit<C>, op: Operation) -> OperationReference {
    //     OperationReference(circuit.expression_set.insert(op))
    //     // match op {
    //         // Operation::Gate(gate) => {
    //         //     let index = circuit.expression_set.insert(gate.gen_expr().to_tensor_expression());
    //         //     OperationReference((index as u64) << 2 | 0b00)
    //         // },
    //         // _ => todo!(),
    //         // Operation::Subcircuit(subcircuit) => {
    //         //     let index = circuit.subcircuits.insert_full(subcircuit).0;
    //         //     OperationReference((index as u64) << 2 | 0b01)
    //         // },
    //     // }
    // }

    // pub fn op_type(&self) -> OperationType {
    //     match self.0 & 0b11 {
    //         0b00 => OperationType::Gate,
    //         0b01 => todo!(),
    //         // 0b01 => OperationType::Subcircuit,
    //         0b10 => OperationType::Control,
    //         _ => panic!("Invalid operation type"),
    //     }
    // }

    pub fn index(&self) -> usize {
        (self.0 >> 2) as usize
    }

    pub fn dereference<'a, C: ComplexScalar>(&'a self, circuit: &'a QuditCircuit<C>) -> &'a Operation {
        match circuit.expression_set.get(&self) {
            Some(s) => s,
            None => panic!("Unable to find expression."),
        }
        // let index = (self.0 >> 2) as usize;
        // match self.0 & 0b11 {
        //     // 0b00 => Operation::Gate(circuit.gates[index].clone()),
        //     _ => todo!(),
        //     // 0b01 => todo!(),
        //     // 0b01 => Operation::Subcircuit(circuit.subcircuits[index].clone()),
        //     // 0b10 => Operation::Control(match self.0 >> 2 {
        //     //     0 => ControlOperation::Measurement,
        //     //     1 => ControlOperation::Reset,
        //     //     2 => ControlOperation::Barrier,
        //     //     _ => panic!("Invalid control operation discriminant"),
        //     // }),
        //     _ => panic!("Invalid operation type"),
        // }
    }
}

impl From<u64> for OperationReference {
    fn from(value: u64) -> OperationReference {
        OperationReference(value)
    }
}

impl HasParams for Operation {
    fn num_params(&self) -> usize {
        match self {
            Operation::Gate(gate) => gate.num_params(),
            Operation::TerminatingMeasurement(s, _) => s.variables.len(),
            Operation::ClassicallyControlled(s, _) => s.num_params(),
            _ => todo!()
            // Operation::Subcircuit(subcircuit) => subcircuit.num_params(),
            // Operation::Control(_) => 0,
        }
    }
}

