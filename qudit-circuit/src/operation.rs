use qudit_core::{ComplexScalar, HasParams, RealScalar};
use qudit_expr::StateExpression;
use qudit_gates::Gate;
use qudit_core::state::StateVector;
// use qudit_tree::ExpressionTree;

use crate::circuit::QuditCircuit;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ControlOperation {
    Measurement,
    Reset,
    Barrier,
}

impl ControlOperation {
    pub fn discriminant(&self) -> u64 {
        match self {
            ControlOperation::Measurement => 0,
            ControlOperation::Reset => 1,
            ControlOperation::Barrier => 2,
        }
    }
}

pub enum OperationType {
    Gate,
    Subcircuit,
    Control,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Operation {
    Gate(Gate),
    // ProjectiveMeasurement(Operators),
    // TerminatingMeasurement(Basis),
    // ClassicallyControlled(Gate, Activation),
    Initialization(StateExpression),
    // TODO: Delay
    // Subcircuit(ImmutableCircuit),
    Reset,
    Barrier,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Copy)]
pub struct OperationReference(u64);

impl OperationReference {
    pub fn new<C: ComplexScalar>(circuit: &mut QuditCircuit<C>, op: Operation) -> OperationReference {
        match op {
            Operation::Gate(gate) => {
                let index = circuit.gates.insert_full(gate).0;
                OperationReference((index as u64) << 2 | 0b00)
            },
            // Operation::Subcircuit(subcircuit) => {
            //     let index = circuit.subcircuits.insert_full(subcircuit).0;
            //     OperationReference((index as u64) << 2 | 0b01)
            // },
            Operation::Control(control_op) => {
                OperationReference(control_op.discriminant() << 2 | 0b10)
            }, 
        }
    }

    pub fn op_type(&self) -> OperationType {
        match self.0 & 0b11 {
            0b00 => OperationType::Gate,
            0b01 => todo!(),
            // 0b01 => OperationType::Subcircuit,
            0b10 => OperationType::Control,
            _ => panic!("Invalid operation type"),
        }
    }

    pub fn index(&self) -> usize {
        (self.0 >> 2) as usize
    }

    pub fn dereference<C: ComplexScalar>(&self, circuit: &QuditCircuit<C>) -> Operation {
        let index = (self.0 >> 2) as usize;
        match self.0 & 0b11 {
            0b00 => Operation::Gate(circuit.gates[index].clone()),
            0b01 => todo!(),
            // 0b01 => Operation::Subcircuit(circuit.subcircuits[index].clone()),
            0b10 => Operation::Control(match self.0 >> 2 {
                0 => ControlOperation::Measurement,
                1 => ControlOperation::Reset,
                2 => ControlOperation::Barrier,
                _ => panic!("Invalid control operation discriminant"),
            }),
            _ => panic!("Invalid operation type"),
        }
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
            // Operation::Subcircuit(subcircuit) => subcircuit.num_params(),
            Operation::Control(_) => 0,
        }
    }
}

