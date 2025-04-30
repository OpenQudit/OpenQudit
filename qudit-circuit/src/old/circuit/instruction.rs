use std::hash::Hash;

use crate::math::BoundedFn;
use crate::math::Function;
use crate::sim::ExpressionTree;
use crate::Gate;
use crate::QuditRadices;
use crate::QuditSystem;

#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum Instruction {
    Quantum(Gate), // TODO: Add two versions: static and dynamic
    ClassicallyControlled(Gate),
    Measurement,
    Reset,
    UnitaryExpression(ExpressionTree),
}

impl Instruction {
    pub fn is_quantum(&self) -> bool {
        if let Instruction::Quantum(_) = self {
            true
        } else {
            false
        }
    }

    pub fn get_gate(&self) -> &Gate {
        match self {
            Instruction::Quantum(g) => g,
            Instruction::ClassicallyControlled(g) => g,
            _ => panic!("Instruction is not a quantum gate."),
        }
    }
}

impl QuditSystem for Instruction {
    fn get_radices(&self) -> QuditRadices {
        match self {
            Instruction::Quantum(g) => g.get_radices(),
            Instruction::ClassicallyControlled(g) => g.get_radices(),
            Instruction::Measurement => {
                panic!("Measurement does not have radices.")
            },
            Instruction::Reset => panic!("Reset does not have radices."),
            Instruction::UnitaryExpression(exp) => exp.get_radices(),
        }
    }
}

impl Function for Instruction {
    fn get_num_params(&self) -> usize {
        match self {
            Instruction::Quantum(g) => g.get_num_params(),
            Instruction::ClassicallyControlled(g) => g.get_num_params(),
            Instruction::Measurement => 0,
            Instruction::Reset => 0,
            Instruction::UnitaryExpression(exp) => exp.get_num_params(),
        }
    }
}

impl BoundedFn for Instruction {
    fn get_bounds(&self) -> Vec<std::ops::Range<f64>> {
        match self {
            Instruction::Quantum(g) => g.get_bounds(),
            Instruction::ClassicallyControlled(g) => g.get_bounds(),
            Instruction::Measurement => vec![],
            Instruction::Reset => vec![],
            Instruction::UnitaryExpression(exp) => exp.get_bounds(),
        }
    }
}

impl From<Gate> for Instruction {
    fn from(gate: Gate) -> Self {
        Instruction::Quantum(gate)
    }
}

// impl<C: ComplexScalar> Eq for Instruction<C> {}
//
// impl<C: ComplexScalar> Hash for Instruction<C> {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         match self {
//             Instruction::Quantum(g) => {
//                 state.write_u8(0);
//                 g.hash(state);
//             },
//             Instruction::ClassicallyControlled(g) => {
//                 state.write_u8(1);
//                 g.hash(state);
//             },
//             Instruction::Measurement => state.write_u8(2),
//             Instruction::Reset => state.write_u8(3),
//             Instruction::UnitaryExpression(exp) => {
//                 state.write_u8(4);
//                 exp.hash(state);
//             },
//         }
//     }
// }
