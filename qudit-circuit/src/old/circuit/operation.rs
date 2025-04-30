// use faer_core::MatMut;

use super::instruction::Instruction;
use super::CircuitLocation;
// use crate::math::unitary::DifferentiableUnitaryFn;
// use crate::math::unitary::DoublyDifferentiableUnitaryFn;
// use crate::math::unitary::UnitaryFn;
// use crate::math::unitary::UnitaryGradient;
// use crate::math::unitary::UnitaryHessian;
use crate::math::BoundedFn;
use crate::math::ComplexScalar;
use crate::math::Function;
use crate::QuditRadices;
use crate::QuditSystem;

#[derive(PartialEq, Clone, Debug)]
pub struct Operation<C: ComplexScalar> {
    inst: Instruction,
    location: CircuitLocation,
    params: Vec<C::Re>,
}

impl<C: ComplexScalar> Operation<C> {
    pub fn new(
        inst: Instruction,
        location: CircuitLocation,
        params: Vec<C::Re>,
    ) -> Operation<C> {
        match &inst {
            Instruction::Quantum(g) => {
                assert_eq!(location.get_num_qudits(), g.get_num_qudits());
                assert!(location.get_num_clbits() == 0);
                assert!(
                    params.is_empty() || params.len() == g.get_num_params()
                );
            },
            Instruction::ClassicallyControlled(g) => {
                assert_eq!(location.get_num_qudits(), g.get_num_qudits());
                assert!(location.get_num_clbits() > 0);
                assert!(
                    params.is_empty() || params.len() == g.get_num_params()
                );
            },
            Instruction::UnitaryExpression(exp) => {
                assert_eq!(location.get_num_qudits(), exp.get_num_qudits());
                assert!(location.get_num_clbits() == 0);
                assert!(
                    params.is_empty() || params.len() == exp.get_num_params()
                );
            },
            Instruction::Measurement => {
                assert!(location.get_num_qudits() != 0);
                assert!(location.get_num_clbits() != 0);
                assert!(params.is_empty());
            },
            Instruction::Reset => {
                assert!(params.is_empty());
            },
        }
        Operation {
            inst,
            location,
            params,
        }
    }

    pub fn instruction(&self) -> &Instruction {
        &self.inst
    }

    pub fn location(&self) -> &CircuitLocation {
        &self.location
    }

    pub fn params(&self) -> &Vec<C::Re> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut Vec<C::Re> {
        &mut self.params
    }

    pub fn get_num_clbits(&self) -> usize {
        self.location.get_num_clbits()
    }

    pub fn set_params(&mut self, params: &[C::Re]) {
        if cfg!(debug_assertions) {
            assert_eq!(params.len(), self.params.len());
        }
        self.params.copy_from_slice(params);
    }
}

impl<C: ComplexScalar> QuditSystem for Operation<C> {
    fn get_num_qudits(&self) -> usize {
        self.location.get_num_qudits()
    }

    fn get_radices(&self) -> QuditRadices {
        self.inst.get_radices()
    }
}

impl<C: ComplexScalar> Function for Operation<C> {
    fn get_num_params(&self) -> usize {
        self.inst.get_num_params()
    }
}

impl<C: ComplexScalar> BoundedFn for Operation<C> {
    fn get_bounds(&self) -> Vec<std::ops::Range<f64>> {
        self.inst.get_bounds()
    }
}

// impl<C: ComplexScalar> UnitaryFn<C> for Operation<C> {
//     fn write_unitary(&self, params: &[C::Re], mut utry: &mut MatMut<C>) {
//         // TODO: Move instruction match statements to instruction.rs
//         match &self.inst {
//             Instruction::Quantum(g) => g.write_unitary(if params.is_empty() { &self.params } else { &params }, &mut utry),
//             Instruction::UnitaryExpression(exp) => exp.write_unitary(if params.is_empty() { &self.params } else { &params }, &mut utry),
//             _ => panic!("Cannot get unitary for non-unitary instruction."),
//         };
//     }
// }
//
// impl<C: ComplexScalar> DifferentiableUnitaryFn<C> for Operation<C> {
//     fn write_gradient(&self, params: &[C::Re], grad: &mut UnitaryGradient<C>) {
//         match &self.inst {
//             Instruction::Quantum(g) => g.write_gradient(if params.is_empty() { &self.params } else { &params }, grad),
//             Instruction::UnitaryExpression(exp) => exp.write_gradient(if params.is_empty() { &self.params } else { &params }, grad),
//             _ => panic!("Cannot get gradient for non-unitary instruction."),
//         }
//     }
//
//     fn write_unitary_and_gradient(
//         &self,
//         params: &[C::Re],
//         utry: &mut MatMut<C>,
//         grad: &mut UnitaryGradient<C>,
//     ) {
//         match &self.inst {
//             Instruction::Quantum(g) => g.write_unitary_and_gradient(if params.is_empty() { &self.params } else { &params }, utry, grad),
//             Instruction::UnitaryExpression(exp) => exp.write_unitary_and_gradient(if params.is_empty() { &self.params } else { &params }, utry, grad),
//             _ => panic!("Cannot get gradient for non-unitary instruction."),
//         }
//     }
// }
//
// impl<C: ComplexScalar> DoublyDifferentiableUnitaryFn<C> for Operation<C> {
//     fn write_hessian(&self, params: &[C::Re], hess: &mut UnitaryHessian<C>) {
//         match &self.inst {
//             Instruction::Quantum(g) => g.write_hessian(if params.is_empty() { &self.params } else { &params }, hess),
//             Instruction::UnitaryExpression(exp) => exp.write_hessian(if params.is_empty() { &self.params } else { &params }, hess),
//             _ => panic!("Cannot get hessian for non-unitary instruction."),
//         }
//     }
//
//     // TODO: implement other hessian write methods
// }
