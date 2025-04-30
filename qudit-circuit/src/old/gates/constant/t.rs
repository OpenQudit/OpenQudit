use std::ops::Range;

use faer_core::MatMut;
use gate_macros::ConstantUnitaryFunction;
use gate_macros::SimpleSingleQubitGate;

use super::super::NamedGate;
use crate::math::matrix::MatGradMut;
use crate::math::matrix::MatHessMut;
use crate::math::unitary::DifferentiableUnitaryFn;
use crate::math::unitary::DoublyDifferentiableUnitaryFn;
use crate::math::unitary::UnitaryFn;
use crate::math::BoundedFn;
use crate::math::ComplexScalar;
use crate::math::Function;
use crate::radices;
use crate::QuditRadices;
use crate::QuditSystem;

/// The one-qubit T gate.
/// TODO: Generalize: <https://arxiv.org/pdf/1206.1598.pdf>, <https://arxiv.org/pdf/2204.00552.pdf>
#[derive(
    Hash,
    PartialEq,
    Eq,
    Clone,
    Debug,
    SimpleSingleQubitGate,
    ConstantUnitaryFunction,
)]
pub struct TGate;

impl<C: ComplexScalar> UnitaryFn<C> for TGate {
    #[inline]
    fn write_unitary(&self, _params: &[C::Re], utry: &mut MatMut<C>) {
        utry.write(1, 1, C::complex(0.7071067811865476, 0.7071067811865475));
    }
}
