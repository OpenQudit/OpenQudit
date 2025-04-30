use std::ops::Range;

use faer_core::MatMut;
use gate_macros::ConstantUnitaryFunction;
use gate_macros::SimpleSingleQuditGate;

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

/// An identity or No-OP gate.
#[derive(
    Hash,
    PartialEq,
    Eq,
    Clone,
    Debug,
    SimpleSingleQuditGate,
    ConstantUnitaryFunction,
)]
pub struct IGate {
    radix: usize,
}

impl<C: ComplexScalar> UnitaryFn<C> for IGate {
    #[inline]
    fn write_unitary(&self, _params: &[C::Re], _utry: &mut MatMut<C>) {}
}
