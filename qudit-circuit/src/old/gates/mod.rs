use std::collections::BTreeMap;
use std::hash::Hash;
use std::ops::Range;

use faer_core::MatMut;
use gate_macros::enum_dispatch;

use crate::math::matrix::MatGradMut;
use crate::math::matrix::MatHessMut;
use crate::math::unitary::DifferentiableUnitaryFn;
use crate::math::unitary::DoublyDifferentiableUnitaryFn;
use crate::math::unitary::UnitaryFn;
use crate::math::unitary::UnitaryGradient;
use crate::math::unitary::UnitaryHessian;
use crate::math::unitary::UnitaryMatrix;
use crate::math::BoundedFn;
use crate::math::ComplexScalar;
use crate::math::Function;
use crate::radices;
use crate::QuditRadices;
use crate::QuditSystem;

pub mod named;
pub use named::NamedGate;

// TODO: Explore custom sin, cos methods for gates
// TODO: Explore simd with pulp in gate calculations

pub mod constant {
    pub mod h;
    pub mod i;
    pub mod swap;
    pub mod t;
    pub mod x;
    pub mod z;
}
pub mod parameterized {
    pub mod p;
    pub mod u3;
}

pub mod composed {
    pub mod control;
    pub mod dagger;
}

pub use composed::control::ControlledGate;
pub use composed::dagger::DaggerGate;
pub use constant::h::HGate;
pub use constant::i::IGate;
pub use constant::swap::SwapGate;
pub use constant::t::TGate;
pub use constant::x::XGate;
pub use constant::z::ZGate;
pub use parameterized::p::PGate;
pub use parameterized::u3::U3Gate;

#[enum_dispatch]
pub enum Gate {
    // Constant Gates
    IGate,
    HGate,
    SwapGate,
    TGate,
    XGate,
    ZGate,

    // Parameterized Gates
    PGate,
    U3Gate,

    // Composed Gates
    ControlledGate,
    DaggerGate,
}

static mut CP_CACHE: Option<Gate> = None;
static mut H_CACHE: BTreeMap<usize, Gate> = BTreeMap::new();

impl Gate {
    #[allow(missing_docs)]
    #[allow(non_snake_case)]
    #[inline]
    pub fn I(radix: usize) -> Self {
        Gate::IGate(IGate::new(radix))
    }

    #[allow(missing_docs)]
    #[allow(non_snake_case)]
    #[inline]
    pub fn H(radix: usize) -> Self {
        unsafe {
            if let Some(gate) = H_CACHE.get(&radix) {
                gate.clone()
            } else {
                let gate = Gate::HGate(HGate::new(radix));
                H_CACHE.insert(radix, gate.clone());
                gate
            }
        }
        // Gate::HGate(HGate::new(radix))
    }

    #[allow(missing_docs)]
    #[allow(non_snake_case)]
    #[inline]
    pub fn Swap(radix: usize) -> Self {
        Gate::SwapGate(SwapGate::new(radix))
    }

    #[allow(missing_docs)]
    #[allow(non_snake_case)]
    #[inline]
    pub fn T() -> Self {
        Gate::TGate(TGate {})
    }

    #[allow(missing_docs)]
    #[allow(non_snake_case)]
    #[inline]
    pub fn Tdg() -> Self {
        Gate::DaggerGate(DaggerGate::new(Gate::T()))
    }

    #[allow(missing_docs)]
    #[allow(non_snake_case)]
    #[inline]
    pub fn X(radix: usize) -> Self {
        Gate::XGate(XGate::new(radix))
    }

    #[allow(missing_docs)]
    #[allow(non_snake_case)]
    #[inline]
    pub fn Z(radix: usize) -> Self {
        Gate::ZGate(ZGate::new(radix))
    }

    #[allow(missing_docs)]
    #[allow(non_snake_case)]
    #[inline]
    pub fn P(radix: usize) -> Self {
        Gate::PGate(PGate::new(radix))
    }

    #[allow(missing_docs)]
    #[allow(non_snake_case)]
    #[inline]
    pub fn U3() -> Self {
        Gate::U3Gate(U3Gate {})
    }

    #[allow(missing_docs)]
    #[allow(non_snake_case)]
    #[inline]
    pub fn Dagger(gate: Gate) -> Self {
        Gate::DaggerGate(DaggerGate::new(gate))
    }

    #[allow(missing_docs)]
    #[allow(non_snake_case)]
    #[inline]
    pub fn Controlled(
        gate: Gate,
        control_radices: QuditRadices,
        control_levels: Vec<Vec<usize>>,
    ) -> Self {
        Gate::ControlledGate(ControlledGate::new(
            gate,
            control_radices,
            control_levels,
        ))
    }

    #[allow(missing_docs)]
    #[allow(non_snake_case)]
    #[inline]
    pub fn CZ() -> Self {
        Gate::ControlledGate(ControlledGate::new(
            Gate::Z(2),
            radices![2],
            vec![vec![1]],
        ))
    }

    #[allow(missing_docs)]
    #[allow(non_snake_case)]
    #[inline]
    pub fn CX() -> Self {
        Gate::ControlledGate(ControlledGate::new(
            Gate::X(2),
            radices![2],
            vec![vec![1]],
        ))
    }

    // gate_shorthand!{
    //     CP() => {
    //         Gate::ControlledGate(ControlledGate::new(Gate::P(2), radices![2],
    // vec![vec![1]]))     }
    // }
    // Make these static references

    #[allow(missing_docs)]
    #[allow(non_snake_case)]
    #[inline]
    pub fn CP() -> Self {
        unsafe {
            if let Some(gate) = &CP_CACHE {
                gate.clone()
            } else {
                let gate = Gate::ControlledGate(ControlledGate::new(
                    Gate::P(2),
                    radices![2],
                    vec![vec![1]],
                ));
                CP_CACHE = Some(gate.clone());
                gate
            }
        }
        // Gate::ControlledGate(ControlledGate::new(Gate::P(2), radices![2],
        // vec![vec![1]]))
    }
}
