use std::ops::Range;

use crate::RealScalar;

/// A parameterized object.
pub trait HasParams {
    /// The number of parameters this object requires.
    fn num_params(&self) -> usize;
}

/// A bounded, parameterized object.
pub trait HasBounds<R: RealScalar>: HasParams {
    /// The bounds for each variable of the function
    fn bounds(&self) -> Vec<Range<R>>;
}

/// A periodic, parameterized object
pub trait HasPeriods<R: RealScalar>: HasParams {
    /// The core period for each variable of the function
    fn periods(&self) -> Vec<Range<R>>;
}

// #[cfg(test)]
// pub mod strategies {
//     use std::ops::Range;

//     use proptest::prelude::*;

//     use super::BoundedFn;

//     pub fn params(num_params: usize) -> impl Strategy<Value = Vec<f64>> {
//         prop::collection::vec(
//             prop::num::f64::POSITIVE
//                 | prop::num::f64::NEGATIVE
//                 | prop::num::f64::NORMAL
//                 | prop::num::f64::SUBNORMAL
//                 | prop::num::f64::ZERO,
//             num_params,
//         )
//     }

//     pub fn params_with_bounds(
//         bounds: Vec<Range<f64>>,
//     ) -> impl Strategy<Value = Vec<f64>> {
//         bounds
//     }

//     pub fn arbitrary_with_params_strategy<F: Clone + BoundedFn + Arbitrary>(
//     ) -> impl Strategy<Value = (F, Vec<f64>)> {
//         any::<F>().prop_flat_map(|f| (Just(f.clone()), f.get_bounds()))
//     }
// }
