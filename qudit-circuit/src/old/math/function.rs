use std::ops::Range;

/// Abstract type for a function
pub trait Function {
    fn get_num_params(&self) -> usize;
}

pub trait BoundedFn {
    fn get_bounds(&self) -> Vec<Range<f64>>;

    // TODO: Determine if this should be merged into Function with a default
    // implementation of (NAN, NAN)xN

    // TODO: Consider changing name to periodic, since UnitaryFn's should not
    // error on out of bounds values
}

#[cfg(test)]
pub mod strategies {
    use std::ops::Range;

    use proptest::prelude::*;

    use super::BoundedFn;

    pub fn params(num_params: usize) -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(
            prop::num::f64::POSITIVE
                | prop::num::f64::NEGATIVE
                | prop::num::f64::NORMAL
                | prop::num::f64::SUBNORMAL
                | prop::num::f64::ZERO,
            num_params,
        )
    }

    pub fn params_with_bounds(
        bounds: Vec<Range<f64>>,
    ) -> impl Strategy<Value = Vec<f64>> {
        bounds
    }

    pub fn arbitrary_with_params_strategy<F: Clone + BoundedFn + Arbitrary>(
    ) -> impl Strategy<Value = (F, Vec<f64>)> {
        any::<F>().prop_flat_map(|f| (Just(f.clone()), f.get_bounds()))
    }
}
