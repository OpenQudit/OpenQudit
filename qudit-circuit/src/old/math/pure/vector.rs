// use std::ops::Deref;

// use faer_core::ColRef;
use faer_core::c64;
use faer_core::Col;

use crate::QuditRadices;
use crate::QuditSystem;

pub struct StateVector {
    _data: Col<c64>,
    radices: QuditRadices,
}

impl StateVector {
    pub fn new(radices: QuditRadices, _data: Col<c64>) -> Self {
        Self { _data, radices }
    }

    pub fn zero(radices: QuditRadices) -> Self {
        let mut state = Col::zeros(radices.get_dimension());
        state[0] = c64::new(1.0, 0.0);
        Self::new(radices, state)
    }
}

impl QuditSystem for StateVector {
    fn get_radices(&self) -> QuditRadices {
        self.radices.clone()
    }
}

// impl<'a> Deref for &'a StateVector
// {
//     type Target = ColRef<'a, c64>;

//     fn deref(&'a self) -> Self::Target { self.data.as_ref() }
// }
