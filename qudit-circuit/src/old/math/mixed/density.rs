use std::ops::Deref;

use crate::{QuditRadices, QuditSystem};

pub struct DensityMatrix {
    data: Array2<c64>,
    radices: QuditRadices,
}

impl DensityMatrix {
    pub fn new(radices: QuditRadices, data: Array2<c64>) -> Self {
        Self { data, radices }
    }

    pub fn zero(radices: QuditRadices) -> Self {
        let mut state = Array2::zeros(radices.get_dimension());
        state[[0, 0]] = r!(1.0);
        Self::new(radices, state)
    }
}

impl QuditSystem for DensityMatrix {
    fn get_radices(&self) -> QuditRadices {
        self.radices.clone()
    }
}

impl Deref for DensityMatrix {
    type Target = Array2<c64>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}