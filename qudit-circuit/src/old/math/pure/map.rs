use super::StateVector;
use crate::QuditSystem;

pub trait StateVectorMap: QuditSystem {
    fn map(&self, vector: &StateVector) -> StateVector;

    fn zero_map(&self) -> StateVector {
        self.map(&StateVector::zero(self.get_radices()))
    }

    fn many_map(&self, vectors: &[StateVector]) -> Vec<StateVector> {
        vectors.iter().map(|v| self.map(v)).collect()
    }
}
