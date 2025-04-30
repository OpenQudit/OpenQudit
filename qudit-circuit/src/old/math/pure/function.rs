use super::StateVector;
use crate::math::Function;
use crate::QuditSystem;

pub trait StateVectorMapFn: QuditSystem + Function {
    fn map(&self, vector: &StateVector, params: &[f64]) -> StateVector;

    fn zero_map(&self, params: &[f64]) -> StateVector {
        self.map(&StateVector::zero(self.get_radices()), params)
    }

    fn many_map(
        &self,
        vectors: &[StateVector],
        params: &[f64],
    ) -> Vec<StateVector> {
        vectors.iter().map(|v| self.map(v, params)).collect()
    }
}

pub trait DifferentiableStateVectorMapFn: StateVectorMapFn {
    fn get_gradient(&self, vector: &StateVector, params: &[f64])
        -> StateVector;
    fn get_map_and_gradient(
        &self,
        vector: &StateVector,
        params: &[f64],
    ) -> (StateVector, StateVector) {
        (self.map(vector, params), self.get_gradient(vector, params))
    }
    // Todo: many_map, zero__map?
}

pub trait DoublyDifferentiableStateVectorMapFn:
    DifferentiableStateVectorMapFn
{
    fn get_hessian(&self, vector: &StateVector, params: &[f64]) -> StateVector;
}
