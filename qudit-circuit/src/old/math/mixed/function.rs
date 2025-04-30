use crate::{math::Function, QuditSystem};

use super::DensityMatrix;

pub trait DensityMatrixMapFn: QuditSystem + Function {
    fn evolve(&self, matix: &DensityMatrix, params: &[f64]) -> DensityMatrix;

    fn zero_evolve(&self, params: &[f64]) -> DensityMatrix {
        self.evolve(&DensityMatrix::zero(self.get_radices()), params)
    }

    fn many_evolve(&self, matices: &[DensityMatrix], params: &[f64]) -> Vec<DensityMatrix> {
        matices.iter().map(|v| self.evolve(v, params)).collect()
    }
}

pub trait DifferentiableDensityMatrixMapFn: DensityMatrixMapFn {
    fn get_gradient(&self, matix: &DensityMatrix, params: &[f64]) -> DensityMatrix;
    fn get_evolve_and_gradient(&self, matix: &DensityMatrix, params: &[f64]) -> (DensityMatrix, DensityMatrix) {
        (self.evolve(matix, params), self.get_gradient(matix, params))
    }
    // TODO: many_evolve, zero__evolve, name-clashing?
}

pub trait DoublyDifferentiableDensityMatrixMapFn: DifferentiableDensityMatrixMapFn {
    fn get_hessian(&self, matix: &DensityMatrix, params: &[f64]) -> DensityMatrix;
}
