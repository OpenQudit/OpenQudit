use crate::QuditSystem;

use super::DensityMatrix;

pub trait DensityMatrixMap: QuditSystem {
    fn evolve(&self, matrix: &DensityMatrix) -> DensityMatrix;

    fn zero_evolve(&self) -> DensityMatrix {
        self.evolve(&DensityMatrix::zero(self.get_radices()))
    }

    fn many_evolve(&self, matrices: &[DensityMatrix]) -> Vec<DensityMatrix> {
        matrices.iter().evolve(|v| self.map(v)).collect()
    }
}