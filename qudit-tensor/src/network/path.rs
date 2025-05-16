use std::collections::BTreeSet;

use bit_set::BitSet;

use super::SubNetwork;


#[derive(Debug, Clone)]
pub struct ContractionPath {
    pub cost: usize,
    pub open_indices: BTreeSet<usize>,
    pub path: Vec<usize>,
    pub subnetwork: SubNetwork,
    pub param_indices: BitSet,
}


impl ContractionPath {
    pub fn calculate_cost(T_a: &Self, T_b: &Self) -> usize {
        let total_indices = T_a.open_indices
            .union(&T_b.open_indices)
            .copied()
            .collect::<Vec<_>>();
        T_a.cost + T_b.cost + usize::pow(2, total_indices.len() as u32)
    }

    pub fn contract(&self, other: &Self) -> Self {
        let subnetwork = self.subnetwork | other.subnetwork;
        let open_indices = self.open_indices
            .symmetric_difference(&other.open_indices)
            .copied()
            .collect();
        let cost = Self::calculate_cost(self, other);
        let path = self.path
            .iter()
            .chain(other.path.iter())
            .copied()
            .chain(std::iter::once(usize::MAX))
            .collect();
        let mut param_indices = self.param_indices.clone();
        param_indices.union_with(&other.param_indices);
        ContractionPath {
            cost,
            open_indices,
            path,
            subnetwork,
            param_indices,
        }
    }

    pub fn trivial(idx: usize, indices: &[usize], param_indices: BitSet) -> Self {
        let open_indices = indices.iter().copied().collect();
        let path = vec![idx];
        let cost = 0;
        let subnetwork = 1 << idx;
        ContractionPath {
            cost,
            open_indices,
            path,
            subnetwork,
            param_indices,
        }
    }
}
