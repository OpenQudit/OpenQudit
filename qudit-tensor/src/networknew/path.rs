use std::collections::BTreeSet;

use crate::networknew::index::{IndexId, IndexSize};

/// A bitmask representing a set of tensors in a network.
pub type SubNetwork = u64;

/// Represents a path of tensor contractions, tracking the state and cost of the operations.
///
/// This struct is used in tensor network contraction algorithms to keep track of the
/// accumulated cost, the resulting open indices, the output indices, the sequence of
/// contractions, and the set of original tensors involved in a contraction path.
#[derive(Debug, Clone)]
pub struct ContractionPath {
    /// The cost of all contractions in this path.
    pub cost: usize,

    /// The open indices of the resulting tensor after performing all contractions in the path.
    pub indices: BTreeSet<(IndexId, IndexSize)>,

    /// These indices are a subset of `indices` and are marked as output for the final network.
    /// These are not to be summed over, and if appear on both sides of a contraction should be
    /// treated as a batch index.
    pub output_indices: BTreeSet<IndexId>,

    /// The tensor path data. Stored as a stack, with usize::MAX marking contractions.
    pub path: Vec<usize>,

    /// The set of original tensors accounted for in this path.
    pub subnetwork: SubNetwork,
}


impl ContractionPath {
    /// Calculates the computational cost of contracting two `ContractionPath`s.
    ///
    /// The cost is determined by summing the costs of the two input paths
    /// and adding the product of the sizes of all unique indices present
    /// in either path after the contraction. This typically represents the
    /// complexity of the resulting matrix multiplication.
    ///
    /// Arguments:
    /// * `T_a` - The first `ContractionPath`.
    /// * `T_b` - The second `ContractionPath`.
    ///
    /// Returns:
    /// The calculated cost as a `usize`.
    pub fn calculate_cost(T_a: &Self, T_b: &Self) -> usize {
        let total_indices = T_a.indices
            .union(&T_b.indices)
            .copied()
            .collect::<Vec<_>>();
        T_a.cost + T_b.cost + total_indices.iter().map(|(_, size)| size).product::<usize>()
    }

    /// Contracts two `ContractionPath`s into a new one.
    ///
    /// This method combines the information from two existing contraction paths
    /// to form a new path representing their contraction. It updates the subnetwork,
    /// output indices, open indices, cost, and the contraction path sequence.
    ///
    /// Arguments:
    /// * `other` - The `ContractionPath` to contract with `self`.
    ///
    /// Returns:
    /// A new `ContractionPath` representing the result of the contraction.
    pub fn contract(&self, other: &Self) -> Self {
        let subnetwork = self.subnetwork | other.subnetwork;
        let output_indices: BTreeSet<IndexId> = self.output_indices.union(&other.output_indices).copied().collect();
        let indices = self.indices.iter().chain(&other.indices).filter(|idx| {
            output_indices.contains(&idx.0) || !self.indices.contains(idx) || !other.indices.contains(idx)
        }).copied()
        .collect();
        let cost = Self::calculate_cost(self, other);
        let path = self.path
            .iter()
            .chain(other.path.iter())
            .copied()
            .chain(std::iter::once(usize::MAX))
            .collect();
        ContractionPath {
            cost,
            indices,
            output_indices,
            path,
            subnetwork,
        }
    }

    /// Creates a trivial `ContractionPath` for a single tensor.
    ///
    /// This is the base case for building contraction paths. It represents
    /// a path where no contractions have yet occurred, and it simply
    /// accounts for a single initial tensor.
    ///
    /// Arguments:
    /// * `idx` - The identifier of the single tensor this path represents.
    /// * `indices` - A slice of (IndexId, IndexSize) tuples representing
    ///   the open indices of this trivial tensor.
    /// * `output_indices` - A slice of IndexId values representing which
    ///   of the `indices` are considered output indices.
    ///
    /// Returns:
    /// A new `ContractionPath` representing the trivial case of a single tensor.
    pub fn trivial(idx: IndexId, indices: &[(IndexId, IndexSize)], output_indices: &[IndexId]) -> Self {
        let indices = indices.iter().copied().collect();
        let output_indices = output_indices.iter().copied().collect();
        let path = vec![idx];
        let cost = 0;
        let subnetwork = 1 << idx;
        ContractionPath {
            cost,
            indices,
            output_indices,
            path,
            subnetwork,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    // Helper function to create a simple ContractionPath for testing
    fn create_test_path(
        idx: IndexId,
        indices: &[(IndexId, IndexSize)],
        output_indices: &[IndexId],
        cost: usize,
        subnetwork: SubNetwork,
        path: Vec<usize>,
    ) -> ContractionPath {
        ContractionPath {
            cost,
            indices: indices.iter().copied().collect(),
            output_indices: output_indices.iter().copied().collect(),
            path,
            subnetwork,
        }
    }

    #[test]
    fn test_trivial_path() {
        let indices = vec![(0, 2), (1, 3)];
        let output_indices = vec![0];
        let path = ContractionPath::trivial(0, &indices, &output_indices);

        assert_eq!(path.cost, 0);
        assert_eq!(path.indices, indices.iter().copied().collect());
        assert_eq!(path.output_indices, output_indices.iter().copied().collect());
        assert_eq!(path.path, vec![0]);
        assert_eq!(path.subnetwork, 1 << 0);
    }

    #[test]
    fn test_calculate_cost() {
        // Path A: cost 5, indices (0,2), (1,3)
        let path_a = create_test_path(0, &[(0, 2), (1, 3)], &[], 5, 1 << 0, vec![0]);
        // Path B: cost 10, indices (1,3), (2,4)
        let path_b = create_test_path(1, &[(1, 3), (2, 4)], &[], 10, 1 << 1, vec![1]);

        // Expected total indices: (0,2), (1,3), (2,4)
        // Product of sizes: 2 * 3 * 4 = 24
        // Expected cost: 5 + 10 + 24 = 39
        let expected_cost = path_a.cost + path_b.cost + (2 * 3 * 4);
        assert_eq!(ContractionPath::calculate_cost(&path_a, &path_b), expected_cost);

        // Test with no common indices
        let path_c = create_test_path(2, &[(3, 5), (4, 6)], &[], 2, 1 << 2, vec![2]);
        // Expected total indices: (0,2), (1,3), (3,5), (4,6)
        // Product of sizes: 2 * 3 * 5 * 6 = 180
        // Expected cost: 5 + 2 + 180 = 187
        let expected_cost_no_common = path_a.cost + path_c.cost + (2 * 3 * 5 * 6);
        assert_eq!(ContractionPath::calculate_cost(&path_a, &path_c), expected_cost_no_common);

        // Test with empty paths (unlikely in real use, but good for robustness)
        let path_empty_a = create_test_path(0, &[], &[], 0, 0, vec![]);
        let path_empty_b = create_test_path(1, &[], &[], 0, 0, vec![]);
        assert_eq!(ContractionPath::calculate_cost(&path_empty_a, &path_empty_b), 0);
    }

    #[test]
    fn test_contract_paths() {
        // Path A: cost 5, indices (0,2), (1,3), output {0}
        let path_a = create_test_path(0, &[(0, 2), (1, 3)], &[0], 5, 1 << 0, vec![0]);
        // Path B: cost 10, indices (1,3), (2,4), output {2}
        let path_b = create_test_path(1, &[(1, 3), (2, 4)], &[2], 10, 1 << 1, vec![1]);

        let contracted_path = path_a.contract(&path_b);

        // Expected subnetwork: 1 << 0 | 1 << 1 = 3
        assert_eq!(contracted_path.subnetwork, (1 << 0) | (1 << 1));

        // Expected output indices: {0, 2}
        let mut expected_output_indices = BTreeSet::new();
        expected_output_indices.insert(0);
        expected_output_indices.insert(2);
        assert_eq!(contracted_path.output_indices, expected_output_indices);

        // Expected indices:
        // (0,2) from A, (1,3) from A, (1,3) from B, (2,4) from B
        // output_indices {0, 2}
        // index (0,2) is output, so included.
        // index (1,3) is common, not output, so summed over, NOT included.
        // index (2,4) is output, so included.
        let mut expected_indices = BTreeSet::new();
        expected_indices.insert((0, 2)); // from A, is output
        expected_indices.insert((2, 4)); // from B, is output
        assert_eq!(contracted_path.indices, expected_indices);

        // Expected cost: Calculated from calculate_cost
        // Total indices: (0,2), (1,3), (2,4)
        // Product of sizes: 2 * 3 * 4 = 24
        // Cost: 5 + 10 + 24 = 39
        assert_eq!(contracted_path.cost, 39);

        // Expected path: [0, 1, usize::MAX]
        assert_eq!(contracted_path.path, vec![0, 1, usize::MAX]);

        // Test with no common indices, no output indices
        let path_c = create_test_path(2, &[(3, 5), (4, 6)], &[], 2, 1 << 2, vec![2]);
        let contracted_path_ac = path_a.contract(&path_c);
        assert_eq!(contracted_path_ac.subnetwork, (1 << 0) | (1 << 2));
        assert_eq!(contracted_path_ac.output_indices, path_a.output_indices); // Only path_a has output_indices
        
        let mut expected_indices_ac = BTreeSet::new();
        expected_indices_ac.insert((0, 2)); // from A, is output
        expected_indices_ac.insert((1, 3)); // from A
        expected_indices_ac.insert((3, 5)); // from C
        expected_indices_ac.insert((4, 6)); // from C
        assert_eq!(contracted_path_ac.indices, expected_indices_ac);

        // Cost: 5 (A) + 2 (C) + (2*3*5*6) = 7 + 180 = 187
        assert_eq!(contracted_path_ac.cost, 187);
        assert_eq!(contracted_path_ac.path, vec![0, 2, usize::MAX]);
    }

    #[test]
    fn test_contract_with_shared_output_index() {
        // Path A: cost 5, indices (0,2), (1,3), output {0}
        let path_a = create_test_path(0, &[(0, 2), (1, 3)], &[0], 5, 1 << 0, vec![0]);
        // Path B: cost 10, indices (0,2), (2,4), output {0}
        // Index 0 is common and is an output index for both paths.
        let path_b = create_test_path(1, &[(0, 2), (2, 4)], &[0], 10, 1 << 1, vec![1]);

        let contracted_path = path_a.contract(&path_b);

        // Expected subnetwork: 1 << 0 | 1 << 1 = 3
        assert_eq!(contracted_path.subnetwork, (1 << 0) | (1 << 1));

        // Expected output indices: {0} (since 0 is output for both, union is just {0})
        let mut expected_output_indices = BTreeSet::new();
        expected_output_indices.insert(0);
        assert_eq!(contracted_path.output_indices, expected_output_indices);

        // Expected indices:
        // (0,2) is common to both A and B, AND it is an output index (0). So it should be preserved.
        // (1,3) is unique to A.
        // (2,4) is unique to B.
        let mut expected_indices = BTreeSet::new();
        expected_indices.insert((0, 2)); // Common and output, so preserved
        expected_indices.insert((1, 3)); // Unique to A
        expected_indices.insert((2, 4)); // Unique to B
        assert_eq!(contracted_path.indices, expected_indices);

        // Expected cost: Calculated from calculate_cost
        // Total unique indices: (0,2), (1,3), (2,4)
        // Product of sizes: 2 * 3 * 4 = 24
        // Cost: 5 (from A) + 10 (from B) + 24 (product of sizes) = 39
        assert_eq!(contracted_path.cost, 39);

        // Expected path: [0, 1, usize::MAX]
        assert_eq!(contracted_path.path, vec![0, 1, usize::MAX]);
    }
}
