use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};

use bit_set::BitSet;

use crate::network::index::{IndexId, WeightedIndex};

/// A bitmask representing a set of tensors in a network.
// pub type SubNetwork = u64;
pub type SubNetwork = BitSet;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PotentialContraction {
    cost: isize,
    i: usize,
    j: usize,
    k: usize,
    result: ContractionPath,
}

/// Represents a path of tensor contractions, tracking the state and cost of the operations.
///
/// This struct is used in tensor network contraction algorithms to keep track of the
/// accumulated cost, the resulting open indices, the output indices, the sequence of
/// contractions, and the set of original tensors involved in a contraction path.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ContractionPath {
    /// The cost of all contractions in this path.
    pub cost: usize,

    /// The open indices of the resulting tensor after performing all contractions in the path.
    pub indices: BTreeSet<WeightedIndex>,

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
    /// * `t_a` - The first `ContractionPath`.
    /// * `t_b` - The second `ContractionPath`.
    ///
    /// Returns:
    /// The calculated cost as a `usize`.
    pub fn calculate_cost(t_a: &Self, t_b: &Self) -> usize {
        let total_indices = t_a.indices.union(&t_b.indices).copied().collect::<Vec<_>>();
        t_a.cost
            + t_b.cost
            + total_indices
                .iter()
                .map(|(_, size)| size)
                .product::<usize>()
    }

    pub fn total_dimension(&self) -> isize {
        self.indices.iter().map(|x| x.1 as isize).product::<isize>()
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
        let subnetwork = self.subnetwork.union(&other.subnetwork).collect();
        let output_indices: BTreeSet<IndexId> = self
            .output_indices
            .union(&other.output_indices)
            .copied()
            .collect();
        let indices = self
            .indices
            .iter()
            .chain(&other.indices)
            .filter(|idx| {
                output_indices.contains(&idx.0)
                    || !self.indices.contains(idx)
                    || !other.indices.contains(idx)
            })
            .copied()
            .collect();
        let cost = Self::calculate_cost(self, other);
        let path = self
            .path
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
    pub fn trivial(
        idx: IndexId,
        indices: BTreeSet<WeightedIndex>,
        output_indices: BTreeSet<IndexId>,
    ) -> Self {
        let path = vec![idx];
        let cost = 0;
        // let subnetwork = 1 << idx;
        let mut subnetwork = BitSet::new();
        subnetwork.insert(idx);
        ContractionPath {
            cost,
            indices,
            output_indices,
            path,
            subnetwork,
        }
    }

    /// Reference: https://arxiv.org/pdf/1304.6112
    pub fn solve_optimal_simple(initial_paths: Vec<Self>) -> Self {
        let n = initial_paths.len();
        assert!(n > 0, "No tensors to solve"); // TODO: does this need to be an error?

        // contractions[c] = S[c + 1] = list of optimal contractions for c-length subnetworks
        let mut contractions: Vec<Vec<ContractionPath>> = vec![vec![]; n];
        contractions[0] = initial_paths;

        let mut best_costs: HashMap<SubNetwork, usize> = HashMap::new();
        let mut best_contractions = HashMap::new();

        for c in 1..n {
            for d in 0..((c + 1) / 2) {
                let sd = &contractions[d]; // optimal d + 1 tensor paths
                let scd = &contractions[c - 1 - d]; // optimal c - d tensor paths
                for path_a in sd {
                    for path_b in scd {
                        if !path_a.subnetwork.is_disjoint(&path_b.subnetwork) {
                            // Non-disjoint subnetworks
                            continue;
                        }

                        let cost = ContractionPath::calculate_cost(path_a, path_b);

                        let new_subnetwork = path_a.subnetwork.union(&path_b.subnetwork).collect();
                        match best_costs.get(&new_subnetwork) {
                            Some(&best_cost) if best_cost <= cost => {
                                // Already found a better path
                                continue;
                            }
                            _ => {
                                best_costs.insert(new_subnetwork.clone(), cost);
                                best_contractions.insert(new_subnetwork, path_a.contract(path_b));
                            }
                        }
                    }
                }
            }

            // Update the contractions for the current size
            best_contractions.drain().for_each(|(_subnetwork, path)| {
                contractions[c].push(path); // best_contractions has c + 1 length contractions
            });
            best_costs.clear();
        }

        // Retrieve and return the best contraction path for the entire network
        contractions[n - 1]
            .iter()
            .next()
            .unwrap_or_else(|| {
                panic!("No contraction path found for the entire network");
            })
            .clone()
    }

    pub fn solve_by_size_simple(mut initial_paths: Vec<Self>) -> Self {
        let n = initial_paths.len();
        if n == 0 {
            panic!("No tensors to solve a path for.");
        }

        initial_paths.sort_by_key(|x| x.total_dimension());

        while initial_paths.len() > 1 {
            let min1 = initial_paths.pop().expect("Should have atleast 2 paths.");
            let min2 = initial_paths.pop().expect("Should have atleast 2 paths.");
            let new_path = min1.contract(&min2);
            match initial_paths
                .binary_search_by_key(&new_path.total_dimension(), |x| x.total_dimension())
            {
                Ok(pos) => initial_paths.insert(pos, new_path),
                Err(pos) => initial_paths.insert(pos, new_path),
            }
        }

        initial_paths
            .into_iter()
            .next()
            .expect("Exactly one path must be left.")
    }

    // TODO: add parameters for cost function
    // make cost function calculations more robust (log arithmetic)
    // Reference: https://quantum-journal.org/papers/q-2021-03-15-410/pdf/
    pub fn solve_greedy_simple(initial_paths: Vec<Self>) -> Self {
        let n = initial_paths.len();
        if n == 0 {
            panic!("No tensors to solve");
        }

        let mut active_paths = initial_paths
            .into_iter()
            .enumerate()
            .collect::<BTreeMap<_, _>>();
        let mut counter = active_paths.len();
        let mut remaining = BTreeSet::new();
        let mut contractions = BinaryHeap::new();

        for i in 0..active_paths.len() {
            remaining.insert(i);
            for j in (i + 1)..active_paths.len() {
                let path_i = &active_paths.get(&i).expect("Just constructed.");
                let path_j = &active_paths.get(&j).expect("Just constructed.");

                if !path_i.subnetwork.is_disjoint(&path_j.subnetwork) {
                    // Non-disjoint subnetworks
                    continue;
                }

                let path_k = path_i.contract(&path_j);

                let cost = -(path_k.total_dimension()
                    - (path_i.total_dimension() + path_j.total_dimension()));

                let pc = PotentialContraction {
                    i,
                    j,
                    k: counter,
                    result: path_k,
                    cost,
                };
                counter += 1;

                contractions.push(pc);
            }
        }

        while let Some(pc) = contractions.pop() {
            if !remaining.contains(&pc.i) || !remaining.contains(&pc.j) {
                // Rather than remove from BinaryHeap, skip ones already contracted.
                continue;
            }

            active_paths.remove(&pc.i);
            active_paths.remove(&pc.j);
            remaining.remove(&pc.i);
            remaining.remove(&pc.j);
            remaining.insert(pc.k);
            active_paths.insert(pc.k, pc.result);
            let j = pc.k;
            let path_j = active_paths.get(&pc.k).expect("Just inserted.");

            for (&i, path_i) in active_paths
                .iter()
                .filter(|(_path_id, path_i)| path_i.subnetwork.is_disjoint(&path_j.subnetwork))
            {
                let new_path_k = path_i.contract(&path_j);
                let new_cost = -(new_path_k.total_dimension()
                    - (path_i.total_dimension() + path_j.total_dimension()));
                let pc = PotentialContraction {
                    i,
                    j,
                    k: counter,
                    result: new_path_k,
                    cost: new_cost,
                };
                counter += 1;

                contractions.push(pc);
            }
        }
        assert_eq!(active_paths.len(), 1);
        active_paths.into_iter().next().unwrap().1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    // Helper function to create a simple ContractionPath for testing
    fn create_test_path(
        indices: &[WeightedIndex],
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

    // #[test]
    // fn test_trivial_path() {
    //     let indices = vec![(0, 2), (1, 3)];
    //     let output_indices = vec![0];
    //     let path = ContractionPath::trivial(0, &indices, &output_indices);

    //     assert_eq!(path.cost, 0);
    //     assert_eq!(path.indices, indices.iter().copied().collect());
    //     assert_eq!(path.output_indices, output_indices.iter().copied().collect());
    //     assert_eq!(path.path, vec![0]);
    //     assert_eq!(path.subnetwork, 1 << 0);
    // }

    #[test]
    fn test_calculate_cost() {
        // Path A: cost 5, indices (0,2), (1,3)
        let mut set1 = BitSet::new();
        set1.insert(0);
        let path_a = create_test_path(&[(0, 2), (1, 3)], &[], 5, set1, vec![0]);
        // Path B: cost 10, indices (1,3), (2,4)
        let mut set2 = BitSet::new();
        set2.insert(1);
        let path_b = create_test_path(&[(1, 3), (2, 4)], &[], 10, set2, vec![1]);

        // Expected total indices: (0,2), (1,3), (2,4)
        // Product of sizes: 2 * 3 * 4 = 24
        // Expected cost: 5 + 10 + 24 = 39
        let expected_cost = path_a.cost + path_b.cost + (2 * 3 * 4);
        assert_eq!(
            ContractionPath::calculate_cost(&path_a, &path_b),
            expected_cost
        );

        // Test with no common indices
        let mut set3 = BitSet::new();
        set3.insert(2);
        let path_c = create_test_path(&[(3, 5), (4, 6)], &[], 2, set3, vec![2]);
        // Expected total indices: (0,2), (1,3), (3,5), (4,6)
        // Product of sizes: 2 * 3 * 5 * 6 = 180
        // Expected cost: 5 + 2 + 180 = 187
        let expected_cost_no_common = path_a.cost + path_c.cost + (2 * 3 * 5 * 6);
        assert_eq!(
            ContractionPath::calculate_cost(&path_a, &path_c),
            expected_cost_no_common
        );

        // Test with empty paths (unlikely in real use, but good for robustness)
        let path_empty_a = create_test_path(&[], &[], 0, BitSet::new(), vec![]);
        let path_empty_b = create_test_path(&[], &[], 0, BitSet::new(), vec![]);
        assert_eq!(
            ContractionPath::calculate_cost(&path_empty_a, &path_empty_b),
            1
        );
    }

    #[test]
    fn test_contract_paths() {
        // Path A: cost 5, indices (0,2), (1,3), output {0}
        let mut set1 = BitSet::new();
        set1.insert(0);
        let path_a = create_test_path(&[(0, 2), (1, 3)], &[0], 5, set1, vec![0]);
        // Path B: cost 10, indices (1,3), (2,4), output {2}
        let mut set2 = BitSet::new();
        set2.insert(1);
        let path_b = create_test_path(&[(1, 3), (2, 4)], &[2], 10, set2, vec![1]);

        let contracted_path = path_a.contract(&path_b);

        // Expected subnetwork: 1 << 0 | 1 << 1 = 3
        let mut expected = BitSet::new();
        expected.insert(0);
        expected.insert(1);
        assert_eq!(contracted_path.subnetwork, expected);

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
        let mut set = BitSet::new();
        set.insert(2);
        let path_c = create_test_path(&[(3, 5), (4, 6)], &[], 2, set, vec![2]);
        let contracted_path_ac = path_a.contract(&path_c);
        let mut expected = BitSet::new();
        expected.insert(0);
        expected.insert(2);
        assert_eq!(contracted_path_ac.subnetwork, expected);
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
        let mut set1 = BitSet::new();
        set1.insert(0);
        let path_a = create_test_path(&[(0, 2), (1, 3)], &[0], 5, set1, vec![0]);
        // Path B: cost 10, indices (0,2), (2,4), output {0}
        // Index 0 is common and is an output index for both paths.
        let mut set2 = BitSet::new();
        set2.insert(1);
        let path_b = create_test_path(&[(0, 2), (2, 4)], &[0], 10, set2, vec![1]);

        let contracted_path = path_a.contract(&path_b);

        // Expected subnetwork: 1 << 0 | 1 << 1 = 3
        let mut expected = BitSet::new();
        expected.insert(0);
        expected.insert(1);
        assert_eq!(contracted_path.subnetwork, expected);

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
