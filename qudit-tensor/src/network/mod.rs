use std::collections::{BTreeSet, HashMap};

use crate::tree::ExpressionTree;

type SubNetwork = u64;
type Cost = usize;

#[derive(Debug, Clone)]
struct ContractionPath {
    cost: usize,
    open_indices: BTreeSet<usize>,
    path: Vec<usize>,
    subnetwork: SubNetwork,
    param_indices: BitSet,
}


impl ContractionPath {
    fn calculate_cost(T_a: &Self, T_b: &Self) -> usize {
        let total_indices = T_a.open_indices
            .union(&T_b.open_indices)
            .copied()
            .collect::<Vec<_>>();
        T_a.cost + T_b.cost + usize::pow(2, total_indices.len() as u32)
    }

    fn contract(&self, other: &Self) -> Self {
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

    fn trivial(idx: usize, indices: &[usize], param_indices: BitSet) -> Self {
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

#[derive(Debug, Clone)]
enum Wire {
    Empty,
    Connected(usize, usize), // node_id, local_index_id
    Closed,
}

type QuditId = usize;
type TensorId = usize;
type IndexId = usize;
type IndexSize = usize;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum IndexDirection {
    Left,
    Right,
    Up,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LocalTensorIndex {
    direction: IndexDirection,
    index_id: IndexId,
    index_size: IndexSize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum NetworkIndex {
    Open(usize, IndexDirection), // qudit_id for left and right and up_index_id for up
    Shared(usize), // contraction_id 
}

use bit_set::BitSet;
use qudit_expr::TensorExpression;

#[derive(Debug, Clone)]
struct QuditTensor {
    indices: Vec<LocalTensorIndex>,
    local_to_global_index_map: HashMap<usize, NetworkIndex>,
    expression: TensorExpression,
    param_indices: ParamIndices,
}

impl QuditTensor {
    fn new(
        expression: TensorExpression,
        left_indices: Vec<usize>,
        right_indices: Vec<usize>,
        up_indices: Vec<usize>,
        param_indices: ParamIndices,
    ) -> Self {
        let mut indices = Vec::new();
        let mut id_counter = 0;
        for size in right_indices {
            indices.push(
                LocalTensorIndex {
                    direction: IndexDirection::Right,
                    index_id: id_counter,
                    index_size: size,
                }
            );
            id_counter += 1;
        }
        for size in left_indices {
            indices.push(
                LocalTensorIndex {
                    direction: IndexDirection::Left,
                    index_id: id_counter,
                    index_size: size,
                }
            );
            id_counter += 1;
        }
        for size in up_indices {
            indices.push(
                LocalTensorIndex {
                    direction: IndexDirection::Up,
                    index_id: id_counter,
                    index_size: size,
                }
            );
            id_counter += 1;
        }
        QuditTensor {
            indices,
            local_to_global_index_map: HashMap::new(),
            expression,
            param_indices,
        }
    }

    fn network_indices(&self) -> Vec<NetworkIndex> {
        let mut indices = Vec::new();
        for i in 0..self.indices.len() {
            if let Some(global_index) = self.local_to_global_index_map.get(&i) {
                indices.push(global_index.clone());
            } else {
                panic!("Index {} not found in local_to_global_index_map", i);
            }
        }
        indices
    }

    fn right_indices(&self) -> Vec<usize> {
        self.indices.iter()
            .filter_map(|index| match index.direction {
                IndexDirection::Right => Some(index.index_id),
                _ => None,
            })
            .collect()
    }

    fn right_radices(&self) -> QuditRadices {
        self.indices.iter()
            .filter_map(|index| match index.direction {
                IndexDirection::Right => Some(index.index_size as u8),
                _ => None,
            })
            .collect()
    }

    fn left_indices(&self) -> Vec<usize> {
        self.indices.iter()
            .filter_map(|index| match index.direction {
                IndexDirection::Left => Some(index.index_id),
                _ => None,
            })
            .collect()
    }

    fn left_radices(&self) -> QuditRadices {
        self.indices.iter()
            .filter_map(|index| match index.direction {
                IndexDirection::Left => Some(index.index_size as u8),
                _ => None,
            })
            .collect()
    }

    fn up_indices(&self) -> Vec<usize> {
        self.indices.iter()
            .filter_map(|index| match index.direction {
                IndexDirection::Up => Some(index.index_id),
                _ => None,
            })
            .collect()
    }
    
    fn up_radices(&self) -> QuditRadices {
        self.indices.iter()
            .filter_map(|index| match index.direction {
                IndexDirection::Up => Some(index.index_size as u8),
                _ => None,
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
struct QuditContraction {
    left_id: usize,
    right_id: usize,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
    total_dimension: usize,
}

use qudit_core::{ParamIndices, QuditRadices};

// Assume: Each index appears in only one pairwise contraction
// Assume: Optimal Contraction is done with fewer than 64 indices
//
// A QuditCircuitNetwork is referenced by many ways:
// * by *qudit_ids* (0, 1, 2, ...), these are the qudits in the circuit
// * by *tensor_ids* (0, 1, 2, ...), these are the tensors in the circuit
// tensor_ids are unique and don't change even if tensors are moved or deleted
// * by *index_ids*:
// - indices describe the links on and between tensors, which in turn, describe the
//   contractions.
// - Since each tensor is not a static dump of data, and rather associated with an
//   expression that will generate the data in either a 1D, 2D, or 3D shape, we
//   remember each tensor's "local" index ids. These describe the tensor from
//   the tensor generator's perspective. Remembering these allows us to determine
//   the necessary transformations to apply to the data after generation to perform
//   the desired network contractions.
// - The network as a whole has a set of "global" index ids. These ids directly
//   correspond (1:1) with a tensor-tensor contraction. The contraction data
//   stored in `intermediate` maps these global ids to their contraction data.
//   The contraction data captures which tensors are involved and the local ids
//   for each involved.
struct QuditCircuitNetwork {
    tensors: Vec<QuditTensor>,
    unused: BTreeSet<usize>,

    radices: QuditRadices,
    left: Vec<Wire>,
    right: Vec<Wire>,
    // up: ?
    intermediate: Vec<QuditContraction>,
}

impl QuditCircuitNetwork {
    fn new(radices: QuditRadices) -> Self {
        QuditCircuitNetwork {
            tensors: vec![],
            unused: BTreeSet::new(),
            left: vec![Wire::Empty; radices.len()],
            right: vec![Wire::Empty; radices.len()],
            radices,
            intermediate: vec![],
        }
    }

    /// Prepend a tensor onto the circuit network
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to prepend
    ///
    /// * `right_qudit_map` - An array of qudit ids that maps the tensors right local indices to
    ///     the circuit qudit ids, such that `right_qudit_map[i] == qudit_id` means that the
    ///     `tensor.right_indices[i]` will be joined with existing leftmost leg on `qudit_id`.
    ///
    /// * `left_qudit_map` - An array of qudit ids that maps the tensors left local indices to
    ///    the circuit qudit ids, such that `left_qudit_map[i] == qudit_id` means that the
    ///    `tensor.left_indices[i]` will now be the leftmost leg on `qudit_id`.
    fn prepend(&mut self, tensor: QuditTensor, left_qudit_map: Vec<usize>, right_qudit_map: Vec<usize>) {
        let left_tensor_indices = tensor.left_indices();
        let right_tensor_indices = tensor.right_indices();

        if left_tensor_indices.len() != left_qudit_map.len() {
            panic!("Left tensor indices and left qudit map lengths do not match");
        }
        if right_tensor_indices.len() != right_qudit_map.len() {
            panic!("Right tensor indices and right qudit map lengths do not match");
        }
        for (i, qudit_id) in left_qudit_map.iter().enumerate() {
            if *qudit_id >= self.radices.len() {
                panic!("Left qudit id {} is out of bounds", qudit_id);
            }
            if self.radices[*qudit_id] != tensor.left_radices()[i] {
                panic!("Left qudit id {} has different radices", qudit_id);
            }
        }
        for (i, qudit_id) in right_qudit_map.iter().enumerate() {
            if *qudit_id >= self.radices.len() {
                panic!("Right qudit id {} is out of bounds", qudit_id);
            }
            if self.radices[*qudit_id] != tensor.right_radices()[i] { // TODO: do I need
                // radices? isn't the size of the index the radix?
                panic!("Right qudit id {} has different radices", qudit_id);
            }
        }

        // add this as a new node
        let tensor_id = self.tensors.len();
        self.tensors.push(tensor);

        let mut contracting_nodes: HashMap<usize, QuditContraction> = HashMap::new();
        for (i, qudit_id) in right_qudit_map.iter().enumerate() {
            let local_index_id = right_tensor_indices[i];
            // println!("Iterating over right qudit map: i:{} qudit_id:{} local_index_id:{}", i, qudit_id, local_index_id);
            match self.left[*qudit_id] {
                Wire::Empty => {
                    self.right[*qudit_id] = Wire::Connected(tensor_id, local_index_id);
                },
                Wire::Connected(right_id, right_local_index_id) => {
                    // Make a new contraction and update left wire
                    // This may not be the first time we say this node either
                    match contracting_nodes.get_mut(&right_id) {
                        Some(contraction) => {
                            contraction.left_indices.push(local_index_id);
                            contraction.right_indices.push(right_local_index_id);
                            contraction.total_dimension *= self.radices[*qudit_id] as usize;
                        },
                        None => {
                            let new_contraction = QuditContraction {
                                left_id: tensor_id,
                                right_id,
                                left_indices: vec![local_index_id],
                                right_indices: vec![right_local_index_id],
                                total_dimension: self.radices[*qudit_id] as usize,
                            };
                            contracting_nodes.insert(right_id, new_contraction);
                        }
                    }
                }
                Wire::Closed => {
                    panic!("Left qudit {} is closed", qudit_id);
                }
            }
            if left_qudit_map.contains(qudit_id) {
                // If it's in the right qudit map, we need to open it
                let left_i = left_qudit_map.iter().position(|&x| x == *qudit_id).unwrap();
                let outgoing_local_index_id = left_tensor_indices[left_i];
                self.left[*qudit_id] = Wire::Connected(tensor_id, outgoing_local_index_id);
            } else {
                // If it's not in the right qudit map, we need to close it
                self.left[*qudit_id] = Wire::Closed;
            }
        }

        for (i, qudit_id) in left_qudit_map.iter().enumerate() {
            if left_qudit_map.contains(&qudit_id) {
                // Already processed
                continue;
            }
            let local_index_id = left_tensor_indices[i];
            match self.left[*qudit_id] {
                Wire::Closed => {
                    // If it's closed, we need to open it
                    self.left[*qudit_id] = Wire::Connected(tensor_id, local_index_id);
                },
                _ => {
                    // If it's not closed then error.
                    panic!("Right qudit {} is closed", qudit_id);
                }
            }
        }

        for (i, qudit_id) in left_qudit_map.iter().enumerate() {
            self.tensors[tensor_id].local_to_global_index_map.insert(
                left_tensor_indices[i],
                NetworkIndex::Open(*qudit_id, IndexDirection::Left),
            );
        }

        for (i, qudit_id) in right_qudit_map.iter().enumerate() {
            self.tensors[tensor_id].local_to_global_index_map.insert(
                right_tensor_indices[i],
                NetworkIndex::Open(*qudit_id, IndexDirection::Right),
            );
        }

        for (_, contraction) in contracting_nodes.drain() {
            let global_index_id = self.intermediate.len();
            println!("Contraction {}: {:?}", global_index_id, contraction);
            // update the left tensor
            for i in &contraction.left_indices {
                self.tensors[contraction.left_id].local_to_global_index_map.insert(*i, NetworkIndex::Shared(global_index_id));
            }

            // update the right tensor
            for i in &contraction.right_indices {
                self.tensors[contraction.right_id].local_to_global_index_map.insert(*i, NetworkIndex::Shared(global_index_id));
            }

            // add contraction to the list of intermediate contractions
            self.intermediate.push(contraction);
        }
    }

    fn build_trivial_contraction_paths(&self) -> Vec<ContractionPath> {
        let mut open_index_counter = self.intermediate.len();
        self.tensors.iter()
            .enumerate()
            .filter(|(i, _)| !self.unused.contains(&i))
            .map(|(i, tensor)| {
                let mut indices = Vec::new();
                for index in tensor.network_indices() {
                    // println!("Tensor {} index: {:?}", i, index);
                    match index {
                        NetworkIndex::Open(_qudit_id, _direction) => {
                            indices.push(open_index_counter);
                            open_index_counter += 1;
                        },
                        NetworkIndex::Shared(contraction_id) => {
                            indices.push(contraction_id);
                        }
                    }
                }
                println!("Tensor {} indices: {:?}", i, indices);
                ContractionPath::trivial(i, &indices, tensor.param_indices.to_bitset())
            })
            .collect()
    }

    /// Reference: https://arxiv.org/pdf/1304.6112
    fn optimize_optimal_simple(&self) -> ContractionPath {
        let n = self.tensors.len();
        if n == 0 {
            panic!("No tensors in the network");
        }

        // contractions[c] = S[c] = list of optimal contractions for c-length subnetworks
        let mut contractions: Vec<Vec<ContractionPath>> = vec![vec![]; n];
        contractions[0] = self.build_trivial_contraction_paths();

        let mut best_costs = HashMap::new();
        let mut best_contractions = HashMap::new();

        for c in 1..n {
            for d in 0..((c+1)/2) {
                let sd = &contractions[d];
                let scd = &contractions[c - 1 - d];
                for path_a in sd {
                    for path_b in scd {
                        if path_a.subnetwork & path_b.subnetwork != 0 {
                            // Non-disjoint subnetworks
                            continue;
                        }

                        let cost = ContractionPath::calculate_cost(path_a, path_b);

                        let new_subnetwork = path_a.subnetwork | path_b.subnetwork;
                        match best_costs.get(&new_subnetwork) {
                            Some(&best_cost) if best_cost <= cost => {
                                // Found a better path
                                continue;
                            }
                            _ => {
                                best_costs.insert(new_subnetwork, cost);
                            }
                        }

                        best_contractions.insert(new_subnetwork, path_a.contract(path_b));
                    }
                }
            }

            // Update the contractions for the current size
            best_contractions.drain().for_each(|(subnetwork, path)| {
                contractions[c].push(path);
            });
            best_costs.clear();
        }

        // Retrieve and return the best contraction path for the entire network
        contractions[n - 1].iter().next().unwrap_or_else(|| {
            panic!("No contraction path found for the entire network");
        }).clone()
    }

    fn path_to_expression_tree(&self, path: &ContractionPath) -> ExpressionTree {
        let mut tree_stack: Vec<PartialTree> = Vec::new();
        for path_element in path.path.iter() {
            if *path_element == usize::MAX {
                let right = tree_stack.pop().unwrap();
                let left = tree_stack.pop().unwrap();
                let contraction_indices = left.open_indices
                    .iter()
                    .filter(|&i| right.open_indices.contains(&i))
                    .cloned()
                    .collect::<Vec<_>>();

                let left_goal_index_order = left.open_indices
                    .iter()
                    .filter(|&i| !contraction_indices.contains(&i))
                    .chain(contraction_indices.iter())
                    .cloned()
                    .collect::<Vec<_>>();
                println!("Left goal index order: {:?}", left_goal_index_order);

                let left_split = left.open_indices.len() - contraction_indices.len();

                let left_index_transpose = left_goal_index_order
                    .iter()
                    .map(|i| left.open_indices.iter().position(|x| x == i).unwrap())
                    .collect::<Vec<_>>();

                let right_goal_index_order = contraction_indices
                    .iter()
                    .chain(right.open_indices.iter().filter(|&i| !contraction_indices.contains(&i)))
                    .cloned()
                    .collect::<Vec<_>>();
                println!("Right goal index order: {:?}", right_goal_index_order);

                let right_split = contraction_indices.len();

                let right_index_transpose = right_goal_index_order
                    .iter()
                    .map(|i| right.open_indices.iter().position(|x| x == i).unwrap())
                    .collect::<Vec<_>>();

                let left_tree = left.tree.transpose(left_index_transpose, left_split);
                let right_tree = right.tree.transpose(right_index_transpose, right_split);
                let matmul_tree = left_tree.matmul(right_tree);

                let output_index_order = left_goal_index_order
                    .iter()
                    .chain(right_goal_index_order.iter())
                    .filter(|&i| !contraction_indices.contains(&i))
                    .cloned()
                    .collect::<Vec<_>>();
                println!("Output index order: {:?}", output_index_order);

                tree_stack.push(PartialTree {
                    tree: matmul_tree,
                    open_indices: output_index_order,
                });
                println!("");

            } else {
                let tensor = self.tensors[*path_element].clone();
                let open_indices = tensor.network_indices();
                println!("Tensor {}: {:?}", path_element, open_indices);
                let tree = ExpressionTree::leaf(tensor.expression, tensor.param_indices);
                tree_stack.push(PartialTree { tree, open_indices });
            }
        }
        if tree_stack.len() != 1 {
            panic!("Tree stack should have exactly one element");
        }
        let partial_tree = tree_stack.pop().unwrap();
        let open_indices = partial_tree.open_indices;
        let tree = partial_tree.tree;
        for i in open_indices.iter() {
            if let NetworkIndex::Shared(_id) = i {
                panic!("Open index is a shared index");
            }
        }
        // argsort the open indices so all right first, then left, then up, ordered by id
        let mut indices = open_indices.iter().enumerate().collect::<Vec<(usize, &NetworkIndex)>>();
        indices.sort_by(|a, b| {
            if let (NetworkIndex::Open(id_a, dir_a), NetworkIndex::Open(id_b, dir_b)) = (a.1, b.1) {
                if dir_a == dir_b {
                    id_a.cmp(&id_b)
                } else {
                    if *dir_a == IndexDirection::Right {
                        std::cmp::Ordering::Less
                    } else if *dir_b == IndexDirection::Right {
                        std::cmp::Ordering::Greater
                    } else if *dir_a == IndexDirection::Left {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    }
                }
            } else {
                panic!("Open index is not a shared index");
            }
        });
        let mut split_at = 0;
        for (i, idx) in indices.iter() {
            if let NetworkIndex::Open(_id, dir) = idx {
                if *dir == IndexDirection::Left {
                    split_at += 1;
                }
            }
        }
        let transpose_order = indices.into_iter().map(|(i, _)| i).collect::<Vec<usize>>();

        tree.transpose(transpose_order, split_at)
    }
}

struct PartialTree {
    tree: ExpressionTree,
    open_indices: Vec<NetworkIndex>,
}

#[cfg(test)]
mod tests {
    use crate::tree::TreeOptimizer;

    use super::*;

    #[test]
    fn test_optimize_optimal() {
        let mut network = QuditCircuitNetwork::new(QuditRadices::new(&vec![2, 2, 2]));
        let expr = TensorExpression::new("A() {
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]
        }");
        network.prepend(
            QuditTensor::new(expr.clone(), vec![2, 2], vec![2, 2], vec![], ParamIndices::constant()),
            vec![0, 1],
            vec![0, 1],
        );
        network.prepend(
            QuditTensor::new(expr.clone(), vec![2, 2], vec![2, 2], vec![], ParamIndices::constant()),
            vec![1, 2],
            vec![1, 2],
        );
        network.prepend(
            QuditTensor::new(expr.clone(), vec![2, 2], vec![2, 2], vec![], ParamIndices::constant()),
            vec![0, 1],
            vec![0, 1],
        );
        network.prepend(
            QuditTensor::new(expr.clone(), vec![2, 2], vec![2, 2], vec![], ParamIndices::constant()),
            vec![1, 2],
            vec![1, 2],
        );
        println!("expr shape: {:?}", expr.shape);
        let optimal_path = network.optimize_optimal_simple();
        println!("Optimal Path: {:?}", optimal_path.path);
        let tree = network.path_to_expression_tree(&optimal_path);
        println!("Expression Tree: {:?}", tree);
        let tree  = TreeOptimizer::new().optimize(tree);
        println!("Expression Tree: {:?}", tree);
        // network.prepend(
        //     QuditTensor::new(vec![2, 2], vec![2, 2], vec![]),
        //     vec![0, 1],
        //     vec![0, 1],
        // );
        // network.prepend(
        //     QuditTensor::new(vec![2, 2], vec![2, 2], vec![]),
        //     vec![1, 2],
        //     vec![1, 2],
        // );
        // network.prepend(
        //     QuditTensor::new(vec![2, 2], vec![2, 2], vec![]),
        //     vec![0, 1],
        //     vec![0, 1],
        // );
        // network.prepend(
        //     QuditTensor::new(vec![2, 2], vec![2, 2], vec![]),
        //     vec![1, 2],
        //     vec![1, 2],
        // );
        assert!(optimal_path.cost > 0);
    }
}
