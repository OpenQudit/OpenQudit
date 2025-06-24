use qudit_core::{ParamIndices, QuditRadices};
use super::path::ContractionPath;
use super::IndexDirection;
use super::NetworkIndex;

use super::QuditTensor;
use super::Wire;
use super::IndexSize;
use super::contraction::QuditContraction;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;

use crate::tree::ExpressionTree;

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
pub struct QuditCircuitNetwork {
    tensors: Vec<QuditTensor>,
    unused: BTreeSet<usize>,

    radices: QuditRadices,
    left: Vec<Wire>,
    right: Vec<Wire>,
    // up_index_tensor_map: BTreeMap<String,Vec<usize>>,
    up_indices: Vec<(String,IndexSize)>,
    intermediate: Vec<QuditContraction>,
}

impl QuditCircuitNetwork {
    pub fn new(radices: QuditRadices) -> Self {
        QuditCircuitNetwork {
            tensors: vec![],
            unused: BTreeSet::new(),
            left: vec![Wire::Empty; radices.len()],
            right: vec![Wire::Empty; radices.len()],
            // up_index_tensor_map: BTreeMap::new(),
            up_indices: Vec::new(),
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
    pub fn prepend(
        &mut self,
        tensor: QuditTensor,
        left_qudit_map: Vec<usize>,
        right_qudit_map: Vec<usize>,
        batch_index_map: Vec<String>,
    ) {
        let left_tensor_indices = tensor.output_indices();
        let right_tensor_indices = tensor.input_indices();
        let up_tensor_indices = tensor.up_indices();

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
            if self.radices[*qudit_id] != tensor.input_radices()[i] {
                panic!("Left qudit id {} has different radices", qudit_id);
            }
        }
        for (i, qudit_id) in right_qudit_map.iter().enumerate() {
            if *qudit_id >= self.radices.len() {
                panic!("Right qudit id {} is out of bounds", qudit_id);
            }
            if self.radices[*qudit_id] != tensor.output_radices()[i] {
                panic!("Right qudit id {} has different radices", qudit_id);
            }
        }
        if batch_index_map.len() != up_tensor_indices.len() {
            panic!("Batch index map is invalid.");
        }
        for (i, batch_index) in batch_index_map.iter().enumerate() {
            let batch_dim = tensor.up_radices()[i] as usize; 
            for (up_index, up_dim) in self.up_indices.iter() {
                if up_index == batch_index && *up_dim != batch_dim {
                    panic!("Batch dimension of new tensor index does not match existing network index dimension.");
                }
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
                NetworkIndex::Open(*qudit_id, IndexDirection::Output),
            );
        }

        for (i, qudit_id) in right_qudit_map.iter().enumerate() {
            self.tensors[tensor_id].local_to_global_index_map.insert(
                right_tensor_indices[i],
                NetworkIndex::Open(*qudit_id, IndexDirection::Input),
            );
        }
        for (i, &idx_id) in self.tensors[tensor_id].up_indices().iter().enumerate() {
            let batch_index_name = &batch_index_map[i];
            let dim = self.tensors[tensor_id].up_radices()[i] as usize;

            let mut global_up_index_id = None;
            for (j, (existing_name, _)) in self.up_indices.iter().enumerate() {
                if existing_name == batch_index_name {
                    global_up_index_id = Some(j);
                    break;
                }
            }

            let final_idx_id = if let Some(id) = global_up_index_id { 
                id
            } else {
                let new_id = self.up_indices.len();
                self.up_indices.push((batch_index_name.clone(), dim));
                new_id
            };

            self.tensors[tensor_id].local_to_global_index_map.insert(
                idx_id,
                NetworkIndex::Shared(final_idx_id, IndexDirection::Up)
            );
        }

        // println!("local_to_global_index_map for tensor {}: {:?}", tensor_id, self.tensors[tensor_id].local_to_global_index_map);

        for (_, contraction) in contracting_nodes.drain() {
            let global_index_id = self.intermediate.len();
            println!("Contraction {}: {:?}", global_index_id, contraction);
            // update the left tensor
            for i in &contraction.left_indices {
                self.tensors[contraction.left_id].local_to_global_index_map.insert(*i, NetworkIndex::Contracted(global_index_id));
            }
            println!("local_to_global_index_map for tensor {}: {:?}", contraction.left_id, self.tensors[contraction.left_id].local_to_global_index_map);

            // update the right tensor
            for i in &contraction.right_indices {
                self.tensors[contraction.right_id].local_to_global_index_map.insert(*i, NetworkIndex::Contracted(global_index_id));
            }
            println!("local_to_global_index_map for tensor {}: {:?}", contraction.right_id, self.tensors[contraction.right_id].local_to_global_index_map);

            // add contraction to the list of intermediate contractions
            self.intermediate.push(contraction);
        }
    }

    fn build_trivial_contraction_paths(&self) -> Vec<ContractionPath> {
        let mut open_index_counter = self.intermediate.len() + self.up_indices.len();
        self.tensors.iter()
            .enumerate()
            .filter(|(i, _)| !self.unused.contains(&i))
            .map(|(i, tensor)| {
                let mut indices = Vec::new();
                let mut unsummed_indices = Vec::new();
                for index in tensor.network_indices() {
                    match index {
                        NetworkIndex::Open(_qudit_id, _direction) => {
                            indices.push(open_index_counter);
                            open_index_counter += 1;
                        },
                        NetworkIndex::Contracted(contraction_id) => {
                            indices.push(contraction_id);
                        }
                        NetworkIndex::Shared(up_index, _direction) => {
                            indices.push(self.intermediate.len() + up_index);
                            unsummed_indices.push(self.intermediate.len() + up_index);
                        }
                    }
                }
                println!("Tensor {} indices: {:?}", i, indices);
                ContractionPath::trivial(i, &indices, &unsummed_indices, tensor.param_indices.to_bitset())
            })
            .collect()
    }

    /// Reference: https://arxiv.org/pdf/1304.6112
    pub fn optimize_optimal_simple(&self) -> ContractionPath {
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

    pub fn path_to_expression_tree(&self, path: &ContractionPath) -> ExpressionTree {
        let mut tree_stack: Vec<PartialTree> = Vec::new();
        for path_element in path.path.iter() {
            if *path_element == usize::MAX {
                let right = tree_stack.pop().unwrap();
                let left = tree_stack.pop().unwrap();
                let shared_indices = left.open_indices
                    .iter()
                    .filter(|&i| i.is_shared())
                    .filter(|i| right.open_indices.contains(i))
                    .cloned()
                    .collect::<Vec<_>>();
                let contraction_indices = left.open_indices
                    .iter()
                    .filter(|i| right.open_indices.contains(i))
                    .filter(|&i| !i.is_shared()) // We don't contract over shared indices
                    .cloned()
                    .collect::<Vec<_>>();

                let left_goal_index_order = shared_indices
                    .iter()
                    .chain(left.open_indices
                            .iter()
                            .filter(|i| !contraction_indices.contains(i))
                            .filter(|i| !shared_indices.contains(i))
                    ).chain(contraction_indices.iter())
                    .cloned()
                    .collect::<Vec<_>>();
                println!("Left goal index order: {:?}", left_goal_index_order);

                let left_split = left.open_indices.len() - contraction_indices.len();

                let left_index_transpose = left_goal_index_order
                    .iter()
                    .map(|i| left.open_indices.iter().position(|x| x == i).unwrap())
                    .collect::<Vec<_>>();

                let right_goal_index_order = shared_indices
                    .iter()
                    .chain(contraction_indices
                            .iter()
                            .chain(right.open_indices
                                    .iter()
                                    .filter(|i| !contraction_indices.contains(i))
                                    .filter(|i| !shared_indices.contains(i))
                            )
                    )
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

        if open_indices.len() == 0 {
            return tree.reshape(qudit_core::TensorShape::Scalar);
        }

        for i in open_indices.iter() {
            if let NetworkIndex::Contracted(_id) = i {
                panic!("Open index is a shared index");
            }
        }
        // argsort the open indices so all up first, then output, then input, ordered by id
        let mut indices = open_indices.iter().enumerate().collect::<Vec<(usize, &NetworkIndex)>>();
        indices.sort_by(|a, b| {
            if let (NetworkIndex::Open(id_a, dir_a), NetworkIndex::Open(id_b, dir_b)) = (a.1, b.1) {
                if dir_a == dir_b {
                    id_a.cmp(&id_b)
                } else {
                    if *dir_a == IndexDirection::Up {
                        std::cmp::Ordering::Less
                    } else if *dir_b == IndexDirection::Up {
                        std::cmp::Ordering::Greater
                    } else if *dir_a == IndexDirection::Output {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    }
                    // if *dir_a == IndexDirection::Output {
                    //     std::cmp::Ordering::Less
                    // } else if *dir_b == IndexDirection::Output {
                    //     std::cmp::Ordering::Greater
                    // } else if *dir_a == IndexDirection::Input {
                    //     std::cmp::Ordering::Less
                    // } else {
                    //     std::cmp::Ordering::Greater
                    // }
                    // if *dir_a == IndexDirection::Input {
                    //     std::cmp::Ordering::Less
                    // } else if *dir_b == IndexDirection::Input {
                    //     std::cmp::Ordering::Greater
                    // } else if *dir_a == IndexDirection::Output {
                    //     std::cmp::Ordering::Less
                    // } else {
                    //     std::cmp::Ordering::Greater
                    // }
                }
            } else {
                panic!("Open index is not a shared index");
            }
        });
        let mut split_at = 0;
        for (i, idx) in indices.iter() {
            if let NetworkIndex::Open(_id, dir) = idx {
                if *dir == IndexDirection::Output {
                    split_at += 1;
                }
            }
        }

        let mut total_output_dim = None;
        let mut total_input_dim = None;
        let mut total_z_dim = None;

        for (_i, idx) in indices.iter() {
            if let NetworkIndex::Open(id, dir) = idx {
                match *dir {
                    IndexDirection::Input => {
                        let r = self.radices[*id] as usize;
                        if let Some(value) = total_input_dim.as_mut() {
                            *value *= r;
                        } else {
                            total_input_dim = Some(r);
                        }
                    }
                    IndexDirection::Output => {
                        let r = self.radices[*id] as usize;
                        if let Some(value) = total_output_dim.as_mut() {
                            *value *= r;
                        } else {
                            total_output_dim = Some(r);
                        }
                    }
                    IndexDirection::Up => {
                        let r = self.up_indices[*id].1;
                        if let Some(value) = total_z_dim.as_mut() {
                            *value *= r;
                        } else {
                            total_z_dim = Some(r);
                        }
                    }
                }
            }
        }

        let transpose_order = indices.into_iter().map(|(i, _)| i).collect::<Vec<usize>>();
        let permuted_tree = tree.transpose(transpose_order, split_at);


        let output_tree = match (total_input_dim, total_output_dim, total_z_dim) {
            (None, None, None) => {
                permuted_tree.reshape(qudit_core::TensorShape::Scalar)
            }
            (Some(b), None, None) => {
                permuted_tree.reshape(qudit_core::TensorShape::Vector(b))
            }
            (None, Some(a), None) => {
                permuted_tree.reshape(qudit_core::TensorShape::Vector(a))
            }
            (None, None, Some(c)) => {
                permuted_tree.reshape(qudit_core::TensorShape::Vector(c))
            }
            (Some(b), Some(a), None) => {
                permuted_tree.reshape(qudit_core::TensorShape::Matrix(a, b))
            }
            (Some(b), None, Some(c)) => {
                permuted_tree.reshape(qudit_core::TensorShape::Matrix(b, c))
            }
            (None, Some(a), Some(c)) => {
                permuted_tree.reshape(qudit_core::TensorShape::Matrix(a, c))
            }
            (Some(b), Some(a), Some(c)) => {
                permuted_tree.reshape(qudit_core::TensorShape::Tensor3D(c, a, b))
            }
        };

        output_tree
    }
}

struct PartialTree {
    tree: ExpressionTree,
    open_indices: Vec<NetworkIndex>,
}
