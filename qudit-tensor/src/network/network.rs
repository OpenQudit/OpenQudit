use std::{collections::{BTreeMap, BTreeSet, HashMap}, hash::Hash};

use qudit_core::{QuditRadices, TensorShape};
use qudit_expr::GenerationShape;

use crate::tree::ExpressionTree;
use super::path::ContractionPath;
use super::tensor::QuditTensor;
use super::index::NetworkIndex;
use super::index::TensorIndex;
use super::index::ContractionIndex;
use super::index::WeightedIndex;
use super::index::IndexDirection;
use super::index::IndexId;
use super::index::IndexSize;
use super::TensorId;

pub type NetworkEdge = (NetworkIndex, BTreeSet<TensorId>);

pub struct QuditTensorNetwork {
    tensors: Vec<QuditTensor>,
    local_to_network_index_map: Vec<Vec<IndexId>>,
    indices: Vec<NetworkEdge>,
}

// TODO: handle multiple disjoint (potentially empty) subnetworks
// TODO: handle partial trace
impl QuditTensorNetwork {
    pub fn new(tensors: Vec<QuditTensor>, local_to_network_index_map: Vec<Vec<IndexId>>, indices: Vec<NetworkEdge>) -> Self {
        for (index, edge) in indices.iter() {
            if edge.is_empty() {
                panic!("Index not attached to any tensor detected. Empty indices, must have explicit identity/copy tensors attached before final network construction.");
            }
        }

        QuditTensorNetwork {
            tensors,
            local_to_network_index_map,
            indices,
        }
    }

    // fn get_num_outputs(&self) -> usize {
    //     todo!()
    // }

    fn num_indices(&self) -> usize {
        self.indices.len()
    }
    
    fn index_id(&self, idx: &NetworkIndex) -> Option<IndexId> {
        self.indices.iter().position(|x| &x.0 == idx)
    }

    fn index_size(&self, idx_id: IndexId) -> Option<IndexSize> {
        if idx_id >= self.num_indices() {
            return None
        }

        // Safety: checked bounds
        unsafe { Some(self.index_size_unchecked(idx_id)) }
    }

    unsafe fn index_size_unchecked(&self, idx_id: IndexId) -> IndexSize {
        match &self.indices[idx_id].0 {
            NetworkIndex::Output(tidx) => tidx.index_size(),
            NetworkIndex::Contracted(con) => con.index_size(),
        }
    }

    fn get_output_indices(&self) -> Vec<TensorIndex> {
        self.indices.iter().filter_map(|x| match &x.0 {
            NetworkIndex::Output(idx) => Some(idx),
            NetworkIndex::Contracted(_) => None,
        }).copied().collect()
    }

    fn get_output_shape(&self) -> GenerationShape {
        /// Calculate dimension totals for each direction
        let mut total_batch_dim = None;
        let mut total_output_dim = None;
        let mut total_input_dim = None;
        for idx in self.get_output_indices() {
            match idx.direction() {
                IndexDirection::Derivative => panic!("Derivatives should not be explicit in networks."),
                IndexDirection::Batch => {
                    if let Some(value) = total_batch_dim.as_mut() {
                        *value *= idx.index_size();
                    } else {
                        total_batch_dim = Some(idx.index_size());
                    }
                }
                IndexDirection::Output => {
                    if let Some(value) = total_output_dim.as_mut() {
                        *value *= idx.index_size();
                    } else {
                        total_output_dim = Some(idx.index_size());
                    }
                }
                IndexDirection::Input => {
                    if let Some(value) = total_input_dim.as_mut() {
                        *value *= idx.index_size();
                    } else {
                        total_input_dim = Some(idx.index_size());
                    }
                }
            }
        }

        match (total_batch_dim, total_output_dim, total_input_dim) {
            (None, None, None) => GenerationShape::Scalar,
            (Some(nbatches), None, None) => GenerationShape::Vector(nbatches),
            (None, Some(nrows), None) => GenerationShape::Matrix(nrows, 1), // Ket
            (None, None, Some(ncols)) => GenerationShape::Vector(ncols), // Bra
            (Some(nbatches), Some(nrows), None) => GenerationShape::Tensor3D(nbatches, nrows, 1),
            (Some(nbatches), None, Some(ncols)) => GenerationShape::Matrix(nbatches, ncols),
            (None, Some(nrows), Some(ncols)) => GenerationShape::Matrix(nrows, ncols),
            (Some(nmats), Some(nrows), Some(ncols)) => GenerationShape::Tensor3D(nmats, nrows, ncols),
        }
    }

    fn get_tensor_unique_network_indices(&self, tensor_id: TensorId) -> BTreeSet<NetworkIndex> {
        self.local_to_network_index_map[tensor_id].iter().map(|&idx_id| self.indices[idx_id].0.clone()).collect()
    }

    fn get_tensor_unique_flat_indices(&self, tensor_id: TensorId) -> BTreeSet<WeightedIndex> {
        self.local_to_network_index_map[tensor_id].iter().map(|&idx_id| (idx_id, self.index_size(idx_id).expect("Index id unexpectedly not found"))).collect()
    }

    fn get_tensor_output_index_ids(&self, tensor_id: TensorId) -> BTreeSet<IndexId> {
        self.local_to_network_index_map[tensor_id].iter().filter(|&idx_id| self.indices[*idx_id].0.is_output()).copied().collect()
   }

    // fn convert_network_to_flat_index(&self, idx: &NetworkIndex) -> WeightedIndex {
    //     match idx {
    //         NetworkIndex::Output(idx) => (*idx, self.output_indices[*idx].index_size()),
    //         NetworkIndex::Contracted(idx) => (*idx + self.get_num_outputs(), self.contractions[*idx].total_dimension),
    //     }
    // }
    
    fn get_neighbors(&self, tensor: TensorId) -> BTreeSet<TensorId> {
        let mut neighbors = BTreeSet::new();
        for idx_id in &self.local_to_network_index_map[tensor] {
            neighbors.extend(self.indices[*idx_id].1.iter());
        }
        neighbors
    }
    
    fn get_subnetworks(&self) -> Vec<Vec<TensorId>> {
        let mut subnetworks: Vec<Vec<TensorId>> = Vec::new();
        let mut visited = vec![false; self.tensors.len()];

        for current_tensor_id in 0..self.tensors.len() {
            if visited[current_tensor_id] {
                continue;
            }

            let mut current_subnetwork = Vec::new();
            let mut queue = vec![current_tensor_id];

            while let Some(tensor_id) = queue.pop() {
                if visited[tensor_id] {
                    continue
                }
                visited[tensor_id] = true;
                current_subnetwork.push(tensor_id);

                for neighbor in self.get_neighbors(tensor_id) {    
                    if !visited[neighbor] {
                        queue.push(neighbor);
                    }
                }
            }

            subnetworks.push(current_subnetwork);
        }
        subnetworks
    }

    pub fn solve_for_path(&self) -> ContractionPath {
        let mut disjoint_paths = Vec::new();

        for subgraph in self.get_subnetworks() {
            let input = self.build_trivial_contraction_paths(subgraph);
            let path = if input.len() < 8 {
                ContractionPath::solve_optimal_simple(input)
            } else {
                ContractionPath::solve_greedy_simple(input)
            };
            disjoint_paths.push(path);
        }

        ContractionPath::solve_by_size_simple(disjoint_paths)
        // pick smallest two and contract (TODO: add new operation to TTGT tree method to
        // determine function. If contracted indices.len() == 0 then just KRON (need batch kron
        // too)).
    }

    fn build_trivial_contraction_paths(&self, subnetwork: Vec<TensorId>) -> Vec<ContractionPath> {
        subnetwork.iter()
            .map(|&tensor_id| {
                let flat_indices = self.get_tensor_unique_flat_indices(tensor_id);
                let output_indices = self.get_tensor_output_index_ids(tensor_id);
                ContractionPath::trivial(tensor_id, flat_indices, output_indices)
            }).collect()
    }

    pub fn path_to_expression_tree(&self, path: ContractionPath) -> ExpressionTree {
        let mut tree_stack: Vec<ExpressionTree> = Vec::new();

        for path_element in path.path.iter() {
            if *path_element == usize::MAX {
                let left = tree_stack.pop().unwrap();
                let right = tree_stack.pop().unwrap();

                let left_network_index_ids: Vec<IndexId> = left.indices().iter().map(|&idx| idx.index_id()).collect();
                let right_network_index_ids: Vec<IndexId> = right.indices().iter().map(|&idx| idx.index_id()).collect();

                let intersection: Vec<IndexId> = left_network_index_ids.iter()
                    .filter(|&id| right_network_index_ids.contains(id))
                    .copied()
                    .collect();

                // Shared indices appear in a contraction on both sides, but are not summed over.
                // These are realized as indices that are output to the network that appear on
                // in both left and right index sets.
                let shared_ids: Vec<IndexId> = intersection.iter()
                    .filter(|&id| self.indices[*id].0.is_output())
                    .copied()
                    .collect();

                let contraction_ids: Vec<IndexId> = intersection.into_iter()
                    .filter(|id| !shared_ids.contains(id))
                    .collect();

                tree_stack.push(left.contract(right, shared_ids, contraction_ids));
            } else {
                let tensor = self.tensors[*path_element].clone();
                // [5, 1, 5, 0, 1, 2] (5 contracted, 1 traced)
                let mut network_idx_ids = self.local_to_network_index_map[*path_element].clone();
                let leaf_node = ExpressionTree::leaf(tensor.expression.clone(), tensor.param_indices.clone());

                // Perform partial traces if necessary
                // find any indices that appear twice in indices and are only connected to this
                let mut looped_index_map: HashMap<IndexId, Vec<usize>> = HashMap::new();
                for (local_idx, &network_idx_id) in network_idx_ids.iter().enumerate() {
                    let index_edge = &self.indices[network_idx_id];
                    if !index_edge.0.is_output() && index_edge.1.len() == 1 {
                        looped_index_map.entry(network_idx_id).or_default().push(local_idx);
                    }
                }
                // looped_index_map = {1 : (1, 4)}

                // Assert that each looped index vector is exactly length 2 and convert them to pairs
                let mut to_remove = Vec::with_capacity(looped_index_map.len() * 2);
                let looped_index_pairs: Vec<(usize, usize)> = looped_index_map.into_iter().map(|(index_id, local_indices)| {
                    assert_eq!(local_indices.len(), 2, "Looped index {:?} did not have exactly two occurrences. It had {}.", index_id, local_indices.len());
                    to_remove.extend(local_indices.clone());
                    (local_indices[0], local_indices[1])
                }).collect();

                to_remove.sort();
                for traced_local_index in to_remove.iter().rev() {
                    network_idx_ids.remove(*traced_local_index);
                }
                // indices = [5, 5, 0, 2]

                let traced_node = leaf_node.trace(looped_index_pairs);
                // traced_node(leaf).indices = ((0, output), (2, output), (3, input), (5, input))

                // need to argsort indices such equal elements are consecutive without changing
                // direction ordering
                //
                // This way a tensor with local_to_network map like [5, 6, 5, 0, 1, 2]
                // can have the two from its generation shape be treated as one, for
                // future operations.
                let argsorted_indices = {
                    let mut argsorted_indices = (0..network_idx_ids.len()).collect::<Vec<_>>();
                    argsorted_indices.sort_by_key(|&i| (traced_node.indices()[i].direction(), network_idx_ids[i]));
                    argsorted_indices 
                };

                let new_directions = argsorted_indices.iter().map(|id| traced_node.indices()[*id].direction()).collect();

                let tranposed_node = traced_node.transpose(argsorted_indices.clone(), new_directions);

                let new_node_indices = {
                    let mut last = None;
                    let mut new_node_indices = Vec::new();
                    let mut dimension_acm = 1;
                    let mut old_node_indices = tranposed_node.indices();
                    for (id, idx) in argsorted_indices.iter().enumerate().map(|(id, &i)| (id, network_idx_ids[i])) {
                        dimension_acm *= old_node_indices[id].index_size();
                        if last != Some(idx) {
                            last = Some(idx);
                            new_node_indices.push(TensorIndex::new(old_node_indices[id].direction(), idx, dimension_acm));
                            dimension_acm = 1;
                        }
                    }
                    new_node_indices
                };

                tree_stack.push(tranposed_node.reindex(new_node_indices));
            }
        }
        if tree_stack.len() != 1 {
            panic!("Tree stack should have exactly one element.");
        }

        let tree = tree_stack.pop().unwrap();

        // Perform final Transpose
        let mut goal_index_order = tree.indices();
        goal_index_order.sort_by_key(|x| &self.indices[x.index_id()]);

        let final_transpose = goal_index_order
            .iter()
            .map(|i| tree.indices().iter().position(|x| x.index_id() == i.index_id()).unwrap())
            .collect::<Vec<_>>();

        let final_redirection = goal_index_order
            .iter()
            .map(|i| {
                if let NetworkIndex::Output(tidx) = self.indices[i.index_id()].0 {
                    tidx.direction()
                } else {
                    panic!("Non output index made it to final network output.");
                }
            }).collect();

        // println!("Current direction: {:?}", tree.indices().iter().map(|idx| idx.direction()).collect::<Vec<_>>());
        // println!("Final transpose: {:?} redirection: {:?}", final_transpose, final_redirection);
        tree.transpose(final_transpose, final_redirection) 
    }
}
