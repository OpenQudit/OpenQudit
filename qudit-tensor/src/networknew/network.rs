use std::collections::{BTreeSet, HashMap};

use qudit_core::{QuditRadices, TensorShape};

use super::path::ContractionPath;
use crate::{networknew::{contraction::QuditContraction, index::{IndexDirection, IndexId, IndexSize, NetworkIndex, TensorLeg}}, tree::ExpressionTree};

use super::tensor::QuditTensor;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum Wire {
    Empty,
    Connected(usize, usize), // node_id, local_index_id
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NetworkIndex {
    // Open(usize, IndexDirection), // qudit_id for left and right and up_index_id for up
    Output(usize), // output index id
    Contracted(usize), // contraction_id 
    // Batch(usize, IndexDirection), // batch index id
}

/// A index in the network is either an output index or a contraction index.
pub struct QuditCircuitTensorNetworkBuilder {
    tensors: Vec<QuditTensor>,
    local_to_network_index_map: Vec<HashMap<IndexId, NetworkIndex>>,
    radices: QuditRadices,

    /// Pointer to front (left in math/right in circuit diagram) of the network for each qudit.
    front: Vec<Wire>,

    /// Pointer to rear (right in math/left in circuit diagram) of the network for each qudit.
    rear: Vec<Wire>,
    batch_indices: Vec<(String,IndexSize)>,
    contracted_indices: Vec<QuditContraction>,
}

impl QuditCircuitTensorNetworkBuilder {
    pub fn new(radices: QuditRadices) -> Self {
        QuditCircuitTensorNetworkBuilder {
            tensors: vec![],
            local_to_network_index_map: vec![],
            front: vec![Wire::Empty; radices.len()],
            rear: vec![Wire::Empty; radices.len()],
            batch_indices: Vec::new(),
            radices,
            contracted_indices: vec![],
        }
    }

    pub fn prepend(
        self,
        tensor: QuditTensor,
        input_index_map: Vec<usize>,
        output_index_map: Vec<usize>,
        batch_index_map: Vec<String>,
    ) -> Self {
        // 1. Check error conditions
        // 2. collected_contractions <- Go to the front:
        //      2a. Connect tensor's input indices with front according to input_index_map
        //      2b. Update rear with tensor's output indices according to output_index_map
        // 3. Update tensor's local index to global index map
        // 4. Insert contractions; updating other tensor's local index to global index map
        todo!()
    }

    pub fn append(
        self,
        tensor: QuditTensor,
        left_qudit_map: Vec<usize>,
        right_qudit_map: Vec<usize>,
        batch_index_map: Vec<String>,
    ) -> Self {
        todo!()
    }

    pub fn trace_wire(self, qudit: usize) -> Self {
        todo!()
    }

    pub fn trace_all_open_wires(self) -> Self {
        todo!()
    }
}


pub struct QuditTensorNetwork {
    tensors: Vec<QuditTensor>,
    local_to_network_index_map: Vec<Vec<NetworkIndex>>,
    output_indices: Vec<TensorLeg>,
    contractions: Vec<QuditContraction>,
}

impl QuditTensorNetwork {
    fn get_num_outputs(&self) -> usize {
        self.output_indices.len()
    }

    fn get_network_indices(&self, tensor_id: usize) -> Vec<NetworkIndex> {
        self.local_to_network_index_map[tensor_id].iter().collect::<BTreeSet<&NetworkIndex>>().into_iter().copied().collect::<Vec<NetworkIndex>>()
    }

    /// Flattened ID map: All output ones (0 ... num_outputs-1), All contractions (num_outputs-1...)
    /// returns pairs of (flattened_id, index_size)
    fn get_flattened_ids(&self, tensor_id: usize) -> Vec<(IndexId, IndexSize)> {
        let network_indices = self.get_network_indices(tensor_id);
        network_indices.iter().map(|idx| match idx {
            NetworkIndex::Output(idx) => (*idx, self.output_indices[*idx].index_size()),
            NetworkIndex::Contracted(idx) => (*idx + self.get_num_outputs(), self.contractions[*idx].total_dimension),
        }).collect()
    }

    fn build_trivial_contraction_paths(&self) -> Vec<ContractionPath> {
        self.tensors.iter()
            .enumerate()
            .map(|(tensor_id, tensor)| {
                let flattened_ids = self.get_flattened_ids(tensor_id);
                let output_ids = flattened_ids.iter().filter(|&(id, _)| *id < self.get_num_outputs()).map(|&(id, _)| id).collect::<Vec<usize>>();
                ContractionPath::trivial(tensor_id, &flattened_ids, &output_ids)
            }).collect()
    }

    /// Reference: https://arxiv.org/pdf/1304.6112
    pub fn optimize_optimal_simple(&self) -> ContractionPath {
        let n = self.tensors.len();

        if n == 0 {
            panic!("No tensors in the network");
        }

        // contractions[c] = S[c + 1] = list of optimal contractions for c-length subnetworks
        let mut contractions: Vec<Vec<ContractionPath>> = vec![vec![]; n];
        contractions[0] = self.build_trivial_contraction_paths();

        let mut best_costs = HashMap::new();
        let mut best_contractions = HashMap::new();

        for c in 1..n {
            for d in 0..((c+1)/2) {
                let sd = &contractions[d]; // optimal d + 1 tensor paths
                let scd = &contractions[c - 1 - d]; // optimal c - d tensor paths
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
                                // Already found a better path
                                continue;
                            }
                            _ => {
                                best_costs.insert(new_subnetwork, cost);
                                best_contractions.insert(new_subnetwork, path_a.contract(path_b));
                            }
                        }
                    }
                }
            }

            // Update the contractions for the current size
            best_contractions.drain().for_each(|(subnetwork, path)| {
                contractions[c].push(path);  // best_contractions has c + 1 length contractions
            });
            best_costs.clear();
        }

        // Retrieve and return the best contraction path for the entire network
        contractions[n - 1].iter().next().unwrap_or_else(|| {
            panic!("No contraction path found for the entire network");
        }).clone()
    }

    pub fn path_to_expression_tree(&self, path: ContractionPath) -> ExpressionTree {
        struct PartialTree {
            partial_tree: ExpressionTree,
            indices: Vec<IndexId>,
            output_indices: BTreeSet<IndexId>
        };

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

                let right_index_transpose = right_goal_index_order
                    .iter()
                    .map(|i| right.open_indices.iter().position(|x| x == i).unwrap())
                    .collect::<Vec<_>>();

                let batch_size = shared_indices.iter().map(|i| self.index_size(i)).product::<usize>();
                let contraction_size = contraction_indices.iter().map(|i| self.index_size(i)).product::<usize>();
                let left_nrows = left.open_indices
                            .iter()
                            .filter(|i| !contraction_indices.contains(i))
                            .filter(|i| !shared_indices.contains(i))
                            .map(|i| self.index_size(i))
                            .product::<usize>();
                let right_ncols = right.open_indices
                            .iter()
                            .filter(|i| !contraction_indices.contains(i))
                            .filter(|i| !shared_indices.contains(i))
                            .map(|i| self.index_size(i))
                            .product::<usize>(); 
                let left_shape = if batch_size == 1 { TensorShape::Matrix(left_nrows, contraction_size) } else { TensorShape::Tensor3D(batch_size, left_nrows, contraction_size) };
                let right_shape = if batch_size == 1 { TensorShape::Matrix(contraction_size, right_ncols) } else { TensorShape::Tensor3D(batch_size, contraction_size, right_ncols) };

                let left_tree = left.tree.transpose(left_index_transpose, left_shape);
                let right_tree = right.tree.transpose(right_index_transpose, right_shape);
                let matmul_tree = left_tree.matmul(right_tree);

                let output_index_order = shared_indices
                    .iter()
                    .chain(left_goal_index_order.iter().filter(|i| !shared_indices.contains(i)))
                    .chain(right_goal_index_order.iter().filter(|i| !shared_indices.contains(i))) 
                    .filter(|i| !contraction_indices.contains(i))
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
                tree_stack.push((tree, open_indices));
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
            match (a.1, b.1) {
                (NetworkIndex::Shared(id_a, dir_a), NetworkIndex::Shared(id_b, dir_b)) => {
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
                    }
                },
                (NetworkIndex::Open(id_a, dir_a), NetworkIndex::Open(id_b, dir_b)) => {
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
                    }
                },
                (NetworkIndex::Shared(_, _), NetworkIndex::Open(_, _)) => std::cmp::Ordering::Less,
                (NetworkIndex::Open(_, _), NetworkIndex::Shared(_, _)) => std::cmp::Ordering::Greater,
                _ => panic!("Open index is not a shared index"),
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
            if let NetworkIndex::Shared(id, dir) = idx {
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

        let output_tree = match (total_input_dim, total_output_dim, total_z_dim) {
            (None, None, None) => {
                tree.transpose(transpose_order, qudit_core::TensorShape::Scalar)
            }
            (Some(b), None, None) => {
                tree.transpose(transpose_order, qudit_core::TensorShape::Vector(b))
            }
            (None, Some(a), None) => {
                tree.transpose(transpose_order, qudit_core::TensorShape::Vector(a))
            }
            (None, None, Some(c)) => {
                tree.transpose(transpose_order, qudit_core::TensorShape::Vector(c))
            }
            (Some(b), Some(a), None) => {
                tree.transpose(transpose_order, qudit_core::TensorShape::Matrix(a, b))
            }
            (Some(b), None, Some(c)) => {
                tree.transpose(transpose_order, qudit_core::TensorShape::Matrix(b, c))
            }
            (None, Some(a), Some(c)) => {
                tree.transpose(transpose_order, qudit_core::TensorShape::Matrix(a, c))
            }
            (Some(b), Some(a), Some(c)) => {
                tree.transpose(transpose_order, qudit_core::TensorShape::Tensor3D(c, a, b))
            }
        };

        output_tree
    }

    // pub fn path_to_expression_tree(&self, path: ContractionPath) -> ExpressionTree {
    //     todo!()
    // }
}
