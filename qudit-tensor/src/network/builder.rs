use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::BTreeMap;

use qudit_core::unitary::UnitaryMatrix;
use qudit_core::ComplexScalar;
use qudit_core::{QuditRadices, TensorShape};
use qudit_expr::UnitaryExpression;
use crate::network::network::NetworkEdge;

use super::tensor::QuditTensor;
use super::index::NetworkIndex;
use super::index::TensorIndex;
use super::index::ContractionIndex;
use super::index::WeightedIndex;
use super::index::IndexDirection;
use super::index::IndexId;
use super::index::IndexSize;
use super::network::QuditTensorNetwork;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum Wire {
    Empty,
    Closed,
    Connected(usize, usize), // node_id, local_index_id
}

impl Wire {
    pub fn is_empty(&self) -> bool {
        match self {
            Wire::Empty => true,
            Wire::Closed => false,
            Wire::Connected(_, _) => false,
        }
    }

    pub fn is_active(&self) -> bool {
        match self {
            Wire::Empty => true,
            Wire::Closed => false,
            Wire::Connected(_, _) => true,
        }
    }
}


#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum NetworkBuilderIndex {
    Front(usize),
    Rear(usize),
    Batch(String),
    Contraction(usize),
}

pub struct QuditCircuitTensorNetworkBuilder {
    tensors: Vec<QuditTensor>,
    local_to_network_index_map: Vec<Vec<NetworkBuilderIndex>>,
    radices: QuditRadices,

    /// Pointer to front (left in math/right in circuit diagram) of the network for each qudit.
    front: Vec<Wire>,

    /// Pointer to rear (right in math/left in circuit diagram) of the network for each qudit.
    rear: Vec<Wire>,
    batch_indices: HashMap<String, IndexSize>,
    contracted_indices: Vec<ContractionIndex>,
}

impl QuditCircuitTensorNetworkBuilder {
    pub fn new(radices: QuditRadices) -> Self {
        QuditCircuitTensorNetworkBuilder {
            tensors: vec![],
            local_to_network_index_map: vec![],
            front: vec![Wire::Empty; radices.len()],
            rear: vec![Wire::Empty; radices.len()],
            batch_indices: HashMap::new(),
            radices,
            contracted_indices: vec![],
        }
    }

    /// Output indices stick out from the front of the Circuit Tensor Network.
    ///
    /// These correspond to wires exiting a circuit in a normal circuit diagram.
    pub fn open_output_indices(&self) -> Vec<usize> {
        self.front.iter()
            .enumerate()
            .filter(|(_, wire)| wire.is_active())
            .map(|(id, _)| id)
            .collect()
    }

    /// Input indices stick out from the rear of the Circuit Tensor Network.
    ///
    /// These correspond to wires entering a circuit in a normal circuit diagram.
    pub fn open_input_indices(&self) -> Vec<usize> {
        self.rear.iter()
            .enumerate()
            .filter(|(_, wire)| wire.is_active())
            .map(|(id, _)| id)
            .collect()
    }

    pub fn num_open_output_indices(&self) -> usize {
        self.front.iter()
            .filter(|wire| wire.is_active())
            .count()
    }

    pub fn num_open_input_indices(&self) -> usize {
        self.rear.iter()
            .filter(|wire| wire.is_active())
            .count()
    }

    /// Prepend a tensor onto the circuit network.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to prepend
    ///
    /// * `input_qudit_map` - An array of qudit ids. It maps the tensors input indices to the
    ///     qudits at the front of the network that will be connected. `input_qudit_map[i] ==
    ///     qudit_id` implies that `tensor.input_indices()[i]` will be connected to the
    ///     `front[qudit_id]` edge.
    ///
    /// * `output_qudit_map` - An array of qudit ids. It maps the tensors output indices to
    ///     the qudits at the front of the network which will become the new open edges for
    ///     the network front. `output_qudit_map[i] == qudit_id` implies that
    ///     `tensor.output_tensor_indices()[i]` will be the open edge on qudit_id after the
    ///     operation.
    ///
    /// * `batch_index_map` - An array of strings. It provides names for the tensors batch
    ///     indices. All batch indices in the network with the same name identify the same
    ///     network indices. These indices can appear on both sides of a pairwise contraction
    ///     without being contracted over.
    ///
    /// # Panics
    ///
    /// - If the length of an index map doesn't match the number of indices the tensor has in that
    /// direction.
    ///
    /// - If any of the `qudit_ids` referenced by the index maps are invalid or out of bounds.
    ///
    /// - If the size of a tensor's index doesn't match the radix of the qudit it's mapped to.
    ///
    /// - If a batch index with the same name exists in the network, they must have the same
    /// dimension.
    pub fn prepend(
        mut self,
        tensor: QuditTensor,
        input_index_map: Vec<usize>,
        output_index_map: Vec<usize>,
        batch_index_map: Vec<String>,
    ) -> Self {
        // Check error conditions
        let batch_tensor_indices = tensor.batch_indices();
        let output_tensor_indices = tensor.output_indices();
        let input_tensor_indices = tensor.input_indices();
        let batch_tensor_index_sizes = tensor.batch_sizes();
        let output_tensor_index_sizes = tensor.output_sizes();
        let input_tensor_index_sizes = tensor.input_sizes();

        if batch_tensor_indices.len() != batch_index_map.len() {
            panic!("Batch tensor indices and batch qudit map lengths do not match");
        }

        if output_tensor_indices.len() != output_index_map.len() {
            panic!("Output tensor indices and output qudit map lengths do not match");
        }

        if input_tensor_indices.len() != input_index_map.len() {
            panic!("Input tensor indices and input qudit map lengths do not match");
        }

        for (i, qudit_id) in output_index_map.iter().enumerate() {
            if *qudit_id >= self.radices.len() {
                panic!("Qudit id {qudit_id} is out of bounds from tensor's output map");
            }
            assert_eq!(
                self.radices[*qudit_id] as IndexSize,
                output_tensor_index_sizes[i],
                "Tensor index size doesn't match mapped qudit radix.",
            );
        }

        for (i, qudit_id) in input_index_map.iter().enumerate() {
            if *qudit_id >= self.radices.len() {
                panic!("Qudit id {qudit_id} is out of bounds from tensor's input map");
            }
            assert_eq!(
                self.radices[*qudit_id] as IndexSize,
                input_tensor_index_sizes[i],
                "Tensor index size doesn't match mapped qudit radix.",
            );
        }

        // Add Tensor to network
        let tensor_id = self.tensors.len();
        let mut tensor_local_to_network_map = vec![None; tensor.num_indices()];
        self.tensors.push(tensor);

        // Handle Input Indices
        let mut new_contraction_ids: BTreeMap<usize, usize> = BTreeMap::new();
        for (tensor_input_idx_id, qudit_id) in input_index_map.iter().enumerate() {
            let local_index_id = input_tensor_indices[tensor_input_idx_id];
            match self.front[*qudit_id] {
                Wire::Empty => {
                    self.rear[*qudit_id] = Wire::Connected(tensor_id, local_index_id);
                    tensor_local_to_network_map[local_index_id] = Some(NetworkBuilderIndex::Rear(*qudit_id));
                },
                Wire::Closed => {
                    panic!("Cannot contract tensor index with a closed qudit.");
                }
                Wire::Connected(existing_tensor_id, existing_local_index_id) => {
                    // Record a new or update existing contraction between existing_tensor_id and
                    // tensor_id.

                    let contraction_id = *new_contraction_ids.entry(existing_tensor_id).or_insert_with(|| {
                        let id = self.contracted_indices.len();
                        self.contracted_indices.push(ContractionIndex {
                            left_id: tensor_id,
                            right_id: existing_tensor_id,
                            total_dimension: 1,
                        });
                        id
                    });
                    
                    self.contracted_indices[contraction_id].total_dimension *= self.radices[*qudit_id] as IndexSize;
                    self.local_to_network_index_map[existing_tensor_id][existing_local_index_id] = NetworkBuilderIndex::Contraction(contraction_id);
                    tensor_local_to_network_map[local_index_id] = Some(NetworkBuilderIndex::Contraction(contraction_id));
                }
            }

            if !output_index_map.contains(qudit_id) {
                // The network wire becomes inactive/closed if this tensor is the first one on it
                // without a corresponding open (output) edge leaving it.
                self.front[*qudit_id] = Wire::Closed;
            }
        }

        // Handle Output Indices
        for (tensor_output_idx_id, qudit_id) in output_index_map.iter().enumerate() {
            let local_index_id = output_tensor_indices[tensor_output_idx_id];
            if self.front[*qudit_id].is_active() && !input_index_map.contains(&qudit_id) {
                panic!("Cannot map a tensor output qudit over an active edge without connecting or closing it.");
            }
            tensor_local_to_network_map[local_index_id] = Some(NetworkBuilderIndex::Front(*qudit_id));
            self.front[*qudit_id] = Wire::Connected(tensor_id, local_index_id);
        }

        // Handle Batch Indices
        for (tensor_batch_idx_id, batch_idx_name) in batch_index_map.into_iter().enumerate() {
            let local_index_id = batch_tensor_indices[tensor_batch_idx_id];
            let batch_tensor_index_size = batch_tensor_index_sizes[tensor_batch_idx_id];

            match self.batch_indices.get(&batch_idx_name) {
                Some(index_size) => {assert_eq!(batch_tensor_index_size, *index_size);}
                None => {self.batch_indices.insert(batch_idx_name.clone(), batch_tensor_index_size);}
            }

            tensor_local_to_network_map[local_index_id] = Some(NetworkBuilderIndex::Batch(batch_idx_name));
        }

        // Finalize tensor addition
        self.local_to_network_index_map.push(
            tensor_local_to_network_map.into_iter().map(|idx| match idx {
                Some(idx) => idx,
                None => panic!("Failed to map local tensor index to network index."),
            }).collect()
        );

        self
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

    pub fn prepend_unitary<C: ComplexScalar>(self, utry: UnitaryMatrix<C>, qudits: Vec<usize>) -> Self {
        self.prepend(QuditTensor::from(utry), qudits.clone(), qudits, vec![])
    }

    pub fn trace_wire(mut self, front_qudit: usize, rear_qudit: usize) -> Self {
        assert_eq!(self.radices[front_qudit], self.radices[rear_qudit]);
        assert!(self.front[front_qudit].is_active() && self.rear[rear_qudit].is_active());

        if self.front[front_qudit].is_empty() {
            let identity = QuditTensor::identity(self.radices[front_qudit].into());
            self = self.prepend(identity, [front_qudit].into(), [front_qudit].into(), vec![]);
        }

        if self.rear[rear_qudit].is_empty() {
            let identity = QuditTensor::identity(self.radices[rear_qudit].into());
            self = self.prepend(identity, [rear_qudit].into(), [rear_qudit].into(), vec![]);
        }

        match (&self.front[front_qudit], &self.rear[rear_qudit]) {
            (Wire::Connected(tid_f, local_id_f), Wire::Connected(tid_r, local_id_r)) => {
                debug_assert_eq!(self.local_to_network_index_map[*tid_f][*local_id_f], NetworkBuilderIndex::Front(front_qudit));
                debug_assert_eq!(self.local_to_network_index_map[*tid_r][*local_id_r], NetworkBuilderIndex::Rear(rear_qudit));

                // Find an existing contraction between tid_f and tid_r or create new one
                let contraction_id = {
                    let mut contraction_id = None;
                    for net_index in &self.local_to_network_index_map[*tid_f] {
                        if let NetworkBuilderIndex::Contraction(cid) = net_index {
                            let contraction = &self.contracted_indices[*cid];
                            if contraction.left_id == *tid_r || contraction.right_id == *tid_r {
                                contraction_id = Some(*cid);
                                break;
                            }
                        }
                    }
                    contraction_id.unwrap_or_else(|| {
                        let cid = self.contracted_indices.len();
                        self.contracted_indices.push(ContractionIndex {
                            left_id: *tid_f,
                            right_id: *tid_r,
                            total_dimension: self.radices[front_qudit] as IndexSize,
                        });
                        cid
                    })
                };

                self.local_to_network_index_map[*tid_f][*local_id_f] = NetworkBuilderIndex::Contraction(contraction_id);
                self.local_to_network_index_map[*tid_r][*local_id_r] = NetworkBuilderIndex::Contraction(contraction_id);
             },
            _ => panic!("Cannot connect a closed wire to another wire.")
        }

        self.front[front_qudit] = Wire::Closed;
        self.rear[rear_qudit] = Wire::Closed;
        self
    }

    pub fn trace_all_open_wires(mut self) -> Self {
        assert_eq!(self.num_open_input_indices(), self.num_open_output_indices());
        for (f, r) in self.open_output_indices().into_iter().zip(self.open_input_indices().into_iter()) {
            self = self.trace_wire(f, r);
        }
        self
    }

    pub fn build(self) -> QuditTensorNetwork {
        let QuditCircuitTensorNetworkBuilder {
            mut tensors,
            mut local_to_network_index_map,
            front,
            rear,
            batch_indices,
            contracted_indices,
            ..
        } = self;

        // Build network indices, while building map from builder to network
        let mut indices = Vec::new();
        let mut builder_to_network_map = HashMap::new();

        for (batch_idx_name, batch_idx_size) in batch_indices.into_iter() {
            let index_id = indices.len();
            indices.push(NetworkIndex::Output(
                TensorIndex::new(
                    IndexDirection::Batch,
                    index_id,
                    batch_idx_size,
                )
            ));
            builder_to_network_map.insert(
                NetworkBuilderIndex::Batch(batch_idx_name),
                index_id
            );
        }

        for (qudit_id, wire) in front.into_iter().enumerate() {
            if wire.is_empty() {
                // Cannot have empty indices in network, so we need to explicitly add identity.
                let identity = QuditTensor::identity(self.radices[qudit_id].into());
                let tensor_id = tensors.len();
                tensors.push(identity);
                local_to_network_index_map.push(vec![NetworkBuilderIndex::Front(qudit_id), NetworkBuilderIndex::Rear(qudit_id)]);
            }

            if wire.is_active() {
                let index_id = indices.len();
                indices.push(NetworkIndex::Output(
                    TensorIndex::new(
                        IndexDirection::Output,
                        index_id,
                        self.radices[qudit_id] as IndexSize,
                    )
                ));
                builder_to_network_map.insert(
                    NetworkBuilderIndex::Front(qudit_id),
                    index_id
                );
            }
        }

        for (qudit_id, wire) in rear.into_iter().enumerate() {
            if wire.is_active() {
                let index_id = indices.len();
                indices.push(NetworkIndex::Output(
                    TensorIndex::new(
                        IndexDirection::Input,
                        index_id,
                        self.radices[qudit_id] as IndexSize,
                    )
                ));
                builder_to_network_map.insert(
                    NetworkBuilderIndex::Rear(qudit_id),
                    index_id
                );
            }
        }

        for (cidx_id, contraction_index) in contracted_indices.into_iter().enumerate() {
            let index_id = indices.len();
            indices.push(NetworkIndex::Contracted(contraction_index));
            builder_to_network_map.insert(NetworkBuilderIndex::Contraction(cidx_id), index_id);
        }

        let mut index_edges: Vec<NetworkEdge> = indices.into_iter().map(|x| (x, BTreeSet::new())).collect();

        let new_index_map = local_to_network_index_map.into_iter()
            .enumerate()
            .map(|(tid, tidx_map)| {
                tidx_map.into_iter()
                    .map(|index| {
                        let network_index = builder_to_network_map[&index];
                        index_edges[network_index].1.insert(tid);
                        network_index
                    })
                    .collect::<Vec<IndexId>>()
            })
            .collect::<Vec<Vec<IndexId>>>();

        QuditTensorNetwork::new(
            tensors,
            new_index_map,
            index_edges,
        )
    }
}


