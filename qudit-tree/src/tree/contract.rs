use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
use std::ops::Deref;

use qudit_core::HasPeriods;
use qudit_core::HasParams;
use qudit_core::RealScalar;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;

use super::fmt::PrintTree;
use super::tree::ExpressionTree;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct ContractNode {
    /// The left node to be contracted.
    pub left: Box<ExpressionTree>,

    /// The right node to be contracted.
    pub right: Box<ExpressionTree>,

    // The qudit indices of the left node in circuit space.
    pub left_qudits: Vec<usize>,

    // The qudit indices of the right node in circuit space.
    pub right_qudits: Vec<usize>,

    /// The number of parameters in the left node.
    left_params: usize,

    /// The number of parameters in the right node.
    right_params: usize,

    /// The normal unfused output dimension of this node.
    dimension: usize,

    /// The normal output tensor shape after contraction and final permutation.
    out_tensor_shape: Vec<u8>,

    /// The shape of the left node as a tensor.
    pub left_tensor_shape: Vec<u8>,

    /// The permutation of the left node's indices as a tensor.
    pub left_perm: Vec<usize>,

    /// The shape of the left node after permutation before contraction.
    pub left_contraction_shape: (usize, usize),

    /// The shape of the right node as a tensor.
    pub right_tensor_shape: Vec<u8>,

    /// The permutation of the right node's indices as a tensor.
    pub right_perm: Vec<usize>,

    /// The shape of the right node after permutation before contraction.
    pub right_contraction_shape: (usize, usize),

    /// The output tensor shape after contraction before final permutation.
    pub pre_out_tensor_shape: Vec<usize>,

    /// The final permutation of this node's indices as a tensor.
    pub pre_out_perm: Vec<usize>,

    // The shape of the output matrix after contraction before permutation.
    pub out_matrix_shape: (usize, usize),

    // If the left node is already properly permuted, we skip the
    // left pre-permutation. This is always initially false and can be
    // set by the [TreeOptimizer](struct.TreeOptimizer).
    pub skip_left: bool,

    // If the right node is already properly permuted, we skip the
    // right pre-permutation. This is always initially false and can be
    // set by the [TreeOptimizer](struct.TreeOptimizer).
    pub skip_right: bool,
}

impl ContractNode {
    /// Creates a new ContractNode that contracts two nodes.
    ///
    /// The left and right nodes will be contracted along shared qudits.
    ///
    /// # Arguments
    ///
    /// * `left` - The left node to be contracted.
    /// * `right` - The right node to be contracted.
    /// * `left_qudits` - The qudit indices of the left node in circuit space.
    /// * `right_qudits` - The qudit indices of the right node in circuit space.
    /// # Panics
    ///
    /// * If there are no overlapping qudits between the left and right nodes.
    /// * If the indices being contracted have different dimensions/radix.
    pub fn new(
        left: ExpressionTree,
        right: ExpressionTree,
        left_qudits: Vec<usize>, // Change to CircuitLocation
        right_qudits: Vec<usize>,
    ) -> ContractNode {
        // The radices of each node
        let left_radices = left.radices();
        let right_radices = right.radices();

        // The qudits shared in left_qudits and right_qudits will be contracted.
        let left_qudit_set =
            left_qudits.iter().map(|&x| x).collect::<HashSet<_>>();
        let right_qudit_set =
            right_qudits.iter().map(|&x| x).collect::<HashSet<_>>();
        let contracting_qudits = left_qudit_set
            .intersection(&right_qudit_set)
            .map(|&x| x)
            .collect::<Vec<_>>();
        let mut all_qudits: Vec<_> = left_qudit_set
            .union(&right_qudit_set)
            .cloned()
            .collect();
        all_qudits.sort();

        if contracting_qudits.len() == 0 {
            panic!("There must be at least one overlapping qudit between the left and right nodes.")
        }

        // The radix_map maps qudit indices in circuit space to their radix.
        let mut radix_map: HashMap<usize, u8> = HashMap::new();
        for q in all_qudits.iter() {
            if contracting_qudits.deref().contains(q) {
                let left_qudit_index =
                    left_qudits.iter().position(|x| x == q).unwrap();
                let right_qudit_index =
                    right_qudits.iter().position(|x| x == q).unwrap();
                let left_radix = &left_radices[left_qudit_index];
                let right_radix = &right_radices[right_qudit_index];

                if left_radix != right_radix {
                    panic!("The indices being contracted must have the same dimension/radix.")
                }

                radix_map.insert(*q, *left_radix);
            } else if left_qudit_set.contains(q) {
                let left_qudit_index =
                    left_qudits.iter().position(|x| x == q).unwrap();
                let left_radix = &left_radices[left_qudit_index];
                radix_map.insert(*q, *left_radix);
            } else {
                let right_qudit_index =
                    right_qudits.iter().position(|x| x == q).unwrap();
                let right_radix = &right_radices[right_qudit_index];
                radix_map.insert(*q, *right_radix);
            }
        }

        // `left_perm` captures the permutation necessary to pre-process
        // the left node's tensor indices before contraction as a
        // reshape-matmul operation.
        let mut left_perm = Vec::new();
        for q in left_qudits.iter() {
            if contracting_qudits.deref().contains(&q) {
                left_perm
                    .push(left_qudits.iter().position(|x| x == q).unwrap());
            }
        }
        for q in left_qudits.iter() {
            if !contracting_qudits.deref().contains(&q) {
                left_perm
                    .push(left_qudits.iter().position(|x| x == q).unwrap());
            }
        }
        for q in 0..left_qudits.len() {
            left_perm.push(q + left_qudits.len());
        }

        // `right_perm` captures the permutation necessary to pre-process
        // the right node's tensor indices before contraction as a
        // reshape-matmul operation.
        let mut right_perm = Vec::new();
        for q in 0..right_qudits.len() {
            right_perm.push(q);
        }
        for q in right_qudits.iter() {
            if !contracting_qudits.deref().contains(&q) {
                right_perm.push(
                    right_qudits.iter().position(|x| x == q).unwrap()
                        + right_qudits.len(),
                );
            }
        }
        for q in right_qudits.iter() {
            if contracting_qudits.deref().contains(&q) {
                right_perm.push(
                    right_qudits.iter().position(|x| x == q).unwrap()
                        + right_qudits.len(),
                );
            }
        }

        // `pre_out_perm` captures the permutation necessary to post-process
        // the output of the contraction as a reshape-matmul operation.
        // In order to achieve this, we track how the operation will permute
        // the uncontracted qudit indices in the local space.
        let mut left_idx_to_qudit_map: Vec<String> = left_qudits
            .iter()
            .map(|q| format!("{}r", q)) // r for right
            .chain(left_qudits.iter().map(|q| format!("{}l", q))) // l for left
            .collect(); // Build qudit index labels in circuit space

        // Apply the permutation to the labels
        left_idx_to_qudit_map = left_perm
            .iter()
            .map(|&i| left_idx_to_qudit_map[i].clone())
            .collect();

        // Do the same with the right qudit index labels
        let mut right_idx_to_qudit_map: Vec<String> = right_qudits
            .iter()
            .map(|q| format!("{}r", q))
            .chain(right_qudits.iter().map(|q| format!("{}l", q)))
            .collect();

        // Apply the permutation to the labels
        right_idx_to_qudit_map = right_perm
            .iter()
            .map(|&i| right_idx_to_qudit_map[i].clone())
            .collect();

        // Build the correct output order of qudit index labels
        let correct_order: Vec<String> = all_qudits
            .iter()
            .map(|q| format!("{}r", q))
            .chain(all_qudits.iter().map(|q| format!("{}l", q)))
            .collect();

        // Build the pre-permutation output order of qudit index labels
        let num_contracting_qudits = contracting_qudits.len();
        let right_pre_out_order: Vec<String> = right_idx_to_qudit_map
            [..right_idx_to_qudit_map.len() - num_contracting_qudits]
            .to_vec();
        let left_pre_out_order: Vec<String> =
            left_idx_to_qudit_map[num_contracting_qudits..].to_vec();
        let pre_out_order: Vec<&String> = right_pre_out_order
            .iter()
            .chain(left_pre_out_order.iter())
            .collect();

        // The permutation necessary to post-process the output of the
        // contraction is now given as the permutation that maps the
        // pre_out_order to the correct_order
        let pre_out_perm: Vec<usize> = correct_order
            .iter()
            .map(|idx| pre_out_order.iter().position(|&q| q == idx).unwrap())
            .collect();
        // Note: this output permutation is a permutation of tensor indices
        // that cannot be captured by a QuditPermutation object, since it
        // is asymmetric. This means output/right indices might be permuted
        // with input/left indices in the contraction.

        let overlap_dimension = contracting_qudits
            .iter()
            .map(|q| radix_map[q] as usize)
            .product::<usize>();

        let pre_out_tensor_shape: Vec<usize> = pre_out_order
            .iter()
            .map(|qstr| {
                radix_map[&qstr[..qstr.len() - 1].parse::<usize>().unwrap()].into()
            })
            .collect();

        let out_tensor_shape: Vec<u8> = correct_order
            .iter()
            .map(|qstr| {
                radix_map[&qstr[..qstr.len() - 1].parse::<usize>().unwrap()]
            })
            .collect();

        let left_dimension = left.dimension();
        let right_dimension = right.dimension();
        let left_params = left.num_params();
        let right_params = right.num_params();
        let dimension = radix_map.values().map(|&r| r as usize).product();

        let left_contraction_dim =
            left_dimension * left_dimension / overlap_dimension;
        let left_contraction_shape = (overlap_dimension, left_contraction_dim);
        let left_tensor_shape = left_radices
            .iter()
            .chain(left_radices.iter())
            .map(|&r| r)
            .collect::<Vec<_>>();

        let right_contraction_dim =
            right_dimension * right_dimension / overlap_dimension;
        let right_contraction_shape =
            (right_contraction_dim, overlap_dimension);
        let right_tensor_shape = right_radices
            .iter()
            .chain(right_radices.iter())
            .map(|&r| r)
            .collect::<Vec<_>>();

        let out_matrix_shape = (dimension, dimension);

        ContractNode {
            left: Box::new(left),
            right: Box::new(right),
            left_qudits,
            right_qudits,
            left_params,
            right_params,
            dimension,
            out_tensor_shape,

            left_tensor_shape,
            left_perm,
            left_contraction_shape,

            right_tensor_shape,
            right_perm,
            right_contraction_shape,

            pre_out_tensor_shape,
            pre_out_perm,
            out_matrix_shape,

            skip_left: false,
            skip_right: false,
        }
    }

    pub(super) fn skip_left_permutation(&mut self) {
        self.skip_left = true;
    }

    pub(super) fn skip_right_permutation(&mut self) {
        self.skip_right = true;
    }

    pub(super) fn fuse_output_perm(
        &mut self,
        perm: Vec<usize>,
        new_shape: (usize, usize),
    ) {
        // permute pre_out_perm according to perm
        self.pre_out_perm =
            perm.iter().map(|&i| self.pre_out_perm[i]).collect();

        self.out_matrix_shape = new_shape;
    }

    // TODO: Optimize permutation shape (consecutive indices do not need to be
    // split)
}

impl HasParams for ContractNode {
    fn num_params(&self) -> usize {
        self.left_params + self.right_params
    }
}

impl<R: RealScalar> HasPeriods<R> for ContractNode {
    fn periods(&self) -> Vec<std::ops::Range<R>> {
        self.left
            .periods()
            .iter()
            .chain(self.right.periods().iter())
            .cloned()
            .collect()
    }
}

impl QuditSystem for ContractNode {
    fn radices(&self) -> QuditRadices {
        QuditRadices::from_iter(
            (0..(self.out_tensor_shape.len() / 2))
                .map(|x| self.out_tensor_shape[x])
        )
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

impl PrintTree for ContractNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(
            fmt,
            "{}Contract({:?} + {:?}; {}, {})",
            prefix,
            self.left_qudits,
            self.right_qudits,
            self.skip_left,
            self.skip_right
        )
        .unwrap();
        let left_prefix = self.modify_prefix_for_child(prefix, false);
        let right_prefix = self.modify_prefix_for_child(prefix, true);
        self.left.write_tree(&left_prefix, fmt);
        self.right.write_tree(&right_prefix, fmt);
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // use crate::math::UnitaryBuilder;
    // use crate::sim::kron::KronNode;
    // use crate::sim::leaf::LeafStruct;
    // use crate::sim::node::Node;
    // use crate::Gate;

    // #[test]
    // fn two_qubit_test() {
    //     let qudits1 = vec![0, 1];
    //     let qudits2 = vec![1];

    //     let mut leaf_r2 = Node::Leaf(LeafStruct::new(Gate::RZ));

    //     let mut node1 = Node::Kron(KronNode::new(leaf_r2.clone(),
    // leaf_r2.clone()));

    //     let mut contract_node = Node::Contract(ContractNode::new(
    //         node1.clone(),
    //         leaf_r2.clone(),
    //         qudits1,
    //         qudits2,
    //     ));

    //     let contract_utry = contract_node.get_unitary_ref(&[1.0, 2.0, 3.0]);

    //     let mut builder = UnitaryBuilder::new(QuditRadices::new(vec![2, 2]));
    //     builder.apply_right(node1.get_unitary_ref(&[1.0, 2.0]).view(), &[0,
    // 1], false);     builder.apply_right(leaf_r2.get_unitary_ref(&[3.0]).
    // view(), &[1], false);     let ans_utry = builder.get_unitary();
    //     assert!((contract_utry - ans_utry).opnorm_fro().unwrap() < 1e-8);
    // }

    // #[test]
    // fn five_qudit_test() {
    //     let qudits1 = vec![4, 11, 8];
    //     let qudits2 = vec![7, 13, 11];

    //     let leaf_r2 = Node::Leaf(LeafStruct::new(Gate::RZ));
    //     let leaf_r3 = Node::Leaf(LeafStruct::new(Gate::QutritRZ));
    //     let leaf_r4 = Node::Leaf(LeafStruct::new(Gate::QuditT(4, 0, 3)));

    //     let mut node1 = Node::Kron(KronNode::new(
    //         Node::Kron(KronNode::new(leaf_r3.clone(), leaf_r4.clone())),
    //         leaf_r2.clone(),
    //     ));
    //     let mut node2 = Node::Kron(KronNode::new(
    //         Node::Kron(KronNode::new(leaf_r2.clone(), leaf_r2.clone())),
    //         leaf_r4.clone(),
    //     ));

    //     let mut contract_node = Node::Contract(ContractNode::new(
    //         node1.clone(),
    //         node2.clone(),
    //         qudits1,
    //         qudits2,
    //     ));

    //     let contract_utry = contract_node.get_unitary_ref(&[1.0, 1.0, 1.0,
    // 1.0]);

    //     let mut builder = UnitaryBuilder::new(QuditRadices::new(vec![3, 2, 2,
    // 4, 2]));     builder.apply_right(node1.get_unitary_ref(&[1.0,
    // 1.0]).view(), &[0, 3, 2], false);     builder.apply_right(node2.
    // get_unitary_ref(&[1.0, 1.0]).view(), &[1, 4, 3], false);
    //     let ans_utry = builder.get_unitary();
    //     assert!((contract_utry - ans_utry).opnorm_fro().unwrap() < 1e-8);
    // }
}
