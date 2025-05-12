use std::fmt;
use std::hash::Hash;

use qudit_core::HasPeriods;
use qudit_core::HasParams;
use qudit_core::ParamIndices;
use qudit_core::RealScalar;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;
use qudit_core::QuditPermutation;
use qudit_expr::TensorGenerationShape;

use super::fmt::PrintTree;
use super::tree::ExpressionTree;

/// A permutation node in the computation tree.
/// This node wraps another node and applies a permutation to its output.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TransposeNode {
    /// The child node to be permuted.
    pub child: Box<ExpressionTree>,

    /// The permutation to apply to the child node.
    pub perm: Vec<usize>,

    pub split_at: usize,

    // num_params: usize,
    //
    /// The dimension of each of the resulting tensor's indices.
    pub dimensions: QuditRadices,

    /// The shape of the tensor after computation.
    pub generation_shape: TensorGenerationShape,
}

impl TransposeNode {
    pub fn new(child: ExpressionTree, perm: Vec<usize>, split_at: usize) -> TransposeNode {
        let child_dimensions = child.dimensions();
        let dimensions: QuditRadices = (0..perm.len()).map(|x| child_dimensions[perm[x]]).collect();
        // let dimensions: QuditRadices = child.dimensions().iter().enumerate().map(|(i, x)| perm[*x as usize]).collect();
        let left_dimension = dimensions[0..split_at].iter().map(|x| *x as usize).product();
        let right_dimension = dimensions[split_at..].iter().map(|x| *x as usize).product();
        let shape = TensorGenerationShape::Matrix(left_dimension, right_dimension);
        // let num_params = child.num_params();
        // let _radices = child.radices();

        // if perm.num_qudits() != child.num_qudits() {
        //     panic!("Number of qudits in permutation must match number of qudits in node.");
        // }

        // if perm.radices() != child.radices() {
        //     panic!("Radices of permutation must match radices of node.");
        // }

        TransposeNode {
            child: Box::new(child),
            perm,
            // num_params,
            split_at,
            dimensions,
            generation_shape: shape,
        }
    }

    pub fn dimensions(&self) -> QuditRadices {
        self.dimensions.clone()
    }

    pub fn generation_shape(&self) -> TensorGenerationShape {
        self.generation_shape.clone()
    }

    pub fn set_generation_shape(&mut self, gen_shape: TensorGenerationShape) {
        self.generation_shape = gen_shape;
    }

    pub fn param_indices(&self) -> ParamIndices {
        self.child.param_indices()
    }
}

// impl HasParams for PermNode {
//     fn num_params(&self) -> usize {
//         self.num_params
//     }
// }

// impl<R: RealScalar> HasPeriods<R> for PermNode {
//     fn periods(&self) -> Vec<std::ops::Range<R>> {
//         self.child.periods()
//     }
// }

// impl QuditSystem for PermNode {
//     fn radices(&self) -> QuditRadices {
//         self.perm.permuted_radices()
//     }

//     fn dimension(&self) -> usize {
//         self.dimension
//     }
// }
// //

impl fmt::Debug for TransposeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Transpose")
            .field("child", &self.child)
            .field("perm", &self.perm)
            .finish()
    }
}

impl PrintTree for TransposeNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}Transpose({:?})", prefix, self.perm).unwrap();
        let child_prefix = self.modify_prefix_for_child(prefix, true);
        self.child.write_tree(&child_prefix, fmt);
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::math::perm::strategies::{perms, perms_with_radices};
//     use crate::math::UnitaryBuilder;
//     use crate::sim::node::{strategies::arbitrary_nodes, Node};
//     // use crate::strategies::params;
//     use proptest::prelude::*;

//     fn nodes_and_perms() -> impl Strategy<Value = (Node, QuditPermutation)> {
//         arbitrary_nodes()
//             .prop_flat_map(|node| (Just(node.clone()),
// perms_with_radices(node.get_radices())))     }

//     fn nodes_and_perms_with_params() -> impl Strategy<Value = ((Node,
// QuditPermutation), Vec<f64>)>     {
//         nodes_and_perms().prop_flat_map(|(node, perm)| {
//             let num_params = node.get_num_params().clone();
//             (Just((node, perm)), params(num_params))
//         })
//     }

//     fn nodes_with_mismatched_size_perms() -> impl Strategy<Value = (Node,
// QuditPermutation)> {         (arbitrary_nodes(), 2..8usize)
//             .prop_filter(
//                 "Number of qudits in permutation must mismatch with node
// size.",                 |(node, n)| node.get_num_qudits() != *n,
//             )
//             .prop_flat_map(|(node, n)| (Just(node.clone()), perms(n)))
//     }

//     fn nodes_with_mismatched_radices_perms() -> impl Strategy<Value = (Node,
// QuditPermutation)> {         arbitrary_nodes()
//             .prop_flat_map(|node| (Just(node.clone()),
// perms(node.get_num_qudits())))             .prop_filter(
//                 "Permutation radices must mismatch with node radices",
//                 |(node, perm)| node.get_radices() != perm.get_radices(),
//             )
//     }

//     proptest! {
//         #[test]
//         fn does_not_crash((node, perm) in nodes_and_perms()) {
//             let _ = Node::Perm(PermNode::new(node, perm));
//         }

//         #[test]
//         #[should_panic(expected = "Number of qudits in permutation must match
// number of qudits in node.")]         fn
// panics_on_perm_mismatch_size_with_node((node, perm) in
// nodes_with_mismatched_size_perms()) {
// std::panic::set_hook(Box::new(|_| {})); // To not print stack trace
//             let _ = Node::Perm(PermNode::new(node, perm));
//         }

//         #[test]
//         #[should_panic(expected = "Radices of permutation must match radices
// of node.")]         fn panics_on_perm_mismatch_radices_with_node((node, perm)
// in nodes_with_mismatched_radices_perms()) {
// std::panic::set_hook(Box::new(|_| {})); // To not print stack trace
//             let _ = Node::Perm(PermNode::new(node, perm));
//         }

//         #[test]
//         fn unitary_equals_permuted_node_unitary(((mut node, perm), params) in
// nodes_and_perms_with_params()) {             let mut perm_node =
// Node::Perm(PermNode::new(node.clone(), perm.clone()));             let
// utry_ref = node.get_unitary_ref(&params);             let perm_utry_ref =
// perm_node.get_unitary_ref(&params);             let p = perm.get_matrix();
//             assert_eq!(p.t().dot(utry_ref).dot(&p), perm_utry_ref);
//         }

//         // #[test]
//         // fn unitary_equals_unitary_builder_unitary(((mut node, perm),
// params) in nodes_and_perms_with_params()) {         //     let mut perm_node
// = Node::Perm(PermNode::new(node.clone(), perm.clone()));         //     let
// mut builder = UnitaryBuilder::new(node.get_radices());         //
// builder.apply_right(node.get_unitary_ref(&params).view(), &perm, false);
//         //     let utry = builder.get_unitary();
//         //     let perm_utry_ref = perm_node.get_unitary_ref(&params);
//         //     assert_eq!(utry, perm_utry_ref);
//         // }

//         #[test]
//         fn gradient_equals_permuted_node_gradient(((mut node, perm), params)
// in nodes_and_perms_with_params()) {             let mut perm_node =
// Node::Perm(PermNode::new(node.clone(), perm.clone()));             let
// grad_ref = node.get_gradient_ref(&params);             let perm_grad_ref =
// perm_node.get_gradient_ref(&params);             let p = perm.get_matrix();
//             for (i, d_m) in grad_ref.outer_iter().enumerate() {
//                 let perm_d_m = perm_grad_ref.index_axis(Axis(0), i);
//                 assert_eq!(p.t().dot(&d_m).dot(&p), perm_d_m);
//             }
//         }

//         #[test]
//         fn unitary_and_gradient_equals_permuted_node(((mut node, perm),
// params) in nodes_and_perms_with_params()) {             let mut perm_node =
// Node::Perm(PermNode::new(node.clone(), perm.clone()));             let
// (utry_ref, grad_ref) = node.get_unitary_and_gradient_ref(&params);
//             let (perm_utry_ref, perm_grad_ref) =
// perm_node.get_unitary_and_gradient_ref(&params);             let p =
// perm.get_matrix();             assert_eq!(p.t().dot(utry_ref).dot(&p),
// perm_utry_ref);             for (i, d_m) in grad_ref.outer_iter().enumerate()
// {                 let perm_d_m = perm_grad_ref.index_axis(Axis(0), i);
//                 assert_eq!(p.t().dot(&d_m).dot(&p), perm_d_m);
//             }
//         }

//         #[test]
//         fn num_params_equals_node((node, perm) in nodes_and_perms()) {
//             let perm_node = Node::Perm(PermNode::new(node.clone(),
// perm.clone()));             assert_eq!(node.get_num_params(),
// perm_node.get_num_params());         }

//         #[test]
//         fn radices_equals_perm_radices((node, perm) in nodes_and_perms()) {
//             let perm_node = Node::Perm(PermNode::new(node.clone(),
// perm.clone()));             assert_eq!(perm.get_permuted_radices(),
// perm_node.get_radices());         }

//         #[test]
//         fn radices_equals_permuted_node_radices((node, perm) in
// nodes_and_perms()) {             let perm_node =
// Node::Perm(PermNode::new(node.clone(), perm.clone()));             let
// node_radices = node.get_radices();             let perm_radices =
// perm_node.get_radices();             assert!(perm
//                 .iter()
//                 .enumerate()
//                 .all(|(s, d)| node_radices[*d] == perm_radices[s])
//             );
//         }

//         #[test]
//         fn dimension_equals_node((node, perm) in nodes_and_perms()) {
//             let perm_node = Node::Perm(PermNode::new(node.clone(),
// perm.clone()));             assert_eq!(node.get_dimension(),
// perm_node.get_dimension());         }

//         #[test]
//         fn is_hashable((node, perm) in nodes_and_perms()) {
//             let perm_node = Node::Perm(PermNode::new(node.clone(),
// perm.clone()));             let mut hasher =
// std::collections::hash_map::DefaultHasher::new();
// perm_node.hash(&mut hasher);             let _ = hasher.finish();
//         }

//         #[test]
//         fn is_hashable_set_insert((node, perm) in nodes_and_perms()) {
//             let perm_node = Node::Perm(PermNode::new(node.clone(),
// perm.clone()));             let mut set = std::collections::HashSet::new();
//             set.insert(perm_node.clone());
//             assert!(set.contains(&perm_node));
//         }

//         #[test]
//         fn equals_have_equal_hashes((node, perm) in nodes_and_perms()) {
//             let perm_node1 = Node::Perm(PermNode::new(node.clone(),
// perm.clone()));             let perm_node2 =
// Node::Perm(PermNode::new(node.clone(), perm.clone()));
// assert_eq!(perm_node1, perm_node2);             let mut hasher1 =
// std::collections::hash_map::DefaultHasher::new();             let mut hasher2
// = std::collections::hash_map::DefaultHasher::new();
// perm_node1.hash(&mut hasher1);             perm_node2.hash(&mut hasher2);
//             assert_eq!(hasher1.finish(), hasher2.finish());
//         }

//     }
// }
