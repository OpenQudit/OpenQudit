use std::hash::Hash;

use super::fmt::PrintTree;
use qudit_core::HasPeriods;
use qudit_core::HasParams;
use qudit_core::RealScalar;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;
use super::tree::ExpressionTree;

/// A kron node in the computation tree that stacks two nodes.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct KronNode {
    /// The left node; in the circuit model, this is the top node.
    pub left: Box<ExpressionTree>,

    /// The right node; in the circuit model, this is the bottom node.
    pub right: Box<ExpressionTree>,

    /// The number of parameters in the left node.
    left_params: usize,

    /// The number of parameters in the right node.
    right_params: usize,

    /// The dimension of the left node.
    left_dimension: usize,

    /// The dimension of the right node.
    right_dimension: usize,
}

impl KronNode {
    /// Create a new kron node from two nodes.
    ///
    /// # Arguments
    ///
    /// * `left` - The left node; the top node in the circuit model.
    /// * `right` - The right node; the bottom node in the circuit model.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::Gate;
    /// use qudit_circuit::sim::node::ExpressionTree;
    /// use qudit_circuit::sim::kron::KronNode;
    /// use qudit_circuit::QuditSystem;
    /// let cz_node = ExpressionTree::Leaf(Gate::CZ());
    /// let u3_node = ExpressionTree::Leaf(Gate::U3());
    /// let kron_node = ExpressionTree::Kron(KronNode::new(cz_node, u3_node));
    /// assert_eq!(kron_node.get_num_qudits(), 3);
    /// ```
    pub fn new(left: ExpressionTree, right: ExpressionTree) -> KronNode {
        let left_params = left.num_params();
        let right_params = right.num_params();
        let left_dimension = left.dimension();
        let right_dimension = right.dimension();
        let _left_radices = left.radices();
        let _right_radices = right.radices();

        KronNode {
            left: Box::new(left),
            right: Box::new(right),
            left_params,
            right_params,
            left_dimension,
            right_dimension,
        }
    }
}

impl HasParams for KronNode {
    fn num_params(&self) -> usize {
        self.left_params + self.right_params
    }
}

impl<R: RealScalar> HasPeriods<R> for KronNode {
    fn periods(&self) -> Vec<std::ops::Range<R>> {
        self.left
            .periods()
            .iter()
            .chain(self.right.periods().iter())
            .cloned()
            .collect()
    }
}

impl QuditSystem for KronNode {
    /// Returns the radices of the qudit system this node outputs.
    fn radices(&self) -> QuditRadices {
        self.left.radices() + self.right.radices()
    }

    /// Returns the dimension of this node's unitary.
    fn dimension(&self) -> usize {
        self.left_dimension * self.right_dimension
    }
}

impl PrintTree for KronNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}Kron", prefix).unwrap();
        let left_prefix = self.modify_prefix_for_child(prefix, false);
        let right_prefix = self.modify_prefix_for_child(prefix, true);
        self.left.write_tree(&left_prefix, fmt);
        self.right.write_tree(&right_prefix, fmt);
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::math::UnitaryBuilder;
//     use crate::sim::node::strategies::{arbitrary_nodes, nodes_and_params};
//     use crate::sim::node::Node;
//     use proptest::prelude::*;

//     proptest! {
//         #[test]
//         fn does_not_crash(node1 in arbitrary_nodes(), node2 in
// arbitrary_nodes()) {             let _ = Node::Kron(KronNode::new(node1,
// node2));         }

//         // #[test]
//         // fn unitary_equals_unitary_builder(
//         //     (mut node1, params1) in nodes_and_params(),
//         //     (mut node2, params2) in nodes_and_params(),
//         // ) {
//         //     let mut kron_node = Node::Kron(KronNode::new(node1.clone(),
// node2.clone()));         //     let radices = kron_node.get_radices();
//         //     let kron_utry = kron_node.get_unitary_ref(&[params1.clone(),
// params2.clone()].concat());

//         //     let mut builder = UnitaryBuilder::new(radices);
//         //     let loc1 = Vec::from_iter(0..node1.get_num_qudits());
//         //     let loc2 =
// Vec::from_iter(node1.get_num_qudits()..node1.get_num_qudits() +
// node2.get_num_qudits());         //
// builder.apply_right(node1.get_unitary_ref(&params1).view(), &loc1, false);
//         //     builder.apply_right(node2.get_unitary_ref(&params2).view(),
// &loc2, false);         //     let utry = builder.get_unitary();
//         //     assert!((kron_utry - utry).opnorm_fro().unwrap() < 1e-8);
//         // }

//         #[test]
//         fn num_params_equals_sum_nodes(node1 in arbitrary_nodes(), node2 in
// arbitrary_nodes())         {
//             let kron_node = Node::Kron(KronNode::new(node1.clone(),
// node2.clone()));             let num_params = kron_node.get_num_params();
//             assert_eq!(num_params, node1.get_num_params() +
// node2.get_num_params());         }

//         #[test]
//         fn radices_equals_concat_radices(node1 in arbitrary_nodes(), node2 in
// arbitrary_nodes())         {
//             let kron_node = Node::Kron(KronNode::new(node1.clone(),
// node2.clone()));             let radices = kron_node.get_radices();
//             let concat_radices = node1.get_radices() + node2.get_radices();
//             assert_eq!(radices, concat_radices);
//         }

//         #[test]
//         fn dimension_equals_product_dimension(node1 in arbitrary_nodes(),
// node2 in arbitrary_nodes())         {
//             let kron_node = Node::Kron(KronNode::new(node1.clone(),
// node2.clone()));             let dim = kron_node.get_dimension();
//             let product_dim = node1.get_dimension() * node2.get_dimension();
//             assert_eq!(dim, product_dim);
//         }

//         #[test]
//         fn is_hashable(node1 in arbitrary_nodes(), node2 in
// arbitrary_nodes()) {             let kron_node =
// Node::Kron(KronNode::new(node1.clone(), node2.clone()));             let mut
// hasher = std::collections::hash_map::DefaultHasher::new();
// kron_node.hash(&mut hasher);             let _ = hasher.finish();
//         }

//         #[test]
//         fn is_hashable_set_insert(node1 in arbitrary_nodes(), node2 in
// arbitrary_nodes()) {             let mut set =
// std::collections::HashSet::new();             let kron_node =
// Node::Kron(KronNode::new(node1.clone(), node2.clone()));
// set.insert(kron_node.clone());             assert!(set.contains(&kron_node));
//         }

//         #[test]
//         fn equals_have_equal_hashes(node1 in arbitrary_nodes(), node2 in
// arbitrary_nodes()) {             let kron_node1 =
// Node::Kron(KronNode::new(node1.clone(), node2.clone()));             let
// kron_node2 = Node::Kron(KronNode::new(node1.clone(), node2.clone()));
//             assert_eq!(kron_node1, kron_node2);
//             let mut hasher1 =
// std::collections::hash_map::DefaultHasher::new();             let mut hasher2
// = std::collections::hash_map::DefaultHasher::new();
// kron_node1.hash(&mut hasher1);             kron_node2.hash(&mut hasher2);
//             assert_eq!(hasher1.finish(), hasher2.finish());
//         }

//         // TODO: Implement kron of pauli test

//         // TODO: Reimplement below tests with circuit.get_unitary() and
// circuit.get_gradient()...         #[test]
//         fn unitary_equals_kron_unitaries(
//             (mut node1, params1) in nodes_and_params(),
//             (mut node2, params2) in nodes_and_params(),
//         ) {
//             let mut kron_node = Node::Kron(KronNode::new(node1.clone(),
// node2.clone()));             let dim = kron_node.get_dimension();
//             let subdim = node2.get_dimension();
//             let kron_utry = kron_node.get_unitary_ref(&[params1.clone(),
// params2.clone()].concat());             let utry1 =
// node1.get_unitary_ref(&params1);             let utry2 =
// node2.get_unitary_ref(&params2);             let mut kron_correct_utry =
// Array2::<c64>::zeros((dim, dim));             kron(subdim, utry1, utry2, &mut
// kron_correct_utry);             assert_eq!(kron_utry, kron_correct_utry);
//         }

//         #[test]
//         fn gradient_equals_kron_gradients(
//             (mut node1, params1) in nodes_and_params(),
//             (mut node2, params2) in nodes_and_params(),
//         ) {
//             let mut kron_node = Node::Kron(KronNode::new(node1.clone(),
// node2.clone()));             let dim = kron_node.get_dimension();
//             let subdim = node2.get_dimension();
//             let num_params = node1.get_num_params() + node2.get_num_params();
//             let kron_grad = kron_node.get_gradient_ref(&[params1.clone(),
// params2.clone()].concat());             let (utry1, grad1) =
// node1.get_unitary_and_gradient_ref(&params1);             let (utry2, grad2)
// = node2.get_unitary_and_gradient_ref(&params2);             let mut
// kron_correct_grad = Array3::<c64>::zeros((num_params, dim, dim));
// let mut grad_idx = 0;             for d_m in grad1.outer_iter() {
//                 let mut grad_ref = kron_correct_grad.index_axis_mut(Axis(0),
// grad_idx);                 kron(subdim, &d_m, utry2, &mut grad_ref);
//                 grad_idx += 1;
//             }
//             for d_m in grad2.outer_iter() {
//                 let mut grad_ref = kron_correct_grad.index_axis_mut(Axis(0),
// grad_idx);                 kron(subdim, utry1, &d_m, &mut grad_ref);
//                 grad_idx += 1;
//             }
//             assert_eq!(kron_grad, kron_correct_grad);
//         }
//     }
// }
