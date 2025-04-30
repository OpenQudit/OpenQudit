use std::hash::Hash;

use super::fmt::PrintTree;
use qudit_core::HasPeriods;
use qudit_core::HasParams;
use qudit_core::RealScalar;
use qudit_core::QuditRadices;
use qudit_core::QuditSystem;

/// A leaf node in the computation tree that wraps an individual gate.
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct IdentityNode {
    /// The radices of the qudit system this identity represents.
    radices: QuditRadices,
}

impl IdentityNode {
    /// Create a new Identity node of a given dimension.
    ///
    /// # Arguments
    ///
    /// * `radices` - The dimension of the identity matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::sim::ExpressionTree;
    /// use qudit_circuit::sim::identity::IdentityNode;
    /// use qudit_circuit::{QuditRadices, radices};
    /// let qubit_identity = ExpressionTree::Identity(IdentityNode::new(radices![2]));
    /// ```
    pub fn new(radices: QuditRadices) -> IdentityNode {
        IdentityNode { radices }
    }
}

impl QuditSystem for IdentityNode {
    /// Returns the radices of the qudit system this node represents.
    fn radices(&self) -> QuditRadices {
        self.radices.clone()
    }

    /// Returns the dimension of this node's unitary.
    fn dimension(&self) -> usize {
        self.radices.dimension()
    }
}

impl HasParams for IdentityNode {
    fn num_params(&self) -> usize {
        0
    }
}

impl<R: RealScalar> HasPeriods<R> for IdentityNode {
    fn periods(&self) -> Vec<std::ops::Range<R>> {
        vec![]
    }
}

impl PrintTree for IdentityNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}Identity({})", prefix, self.radices).unwrap();
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     // use crate::radices::strategies::radices;
//     use crate::sim::node::Node;
//     use proptest::prelude::*;

//     proptest! {
//         #[test]
//         fn does_not_crash(radices in radices(5)) {
//             let _ = Node::Identity(IdentityNode::new(radices));
//         }

//         #[test]
//         fn unitary_is_identity(radices in radices(5)) {
//             let mut node =
// Node::Identity(IdentityNode::new(radices.clone()));             let utry_ref
// = node.get_unitary_ref(&[] as &[f64]);             assert_eq!(utry_ref,
// &Array2::eye(radices.get_dimension()));         }

//         #[test]
//         fn gradient_is_empty(radices in radices(5)) {
//             let mut node = Node::Identity(IdentityNode::new(radices));
//             let grad_ref = node.get_gradient_ref(&[] as &[f64]);
//             assert_eq!(grad_ref.len(), 0);
//         }

//         #[test]
//         fn unitary_and_gradient_as_above(radices in radices(5)) {
//             let mut node =
// Node::Identity(IdentityNode::new(radices.clone()));             let
// (utry_ref, grad_ref) = node.get_unitary_and_gradient_ref(&[] as &[f64]);
//             assert_eq!(utry_ref, &Array2::eye(radices.get_dimension()));
//             assert_eq!(grad_ref.len(), 0);
//         }

//         #[test]
//         fn is_hashable(radices in radices(5)) {
//             let node = Node::Identity(IdentityNode::new(radices));
//             let mut hasher =
// std::collections::hash_map::DefaultHasher::new();             node.hash(&mut
// hasher);             let _ = hasher.finish();
//         }

//         #[test]
//         fn is_hashable_set_insert(radices in radices(5)) {
//             let node = Node::Identity(IdentityNode::new(radices));
//             let mut set = std::collections::HashSet::new();
//             set.insert(node.clone());
//             assert!(set.contains(&node));
//         }

//         #[test]
//         fn equals_have_equal_hashes(radices in radices(5)) {
//             let node1 = Node::Identity(IdentityNode::new(radices.clone()));
//             let node2 = Node::Identity(IdentityNode::new(radices));
//             assert_eq!(node1, node2);
//             let mut hasher1 =
// std::collections::hash_map::DefaultHasher::new();             let mut hasher2
// = std::collections::hash_map::DefaultHasher::new();
// node1.hash(&mut hasher1);             node2.hash(&mut hasher2);
//             assert_eq!(hasher1.finish(), hasher2.finish());
//         }
//     }
// }
