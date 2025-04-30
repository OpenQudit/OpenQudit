use std::hash::Hash;

use super::tree::PrintTree;
use crate::{
    math::{BoundedFn, Function},
    QuditRadices, QuditSystem,
};

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

    // /// Return the number of input parameters for this node.
    // pub(super) fn get_num_params(&self) -> usize {
    //     0
    // }

    // /// Calculate the unitary matrix for this node and return a reference.
    // pub(super) fn get_unitary_ref(&mut self, _params: &[f64]) -> &Array2<c64>
    // {     &self.out
    // }

    // /// Calculate the gradient matrix for this node and return a reference.
    // pub(super) fn get_gradient_ref(&mut self, _params: &[f64]) ->
    // &Array3<c64> {     &self.out_grad
    // }

    // /// Calculate the unitary and gradient and return a reference.
    // pub(super) fn get_unitary_and_gradient_ref(
    //     &mut self,
    //     _params: &[f64],
    // ) -> (&Array2<c64>, &Array3<c64>) {
    //     (&self.out, &self.out_grad)
    // }
}

impl QuditSystem for IdentityNode {
    /// Returns the radices of the qudit system this node represents.
    fn get_radices(&self) -> QuditRadices {
        self.radices.clone()
    }

    /// Returns the dimension of this node's unitary.
    fn get_dimension(&self) -> usize {
        self.radices.get_dimension()
    }
}

impl Function for IdentityNode {
    fn get_num_params(&self) -> usize {
        0
    }
}

impl BoundedFn for IdentityNode {
    fn get_bounds(&self) -> Vec<std::ops::Range<f64>> {
        vec![]
    }
}

// impl<C: ComplexScalar> UnitaryFn<C> for IdentityNode<C> {
//     fn write_unitary(&self, _params: &[C::Re], _out_utry: &mut MatMut<C>) {
//         // NO-OP Since write_unitary guarantees out_utry is called with identity
//     }
// }
//
// impl<C: ComplexScalar> DifferentiableUnitaryFn<C> for IdentityNode<C> {
//     fn write_gradient(
//         &self,
//         _params: &[C::Re],
//         _out_grad: &mut UnitaryGradient<C>,
//     ) {
//         // NO-OP Since write_gradient guarantees out_grad is called with zeros
//     }
//
//     fn write_unitary_and_gradient(
//         &self,
//         _params: &[C::Re],
//         _out_utry: &mut MatMut<C>,
//         _out_grad: &mut UnitaryGradient<C>,
//     ) {
//         // NO-OP Since write_unitary_and_gradient guarantees out_utry is called
//         // with identity and out_grad is called with zeros
//     }
// }
//
// impl<C: ComplexScalar> DoublyDifferentiableUnitaryFn<C> for IdentityNode<C> {
//     fn write_hessian(
//         &self,
//         _params: &[C::Re],
//         _out_hess: &mut UnitaryHessian<C>,
//     ) {
//         // NO-OP Since write_hessian guarantees out_hess is called with zeros
//     }
//
//     fn write_unitary_and_hessian(
//         &self,
//         _params: &[C::Re],
//         _out_utry: &mut MatMut<C>,
//         _out_hess: &mut UnitaryHessian<C>,
//     ) {
//         // NO-OP Since write_unitary_and_hessian guarantees out_utry is called
//         // with identity and out_hess is called with zeros
//     }
//
//     fn write_gradient_and_hessian(
//         &self,
//         _params: &[C::Re],
//         _out_grad: &mut UnitaryGradient<C>,
//         _out_hess: &mut UnitaryHessian<C>,
//     ) {
//         // NO-OP Since write_gradient_and_hessian guarantees out_grad is called
//         // with zeros and out_hess is called with zeros
//     }
//
//     fn write_unitary_gradient_and_hessian(
//         &self,
//         _params: &[C::Re],
//         _out_utry: &mut MatMut<C>,
//         _out_grad: &mut UnitaryGradient<C>,
//         _out_hess: &mut UnitaryHessian<C>,
//     ) {
//         // NO-OP Since write_unitary_gradient_and_hessian guarantees out_utry is
//         // called with identity, out_grad is called with zeros, and out_hess is
//         // called with zeros
//     }
// }

// impl PartialEq for IdentityNode {
//     /// Compares the gate that this node wraps.
//     fn eq(&self, other: &Self) -> bool { self.radices == other.radices }
// }
//
// impl<C: ComplexScalar> Eq for IdentityNode<C> {}
//
// impl<C: ComplexScalar> Hash for IdentityNode<C> {
//     /// Hashes the gate that this node wraps.
//     fn hash<H: Hasher>(&self, state: &mut H) { self.radices.hash(state); }
// }
//
// impl<C: ComplexScalar> fmt::Debug for IdentityNode<C> {
//     /// Formats the node as a struct with a `radices` field.
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_struct("Identity")
//             .field("radices", &self.radices)
//             .finish()
//     }
// }

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
