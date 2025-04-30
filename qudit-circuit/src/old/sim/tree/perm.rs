// use std::cell::UnsafeCell;
use std::fmt;

// use faer_core::Mat;
// use faer_core::MatMut;

use super::tree::ExpressionTree;
use super::tree::PrintTree;
// use crate::math::unitary::DifferentiableUnitaryFn;
// use crate::math::unitary::DoublyDifferentiableUnitaryFn;
// use crate::math::unitary::UnitaryFn;
// use crate::math::unitary::UnitaryGradient;
// use crate::math::unitary::UnitaryHessian;
use crate::math::BoundedFn;
// use crate::math::ComplexScalar;
use crate::math::Function;
use crate::QuditPermutation;
use crate::QuditRadices;
use crate::QuditSystem;

/// A permutation node in the computation tree.
/// This node wraps another node and applies a permutation to its output.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PermNode {
    /// The child node to be permuted.
    pub child: Box<ExpressionTree>,

    /// The permutation to apply to the child node.
    perm: QuditPermutation,

    num_params: usize,
    dimension: usize,
    // child_buf: Mat<C>,
    // child_grad_buf: UnsafeCell<UnitaryGradient<C>>,
    // child_hess_buf: UnsafeCell<UnitaryHessian<C>>,
}

impl PermNode {
    /// Create a new permutation node from a child node and a permutation.
    /// The permutation must have the same number of qudits and matching
    /// radices as the child node.
    ///
    /// # Arguments
    ///
    /// * `child` - The child node to permute.
    /// * `perm` - The permutation to apply to the child node.
    ///
    /// # Panics
    ///
    /// If the number of qudits in the permutation does not match the number
    /// of qudits in the child node, or if the radices of the permutation do
    /// not match the radices of the child node.
    ///
    /// # Examples
    ///
    /// ```
    /// use qudit_circuit::Gate;
    /// use qudit_circuit::math::perm::QuditPermutation;
    /// use qudit_circuit::QuditSystem;
    /// use qudit_circuit::sim::node::ExpressionTree;
    /// use qudit_circuit::sim::leaf::LeafStruct;
    /// use qudit_circuit::sim::perm::PermNode;
    /// let cz_node = ExpressionTree::Leaf(Gate::CZ());
    /// let perm = QuditPermutation::new(cz_node.get_radices(), vec![1, 0]);
    /// let perm_node = ExpressionTree::Perm(PermNode::new(cz_node, perm));
    /// ```
    pub fn new(child: ExpressionTree, perm: QuditPermutation) -> PermNode {
        let dimension = child.get_dimension();
        let num_params = child.get_num_params();
        let _radices = child.get_radices();

        if perm.get_num_qudits() != child.get_num_qudits() {
            panic!("Number of qudits in permutation must match number of qudits in node.");
        }

        if perm.get_radices() != child.get_radices() {
            panic!("Radices of permutation must match radices of node.");
        }

        PermNode {
            child: Box::new(child),
            perm,
            num_params,
            dimension,
            // child_buf: Mat::identity(radices.get_dimension(), radices.get_dimension()),
            // child_grad_buf: UnitaryGradient::zeros(radices.clone(), num_params).into(),
            // child_hess_buf: UnitaryHessian::zeros(radices.clone(), num_params).into(),
        }
    }

    // // Safety: Caller has to ensure that the child buffer is not mutably aliased.
    // #[inline(always)]
    // unsafe fn child_buf_as_mut(&self) -> MatMut<C> {
    //     faer_core::mat::from_raw_parts_mut(
    //         C::faer_map(self.child_buf.as_ptr(), |ptr| ptr as *mut _),
    //         self.child_buf.nrows(),
    //         self.child_buf.ncols(),
    //         self.child_buf.row_stride(),
    //         self.child_buf.col_stride(),
    //     )
    // }

    // // Safety: Caller has to ensure that the child gradient buffer is not mutably aliased.
    // #[inline(always)]
    // unsafe fn child_grad_buf_as_mut(&self) -> &mut UnitaryGradient<C> {
    //     &mut *self.child_grad_buf.get()
    // }

    // // Safety: Caller has to ensure that the child hessian buffer is not mutably aliased.
    // #[inline(always)]
    // unsafe fn child_hess_buf_as_mut(&self) -> &mut UnitaryHessian<C> {
    //     &mut *self.child_hess_buf.get()
    // }

    // fn calc_grad(
    //     in_grad: &UnitaryGradient<C>,
    //     perm: &QuditPermutation,
    //     out_grad: &mut UnitaryGradient<C>,
    // ) {
    //     in_grad.iter().zip(out_grad.iter_mut())
    //         .for_each(|(d_m, out_d_m)| {
    //             perm.apply_to_buf(d_m, out_d_m)
    //         });
    // }

    // fn calc_hess(
    //     in_hess: &UnitaryHessian<C>,
    //     perm: &QuditPermutation,
    //     out_hess: &mut UnitaryHessian<C>,
    // ) {
    //     in_hess.iter().zip(out_hess.iter_mut())
    //         .for_each(|(d_m, out_d_m)| {
    //             perm.apply_to_buf(d_m, out_d_m)
    //         });
    // }
}

impl Function for PermNode {
    fn get_num_params(&self) -> usize {
        self.num_params
    }
}

impl BoundedFn for PermNode {
    fn get_bounds(&self) -> Vec<std::ops::Range<f64>> {
        self.child.get_bounds()
    }
}

impl QuditSystem for PermNode {
    fn get_radices(&self) -> QuditRadices {
        self.perm.get_permuted_radices()
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }
}
//
// impl<C: ComplexScalar> UnitaryFn<C> for PermNode<C> {
//     fn write_unitary(
//         &self,
//         params: &[C::Re],
//         out_utry: &mut MatMut<C>,
//     ) {
//         // Safety: The child buffer is not mutably aliased.
//         let mut child_utry_buf = unsafe { self.child_buf_as_mut() };
//
//         self.child.write_unitary(params, &mut child_utry_buf);
//         self.perm.apply_to_buf(&self.child_buf, out_utry);
//     }
// }
//
// impl<C: ComplexScalar> DifferentiableUnitaryFn<C> for PermNode<C> {
//     fn write_gradient(
//         &self,
//         params: &[C::Re],
//         out_grad: &mut UnitaryGradient<C>,
//     ) {
//         // Safety: The child gradient buffer is not mutably aliased.
//         let child_grad_buf = unsafe { self.child_grad_buf_as_mut() };
//
//         self.child.write_gradient(params, child_grad_buf);
//         PermNode::calc_grad(child_grad_buf, &self.perm, out_grad);
//     }
//
//     fn write_unitary_and_gradient(
//         &self,
//         params: &[C::Re],
//         out: &mut MatMut<C>,
//         out_grad: &mut UnitaryGradient<C>,
//     ) {
//         // Safety: The child buffer is not mutably aliased.
//         let mut child_utry_buf = unsafe { self.child_buf_as_mut() };
//
//         // Safety: The child gradient buffer is not mutably aliased.
//         let child_grad_buf = unsafe { self.child_grad_buf_as_mut() };
//
//         self.child.write_unitary_and_gradient(
//             params,
//             &mut child_utry_buf,
//             child_grad_buf,
//         );
//         self.perm.apply_to_buf(&self.child_buf, out);
//         PermNode::calc_grad(child_grad_buf, &self.perm, out_grad);
//     }
// }
//
// impl<C: ComplexScalar> DoublyDifferentiableUnitaryFn<C> for PermNode<C> {
//     fn write_hessian(
//         &self,
//         params: &[C::Re],
//         out_hess: &mut UnitaryHessian<C>,
//     ) {
//         // Safety: The child hessian buffer is not mutably aliased.
//         let child_hess_buf = unsafe { self.child_hess_buf_as_mut() };
//
//         self.child.write_hessian(params, child_hess_buf);
//         PermNode::calc_hess(child_hess_buf, &self.perm, out_hess);
//     }
//
//     fn write_unitary_and_hessian(
//         &self,
//         params: &[C::Re],
//         out_utry: &mut MatMut<C>,
//         out_hess: &mut UnitaryHessian<C>,
//     ) {
//         // Safety: The child buffer is not mutably aliased.
//         let mut child_utry_buf = unsafe { self.child_buf_as_mut() };
//
//         // Safety: The child hessian buffer is not mutably aliased.
//         let child_hess_buf = unsafe { self.child_hess_buf_as_mut() };
//
//         self.child.write_unitary_and_hessian(
//             params,
//             &mut child_utry_buf,
//             child_hess_buf,
//         );
//         self.perm.apply_to_buf(&self.child_buf, out_utry);
//         PermNode::calc_hess(child_hess_buf, &self.perm, out_hess);
//     }
//
//     fn write_gradient_and_hessian(
//         &self,
//         params: &[C::Re],
//         out_grad: &mut UnitaryGradient<C>,
//         out_hess: &mut UnitaryHessian<C>,
//     ) {
//         // Safety: The child gradient buffer is not mutably aliased.
//         let child_grad_buf = unsafe { self.child_grad_buf_as_mut() };
//
//         // Safety: The child hessian buffer is not mutably aliased.
//         let child_hess_buf = unsafe { self.child_hess_buf_as_mut() };
//
//         self.child.write_gradient_and_hessian(
//             params,
//             child_grad_buf,
//             child_hess_buf,
//         );
//         PermNode::calc_grad(child_grad_buf, &self.perm, out_grad);
//         PermNode::calc_hess(child_hess_buf, &self.perm, out_hess);
//     }
//
//     fn write_unitary_gradient_and_hessian(
//         &self,
//         params: &[C::Re],
//         out_utry: &mut MatMut<C>,
//         out_grad: &mut UnitaryGradient<C>,
//         out_hess: &mut UnitaryHessian<C>,
//     ) {
//         // Safety: The child buffer is not mutably aliased.
//         let mut child_utry_buf = unsafe { self.child_buf_as_mut() };
//
//         // Safety: The child gradient buffer is not mutably aliased.
//         let child_grad_buf = unsafe { self.child_grad_buf_as_mut() };
//
//         // Safety: The child hessian buffer is not mutably aliased.
//         let child_hess_buf = unsafe { self.child_hess_buf_as_mut() };
//
//         self.child.write_unitary_gradient_and_hessian(
//             params,
//             &mut child_utry_buf,
//             child_grad_buf,
//             child_hess_buf,
//         );
//         self.perm.apply_to_buf(&self.child_buf, out_utry);
//         PermNode::calc_grad(child_grad_buf, &self.perm, out_grad);
//         PermNode::calc_hess(child_hess_buf, &self.perm, out_hess);
//     }
// }

// impl<C: ComplexScalar> PartialEq for PermNode<C> {
//     /// Compares the node by comparing its child and permutation.
//     fn eq(&self, other: &Self) -> bool {
//         (*self.child) == (*other.child) && self.perm == other.perm
//     }
// }
//
// impl<C: ComplexScalar> Eq for PermNode<C> {}

// impl Hash for PermNode<C> {
//     /// Hashes the node by hashing its child and permutation.
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.child.hash(state);
//         self.perm.hash(state);
//     }
// }

impl fmt::Debug for PermNode {
    /// Formats the leaf node as a struct with a `gate` field.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Perm")
            .field("child", &self.child)
            .field("perm", &self.perm)
            .finish()
    }
}

impl PrintTree for PermNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}Perm({})", prefix, self.perm).unwrap();
        let child_prefix = self.modify_prefix_for_child(prefix, true);
        self.child.write_tree(&child_prefix, fmt);
    }
}

// impl<C: ComplexScalar> Clone for PermNode<C> {
//     fn clone(&self) -> Self {
//         PermNode {
//             child: self.child.clone(),
//             perm: self.perm.clone(),
//             num_params: self.num_params,
//             dimension: self.dimension,
//             child_buf: self.child_buf.clone(),
//             child_grad_buf: UnitaryGradient::zeros(self.get_radices(), self.get_num_params()).into(),
//             child_hess_buf: UnitaryHessian::zeros(self.get_radices(), self.get_num_params()).into(),
//         }
//     }
// }

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
