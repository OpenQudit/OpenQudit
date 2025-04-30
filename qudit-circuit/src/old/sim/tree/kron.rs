// use std::cell::UnsafeCell;
// use std::fmt;
use std::hash::Hash;

// use faer_core::Mat;
// use faer_core::MatMut;

use super::tree::ExpressionTree;
use super::tree::PrintTree;
// use crate::math::matrix_kron;
// use crate::math::unitary::DifferentiableUnitaryFn;
// use crate::math::unitary::DoublyDifferentiableUnitaryFn;
// use crate::math::unitary::UnitaryFn;
// use crate::math::unitary::UnitaryGradient;
// use crate::math::unitary::UnitaryHessian;
use crate::math::BoundedFn;
use crate::math::Function;
use crate::QuditRadices;
use crate::QuditSystem;

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
    // left_buf: Mat<C>,
    // right_buf: Mat<C>,
    // left_grad_buf: UnsafeCell<UnitaryGradient<C>>,
    // right_grad_buf: UnsafeCell<UnitaryGradient<C>>,
    // left_hess_buf: UnsafeCell<UnitaryHessian<C>>,
    // right_hess_buf: UnsafeCell<UnitaryHessian<C>>,
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
        let left_params = left.get_num_params();
        let right_params = right.get_num_params();
        let left_dimension = left.get_dimension();
        let right_dimension = right.get_dimension();
        let _left_radices = left.get_radices();
        let _right_radices = right.get_radices();

        KronNode {
            left: Box::new(left),
            right: Box::new(right),
            left_params,
            right_params,
            left_dimension,
            right_dimension,
            // left_buf: Mat::identity(left_radices.get_dimension(), left_radices.get_dimension()),
            // right_buf: Mat::identity(right_radices.get_dimension(), right_radices.get_dimension()),
            // left_grad_buf: UnitaryGradient::zeros(
            //     left_radices.clone(),
            //     left_params,
            // ).into(),
            // right_grad_buf: UnitaryGradient::zeros(
            //     right_radices.clone(),
            //     right_params,
            // ).into(),
            // left_hess_buf: UnitaryHessian::zeros(
            //     left_radices.clone(),
            //     left_params,
            // ).into(),
            // right_hess_buf: UnitaryHessian::zeros(
            //     right_radices.clone(),
            //     right_params,
            // ).into(),
        }
    }

    // // Safety: Caller has to ensure that the child buffer is not mutably aliased.
    // #[inline(always)]
    // unsafe fn child_bufs_as_mut(&self) -> (MatMut<C>, MatMut<C>) {
    //     (
    //         faer_core::mat::from_raw_parts_mut(
    //             C::faer_map(self.left_buf.as_ptr(), |ptr| ptr as *mut _),
    //             self.left_buf.nrows(),
    //             self.left_buf.ncols(),
    //             self.left_buf.row_stride(),
    //             self.left_buf.col_stride(),
    //         ),
    //         faer_core::mat::from_raw_parts_mut(
    //             C::faer_map(self.right_buf.as_ptr(), |ptr| ptr as *mut _),
    //             self.right_buf.nrows(),
    //             self.right_buf.ncols(),
    //             self.right_buf.row_stride(),
    //             self.right_buf.col_stride(),
    //         ),
    //     )
    // }

    // // Safety: Caller has to ensure that the child gradient buffer is not mutably aliased.
    // #[inline(always)]
    // unsafe fn child_grad_bufs_as_mut(&self) -> (&mut UnitaryGradient<C>, &mut UnitaryGradient<C>) {
    //     (&mut *self.left_grad_buf.get(), &mut *self.right_grad_buf.get())
    // }

    // // Safety: Caller has to ensure that the child hessian buffer is not mutably aliased.
    // #[inline(always)]
    // unsafe fn child_hess_bufs_as_mut(&self) -> (&mut UnitaryHessian<C>, &mut UnitaryHessian<C>) {
    //     (&mut *self.left_hess_buf.get(), &mut *self.right_hess_buf.get())
    // }

    // fn calc_grad(
    //     left_utry: &Mat<C>,
    //     left_grad: &UnitaryGradient<C>,
    //     right_utry: &Mat<C>,
    //     right_grad: &UnitaryGradient<C>,
    //     out_grad: &mut UnitaryGradient<C>,
    // ) {
    //     let mut grad_idx = 0;

    //     for d_m in left_grad.iter() {
    //         matrix_kron(
    //             d_m.as_ref(),
    //             right_utry.as_ref(),
    //             out_grad[grad_idx].as_mut(),
    //         );
    //         grad_idx += 1;
    //     }

    //     for d_m in right_grad.iter() {
    //         matrix_kron(
    //             left_utry.as_ref(),
    //             d_m.as_ref(),
    //             out_grad[grad_idx].as_mut(),
    //         );
    //         grad_idx += 1;
    //     }
    // }

    // fn calc_hess(
    //     left_utry: &Mat<C>,
    //     left_grad: &UnitaryGradient<C>,
    //     left_hess: &UnitaryHessian<C>,
    //     right_utry: &Mat<C>,
    //     right_grad: &UnitaryGradient<C>,
    //     right_hess: &UnitaryHessian<C>,
    //     out_hess: &mut UnitaryHessian<C>,
    // ) {
    //     // Upper left block: right_utry * left_hess
    //     for left_hess_row in 0..left_hess.stride() {
    //         for left_hess_col in left_hess_row..left_hess.stride() {
    //             let left_hess_ref = &left_hess[(left_hess_row, left_hess_col)];
    //             let hess_ref =
    //                 out_hess[(left_hess_row, left_hess_col)].as_mut();
    //             matrix_kron(
    //                 left_hess_ref.as_ref(),
    //                 right_utry.as_ref(),
    //                 hess_ref,
    //             );
    //         }
    //     }

    //     // Lower right block: right_hess * left_utry
    //     for right_hess_row in 0..right_hess.stride() {
    //         for right_hess_col in right_hess_row..right_hess.stride() {
    //             let right_hess_ref =
    //                 &right_hess[(right_hess_row, right_hess_col)];
    //             let hess_ref = out_hess[(
    //                 left_hess.stride() + right_hess_row,
    //                 left_hess.stride() + right_hess_col,
    //             )]
    //                 .as_mut();
    //             matrix_kron(
    //                 left_utry.as_ref(),
    //                 right_hess_ref.as_ref(),
    //                 hess_ref,
    //             );
    //         }
    //     }

    //     // Upper right block: right_grad * left_grad
    //     for (left_idx, left_d_m) in left_grad.iter().enumerate() {
    //         for (right_idx, right_d_m) in right_grad.iter().enumerate() {
    //             let hess_ref = out_hess
    //                 [(left_idx, left_hess.stride() + right_idx)]
    //                 .as_mut();
    //             matrix_kron(left_d_m.as_ref(), right_d_m.as_ref(), hess_ref);
    //         }
    //     }
    // }
}

impl Function for KronNode {
    fn get_num_params(&self) -> usize {
        self.left_params + self.right_params
    }
}

impl BoundedFn for KronNode {
    fn get_bounds(&self) -> Vec<std::ops::Range<f64>> {
        self.left
            .get_bounds()
            .iter()
            .chain(self.right.get_bounds().iter())
            .cloned()
            .collect()
    }
}

impl QuditSystem for KronNode {
    /// Returns the radices of the qudit system this node outputs.
    fn get_radices(&self) -> QuditRadices {
        self.left.get_radices() + self.right.get_radices()
    }

    /// Returns the dimension of this node's unitary.
    fn get_dimension(&self) -> usize {
        self.left_dimension * self.right_dimension
    }
}
//
// impl<C: ComplexScalar> UnitaryFn<C> for KronNode<C> {
//     fn write_unitary(
//         &self,
//         params: &[C::Re],
//         out_utry: &mut MatMut<C>,
//     ) {
//         // Safety: We are not aliasing the child buffers.
//         let (mut left_utry_buf, mut right_utry_buf) = unsafe { self.child_bufs_as_mut() };
//
//         let (left, right) = params.split_at(self.left_params);
//         self.left.write_unitary(left, &mut left_utry_buf);
//         self.right.write_unitary(right, &mut right_utry_buf);
//         matrix_kron(
//             self.left_buf.as_ref(),
//             self.right_buf.as_ref(),
//             out_utry.as_mut(),
//         );
//     }
// }
//
// impl<C: ComplexScalar> DifferentiableUnitaryFn<C> for KronNode<C> {
//     fn write_gradient(
//         &self,
//         params: &[C::Re],
//         out_grad: &mut UnitaryGradient<C>,
//     ) {
//         // Safety: We are not aliasing the child buffers.
//         let (mut left_utry_buf, mut right_utry_buf) = unsafe { self.child_bufs_as_mut() };
//
//         // Safety: We are not aliasing the child gradient buffers.
//         let (left_grad_buf, right_grad_buf) = unsafe { self.child_grad_bufs_as_mut() };
//
//         let (left, right) = params.split_at(self.left_params);
//         self.left.write_unitary_and_gradient(
//             left,
//             &mut left_utry_buf,
//             left_grad_buf,
//         );
//         self.right.write_unitary_and_gradient(
//             right,
//             &mut right_utry_buf,
//             right_grad_buf,
//         );
//         KronNode::calc_grad(
//             &self.left_buf,
//             left_grad_buf,
//             &self.right_buf,
//             right_grad_buf,
//             out_grad,
//         );
//     }
//
//     fn write_unitary_and_gradient(
//         &self,
//         params: &[C::Re],
//         out: &mut MatMut<C>,
//         out_grad: &mut UnitaryGradient<C>,
//     ) {
//         // Safety: We are not aliasing the child buffers.
//         let (mut left_utry_buf, mut right_utry_buf) = unsafe { self.child_bufs_as_mut() };
//
//         // Safety: We are not aliasing the child gradient buffers.
//         let (left_grad_buf, right_grad_buf) = unsafe { self.child_grad_bufs_as_mut() };
//
//         let (left, right) = params.split_at(self.left_params);
//         self.left.write_unitary_and_gradient(
//             left,
//             &mut left_utry_buf,
//             left_grad_buf,
//         );
//         self.right.write_unitary_and_gradient(
//             right,
//             &mut right_utry_buf,
//             right_grad_buf,
//         );
//         matrix_kron(
//             self.left_buf.as_ref(),
//             self.right_buf.as_ref(),
//             out.as_mut(),
//         );
//         KronNode::calc_grad(
//             &self.left_buf,
//             left_grad_buf,
//             &self.right_buf,
//             right_grad_buf,
//             out_grad,
//         );
//     }
// }
//
// impl<C: ComplexScalar> DoublyDifferentiableUnitaryFn<C> for KronNode<C> {
//     fn write_hessian(
//         &self,
//         params: &[C::Re],
//         out_hess: &mut UnitaryHessian<C>,
//     ) {
//         // Safety: We are not aliasing the child buffers.
//         let (mut left_utry_buf, mut right_utry_buf) = unsafe { self.child_bufs_as_mut() };
//
//         // Safety: We are not aliasing the child gradient buffers.
//         let (left_grad_buf, right_grad_buf) = unsafe { self.child_grad_bufs_as_mut() };
//
//         // Safety: We are not aliasing the child hessian buffers.
//         let (left_hess_buf, right_hess_buf) = unsafe { self.child_hess_bufs_as_mut() };
//
//         let (left, right) = params.split_at(self.left_params);
//         self.left.write_unitary_gradient_and_hessian(
//             left,
//             &mut left_utry_buf,
//             left_grad_buf,
//             left_hess_buf,
//         );
//         self.right.write_unitary_gradient_and_hessian(
//             right,
//             &mut right_utry_buf,
//             right_grad_buf,
//             right_hess_buf,
//         );
//         KronNode::calc_hess(
//             &self.left_buf,
//             left_grad_buf,
//             left_hess_buf,
//             &self.right_buf,
//             right_grad_buf,
//             right_hess_buf,
//             out_hess,
//         );
//     }
//
//     fn write_unitary_and_hessian(
//         &self,
//         params: &[C::Re],
//         out_utry: &mut MatMut<C>,
//         out_hess: &mut UnitaryHessian<C>,
//     ) {
//         // Safety: We are not aliasing the child buffers.
//         let (mut left_utry_buf, mut right_utry_buf) = unsafe { self.child_bufs_as_mut() };
//
//         // Safety: We are not aliasing the child gradient buffers.
//         let (left_grad_buf, right_grad_buf) = unsafe { self.child_grad_bufs_as_mut() };
//
//         // Safety: We are not aliasing the child hessian buffers.
//         let (left_hess_buf, right_hess_buf) = unsafe { self.child_hess_bufs_as_mut() };
//
//         let (left, right) = params.split_at(self.left_params);
//         self.left.write_unitary_gradient_and_hessian(
//             left,
//             &mut left_utry_buf,
//             left_grad_buf,
//             left_hess_buf,
//         );
//         self.right.write_unitary_gradient_and_hessian(
//             right,
//             &mut right_utry_buf,
//             right_grad_buf,
//             right_hess_buf,
//         );
//         matrix_kron(
//             self.left_buf.as_ref(),
//             self.right_buf.as_ref(),
//             out_utry.as_mut(),
//         );
//         KronNode::calc_hess(
//             &self.left_buf,
//             left_grad_buf,
//             left_hess_buf,
//             &self.right_buf,
//             right_grad_buf,
//             right_hess_buf,
//             out_hess,
//         );
//     }
//
//     fn write_gradient_and_hessian(
//         &self,
//         params: &[C::Re],
//         out_grad: &mut UnitaryGradient<C>,
//         out_hess: &mut UnitaryHessian<C>,
//     ) {
//         // Safety: We are not aliasing the child buffers.
//         let (mut left_utry_buf, mut right_utry_buf) = unsafe { self.child_bufs_as_mut() };
//
//         // Safety: We are not aliasing the child gradient buffers.
//         let (left_grad_buf, right_grad_buf) = unsafe { self.child_grad_bufs_as_mut() };
//
//         // Safety: We are not aliasing the child hessian buffers.
//         let (left_hess_buf, right_hess_buf) = unsafe { self.child_hess_bufs_as_mut() };
//
//         let (left, right) = params.split_at(self.left_params);
//         self.left.write_unitary_gradient_and_hessian(
//             left,
//             &mut left_utry_buf,
//             left_grad_buf,
//             left_hess_buf,
//         );
//         self.right.write_unitary_gradient_and_hessian(
//             right,
//             &mut right_utry_buf,
//             right_grad_buf,
//             right_hess_buf,
//         );
//         KronNode::calc_grad(
//             &self.left_buf,
//             left_grad_buf,
//             &self.right_buf,
//             right_grad_buf,
//             out_grad,
//         );
//         KronNode::<C>::calc_hess(
//             &self.left_buf,
//             left_grad_buf,
//             left_hess_buf,
//             &self.right_buf,
//             right_grad_buf,
//             right_hess_buf,
//             out_hess,
//         );
//     }
//
//     fn write_unitary_gradient_and_hessian(
//         &self,
//         params: &[C::Re],
//         out_utry: &mut MatMut<C>,
//         out_grad: &mut UnitaryGradient<C>,
//         out_hess: &mut UnitaryHessian<C>,
//     ) {
//         // Safety: We are not aliasing the child buffers.
//         let (mut left_utry_buf, mut right_utry_buf) = unsafe { self.child_bufs_as_mut() };
//
//         // Safety: We are not aliasing the child gradient buffers.
//         let (left_grad_buf, right_grad_buf) = unsafe { self.child_grad_bufs_as_mut() };
//
//         // Safety: We are not aliasing the child hessian buffers.
//         let (left_hess_buf, right_hess_buf) = unsafe { self.child_hess_bufs_as_mut() };
//
//         let (left, right) = params.split_at(self.left_params);
//         self.left.write_unitary_gradient_and_hessian(
//             left,
//             &mut left_utry_buf,
//             left_grad_buf,
//             left_hess_buf,
//         );
//         self.right.write_unitary_gradient_and_hessian(
//             right,
//             &mut right_utry_buf,
//             right_grad_buf,
//             right_hess_buf,
//         );
//         matrix_kron(
//             self.left_buf.as_ref(),
//             self.right_buf.as_ref(),
//             out_utry.as_mut(),
//         );
//         KronNode::calc_grad(
//             &self.left_buf,
//             left_grad_buf,
//             &self.right_buf,
//             right_grad_buf,
//             out_grad,
//         );
//         KronNode::calc_hess(
//             &self.left_buf,
//             left_grad_buf,
//             left_hess_buf,
//             &self.right_buf,
//             right_grad_buf,
//             right_hess_buf,
//             out_hess,
//         );
//     }
// }
//
// impl<C: ComplexScalar> PartialEq for KronNode<C> {
//     /// Compare two kron nodes for equality.
//     fn eq(&self, other: &Self) -> bool {
//         (*self.left) == (*other.left) && (*self.right) == (*other.right)
//     }
// }
//
// impl<C: ComplexScalar> Eq for KronNode<C> {}

// impl<C: ComplexScalar> Hash for KronNode<C> {
//     /// Hash a kron node by hashing its components.
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.left.hash(state);
//         self.right.hash(state);
//     }
// }

// impl<C: ComplexScalar> fmt::Debug for KronNode<C> {
//     /// Formats the leaf node as a struct with a `gate` field.
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_struct("Kron")
//             .field("left", &self.left)
//             .field("right", &self.right)
//             .finish()
//     }
// }

impl PrintTree for KronNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}Kron", prefix).unwrap();
        let left_prefix = self.modify_prefix_for_child(prefix, false);
        let right_prefix = self.modify_prefix_for_child(prefix, true);
        self.left.write_tree(&left_prefix, fmt);
        self.right.write_tree(&right_prefix, fmt);
    }
}

// impl<C: ComplexScalar> Clone for KronNode<C> {
//     fn clone(&self) -> Self {
//         KronNode {
//             left: self.left.clone(),
//             right: self.right.clone(),
//             left_params: self.left_params,
//             right_params: self.right_params,
//             left_dimension: self.left_dimension,
//             right_dimension: self.right_dimension,
//             left_buf: self.left_buf.clone(),
//             right_buf: self.right_buf.clone(),
//             left_grad_buf: UnitaryGradient::zeros(
//                 self.left.get_radices(),
//                 self.left_params,
//             ).into(),
//             right_grad_buf: UnitaryGradient::zeros(
//                 self.right.get_radices(),
//                 self.right_params,
//             ).into(),
//             left_hess_buf: UnitaryHessian::zeros(
//                 self.left.get_radices(),
//                 self.left_params,
//             ).into(),
//             right_hess_buf: UnitaryHessian::zeros(
//                 self.right.get_radices(),
//                 self.right_params,
//             ).into(),
//         }
//     }
// }

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
