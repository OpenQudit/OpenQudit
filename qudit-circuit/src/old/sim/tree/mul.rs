// use std::cell::UnsafeCell;
use std::hash::Hash;

// use faer_core::Mat;
// use faer_core::MatMut;
// use faer_core::mul::matmul;
// use faer_core::Parallelism;

use super::tree::ExpressionTree;
use super::tree::PrintTree;
// use crate::math::unitary::DifferentiableUnitaryFn;
// use crate::math::unitary::DoublyDifferentiableUnitaryFn;
// use crate::math::unitary::UnitaryFn;
// use crate::math::unitary::UnitaryGradient;
// use crate::math::unitary::UnitaryHessian;
use crate::math::BoundedFn;
use crate::math::Function;
use crate::QuditRadices;
use crate::QuditSystem;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct MulNode {
    pub left: Box<ExpressionTree>,
    pub right: Box<ExpressionTree>,
    left_params: usize,
    right_params: usize,
    dimension: usize,
    // left_buf: Mat<C>,
    // right_buf: Mat<C>,
    // left_grad_buf: UnsafeCell<UnitaryGradient<C>>,
    // right_grad_buf: UnsafeCell<UnitaryGradient<C>>,
    // left_hess_buf: UnsafeCell<UnitaryHessian<C>>,
    // right_hess_buf: UnsafeCell<UnitaryHessian<C>>,
}

impl MulNode {
    pub fn new(left: ExpressionTree, right: ExpressionTree) -> MulNode {
        if right.get_radices() != left.get_radices() {
            panic!("Left and right node do not have same dimension in multiply node.");
        }

        let left_params = left.get_num_params();
        let right_params = right.get_num_params();
        let _left_radices = left.get_radices();
        let _right_radices = right.get_radices();
        let dimension = left.get_dimension();

        MulNode {
            left: Box::new(left),
            right: Box::new(right),
            left_params,
            right_params,
            dimension,
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
    //         matmul(
    //             out_grad[grad_idx].as_mut(),
    //             right_utry.as_ref(),
    //             d_m.as_ref(),
    //             None,
    //             C::complex(1.0, 0.0),
    //             Parallelism::None,
    //         );
    //         grad_idx += 1;
    //     }

    //     for d_m in right_grad.iter() {
    //         matmul(
    //             out_grad[grad_idx].as_mut(),
    //             d_m.as_ref(),
    //             left_utry.as_ref(),
    //             None,
    //             C::complex(1.0, 0.0),
    //             Parallelism::None,
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
    //             let hess_ref = out_hess[(left_hess_row, left_hess_col)].as_mut();
    //             matmul(
    //                 hess_ref,
    //                 right_utry.as_ref(),
    //                 left_hess_ref.as_ref(),
    //                 None,
    //                 C::complex(1.0, 0.0),
    //                 Parallelism::None,
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
    //             matmul(
    //                 hess_ref,
    //                 right_hess_ref.as_ref(),
    //                 left_utry.as_ref(),
    //                 None,
    //                 C::complex(1.0, 0.0),
    //                 Parallelism::None,
    //             );
    //         }
    //     }

    //     // Upper right block: right_grad * left_grad
    //     for (left_idx, left_d_m) in left_grad.iter().enumerate() {
    //         for (right_idx, right_d_m) in right_grad.iter().enumerate() {
    //             let hess_ref = out_hess
    //                 [(left_idx, left_hess.stride() + right_idx)]
    //                 .as_mut();
    //             matmul(
    //                 hess_ref,
    //                 right_d_m.as_ref(),
    //                 left_d_m.as_ref(),
    //                 None,
    //                 C::complex(1.0, 0.0),
    //                 Parallelism::None,
    //             );
    //         }
    //     }
    // }
}

impl Function for MulNode {
    fn get_num_params(&self) -> usize {
        self.left_params + self.right_params
    }
}

impl BoundedFn for MulNode {
    fn get_bounds(&self) -> Vec<std::ops::Range<f64>> {
        self.left
            .get_bounds()
            .iter()
            .chain(self.right.get_bounds().iter())
            .cloned()
            .collect()
    }
}

impl QuditSystem for MulNode {
    fn get_radices(&self) -> QuditRadices {
        self.left.get_radices()
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }
}
//
// impl<C: ComplexScalar> UnitaryFn<C> for MulNode<C> {
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
//
//         matmul(
//             out_utry.as_mut(),
//             self.right_buf.as_ref(),
//             self.left_buf.as_ref(),
//             None,
//             C::complex(1.0, 0.0),
//             Parallelism::None,
//         );
//     }
// }
//
// impl<C: ComplexScalar> DifferentiableUnitaryFn<C> for MulNode<C> {
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
//         MulNode::calc_grad(
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
//         utry: &mut MatMut<C>,
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
//         matmul(
//             utry.as_mut(),
//             self.right_buf.as_ref(),
//             self.left_buf.as_ref(),
//             None,
//             C::complex(1.0, 0.0),
//             Parallelism::None,
//         );
//         MulNode::calc_grad(
//             &self.left_buf,
//             left_grad_buf,
//             &self.right_buf,
//             right_grad_buf,
//             out_grad,
//         );
//     }
// }
//
// impl<C: ComplexScalar> DoublyDifferentiableUnitaryFn<C> for MulNode<C> {
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
//         MulNode::calc_hess(
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
//         matmul(
//             out_utry.as_mut(),
//             self.right_buf.as_ref(),
//             self.left_buf.as_ref(),
//             None,
//             C::complex(1.0, 0.0),
//             Parallelism::None,
//         );
//         MulNode::calc_hess(
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
//         MulNode::calc_grad(
//             &self.left_buf,
//             left_grad_buf,
//             &self.right_buf,
//             right_grad_buf,
//             out_grad,
//         );
//         MulNode::calc_hess(
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
//         matmul(
//             out_utry.as_mut(),
//             self.right_buf.as_ref(),
//             self.left_buf.as_ref(),
//             None,
//             C::complex(1.0, 0.0),
//             Parallelism::None,
//         );
//         MulNode::calc_grad(
//             &self.left_buf,
//             left_grad_buf,
//             &self.right_buf,
//             right_grad_buf,
//             out_grad,
//         );
//         MulNode::calc_hess(
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
// impl<C: ComplexScalar> PartialEq for MulNode<C> {
//     fn eq(&self, other: &Self) -> bool {
//         (*self.left) == (*other.left)
//             && (*self.right) == (*other.left)
//             && self.left_params == other.left_params
//             && self.right_params == other.right_params
//             && self.dimension == other.dimension
//     }
// }
//
// impl<C: ComplexScalar> Eq for MulNode<C> {}

// impl<C: ComplexScalar> Hash for MulNode<C> {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.left.hash(state);
//         self.right.hash(state);
//         // self.left_params.hash(state);
//         // self.right_params.hash(state);
//         self.dimension.hash(state);
//     }
// }

impl PrintTree for MulNode {
    fn write_tree(&self, prefix: &str, fmt: &mut std::fmt::Formatter<'_>) {
        writeln!(fmt, "{}Mul", prefix).unwrap();
        let left_prefix = self.modify_prefix_for_child(prefix, false);
        let right_prefix = self.modify_prefix_for_child(prefix, true);
        self.left.write_tree(&left_prefix, fmt);
        self.right.write_tree(&right_prefix, fmt);
    }
}

// impl<C: ComplexScalar> Clone for MulNode<C> {
//     fn clone(&self) -> Self {
//         MulNode {
//             left: self.left.clone(),
//             right: self.right.clone(),
//             left_params: self.left_params,
//             right_params: self.right_params,
//             dimension: self.dimension,
//             left_buf: self.left_buf.clone(),
//             right_buf: self.right_buf.clone(),
//             left_grad_buf: UnitaryGradient::zeros(
//                 self.left.get_radices().clone(),
//                 self.left_params,
//             ).into(),
//             right_grad_buf: UnitaryGradient::zeros(
//                 self.right.get_radices().clone(),
//                 self.right_params,
//             ).into(),
//             left_hess_buf: UnitaryHessian::zeros(
//                 self.left.get_radices().clone(),
//                 self.left_params,
//             ).into(),
//             right_hess_buf: UnitaryHessian::zeros(
//                 self.right.get_radices().clone(),
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

//     //TODO: Build strategy for generating nodes with same radices; redo tests

//     proptest! {
//         #[test]
//         fn does_not_crash(node in arbitrary_nodes()) {
//             let _ = Node::Mul(MulNode::new(node.clone(), node));
//         }

//         // #[test]
//         // fn unitary_equals_unitary_builder((mut node, params) in
// nodes_and_params()) {         //     let mut mul_node =
// Node::Mul(MulNode::new(node.clone(), node.clone()));         //     let
// radices = mul_node.get_radices();         //     let mul_utry =
// mul_node.get_unitary_ref(&[params.clone(), params.clone()].concat());

//         //     let mut builder = UnitaryBuilder::new(radices);
//         //     let loc = Vec::from_iter(0..node.get_num_qudits());
//         //     builder.apply_right(node.get_unitary_ref(&params).view(),
// &loc, false);         //
// builder.apply_right(node.get_unitary_ref(&params).view(), &loc, false);
//         //     let utry = builder.get_unitary();
//         //     assert!((mul_utry - utry).opnorm_fro().unwrap() < 1e-8);
//         // }

//         #[test]
//         fn num_params_equals_sum_nodes(node in arbitrary_nodes())
//         {
//             let mul_node = Node::Mul(MulNode::new(node.clone(),
// node.clone()));             let num_params = mul_node.get_num_params();
//             assert_eq!(num_params, node.get_num_params() +
// node.get_num_params());         }

//         #[test]
//         fn radices_equals_same_radices(node in arbitrary_nodes())
//         {
//             let mul_node = Node::Mul(MulNode::new(node.clone(),
// node.clone()));             let radices = mul_node.get_radices();
//             assert_eq!(radices, node.get_radices());
//         }

//         #[test]
//         fn dimension_equals_same_dimension(node in arbitrary_nodes())
//         {
//             let mul_node = Node::Mul(MulNode::new(node.clone(),
// node.clone()));             let radices = mul_node.get_dimension();
//             assert_eq!(radices, node.get_dimension());
//         }

//         #[test]
//         fn is_hashable(node in arbitrary_nodes()) {
//             let mul_node = Node::Mul(MulNode::new(node.clone(),
// node.clone()));             let mut hasher =
// std::collections::hash_map::DefaultHasher::new();
// mul_node.hash(&mut hasher);             let _ = hasher.finish();
//         }

//         #[test]
//         fn is_hashable_set_insert(node in arbitrary_nodes()) {
//             let mut set = std::collections::HashSet::new();
//             let mul_node = Node::Mul(MulNode::new(node.clone(),
// node.clone()));             set.insert(mul_node.clone());
//             assert!(set.contains(&mul_node));
//         }

//         #[test]
//         fn equals_have_equal_hashes(node in arbitrary_nodes()) {
//             let mul_node1 = Node::Mul(MulNode::new(node.clone(),
// node.clone()));             let mul_node2 =
// Node::Mul(MulNode::new(node.clone(), node.clone()));
// assert_eq!(mul_node1, mul_node2);             let mut hasher1 =
// std::collections::hash_map::DefaultHasher::new();             let mut hasher2
// = std::collections::hash_map::DefaultHasher::new();
// mul_node1.hash(&mut hasher1);             mul_node2.hash(&mut hasher2);
//             assert_eq!(hasher1.finish(), hasher2.finish());
//         }

//         // TODO: Implement gradient tests with circuit.get_gradient
//     }
// }
