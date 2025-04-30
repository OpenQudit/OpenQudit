use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
use std::ops::Deref;

use itertools::Itertools;

use super::tree::ExpressionTree;
use super::tree::PrintTree;
// use crate::math::fused_reshape_permuted_reshape_into_impl;
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
pub struct ContractNode {
    /// The left node to be contracted.
    pub left: Box<ExpressionTree>,

    /// The right node to be contracted.
    pub right: Box<ExpressionTree>,

    // The qudit indices of the left node in circuit space.
    left_qudits: Vec<usize>,

    // The qudit indices of the right node in circuit space.
    right_qudits: Vec<usize>,

    /// The number of parameters in the left node.
    left_params: usize,

    /// The number of parameters in the right node.
    right_params: usize,

    /// The normal unfused output dimension of this node.
    dimension: usize,

    /// The normal output tensor shape after contraction and final permutation.
    out_tensor_shape: Vec<usize>,

    /// The shape of the left node as a tensor.
    pub left_tensor_shape: Vec<usize>,

    /// The permutation of the left node's indices as a tensor.
    pub left_perm: Vec<usize>,

    /// The shape of the left node after permutation before contraction.
    pub left_contraction_shape: (usize, usize),

    /// The shape of the right node as a tensor.
    pub right_tensor_shape: Vec<usize>,

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
        let left_radices = left.get_radices();
        let right_radices = right.get_radices();

        // The qudits shared in left_qudits and right_qudits will be contracted.
        let left_qudit_set =
            left_qudits.iter().map(|&x| x).collect::<HashSet<_>>();
        let right_qudit_set =
            right_qudits.iter().map(|&x| x).collect::<HashSet<_>>();
        let contracting_qudits = left_qudit_set
            .intersection(&right_qudit_set)
            .map(|&x| x)
            .collect::<Vec<_>>();
        let all_qudits = left_qudit_set
            .union(&right_qudit_set)
            .sorted()
            .map(|&x| x)
            .collect::<Vec<_>>();

        if contracting_qudits.len() == 0 {
            panic!("There must be at least one overlapping qudit between the left and right nodes.")
        }

        // The radix_map maps qudit indices in circuit space to their radix.
        let mut radix_map: HashMap<usize, usize> = HashMap::new();
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
            .map(|q| radix_map[q])
            .product::<usize>();

        let pre_out_tensor_shape: Vec<usize> = pre_out_order
            .iter()
            .map(|qstr| {
                radix_map[&qstr[..qstr.len() - 1].parse::<usize>().unwrap()]
            })
            .collect();

        let out_tensor_shape: Vec<usize> = correct_order
            .iter()
            .map(|qstr| {
                radix_map[&qstr[..qstr.len() - 1].parse::<usize>().unwrap()]
            })
            .collect();

        let left_dimension = left.get_dimension();
        let right_dimension = right.get_dimension();
        let left_params = left.get_num_params();
        let right_params = right.get_num_params();
        let dimension = radix_map.values().map(|&r| r).product();

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

    // // TODO: Optimize permutation shape (consecutive indices do not need to be
    // // split)

    // /// Produces a view of a permuted left node ready for contraction as matmul.
    // ///
    // /// # Arguments
    // ///
    // /// * `left` - The left node to be contracted.
    // ///
    // /// # Returns
    // ///
    // /// * `left_p` - A view of the left node permuted for contraction.
    // ///
    // /// # Safety
    // ///
    // /// This function is unsafe because it returns a view of a mutable buffer.
    // /// The buffer is only used locally in this function, so it is safe to
    // /// return a view of it, as long as the caller finishes using the view
    // /// before this method is called again.
    // #[inline]
    // unsafe fn pre_permute_left<'a>(&self, left: &'a Mat<C>) -> &'a Mat<C> {
    //     if self.skip_left {
    //         return left;
    //     }

    //     let left_perm_buf = &mut *self.left_perm_buf.get();

    //     let (is, os, dims) = &self.left_frpr;
    //     fused_reshape_permuted_reshape_into_impl(
    //         left.as_ref(),
    //         left_perm_buf.as_mut(),
    //         is,
    //         os,
    //         dims,
    //     );

    //     left_perm_buf
    // }

    // /// Produces a view of a permuted right node ready for contraction as
    // /// matmul.
    // ///
    // /// # Arguments
    // ///
    // /// * `right` - The right node to be contracted.
    // ///
    // /// # Returns
    // ///
    // /// * `right_p` - A view of the right node permuted for contraction.
    // ///
    // /// # Safety
    // ///
    // /// This function is unsafe because it returns a view of a mutable buffer.
    // /// The buffer is only used locally in this function, so it is safe to
    // /// return a view of it, as long as the caller finishes using the view
    // /// before this method is called again.
    // #[inline]
    // unsafe fn pre_permute_right<'a>(&self, right: &'a Mat<C>) -> &'a Mat<C> {
    //     if self.skip_right {
    //         return right;
    //     }

    //     let right_perm_buf = &mut *self.right_perm_buf.get();

    //     let (is, os, dims) = &self.right_frpr;
    //     fused_reshape_permuted_reshape_into_impl(
    //         right.as_ref(),
    //         right_perm_buf.as_mut(),
    //         is,
    //         os,
    //         dims,
    //     );

    //     // Reshape and perform tensor contraction as matmul
    //     right_perm_buf
    // }

    // #[inline]
    // fn post_permute_into_out(&self, pre_out: &Mat<C>, out: &mut MatMut<C>) {
    //     let (is, os, dims) = &self.out_frpr;
    //     unsafe {
    //         fused_reshape_permuted_reshape_into_impl(
    //             pre_out.as_ref(), out.as_mut(), is, os, dims,
    //         );
    //     }
    // }

    // #[inline]
    // fn contract(
    //     &self,
    //     left: &Mat<C>,
    //     right: &Mat<C>,
    //     out: &mut MatMut<C>,
    // ) {
    //     // Safety: left_p is used once and then discarded
    //     let left_p = unsafe { self.pre_permute_left(left) };

    //     // Safety: right_p is used once and then discarded
    //     let right_p = unsafe { self.pre_permute_right(right) };

    //     // Safety: pre_out_perm_buf is only used locally in this function
    //     let pre_out_perm_buf = unsafe { &mut *self.pre_out_perm_buf.get() };

    //     matmul(
    //         pre_out_perm_buf.as_mut(),
    //         right_p.as_ref(),
    //         left_p.as_ref(),
    //         None,
    //         C::one(),
    //         Parallelism::None,
    //     );
    //     self.post_permute_into_out(pre_out_perm_buf, out);
    // }

    // #[inline]
    // fn contract_no_pre_perm(
    //     &self,
    //     left: &Mat<C>,
    //     right: &Mat<C>,
    //     out: &mut MatMut<C>,
    // ) {
    //     // Safety: pre_out_perm_buf is only used locally in this function
    //     let pre_out_perm_buf = unsafe { &mut *self.pre_out_perm_buf.get() };

    //     matmul(
    //         pre_out_perm_buf.as_mut(),
    //         right.as_ref(),
    //         left.as_ref(),
    //         None,
    //         C::one(),
    //         Parallelism::None,
    //     );
    //     self.post_permute_into_out(pre_out_perm_buf, out);
    // }

    // fn calc_grad(
    //     &self,
    //     left: &Mat<C>,
    //     right: &Mat<C>,
    //     left_grad: &UnitaryGradient<C>,
    //     right_grad: &UnitaryGradient<C>,
    //     out_grad: &mut UnitaryGradient<C>,
    // ) {
    //     let mut grad_idx = 0;

    //     // Safety: right_p is used read-only in a loop and then discarded
    //     let right_p = unsafe { self.pre_permute_right(right) };

    //     for d_m in left_grad.iter() {
    //         let grad_ref = &mut out_grad[grad_idx];

    //         // Safety: left_grad_p is used once and then discarded
    //         let left_grad_p = unsafe { self.pre_permute_left(d_m) };

    //         self.contract_no_pre_perm(left_grad_p, right_p, &mut grad_ref.as_mut());
    //         grad_idx += 1;
    //     }

    //     // Safety: left_p is used read-only in a loop and then discarded
    //     let left_p = unsafe { self.pre_permute_left(left) };

    //     for d_m in right_grad.iter() {
    //         let grad_ref = &mut out_grad[grad_idx];

    //         // Safety: right_grad_p is used once and then discarded
    //         let right_grad_p = unsafe { self.pre_permute_right(d_m) };

    //         self.contract_no_pre_perm(&left_p, &right_grad_p, &mut grad_ref.as_mut());
    //         grad_idx += 1;
    //     }
    // }

    // fn calc_hess(
    //     &self,
    //     left: &Mat<C>,
    //     right: &Mat<C>,
    //     left_grad: &UnitaryGradient<C>,
    //     right_grad: &UnitaryGradient<C>,
    //     left_hess: &UnitaryHessian<C>,
    //     right_hess: &UnitaryHessian<C>,
    //     out_hess: &mut UnitaryHessian<C>,
    // ) {
    //     // Safety: right_p is used read-only in a loop and then discarded
    //     let right_p = unsafe { self.pre_permute_right(right) };

    //     // Upper left block: right_utry * left_hess
    //     for (k, dd_m) in left_hess.iter().enumerate() {
    //         let (left_row, left_col) = left_hess.index_to_coords(k);
    //         let hess_ref = &mut out_hess[(left_row, left_col)];
    //
    //         let left_hess_p = unsafe { self.pre_permute_left(dd_m) };

    //         self.contract_no_pre_perm(
    //             left_hess_p,
    //             right_p,
    //             &mut hess_ref.as_mut(),
    //         );
    //     }
    //
    //     // Store size of upper left block dimension offsetting other blocks
    //     let left_offset = left_grad.len();

    //     // Upper right block: right_grad * left_grad
    //     for left_d_m in left_grad.iter() {
    //         let mut hess_row_idx = 0;
    //         // Safety: left_grad_p is used read-only in a loop and then
    //         // discarded
    //         let left_grad_p = unsafe { self.pre_permute_left(&left_d_m) };

    //         for (hess_col_idx, right_d_m) in right_grad.iter().enumerate()
    //         {
    //             let shifted_col_idx = hess_col_idx + left_offset;
    //             let hess_ref = &mut out_hess[(hess_row_idx, shifted_col_idx)];

    //             // Safety: right_grad_p is used once and then discarded
    //             let right_grad_p =
    //                 unsafe { self.pre_permute_right(&right_d_m) };

    //             self.contract_no_pre_perm(
    //                 &left_grad_p,
    //                 &right_grad_p,
    //                 &mut hess_ref.as_mut(),
    //             );
    //             hess_row_idx += 1;
    //         }
    //     }

    //     // Safety: left_p is used read-only in a loop and then discarded
    //     let left_p = unsafe { self.pre_permute_left(left) };

    //     // Lower right block: right_hess * left_utry
    //     for (k, dd_m) in right_hess.iter().enumerate() {
    //         let (right_row, right_col) = right_hess.index_to_coords(k);
    //         let hess_ref = &mut out_hess[(right_row + left_offset, right_col + left_offset)];
    //
    //         let right_hess_p = unsafe { self.pre_permute_right(dd_m) };

    //         self.contract_no_pre_perm(
    //             left_p,
    //             right_hess_p,
    //             &mut hess_ref.as_mut(),
    //         );
    //     }
    // }
}

impl Function for ContractNode {
    fn get_num_params(&self) -> usize {
        self.left_params + self.right_params
    }
}

impl BoundedFn for ContractNode {
    fn get_bounds(&self) -> Vec<std::ops::Range<f64>> {
        self.left
            .get_bounds()
            .iter()
            .chain(self.right.get_bounds().iter())
            .cloned()
            .collect()
    }
}

impl QuditSystem for ContractNode {
    fn get_radices(&self) -> QuditRadices {
        QuditRadices::new(
            (0..(self.out_tensor_shape.len() / 2))
                .map(|x| self.out_tensor_shape[x])
                .collect_vec(),
        )
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }
}
//
// impl<C: ComplexScalar> UnitaryFn<C> for ContractNode<C> {
//     fn write_unitary(&self, params: &[C::Re], utry: &mut MatMut<C>) {
//         // Safety: We are not aliasing the child buffers.
//         let (mut left_utry_buf, mut right_utry_buf) = unsafe { self.child_bufs_as_mut() };
//
//         let (left, right) = params.split_at(self.left_params);
//         self.left.write_unitary(left, &mut left_utry_buf);
//         self.right.write_unitary(right, &mut right_utry_buf);
//         self.contract(
//             &self.left_buf,
//             &self.right_buf,
//             utry,
//         );
//     }
// }
//
// impl<C: ComplexScalar> DifferentiableUnitaryFn<C> for ContractNode<C> {
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
//         self.calc_grad(
//             &self.left_buf,
//             &self.right_buf,
//             left_grad_buf,
//             right_grad_buf,
//             out_grad,
//         );
//     }
//
//     fn write_unitary_and_gradient(
//         &self,
//         params: &[C::Re],
//         out_utry: &mut MatMut<C>,
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
//         self.contract(&self.left_buf, &self.right_buf, out_utry);
//         self.calc_grad(
//             &self.left_buf,
//             &self.right_buf,
//             left_grad_buf,
//             right_grad_buf,
//             out_grad,
//         );
//     }
// }
//
// impl<C: ComplexScalar> DoublyDifferentiableUnitaryFn<C> for ContractNode<C> {
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
//         self.calc_hess(
//             &self.left_buf,
//             &self.right_buf,
//             left_grad_buf,
//             right_grad_buf,
//             left_hess_buf,
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
//         self.contract(&self.left_buf, &self.right_buf, out_utry);
//         self.calc_hess(
//             &self.left_buf,
//             &self.right_buf,
//             left_grad_buf,
//             right_grad_buf,
//             left_hess_buf,
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
//         self.calc_grad(
//             &self.left_buf,
//             &self.right_buf,
//             left_grad_buf,
//             right_grad_buf,
//             out_grad,
//         );
//         self.calc_hess(
//             &self.left_buf,
//             &self.right_buf,
//             left_grad_buf,
//             right_grad_buf,
//             left_hess_buf,
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
//         self.contract(&self.left_buf, &self.right_buf, out_utry);
//         self.calc_grad(
//             &self.left_buf,
//             &self.right_buf,
//             left_grad_buf,
//             right_grad_buf,
//             out_grad,
//         );
//         self.calc_hess(
//             &self.left_buf,
//             &self.right_buf,
//             left_grad_buf,
//             right_grad_buf,
//             left_hess_buf,
//             right_hess_buf,
//             out_hess,
//         );
//     }
// }
//
// impl<C: ComplexScalar> PartialEq for ContractNode<C> {
//     fn eq(&self, other: &Self) -> bool {
//         (*self.left) == (*other.left)
//             && (*self.right) == (*other.left)
//             && self.left_params == other.left_params
//             && self.right_params == other.right_params
//             && self.dimension == other.dimension
//     }
// }
//
// impl<C: ComplexScalar> Eq for ContractNode<C> {}

// impl Hash for ContractNode {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.left.hash(state);
//         self.right.hash(state);
//         self.left_params.hash(state);
//         self.right_params.hash(state);
//         self.dimension.hash(state);
//     }
// }

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
