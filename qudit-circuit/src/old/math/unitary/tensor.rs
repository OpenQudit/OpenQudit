// use faer_core::Mat;
// use faer_core::MatRef;
// use itertools::Itertools;
// use super::UnitaryMatrix;
// use crate::math::*;
// use crate::QuditRadices;

// /// A type to build unitaries using tensor networks
// #[derive(Debug, Clone)]
// pub struct UnitaryTensor<C: ComplexScalar>
// {
//     num_qudits:     usize,
//     num_idxs:       usize,
//     dim:            usize,
//     pi:             Vec<usize>,
//     radices:        QuditRadices,
//     tensor:         Option<Mat<C>>,
//     tensor_buf:     Option<Mat<C>>,
//     original_shape: Vec<usize>,
// }

// impl<C: ComplexScalar> UnitaryTensor<C>
// {
//     pub fn new(radices: QuditRadices) -> Self
//     {
//         let num_qudits = radices.len();
//         let dim = radices.iter().product();
//         let num_idxs = num_qudits * 2;
//         let pi = Vec::from_iter(0..num_idxs);
//         let og_shape = [&radices[..], &radices[..]].concat();

//         // Force faer to allocate the most amount of capacity we will ever
// need         // Then we can reshape for free
//         let max_dim = Self::calculate_max_dim(&radices);
//         let tensor = Mat::identity(max_dim, max_dim);
//         let tensor_buf = tensor.clone();
//         tensor.resize_with(dim, dim, |c| C::zero());
//         tensor_buf.resize_with(dim, dim, |c| C::zero());

//         UnitaryTensor {
//             num_qudits,
//             num_idxs,
//             dim,
//             pi,
//             radices,
//             tensor: Some(tensor),
//             tensor_buf: Some(tensor_buf),
//             original_shape: og_shape,
//         }
//     }

//     fn calculate_max_dim(radices: &QuditRadices) -> usize
//     {
//         let min_radix = radices.iter().min().unwrap().to_owned();
//         let dim = radices.iter().product::<usize>();
//         dim * dim / min_radix
//     }

//     pub fn get_square_data(&mut self) -> Mat<C>
//     {
//         self.reset_idxs();
//         match &self.tensor
//         {
//             Some(t) => t.to_owned(),
//             None => panic!("Tensor was unexpectedly None."),
//         }
//     }

//     pub fn get_unitary(&mut self) -> UnitaryMatrix<C>
//     {
//         let unitary = self.get_square_data();
//         UnitaryMatrix::new(self.radices.clone(), unitary)
//     }

//     pub fn get_revert_permutation(&mut self) -> Vec<usize>
//     {
//         self.pi
//             .iter()
//             .enumerate()
//             .sorted_by(|(_idx_a, a), (_idx_b, b)| a.cmp(b))
//             .map(|(idx, _a)| idx)
//             .collect()
//     }

//     pub fn permute_idxs(
//         &mut self,
//         new_pi: Vec<usize>,
//     )
//     {
//         if new_pi == self.pi
//         {
//             // No permutation necessary when we are already in the correct
// order             return;
//         }

//         let revert = self.get_revert_permutation();
//         let revert_and_apply: Vec<usize> = new_pi.iter().map(|&x|
// revert[x]).collect();

//         let current_shape = self.get_current_shape();
//         let new_shape: Vec<usize> = new_pi
//             .iter()
//             .map(|&x| match x
//             {
//                 r if r < self.num_qudits => self.radices[x],
//                 r if r >= self.num_qudits => self.radices[x -
// self.num_qudits],                 _ => panic!("Tensor index is invalid."),
//             })
//             .collect();

//         // self.tensor =
//         // Some(self.tensor.take().unwrap().permuted_axes(revert_and_apply));

//         let mut tensor = self.tensor.take().unwrap();
//         let mut buf = self.tensor_buf.take().unwrap();

//         fused_reshape_permute_reshape_into_fear_nc(
//             tensor.as_ref(),
//             &current_shape,
//             &revert_and_apply,
//             &buf.as_mut(),
//         );

//         // unsafe {
//         //     let mut tensor_view =
// ArrayViewMutD::from_shape_ptr(current_shape, tensor.as_mut_ptr());         //
// let mut buf_view = ArrayViewMutD::from_shape_ptr(new_shape,
// buf.as_mut_ptr());

//         //     tensor_view = tensor_view.permuted_axes(revert_and_apply);
//         //     buf_view.assign(&tensor_view);
//         // }

//         self.tensor = Some(buf);
//         self.tensor_buf = Some(tensor);

//         self.pi = new_pi;
//     }

//     pub fn reset_idxs(&mut self) {
// self.permute_idxs(Vec::from_iter(0..self.num_idxs)); }

//     pub fn get_current_shape(&self) -> Vec<usize>
//     {
//         self.pi
//             .iter()
//             .map(|&x| match x
//             {
//                 r if r < self.num_qudits => self.radices[x],
//                 r if r >= self.num_qudits => self.radices[x -
// self.num_qudits],                 _ => panic!("Tensor index is invalid."),
//             })
//             .collect()
//     }

//     pub fn apply_right(
//         &mut self,
//         utry: MatRef<C>,
//         location: &[usize],
//         inverse: bool,
//     )
//     {
//         // Permute Tensor Indicies
//         let left_perm = location.iter();
//         let right_perm = (0..self.num_idxs).filter(|x|
// !location.contains(&x));         let mut perm = vec![];
//         perm.extend(left_perm);
//         perm.extend(right_perm);
//         self.permute_idxs(perm);

//         // Reshape
//         let owned_tensor = self.tensor.take().unwrap();
//         let shape = self.get_current_shape();
//         let left_dim: usize = shape[..location.len()].iter().product();
//         let mul_shape = (left_dim, self.dim * self.dim / left_dim);
//         let reshaped = owned_tensor
//             .view()
//             .into_shape(mul_shape)
//             .expect("Cannot reshape tensor to matrix");

//         // Prepare scratch buffer
//         let mut buf = self.tensor_buf.take().unwrap();

//         // Apply Unitary
//         unsafe {
//             let mut buf_view = ArrayViewMut2::from_shape_ptr(mul_shape,
// buf.as_mut_ptr());             if inverse
//             {
//                 general_mat_mul(
//                     C::one(),
//                     &utry.conj().reversed_axes(),
//                     &reshaped,
//                     C::zero(),
//                     &mut buf_view,
//                 )
//             }
//             else
//             {
//                 general_mat_mul(C::one(), &utry, &reshaped, C::zero(), &mut
// buf_view)             }
//         }

//         // Replace buffers
//         self.tensor = Some(buf);
//         self.tensor_buf = Some(owned_tensor);
//     }

//     pub fn squeeze_right(
//         &mut self,
//         right: ArrayView2<C>,
//         middles: ArrayView3<C>,
//         outs: &mut ArrayViewMut3<C>,
//         location: &[usize],
//     )
//     {
//         // Permute Tensor Indicies
//         let left_perm = location.iter();
//         let right_perm = (0..self.num_idxs).filter(|x|
// !location.contains(&x));         let mut perm = vec![];
//         perm.extend(left_perm);
//         perm.extend(right_perm);
//         self.permute_idxs(perm);

//         // Reshape
//         let owned_tensor = self.tensor.take().unwrap();
//         let shape = self.get_current_shape();
//         let left_dim: usize = shape[..location.len()].iter().product();
//         let mul_shape = (left_dim, self.dim * self.dim / left_dim);
//         let reshaped = owned_tensor
//             .view()
//             .into_shape(mul_shape)
//             .expect("Cannot reshape tensor to matrix");

//         // Prepare scratch buffer
//         let mut buf = self.tensor_buf.take().unwrap();

//         for (middle, mut out) in
// middles.outer_iter().zip(outs.outer_iter_mut())         {
//             unsafe {
//                 let mut buf_view =
//
// ArrayViewMutD::from_shape_ptr(self.original_shape.clone(), buf.as_mut_ptr());
//                 let mut out_view = ArrayViewMut2::from_shape_ptr(mul_shape,
// out.as_mut_ptr());

//                 // Multiply: middle * reshaped(left) -> out (as buffer)
//                 general_mat_mul(C::one(), &middle, &reshaped, C::zero(), &mut
// out_view);

//                 // Unpermute out (as buffer)
//                 let new_out_view = out_view
//                     .into_shape(shape.clone())
//                     .expect("Cannot reshape buffer matrix view to tensor")
//                     .permuted_axes(self.get_revert_permutation());

//                 // Unshape out (as buffer) into buffer:
//                 //  - buf_view is now the buffer
//                 //  - the original buffer is still shaped correctly
//                 //  - this assignment circumvents an allocation
//                 buf_view.assign(&new_out_view);
//             }

//             // Multiply: buf * right -> out
//             general_mat_mul(C::one(), &right, &buf, C::zero(), &mut out);
//         }

//         // Replace buffers
//         self.tensor = Some(owned_tensor);
//         self.tensor_buf = Some(buf);
//     }

//     pub fn apply_left(
//         &mut self,
//         utry: ArrayView2<C>,
//         location: &[usize],
//         inverse: bool,
//     )
//     {
//         // Permute Tensor Indicies
//         let right_perm: Vec<usize> = location.iter().map(|x| x +
// self.num_qudits).collect();         let left_perm =
// (0..self.num_idxs).filter(|x| !right_perm.contains(&x));         let mut perm
// = vec![];         perm.extend(left_perm);
//         perm.extend(right_perm);
//         self.permute_idxs(perm);

//         // Reshape
//         let owned_tensor = self.tensor.take().unwrap();
//         // let shape = owned_tensor.shape().clone();
//         let shape = self.get_current_shape();
//         let right_dim: usize = shape[shape.len() -
// location.len()..].iter().product();         let mul_shape = (self.dim *
// self.dim / right_dim, right_dim);         let reshaped = owned_tensor
//             .view()
//             .into_shape(mul_shape)
//             .expect("Cannot reshape tensor to matrix");

//         // Prepare scratch buffer
//         let mut buf = self.tensor_buf.take().unwrap();

//         // Apply Unitary
//         unsafe {
//             let mut buf_view = ArrayViewMut2::from_shape_ptr(mul_shape,
// buf.as_mut_ptr());             if inverse
//             {
//                 general_mat_mul(
//                     C::one(),
//                     &reshaped,
//                     &utry.conj().reversed_axes(),
//                     C::zero(),
//                     &mut buf_view,
//                 );
//             }
//             else
//             {
//                 general_mat_mul(C::one(), &reshaped, &utry, C::zero(), &mut
// buf_view);             }
//         }

//         // Replace buffers
//         self.tensor = Some(buf);
//         self.tensor_buf = Some(owned_tensor);
//     }

//     pub fn squeeze_left(
//         &mut self,
//         left: ArrayView2<C>,
//         middles: ArrayView3<C>,
//         outs: &mut ArrayViewMut3<C>,
//         location: &[usize],
//     )
//     {
//         // permute self
//         // reshape -> reshaped
//         // for each middle
//         // mul reshaped * middle -> buf
//         // unshape buf
//         // permute buf
//         // buf * left -> out

//         // Permute Tensor Indicies
//         let right_perm: Vec<usize> = location.iter().map(|x| x +
// self.num_qudits).collect();         let left_perm =
// (0..self.num_idxs).filter(|x| !right_perm.contains(&x));         let mut perm
// = vec![];         perm.extend(left_perm);
//         perm.extend(right_perm);
//         self.permute_idxs(perm);

//         // Reshape
//         let owned_tensor = self.tensor.take().unwrap();
//         // let shape = owned_tensor.shape().clone();
//         let shape = self.get_current_shape();
//         let right_dim: usize = shape[shape.len() -
// location.len()..].iter().product();         let mul_shape = (self.dim *
// self.dim / right_dim, right_dim);         let reshaped = owned_tensor
//             .view()
//             .into_shape(mul_shape)
//             .expect("Cannot reshape tensor to matrix");

//         // Prepare scratch buffer
//         let mut buf = self.tensor_buf.take().unwrap();

//         for (middle, mut out) in
// middles.outer_iter().zip(outs.outer_iter_mut())         {
//             unsafe {
//                 let mut buf_view =
//
// ArrayViewMutD::from_shape_ptr(self.original_shape.clone(), buf.as_mut_ptr());
//                 let mut out_view = ArrayViewMut2::from_shape_ptr(mul_shape,
// out.as_mut_ptr());

//                 // Multiply: reshaped(right) * middle -> out (as buffer)
//                 general_mat_mul(C::one(), &reshaped, &middle, C::zero(), &mut
// out_view);

//                 // Unpermute out (as buffer)
//                 let new_out_view = out_view
//                     .into_shape(shape.clone())
//                     .expect("Cannot reshape buffer matrix view to tensor")
//                     .permuted_axes(self.get_revert_permutation());

//                 // Unshape out (as buffer) into buffer:
//                 //  - buf_view is now the buffer
//                 //  - the original buffer is still shaped correctly
//                 //  - this assignment circumvents an allocation
//                 buf_view.assign(&new_out_view);
//             }

//             // Multiply: buf * left -> out
//             general_mat_mul(C::one(), &buf, &left, C::zero(), &mut out);
//         }

//         // Replace buffers
//         self.tensor = Some(owned_tensor);
//         self.tensor_buf = Some(buf);
//     }

//     pub fn calc_env_matrix(
//         &mut self,
//         location: &[usize],
//     ) -> Array2<C>
//     {
//         self.reset_idxs();
//         let mut left_perm: Vec<usize> = (0..self.num_qudits)
//             .filter(|x| !location.contains(x))
//             .collect();
//         let left_perm_copy = left_perm.clone();
//         let left_extension = left_perm_copy.iter().map(|x| x +
// self.num_qudits);         left_perm.extend(left_extension);
//         let mut right_perm = location.to_owned();
//         right_perm.extend(location.iter().map(|x| x + self.num_qudits));

//         let mut perm = vec![];
//         perm.append(&mut left_perm);
//         perm.append(&mut right_perm);
//         let a = match &self.tensor
//         {
//             Some(t) => t
//                 .clone()
//                 .into_dyn()
//                 .into_shape(self.original_shape.clone())
//                 .unwrap()
//                 .permuted_axes(perm),
//             None => panic!("Tensor was unexpectedly None."),
//         };
//         let reshaped = a
//             .to_shape([
//                 2usize.pow(self.num_qudits as u32 - location.len() as u32),
//                 2usize.pow(self.num_qudits as u32 - location.len() as u32),
//                 2usize.pow(location.len() as u32),
//                 2usize.pow(location.len() as u32),
//             ])
//             .expect("Failed to reshape in calc_env_matrix.");
//         trace(reshaped.view())
//     }
// }

// #[cfg(test)]
// mod tests
// {
//     use itertools::Itertools;
//     use proptest::prelude::*;

//     use super::*;
//     use crate::math::unitary::UnitaryFn;
//     use crate::math::unitary::UnitaryMatrix;
//     use crate::math::Kronecker;
//     use crate::radices;
//     use crate::Gate;
//     use crate::QuditRadices;

//     proptest! {
//         #[test]
//         fn test_single_full_gate_on_left_equals_utry(radices in
// any::<QuditRadices>()) {             let mut tensor: UnitaryTensor<f64> =
// UnitaryTensor::new(radices.clone());             let utry: UnitaryMatrix<f64>
// = UnitaryMatrix::random(radices.clone());
// tensor.apply_left(utry.view(), &(0..radices.get_num_qudits()).collect_vec(),
// false);             tensor.get_unitary().assert_close_to(&utry);
//         }

//         #[test]
//         fn test_single_full_gate_on_right_equals_utry(radices in
// any::<QuditRadices>()) {             let mut tensor: UnitaryTensor<f64> =
// UnitaryTensor::new(radices.clone());             let utry: UnitaryMatrix<f64>
// = UnitaryMatrix::random(radices.clone());
// tensor.apply_right(utry.view(), &(0..radices.get_num_qudits()).collect_vec(),
// false);             tensor.get_unitary().assert_close_to(&utry);
//         }

//         #[test]
//         fn test_two_full_gates_on_left_equals_product(radices in
// any::<QuditRadices>()) {             let mut tensor: UnitaryTensor<f64> =
// UnitaryTensor::new(radices.clone());             let u1: UnitaryMatrix<f64> =
// UnitaryMatrix::random(radices.clone());             let u2 =
// UnitaryMatrix::random(radices.clone());
// tensor.apply_left(u1.view(), &(0..radices.get_num_qudits()).collect_vec(),
// false);             tensor.apply_left(u2.view(),
// &(0..radices.get_num_qudits()).collect_vec(), false);
// tensor.get_unitary().assert_close_to(&u1.dot(&u2));         }

//         #[test]
//         fn test_two_full_gates_on_right_equals_product(radices in
// any::<QuditRadices>()) {             let mut tensor: UnitaryTensor<f64> =
// UnitaryTensor::new(radices.clone());             let u1: UnitaryMatrix<f64> =
// UnitaryMatrix::random(radices.clone());             let u2 =
// UnitaryMatrix::random(radices.clone());
// tensor.apply_right(u1.view(), &(0..radices.get_num_qudits()).collect_vec(),
// false);             tensor.apply_right(u2.view(),
// &(0..radices.get_num_qudits()).collect_vec(), false);
// tensor.get_unitary().assert_close_to(&u2.dot(&u1));         }

//         #[test]
//         fn test_gate_with_gaps_left(radix1 in 2..5usize, radix2 in 2..5usize)
// {             let radices = radices![radix1, radix1, radix2];
//             let mut tensor: UnitaryTensor<f64> = UnitaryTensor::new(radices);
//             let u1: UnitaryMatrix<f64> =
// UnitaryMatrix::random(radices![radix1, radix2]);
// tensor.apply_left(u1.view(), &[0, 2], false);             let id1 =
// Gate::I(radix1).get_unitary(&[]);             let id2 =
// Gate::I(radix2).get_unitary(&[]);             let swap =
// Gate::Swap(radix1).get_unitary(&[]);             swap.kron(&id2)
//                 .dot(&id1.kron(&u1))
//                 .dot(&swap.kron(&id2))
//                 .assert_close_to(&tensor.get_unitary());
//         }

//         #[test]
//         fn test_two_gates_offset_on_the_left(radices in
// any_with::<QuditRadices>((2, 4, 3, 3))) {             let mut tensor:
// UnitaryTensor<f64> = UnitaryTensor::new(radices.clone());             let u1:
// UnitaryMatrix<f64> = UnitaryMatrix::random(radices[..2].into());
// let u2 = UnitaryMatrix::random(radices[1..].into());
// tensor.apply_left(u1.view(), &[0, 1], false);
// tensor.apply_left(u2.view(), &[1, 2], false);             let id0 =
// Gate::I(radices[0]).get_unitary(&[]);             let id2 =
// Gate::I(radices[2]).get_unitary(&[]);             u1.kron(&id2)
//                 .dot(&id0.kron(&u2))
//                 .assert_close_to(&tensor.get_unitary());
//         }

//         #[test]
//         fn test_two_gates_offset_on_the_right(radices in
// any_with::<QuditRadices>((2, 4, 3, 3))) {             let mut tensor:
// UnitaryTensor<f64> = UnitaryTensor::new(radices.clone());             let u1:
// UnitaryMatrix<f64> = UnitaryMatrix::random(radices[..2].into());
// let u2 = UnitaryMatrix::random(radices[1..].into());
// tensor.apply_right(u1.view(), &[0, 1], false);
// tensor.apply_right(u2.view(), &[1, 2], false);             let id0 =
// Gate::I(radices[0]).get_unitary(&[]);             let id2 =
// Gate::I(radices[2]).get_unitary(&[]);             id0.kron(&u2)
//                 .dot(&u1.kron(&id2))
//                 .assert_close_to(&tensor.get_unitary());
//         }

//         #[test]
//         fn test_two_gates_offset_both_sides(radices in
// any_with::<QuditRadices>((2, 4, 3, 3))) {             let mut tensor:
// UnitaryTensor<f64> = UnitaryTensor::new(radices.clone());             let u1:
// UnitaryMatrix<f64> = UnitaryMatrix::random(radices[..2].into());
// let u2 = UnitaryMatrix::random(radices[1..].into());
// tensor.apply_left(u1.view(), &[0, 1], false);
// tensor.apply_right(u2.view(), &[1, 2], false);             let id0 =
// Gate::I(radices[0]).get_unitary(&[]);             let id2 =
// Gate::I(radices[2]).get_unitary(&[]);             id0.kron(&u2)
//                 .dot(&u1.kron(&id2))
//                 .assert_close_to(&tensor.get_unitary());
//         }

//         #[test]
//         fn test_three_gates_with_gaps_left(radix1 in 2..5usize, radix2 in
// 2..5usize) {             let radices = radices![radix1, radix1, radix2];
//             let mut tensor: UnitaryTensor<f64> =
// UnitaryTensor::new(radices.clone());             let u1: UnitaryMatrix<f64> =
// UnitaryMatrix::random(vec![radix1, radix2].into());             let u2 =
// UnitaryMatrix::random(radices[1..].into());             let u3 =
// UnitaryMatrix::random(vec![radix1, radix2].into());
// tensor.apply_left(u1.view(), &[0, 2], false);
// tensor.apply_left(u2.view(), &[1, 2], false);
// tensor.apply_left(u3.view(), &[0, 2], false);             let id1 =
// Gate::I(radix1).get_unitary(&[]);             let id2 =
// Gate::I(radix2).get_unitary(&[]);             let swap =
// Gate::Swap(radix1).get_unitary(&[]);             swap.kron(&id2)
//                 .dot(&id1.kron(&u1))
//                 .dot(&swap.kron(&id2))
//                 .dot(&id1.kron(&u2))
//                 .dot(&swap.kron(&id2))
//                 .dot(&id1.kron(&u3))
//                 .dot(&swap.kron(&id2))
//                 .assert_close_to(&tensor.get_unitary());
//         }

//         #[test]
//         fn test_three_gates_with_gaps_right(radix1 in 2..5usize, radix2 in
// 2..5usize) {             let radices = radices![radix1, radix1, radix2];
//             let mut tensor: UnitaryTensor<f64> =
// UnitaryTensor::new(radices.clone());             let u1: UnitaryMatrix<f64> =
// UnitaryMatrix::random(vec![radix1, radix2].into());             let u2 =
// UnitaryMatrix::random(radices[1..].into());             let u3 =
// UnitaryMatrix::random(vec![radix1, radix2].into());
// tensor.apply_right(u1.view(), &[0, 2], false);
// tensor.apply_right(u2.view(), &[1, 2], false);
// tensor.apply_right(u3.view(), &[0, 2], false);             let id1 =
// Gate::I(radix1).get_unitary(&[]);             let id2 =
// Gate::I(radix2).get_unitary(&[]);             let swap =
// Gate::Swap(radix1).get_unitary(&[]);             swap.kron(&id2)
//                 .dot(&id1.kron(&u3))
//                 .dot(&swap.kron(&id2))
//                 .dot(&id1.kron(&u2))
//                 .dot(&swap.kron(&id2))
//                 .dot(&id1.kron(&u1))
//                 .dot(&swap.kron(&id2))
//                 .assert_close_to(&tensor.get_unitary());
//         }
//     }
// }
