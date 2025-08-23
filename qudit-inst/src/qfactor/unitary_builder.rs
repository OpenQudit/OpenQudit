use qudit_core::accel::{matmul_unchecked, tensor_fused_reshape_permute_reshape_into_prepare, fused_reshape_permute_reshape_into_impl};
use qudit_core::memory::MemoryBuffer;
use qudit_core::matrix::{Mat, MatRef};
use qudit_core::{ComplexScalar, QuditSystem};
use qudit_core::unitary::UnitaryMatrix;
use faer::mat::AsMatRef;
use qudit_core::memory::Memorable;
use std::ops::Deref;
use std::slice;

//START Helper functions--------------------------------------------------------------------------

/// Calculates the row-major strides for a continguous tensor.
fn calc_contiguous_row_major_strides(dims: &[usize]) -> Vec<isize> {
    let d = dims.len();
    let mut strides: Vec<isize> = vec![0; d];
    strides[d - 1] = 1;
    for i in (0..d - 1).rev() {
        strides[i] = strides[i + 1] * (dims[i + 1] as isize);
    }
    return strides;
}

/// Calculates the col-major strides for a continguous tensor.
fn calc_contiguous_col_major_strides(dims: &[usize]) -> Vec<isize> {
    let d = dims.len();
    let mut strides: Vec<isize> = vec![0; d];
    strides[0] = 1;
    for i in (1..d) {
        strides[i] = strides[i - 1] * (dims[i - 1] as isize);
    }
    return strides;
}

/// Partial trace on first 2 indices of a 4D tensor.
fn partial_trace<C: ComplexScalar>(data_ptr: *const C, dims: &[usize], strides: &[isize]) -> Mat<C> {
    let left_dim = dims[0];
    let right_dim = dims[2];
    assert_eq!(dims[0], dims[1]);
    assert_eq!(dims[2], dims[3]);

    let mut result = Mat::zeros(right_dim, right_dim);

    for i in 0..left_dim {
        let offset = (i as isize) * strides[0] + (i as isize) * strides[1];
        let sub_mat_ptr = unsafe {data_ptr.offset(offset)};
        let sub_mat_ref = unsafe {
            MatRef::from_raw_parts(
                sub_mat_ptr,
                right_dim,
                right_dim,
                strides[2] as isize,
                strides[3] as isize,
            )
        };
        result += sub_mat_ref;
    }
    return result;
}

fn raw_tensor_fused_reshape_permute_reshape_into<'a, C: Memorable>(
    inp: *const C,
    inp_dims: &[usize],
    inp_strides: &[isize],
    out: *mut C,
    out_dims: &[usize],
    out_strides: &[isize],
    shape: &[usize],
    perm: &[usize]
) {
    
    let (is, os, dims) = tensor_fused_reshape_permute_reshape_into_prepare(
        inp_dims,
        inp_strides,
        out_dims,
        out_strides,
        shape,
        perm
    );

    // println!("is: {:?}", is);
    // println!("os: {:?}", os);
    // println!("dims: {:?}", dims);

    // let rev_os = os.into_iter().rev().collect::<Vec<_>>();
    // contiguity_check(&rev_os, &dims);

    unsafe {
        fused_reshape_permute_reshape_into_impl(
            inp, 
            out, 
            &is, 
            &os,
            &dims
        );
    }
}

// Column-major order
fn initialize_matrix_buffer<C: ComplexScalar>(row: usize, col: usize) -> MemoryBuffer<C> {
    let identity = Mat::<C>::identity(row, col);
    let mut identity_elems = Vec::<C>::new();
    
    for i in 0..col {
        identity_elems.extend_from_slice(identity.col_as_slice(i));
    }
    
    let tensor_data = MemoryBuffer::from_slice(64, &identity_elems);
    return tensor_data;
}

fn print_mat<C: ComplexScalar>(data: &MemoryBuffer<C>, left_dim: usize, right_dim: usize, row_stride: usize, col_stride: usize ) {
    let mat_ref = unsafe {MatRef::from_raw_parts(data.as_ptr(), left_dim, right_dim, row_stride as isize, col_stride as isize)};
    println!("{:?}", mat_ref);
}

fn contiguity_check(strides: &[isize], dims: &[usize]) {
    assert_eq!(strides.len(), dims.len());
    assert_eq!(strides[0], 1);
    let mut expected = 1;
    for i in 1..strides.len() {
        expected = expected * dims[i - 1];
        assert_eq!(expected, strides[i] as usize);
    }
}

//END Helper functions--------------------------------------------------------------------------

pub struct UnitaryBuilder<C: ComplexScalar> {
    pub num_qudits: usize,
    pub num_idxs: usize,
    pub dim: usize,
    pub pi: Vec<usize>,
    pub radixes: Vec<usize>,
    
    tensor_data: MemoryBuffer<C>,
    current_dims: Vec<usize>,
    current_strides: Vec<isize>,
}

impl<C: ComplexScalar> UnitaryBuilder<C> {

    pub fn new(num_qudits: usize, radixes: Vec<usize>) -> Self {
        let dim = radixes.iter().product();
        let num_idxs = num_qudits * 2;
        let pi = Vec::from_iter(0..num_idxs);
    
        let tensor_data = initialize_matrix_buffer(dim, dim);
        
        let mut current_dims = Vec::<usize>::new();
        current_dims.extend_from_slice(&radixes);
        current_dims.extend_from_slice(&radixes);

        let current_strides = calc_contiguous_col_major_strides(&current_dims);

        UnitaryBuilder {
            num_qudits,
            num_idxs,
            dim,
            pi,
            radixes,
            
            tensor_data,
            current_dims,
            current_strides,
        }
    }

    pub fn get_utry(&mut self) -> UnitaryMatrix<C> {
        self.reset_idxs();

        let mut matrix_data = initialize_matrix_buffer(self.dim, self.dim);
        let matrix_dims = vec![self.dim, self.dim];
        let matrix_strides = vec![1, self.dim as isize];

        raw_tensor_fused_reshape_permute_reshape_into(
            self.tensor_data.as_ptr(),
            &self.current_dims,
            &self.current_strides,
            matrix_data.as_mut_ptr(),
            &matrix_dims,
            &matrix_strides,
            &self.current_dims,
            &(0..self.num_idxs).collect::<Vec<_>>(),
        );
        let temp_matrix = unsafe{MatRef::from_raw_parts(matrix_data.as_ptr(), self.dim, self.dim, 1, self.dim as isize)};
        
        //println!("Matrix - {:?}", temp_matrix);

        return UnitaryMatrix::<C>::new(&self.radixes, temp_matrix.to_owned());
    }

    pub fn permute_idxs(&mut self, pi: Vec<usize>) {
        
        let mut base_dims = Vec::<usize>::new();
        base_dims.extend_from_slice(&self.radixes);
        base_dims.extend_from_slice(&self.radixes);
        let base_strides = calc_contiguous_col_major_strides(&base_dims);

        let mut new_dims = vec![0; self.num_idxs];
        let mut new_strides = vec![0; self.num_idxs];

        for i in 0..self.num_idxs {
            new_dims[i] = base_dims[pi[i]];
            new_strides[i] = base_strides[pi[i]];
        }

        self.pi = pi;
        self.current_dims = new_dims;
        self.current_strides = new_strides;
    }

    pub fn reset_idxs(&mut self) {
        self.permute_idxs(Vec::from_iter(0..self.num_idxs));
    }

    pub fn get_current_shape(&self) -> Vec<usize> {
        self.current_dims.clone()
    }

    pub fn apply_right(&mut self, utry: &UnitaryMatrix<C>, location: &[usize], inverse: bool) {
        assert_eq!(utry.radices().deref().len(), location.len());

        // Calculate permutation vector
        let left_perm: Vec<usize> = location.iter().cloned().collect();
        let right_perm: Vec<usize> = (0..self.num_idxs).filter(|x| !left_perm.contains(&x)).collect();
        let mut perm: Vec<usize> = left_perm.iter().chain(right_perm.iter()).copied().collect();

        // Reshape
        let left_dims: Vec<usize> = left_perm.iter().map(|&x| self.current_dims[x]).collect();
        let right_dims: Vec<usize> = right_perm.iter().map(|&x| self.current_dims[x]).collect();
        let left_dim = left_dims.iter().product();
        let right_dim = self.dim * self.dim / left_dim;
        let mut reshaped_data = initialize_matrix_buffer(left_dim, right_dim);
        raw_tensor_fused_reshape_permute_reshape_into(
            self.tensor_data.as_ptr(),
            &self.current_dims,
            &self.current_strides,
            reshaped_data.as_mut_ptr(),
            &[left_dim, right_dim],
            &[1, left_dim as isize],
            &self.current_dims,
            &perm,
        );

        // Apply Unitary
        let true_utry = if inverse {
            &utry.dagger()
        } else {
            utry
        };
        let mut prod = Mat::zeros(left_dim, right_dim);
        let reshaped_mat_ref = unsafe {MatRef::<C>::from_raw_parts(reshaped_data.as_ptr(), left_dim, right_dim, 1, left_dim as isize)};
        matmul_unchecked(true_utry.as_mat_ref(), reshaped_mat_ref, prod.as_mut());
        
        // Reshape Back
        let mut intermediate_shape: Vec<usize> = left_dims.iter().chain(right_dims.iter()).copied().collect();
        let mut inverse_perm = vec![0; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inverse_perm[p] = i;
        }
        raw_tensor_fused_reshape_permute_reshape_into(
            prod.as_ptr(),
            &[left_dim, right_dim],
            &[prod.row_stride(), prod.col_stride()],
            self.tensor_data.as_mut_ptr(),
            &self.current_dims,
            &self.current_strides,
            &intermediate_shape,
            &inverse_perm
        );
    }

    pub fn apply_left(&mut self, utry: &UnitaryMatrix<C>, location: &[usize], inverse: bool) {
        // Calculate permutation vector
        let right_perm: Vec<usize> = location.iter().map(|x| x + self.num_qudits).collect();
        let left_perm: Vec<usize> = (0..self.num_idxs).filter(|x| !right_perm.contains(&x)).collect();
        let mut perm: Vec<usize> = left_perm.iter().chain(right_perm.iter()).copied().collect();

        // Reshape
        let left_dims: Vec<usize> = left_perm.iter().map(|&x| self.current_dims[x]).collect();
        let right_dims: Vec<usize> = right_perm.iter().map(|&x| self.current_dims[x]).collect();
        let right_dim = right_dims.iter().product();
        let left_dim = self.dim * self.dim / right_dim;
        let mut reshaped_data = initialize_matrix_buffer(left_dim, right_dim);
        raw_tensor_fused_reshape_permute_reshape_into(
            self.tensor_data.as_ptr(),
            &self.current_dims,
            &self.current_strides,
            reshaped_data.as_mut_ptr(),
            &[left_dim, right_dim],
            &[1, left_dim as isize],
            &self.current_dims,
            &perm,
        );
        
        // Apply Unitary
        let true_utry = if inverse {
            &utry.dagger()
        } else {
            utry
        };
        let mut prod = Mat::zeros(left_dim, right_dim);
        let reshaped_mat_ref = unsafe {MatRef::from_raw_parts(reshaped_data.as_ptr(), left_dim, right_dim, 1, left_dim as isize)};
        matmul_unchecked(reshaped_mat_ref, true_utry.as_mat_ref(), prod.as_mut());
        
        // Reshape Back
        let mut intermediate_shape: Vec<usize> = left_dims.iter().chain(right_dims.iter()).copied().collect();
        let mut inverse_perm = vec![0; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inverse_perm[p] = i;
        }
        raw_tensor_fused_reshape_permute_reshape_into(
            prod.as_ptr(),
            &[left_dim, right_dim],
            &[prod.row_stride(), prod.col_stride()],
            self.tensor_data.as_mut_ptr(),
            &self.current_dims,
            &self.current_strides,
            &intermediate_shape,
            &inverse_perm
        );
    }

    pub fn calc_env_matrix(&mut self, location: &[usize]) -> Mat<C> {
        self.reset_idxs();

        // Calculate permutation vector
        let mut env_perm: Vec<usize> = (0..self.num_qudits).filter(|x| !location.contains(x)).collect();
        env_perm.extend(env_perm.clone().iter().map(|x| x + self.num_qudits));
        let mut op_perm = location.to_owned();
        op_perm.extend(location.iter().map(|x| x + self.num_qudits));
        let perm: Vec<usize> = env_perm.iter().chain(op_perm.iter()).copied().collect();

        // Reshape
        let op_dim = location.iter().map(|&i| self.radixes[i]).product();
        let env_dim = self.dim / op_dim;
        let reshaped_4d_dims = [env_dim, env_dim, op_dim, op_dim];
        let reshaped_4d_strides = calc_contiguous_col_major_strides(&reshaped_4d_dims);
        let mut reshaped_4d_data = initialize_matrix_buffer(self.dim, self.dim);
        raw_tensor_fused_reshape_permute_reshape_into(
            self.tensor_data.as_ptr(),
            &self.current_dims, 
            &self.current_strides,
            reshaped_4d_data.as_mut_ptr(),
            &reshaped_4d_dims,
            &reshaped_4d_strides,
            &self.current_dims,
            &perm,
        );

        // Trace out the environment
        let result = partial_trace(
            reshaped_4d_data.as_ptr(),
            &reshaped_4d_dims,
            &reshaped_4d_strides,
        );
        return result;
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use qudit_core::c64;
    use std::f64::consts::PI;

    // Basic gates for testing
    fn cnot_unitary() -> UnitaryMatrix<c64> {
        let mut cnot = Mat::<c64>::zeros(4, 4);
        cnot[(0, 0)] = c64::new(1.0, 0.0);
        cnot[(1, 1)] = c64::new(1.0, 0.0);
        cnot[(2, 3)] = c64::new(1.0, 0.0);
        cnot[(3, 2)] = c64::new(1.0, 0.0);
        return UnitaryMatrix::new(&[2, 2], cnot);
    }

    fn swap_unitary() -> UnitaryMatrix<c64> {
        let mut swap = Mat::<c64>::zeros(4, 4);
        swap[(0, 0)] = c64::new(1.0, 0.0);
        swap[(1, 2)] = c64::new(1.0, 0.0);
        swap[(2, 1)] = c64::new(1.0, 0.0);
        swap[(3, 3)] = c64::new(1.0, 0.0);
        return UnitaryMatrix::new(&[2, 2], swap);
    }

    fn pauli_x() -> UnitaryMatrix<c64> {
        let mut x = Mat::<c64>::zeros(2, 2);
        x[(0, 1)] = c64::new(1.0, 0.0);
        x[(1, 0)] = c64::new(1.0, 0.0);
        return UnitaryMatrix::new(&[2], x);
    }

    fn pauli_y() -> UnitaryMatrix<c64> {
        let mut y = Mat::<c64>::zeros(2, 2);
        y[(0, 1)] = c64::new(0.0, -1.0);
        y[(1, 0)] = c64::new(0.0, 1.0);
        return UnitaryMatrix::new(&[2], y);
    }

    fn pauli_z() -> UnitaryMatrix<c64> {
        let mut z = Mat::<c64>::zeros(2, 2);
        z[(0, 0)] = c64::new(1.0, 0.0);
        z[(1, 1)] = c64::new(-1.0, 0.0);
        return UnitaryMatrix::new(&[2], z);
    }

    fn hadamard() -> UnitaryMatrix<c64> {
        let factor = 1.0 / (2.0f64).sqrt();
        let mut h = Mat::<c64>::zeros(2, 2);
        h[(0, 0)] = c64::new(factor, 0.0);
        h[(0, 1)] = c64::new(factor, 0.0);
        h[(1, 0)] = c64::new(factor, 0.0);
        h[(1, 1)] = c64::new(-factor, 0.0);
        return UnitaryMatrix::new(&[2], h);
    }

    //------------------------------------------------------------------------------------------

    #[test]
    fn simple_right_inverse_test() {
        let mut test_subject = UnitaryBuilder::<c64>::new(2, vec![2, 2]);
        test_subject.apply_right(&swap_unitary(), &[0, 1], false);
        test_subject.apply_right(&swap_unitary(), &[0, 1], true);

        let attempt = test_subject.get_utry();
        let answer = Mat::<c64>::identity(4, 4);

        for r in 0..4 {
            for c in 0..4 {
                assert!((attempt.as_mat_ref()[(r, c)] - answer[(r, c)]).norm() < 1e-9);
            }
        }
    }

    #[test]
    fn simple_left_inverse_test() {
        let mut test_subject = UnitaryBuilder::<c64>::new(2, vec![2, 2]);
        test_subject.apply_left(&swap_unitary(), &[0, 1], false);
        test_subject.apply_left(&swap_unitary(), &[0, 1], true);

        let attempt = test_subject.get_utry();
        let answer = Mat::<c64>::identity(4, 4);

        for r in 0..4 {
            for c in 0..4 {
                assert!((attempt.as_mat_ref()[(r, c)] - answer[(r, c)]).norm() < 1e-9);
            }
        }
    }

    #[test]
    fn simple_env_matrix_test() {
        let mut test_subject = UnitaryBuilder::<c64>::new(2, vec![2, 2]);
        test_subject.apply_right(&cnot_unitary(), &[0, 1], false);
        let attempt = test_subject.calc_env_matrix(&[0]);
        
        // CNOT = P_0 ⊗ I + P_1 ⊗ X. 
        // Thus, Tr_{1}(CNOT) = P_0 x Tr(I) + P_1 x Tr(X) = 2 P_0.
        let mut answer = Mat::<c64>::zeros(2, 2);
        answer[(0, 0)] = c64::new(2.0, 0.0);

        for r in 0..2 {
            for c in 0..2 {
                assert!((attempt[(r, c)] - answer[(r, c)]).norm() < 1e-9);
            }
        }
    }

    #[test]
    fn three_qubit_single_gate_test() {
        let mut test_subject = UnitaryBuilder::<c64>::new(3, vec![2, 2, 2]);
        test_subject.apply_right(&pauli_x(), &[1], false);
        let result = test_subject.get_utry();

        // Expected: I ⊗ X ⊗ I
        let mut expected = Mat::<c64>::zeros(8, 8);
        expected[(0, 2)] = c64::new(1.0, 0.0);
        expected[(1, 3)] = c64::new(1.0, 0.0);
        expected[(2, 0)] = c64::new(1.0, 0.0);
        expected[(3, 1)] = c64::new(1.0, 0.0);
        expected[(4, 6)] = c64::new(1.0, 0.0);
        expected[(5, 7)] = c64::new(1.0, 0.0);
        expected[(6, 4)] = c64::new(1.0, 0.0);
        expected[(7, 5)] = c64::new(1.0, 0.0);

        for r in 0..8 {
            for c in 0..8 {
                assert!((result.as_mat_ref()[(r, c)] - expected[(r, c)]).norm() < 1e-9);
            }
        }
    }
    
   #[test]
   fn xyz_test() {
        let mut test_subject = UnitaryBuilder::<c64>::new(1, vec![2]);
        test_subject.apply_right(&pauli_x(), &[0], false);
        test_subject.apply_right(&pauli_y(), &[0], false);
        test_subject.apply_right(&pauli_z(), &[0], false);
        let result = test_subject.get_utry();

        // XYZ = -iI
        let mut expected = Mat::<c64>::zeros(2, 2);
        expected[(0, 0)] = c64::new(0.0, -1.0);
        expected[(1, 1)] = c64::new(0.0, -1.0);

        for r in 0..2 {
            for c in 0..2 {
                assert!((result.as_mat_ref()[(r, c)] - expected[(r, c)]).norm() < 1e-12);
            }
        }
    }

   #[test]
   fn h_cx_test() {
        let mut test_subject = UnitaryBuilder::<c64>::new(2, vec![2, 2]);
        test_subject.apply_right(&hadamard(), &[0], false);
        test_subject.apply_right(&cnot_unitary(), &[0, 1], false);
        let result = test_subject.get_utry();
      
        let factor = 1.0 / (2.0f64).sqrt();
        let mut expected = Mat::<c64>::zeros(4, 4);
        expected[(0, 0)] = c64::new(factor, 0.0);
        expected[(0, 2)] = c64::new(factor, 0.0);
    
        expected[(1, 1)] = c64::new(factor, 0.0);
        expected[(1, 3)] = c64::new(factor, 0.0);

        expected[(2, 1)] = c64::new(factor, 0.0);
        expected[(2, 3)] = c64::new(-factor, 0.0);
    
        expected[(3, 0)] = c64::new(factor, 0.0);
        expected[(3, 2)] = c64::new(-factor, 0.0);

        for r in 0..4 {
            for c in 0..4 {
                assert!((result.as_mat_ref()[(r, c)] - expected[(r, c)]).norm() < 1e-12);
            }
        }
    }

   #[test]
   fn env_matrix_entangled_state_test() {
        let mut test_subject = UnitaryBuilder::<c64>::new(3, vec![2, 2, 2]);
        test_subject.apply_right(&hadamard(), &[0], false);
        test_subject.apply_right(&cnot_unitary(), &[0, 1], false);
        test_subject.apply_right(&cnot_unitary(), &[0, 2], false);

        let env = test_subject.calc_env_matrix(&[0]);
        
        // U = (P_0 ⊗ I ⊗ I + P_1 ⊗ I ⊗ X)(P_0 ⊗ I ⊗ I + P_1 ⊗ X ⊗ I)(H ⊗ I ⊗ I)
        //   = P_0 H ⊗ I ⊗ I + P_0 P_1 H ⊗ X ⊗ I + P_1 P_0 H ⊗ I ⊗ X + P_1 H ⊗ X ⊗ X
        //   = P_0 H ⊗ I ⊗ I + P_1 H ⊗ X ⊗ X.
        // Apparently, Tr_{23}(U) = 4 P_0 H
        let mut expected = Mat::<c64>::zeros(2, 2);
        let factor = 2.0 * 2.0_f64.sqrt();
        expected[(0, 0)] = c64::new(factor, 0.0);
        expected[(0, 1)] = c64::new(factor, 0.0);

        for r in 0..2 {
            for c in 0..2 {
                assert!((env[(r, c)] - expected[(r, c)]).norm() < 1e-12);
            }
        }
    }

}
