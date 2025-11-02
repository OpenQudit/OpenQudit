//! Functions to efficiently perform the Kronecker product and fused Kronecker-add operations.

use std::ops::AddAssign;
use std::ops::Mul;

use faer::reborrow::ReborrowMut;
use faer_traits::ComplexField;

use super::cartesian_match;
use faer::MatMut;
use faer::MatRef;
use crate::ComplexScalar;

// TODO: Add proper documentation to raw methods and add higher level
// functions that call them with the cartesian_match for loop unrolling.
/// Perform a kroneckor product between two matrix buffers.
pub unsafe fn kron_kernel_raw<C: Mul<Output = C> + Copy>(
    dst: *mut C,
    dst_rs: isize,
    dst_cs: isize,
    lhs: *const C,
    lhs_nrows: usize,
    lhs_ncols: usize,
    lhs_rs: isize,
    lhs_cs: isize,
    rhs: *const C,
    rhs_nrows: usize,
    rhs_ncols: usize,
    rhs_rs: isize,
    rhs_cs: isize,
) {
    for lhs_j in 0..lhs_ncols {
        for lhs_i in 0..lhs_nrows {
            let lhs_val = *lhs.offset(lhs_i as isize * lhs_rs + lhs_j as isize * lhs_cs);

            let dst_major_row = lhs_i * rhs_nrows;
            let dst_major_col = lhs_j * rhs_ncols;


            for rhs_j in 0..rhs_ncols {
                for rhs_i in 0..rhs_nrows {
                    let rhs_val = *rhs.offset(rhs_i as isize * rhs_rs + rhs_j as isize * rhs_cs);

                    let dst_row = dst_major_row + rhs_i;
                    let dst_col = dst_major_col + rhs_j;

                    let dst_offset = dst_row as isize * dst_rs + dst_col as isize * dst_cs;
                    
                    *dst.offset(dst_offset) = lhs_val * rhs_val;
                }
            }
        }
    }
}

/// Perform a kroneckor product between two matrix buffers and add the result to the output.
pub unsafe fn kron_kernel_add_raw<C: Mul<Output = C> + Copy + AddAssign>(
    dst: *mut C,
    dst_rs: isize,
    dst_cs: isize,
    lhs: *const C,
    lhs_nrows: usize,
    lhs_ncols: usize,
    lhs_rs: isize,
    lhs_cs: isize,
    rhs: *const C,
    rhs_nrows: usize,
    rhs_ncols: usize,
    rhs_rs: isize,
    rhs_cs: isize,
) {
    for lhs_j in 0..lhs_ncols {
        for lhs_i in 0..lhs_nrows {
            let lhs_val = *lhs.offset(lhs_i as isize * lhs_rs + lhs_j as isize * lhs_cs);

            let dst_major_row = lhs_i * rhs_nrows;
            let dst_major_col = lhs_j * rhs_ncols;


            for rhs_j in 0..rhs_ncols {
                for rhs_i in 0..rhs_nrows {
                    let rhs_val = *rhs.offset(rhs_i as isize * rhs_rs + rhs_j as isize * rhs_cs);

                    let dst_row = dst_major_row + rhs_i;
                    let dst_col = dst_major_col + rhs_j;

                    let dst_offset = dst_row as isize * dst_rs + dst_col as isize * dst_cs;
                    
                    *dst.offset(dst_offset) += lhs_val * rhs_val;
                }
            }
        }
    }
}

/// The inner kernel that performs the Kronecker product of two matrices 
/// without checking assumptions. 
///
/// # Safety
///
/// * The dimensions of `dst` must be at least `lhs_rows * rhs_rows` by `lhs_cols * rhs_cols`.
///
/// # See also
/// 
/// * [`kron`] for a safe version of this function.
/// 
unsafe fn kron_kernel<C: ComplexField>(
    mut dst: MatMut<C>,
    lhs: MatRef<C>,
    rhs: MatRef<C>,
    lhs_rows: usize,
    lhs_cols: usize,
    rhs_rows: usize,
    rhs_cols: usize,
) {
    for lhs_j in 0..lhs_cols {
        for lhs_i in 0..lhs_rows {
            
            let lhs_val = lhs.get_unchecked(lhs_i, lhs_j);
            
            for rhs_j in 0..rhs_cols {
                for rhs_i in 0..rhs_rows {
                    
                    let rhs_val = rhs.get_unchecked(rhs_i, rhs_j);
                    
                    *(dst
                        .rb_mut()
                        .get_mut_unchecked(lhs_i * rhs_rows + rhs_i, lhs_j * rhs_cols + rhs_j)) =
                        lhs_val.mul_by_ref(rhs_val);
                }
            }
        }
    }
}

/// Performs the Kronecker product of two matrices and adds this to the destination 
/// without checking assumptions.
/// 
/// More efficient that performing the Kronecker product followed by addition;
/// we only look up each element of `dst` once, rather than twice.
/// 
/// # Safety
///
/// * The dimensions of `dst` must be at least `lhs_rows * rhs_rows` by `lhs_cols * rhs_cols`.
///
/// # See also
/// 
/// * [`kron_add`] for a safe version of this function.
/// 
unsafe fn kron_kernel_add<C: ComplexScalar>(
    mut dst: MatMut<C>,
    lhs: MatRef<C>,
    rhs: MatRef<C>,
    lhs_rows: usize,
    lhs_cols: usize,
    rhs_rows: usize,
    rhs_cols: usize,
) {
    for lhs_j in 0..lhs_cols {
        for lhs_i in 0..lhs_rows {
            
            let lhs_val = lhs.get_unchecked(lhs_i, lhs_j);
            
            for rhs_j in 0..rhs_cols {
                for rhs_i in 0..rhs_rows {
                    
                    let rhs_val = rhs.get_unchecked(rhs_i, rhs_j);
                    
                    // Notice that each element of `dst` is only looked up once throughout the loops.
                    *(dst
                        .rb_mut()
                        .get_mut_unchecked(lhs_i * rhs_rows + rhs_i, lhs_j * rhs_cols + rhs_j)) +=
                        lhs_val.mul_by_ref(rhs_val);
                }
            }
        }
    }
}

/// Performs the Kronecker product of two matrices without checking assumptions.
///
/// # Safety
///
/// * The dimensions of `dst` must be at least `lhs.nrows() * rhs.nrows()` by `lhs.ncols() * rhs.ncols()`.
///
/// # See also
///
/// * [`kron`] for a safe version of this function.
/// 
pub unsafe fn kron_unchecked<C: ComplexField>(dst: MatMut<C>, lhs: MatRef<C>, rhs: MatRef<C>) {
    let lhs_rows = lhs.nrows();
    let lhs_cols = lhs.ncols();
    let rhs_rows = rhs.nrows();
    let rhs_cols = rhs.ncols();

    cartesian_match!(
        { kron_kernel(dst, lhs, rhs, lhs_rows, lhs_cols, rhs_rows, rhs_cols) },
        (lhs_rows, (lhs_cols, (rhs_rows, (rhs_cols, ())))),
        ((2, 3, 4, _), ((2, 3, 4, _), ((2, 3, 4, _), ((2, 3, 4, _), ()))))
    );
}

/// Performs the Kronecker product of two square matrices without checking assumptions.
///
/// # Safety
///
/// * The dimensions of `dst` must be at least `lhs.nrows() * rhs.nrows()` by `lhs.ncols() * rhs.ncols()`.
/// * The matrices must be square.
///
/// # See also
/// 
/// * [`kron`] for a safe version of this function.
/// 
pub unsafe fn kron_sq_unchecked<C: ComplexField>(dst: MatMut<C>, lhs: MatRef<C>, rhs: MatRef<C>) {
    let lhs_dim = lhs.nrows();
    let rhs_dim = rhs.nrows();

    cartesian_match!(
        { kron_kernel(dst, lhs, rhs, lhs_dim, lhs_dim, rhs_dim, rhs_dim) },
        (lhs_dim, (rhs_dim, ())),
        (
            (2, 3, 4, 6, 8, 9, 16, 27, 32, 64, 81, _),
            ((2, 3, 4, 6, 8, 9, 16, 27, 32, 64, 81, _), ())
        )
    );
}

/// Kronecker product of two matrices.
///
/// The Kronecker product of two matrices `A` and `B` is a block matrix
/// `C` with the following structure:
///
/// ```text
/// C = [ a00 * B, a01 * B, ..., a0n * B ]
///     [ a10 * B, a11 * B, ..., a1n * B ]
///     [ ...    , ...    , ..., ...     ]
///     [ am0 * B, am1 * B, ..., amn * B ]
/// ```
/// where `a_ij` is the element at position `(i, j)` of `A`.
///
/// # Panics
///
/// * If `dst` does not have the correct dimensions. The dimensions
///     of `dst` must be `nrows(A) * nrows(B)` by `ncols(A) * ncols(B)`.
///
/// # Example
/// ```
/// use qudit_core::matrix::mat;
/// use qudit_core::matrix::Mat;
/// use qudit_core::accel::kron;
///
/// let a = mat![
///     [1.0, 2.0],
///     [3.0, 4.0],
/// ];
/// let b = mat![
///     [0.0, 5.0],
///     [6.0, 7.0],
/// ];
/// let c = mat![
///     [0.0 , 5.0 , 0.0 , 10.0],
///     [6.0 , 7.0 , 12.0, 14.0],
///     [0.0 , 15.0, 0.0 , 20.0],
///     [18.0, 21.0, 24.0, 28.0],
/// ];
/// 
/// let mut dst = Mat::new();
/// dst.resize_with(4, 4, |_, _| 0f64);
/// 
/// kron(a.as_ref(), b.as_ref(), dst.as_mut());
/// 
/// assert_eq!(dst, c);
/// ```
/// 
pub fn kron<C: ComplexField>(lhs: MatRef<C>, rhs: MatRef<C>, dst: MatMut<C>) {
    let mut lhs = lhs;
    let mut rhs = rhs;
    let mut dst = dst;

    // Ensures that `dst` is in column-major order.
    if dst.col_stride().unsigned_abs() < dst.row_stride().unsigned_abs() {
        dst = dst.transpose_mut();
        lhs = lhs.transpose();
        rhs = rhs.transpose();
    }

    // Checks that the dimensions of `dst` matches the expected dimensions of the Kronecker product of
    // `lhs` and `rhs`. Also checks that no overflow occurs during the multiplication.
    assert!(Some(dst.nrows()) == lhs.nrows().checked_mul(rhs.nrows()));
    assert!(Some(dst.ncols()) == lhs.ncols().checked_mul(rhs.ncols()));

    // Uses a specialized kernel for square matrices if both `lhs` and `rhs` are square.
    if lhs.nrows() == lhs.ncols() && rhs.nrows() == rhs.ncols() {
        // Safety: The dimensions have been checked.
        unsafe { kron_sq_unchecked(dst, lhs, rhs) }
    } else {
        // Safety: The dimensions have been checked.
        unsafe { kron_unchecked(dst, lhs, rhs) }
    }
}

/// Computes the Kronecker product of two matrices and adds the result to a destination matrix.
/// 
/// For `A` ∈ M(R_a, C_a), `B` ∈ M(R_b, C_b), `C` ∈ M(R_a * R_b, C_a * C_b), this function mutates `C` 
/// such C_{i * R_b + k , j * C_b + l} -> C_{i * R_b + k , j * C_b + l} + A_{i, j} * B_{k, l}.
/// 
/// # Arguments
/// 
/// * `lhs` -  The left hand-side matrix for the kronecker product. `A` in the description above.
/// * `rhs` - The right hand-side matrix for the kronecker product. `B` in the description above.
/// * `dst` - The matrix to be summed (mutated) by the kronercker product of `lhs` and `rhs`. 
///     `C` in the description above.
/// 
/// # Panics
/// 
/// * If `dst.nrows()` doesn't match `lhs.nrows()` times `rhs.nrows()`
/// * If `dst.ncols()` doesn't match `lhs.ncols()` times `rhs.ncols()`
/// * If an overflow occurs when calculating the expected dimensions.
/// 
/// # Example
/// ```
/// use qudit_core::matrix::{mat, Mat};
/// use qudit_core::accel::kron_add;
/// use qudit_core::c64;
///
/// let mut dst = Mat::<c64>::zeros(4, 4);
///
/// let lhs = Mat::<c64>::from_fn(2, 2, |i, j| -> c64 {c64::new((2*i+1+j) as f64, 0.0)});
/// let rhs = Mat::<c64>::from_fn(2, 2, |i, j| -> c64 {c64::new((2*i+5+j) as f64, 0.0)});
///
/// kron_add(lhs.as_ref(), rhs.as_ref(), dst.as_mut());
/// 
/// let expected_data = [
///      [c64::new(5.0, 0.0), c64::new(6.0, 0.0), c64::new(10.0, 0.0), c64::new(12.0, 0.0)],
///      [c64::new(7.0, 0.0), c64::new(8.0, 0.0), c64::new(14.0, 0.0), c64::new(16.0, 0.0)],
///      [c64::new(15.0, 0.0), c64::new(18.0, 0.0), c64::new(20.0, 0.0), c64::new(24.0, 0.0)],
///      [c64::new(21.0, 0.0), c64::new(24.0, 0.0), c64::new(28.0, 0.0), c64::new(32.0, 0.0)]
/// ];
/// let expected = Mat::from_fn(4, 4, |i, j| -> c64 {expected_data[i][j]});
/// assert_eq!(dst, expected);
/// ```
/// 
pub fn kron_add<C: ComplexScalar>(lhs: MatRef<C>, rhs: MatRef<C>, dst: MatMut<C>) {
    let mut lhs = lhs; 
    let mut rhs = rhs; 
    let mut dst = dst; 

    // Makes `dst` is in column-major order. To maintain the same computation, we transpose `lhs` and `rhs` as well. 
    // This is allowed because the transpose is distributive over the Kronecker product and addition. Notice we are 
    // transposing input views, so we need not re-transpose our matrices after mutating the underlying data of dst.
    if dst.col_stride().unsigned_abs() < dst.row_stride().unsigned_abs() {
        dst = dst.transpose_mut(); 
        lhs = lhs.transpose(); 
        rhs = rhs.transpose();
    }
    // Makes sure the dimesion of the Kronecker product between lhs, rhs matches that of dst.
    // Recall (F_r, F_c) = (D_r * E_r, D_c * E_c) where F = D (x) E. 
    // Also makes sure overflows do not occur during the multiplications.
    assert!(Some(dst.nrows()) == lhs.nrows().checked_mul(rhs.nrows()));
    assert!(Some(dst.ncols()) == lhs.ncols().checked_mul(rhs.ncols()));
    
    // Performs the actual Kronecker product followed by sum.
    unsafe {
        kron_kernel_add(
            dst,
            lhs,
            rhs,
            lhs.nrows(),  
            lhs.ncols(),
            rhs.nrows(),
            rhs.ncols(),
        );
    }
}

#[cfg(test)]
mod kron_tests {
    use super::*;
    use faer::Mat;
    use faer::mat;
    use crate::c64;
    use qudit_macros::complex_mat;

    #[test]
    fn kron_add_test() {
        let mut dst = complex_mat!([
            [1.0-8.0j, 2.0+67.0j, 3.0, 4.0], 
            [5.0, 6.0, 7.0, 8.0], 
            [9.0, 10.0, 11.0, 12.0], 
            [13.0, 14.0, 15.0, 16.0]]);
        let lhs= complex_mat!([
            [1.0+9.0j, 2.0], 
            [3.0, 4.0]
        ]);
        let rhs = complex_mat!([
            [5.0-8.0j, 6.0],
            [7.0, 8.0]
        ]);

        kron_add(lhs.as_ref(), rhs.as_ref(), dst.as_mut());

        let expected = complex_mat!([
            [78.0+29.0j, 8.0+121.0j, 13.0-16.0j, 16.0],
            [12.0+63.0j, 14.0+72.0j, 21.0, 24.0],
            [24.0-24.0j, 28.0, 31.0-32.0j, 36.0],
            [34.0, 38.0, 43.0, 48.0]
        ]);

        assert_eq!(dst, expected);
    }

    #[test]
    fn kron_test () {
        let a = mat![
            [1.0, 2.0],
            [3.0, 4.0],
        ];
        let b = mat![
            [0.0, 5.0],
            [6.0, 7.0],
        ];
        let c = mat![
            [0.0 , 5.0 , 0.0 , 10.0],
            [6.0 , 7.0 , 12.0, 14.0],
            [0.0 , 15.0, 0.0 , 20.0],
            [18.0, 21.0, 24.0, 28.0],
        ];
        let mut dst = Mat::new();
        dst.resize_with(4, 4, |_, _| 0f64);
        kron(a.as_ref(), b.as_ref(),dst.as_mut());
        assert_eq!(dst, c);
    }
}
