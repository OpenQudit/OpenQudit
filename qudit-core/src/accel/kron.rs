use faer::reborrow::ReborrowMut;
use faer_traits::ComplexField;

use super::cartesian_match;
use crate::matrix::MatMut;
use crate::matrix::MatRef;

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

/// Performs the Kronecker product of two matrices without checking assumptions.
///
/// # Safety
///
/// - The dimensions of `dst` must be `nrows(A) * nrows(B)` by `ncols(A) * ncols(B)`.
///
/// - The matrices must all be column-major.
///
/// # See also
///
/// - [`kron`] for a safe version of this function.
pub unsafe fn kron_unchecked<C: ComplexField>(dst: MatMut<C>, lhs: MatRef<C>, rhs: MatRef<C>) {
    let lhs_rows = lhs.nrows();
    let lhs_cols = lhs.ncols();
    let rhs_rows = rhs.nrows();
    let rhs_cols = rhs.ncols();
    cartesian_match!(
        { kron_kernel(dst, lhs, rhs, lhs_rows, lhs_cols, rhs_rows, rhs_cols) },
        (lhs_rows, (lhs_cols, (rhs_rows, (rhs_cols, ())))),
        (
            (2, 3, 4, _),
            ((2, 3, 4, _), ((2, 3, 4, _), ((2, 3, 4, _), ())))
        )
    );
}

/// Performs the Kronecker product of two square matrices without checking assumptions.
///
/// # Safety
///
/// - The dimensions of `dst` must be `nrows(A) * nrows(B)` by `ncols(A) * ncols(B)`.
///
/// - The matrices must all be column-major.
///
/// - The matrices must be square.
///
/// # See also
///
/// - [`kron`] for a safe version of this function.
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
///
/// where `a_ij` is the element at position `(i, j)` of `A`.
///
/// # Panics
///
/// Panics if `dst` does not have the correct dimensions. The dimensions
/// of `dst` must be `nrows(A) * nrows(B)` by `ncols(A) * ncols(B)`.
///
/// # Example
///
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
/// let mut dst = Mat::new();
/// dst.resize_with(4, 4, |_, _| 0f64);
/// kron(dst.as_mut(), a.as_ref(), b.as_ref());
/// assert_eq!(dst, c);
/// ```
pub fn kron<C: ComplexField>(dst: MatMut<C>, lhs: MatRef<C>, rhs: MatRef<C>) {
    let mut lhs = lhs;
    let mut rhs = rhs;
    let mut dst = dst;
    if dst.col_stride().unsigned_abs() < dst.row_stride().unsigned_abs() {
        dst = dst.transpose_mut();
        lhs = lhs.transpose();
        rhs = rhs.transpose();
    }

    assert!(Some(dst.nrows()) == lhs.nrows().checked_mul(rhs.nrows()));
    assert!(Some(dst.ncols()) == lhs.ncols().checked_mul(rhs.ncols()));

    if lhs.nrows() == lhs.ncols() && rhs.nrows() == rhs.ncols() {
        // Safety: The dimensions have been checked.
        unsafe { kron_sq_unchecked(dst, lhs, rhs) }
    } else {
        // Safety: The dimensions have been checked.
        unsafe { kron_unchecked(dst, lhs, rhs) }
    }
}
