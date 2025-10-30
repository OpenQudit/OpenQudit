//! Functions and structs for efficient generalized matrix multiplication (GEMM).

use coe::is_same;
use nano_gemm::Plan;
use num_traits::One;
use num_traits::Zero;

use crate::c32;
use crate::c64;
use faer::MatMut;
use faer::MatRef;
use crate::ComplexScalar;

/// Stores a plan for a generalized matrix multiplication (GEMM). Based on the dimensions and underlying
/// field of the matrices, the plan will select the appropriate mili/micro-kernels for performance.
pub struct MatMulPlan<C: ComplexScalar> {
    m: usize,
    n: usize,
    k: usize,
    plan: Plan<C>,
}

impl<C: ComplexScalar> MatMulPlan<C> {
    
    /// Creates a new GEMM plan for column-major matrices.
    /// 
    /// # Arguments
    /// 
    /// * `m`: Number of rows in the left-hand side matrix.
    /// * `n`: Number of columns in the right-hand side matrix.
    /// * `k`: Number of columns in the left-hand side matrix. 
    ///     This should equal the number of rows in the right-hand side matrix.
    /// 
    /// # Returns
    /// 
    /// * A `MatMulPlan` instance.
    /// 
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        if is_same::<C, c32>() {
            let plan = Plan::new_colmajor_lhs_and_dst_c32(m, n, k);
            // Safety: This is safe because C is c32. 
            Self { m, n, k, plan: unsafe { std::mem::transmute(plan) } }
        } else {
            let plan = Plan::new_colmajor_lhs_and_dst_c64(m, n, k);
            Self { m, n, k, plan: unsafe { std::mem::transmute(plan) } }
        }
    }

    /// Executes the milikernel of the plan, for matrix multiplication. (`alpha = 0`, `beta = 1`)
    /// We do not perform comprehensive checks.
    /// 
    /// # Arguments
    /// 
    /// * `lhs`: The left-hand side matrix to multiply.
    /// * `rhs`: The right-hand side matrix to multiply.
    /// * `out`: The output matrix where the result will be stored.
    /// 
    /// # Safety 
    /// 
    /// * The matrices must be column-major.
    /// * The dimensions of `out` must be `lhs.nrows() * rhs.nrows()` by `lhs.ncols() * rhs.ncols()`.
    /// 
    /// # Examples
    /// ```
    /// use qudit_core::accel::MatMulPlan;
    /// use qudit_core::matrix::{mat, Mat};
    /// use qudit_core::c64;
    /// 
    /// let mut out = Mat::<c64>::zeros(2, 2);
    /// 
    /// let lhs = mat![
    ///     [c64::new(1.0, 0.0), c64::new(2.0, 0.0)],
    ///     [c64::new(3.0, 0.0), c64::new(4.0, 0.0)]
    /// ];
    /// let rhs = mat![
    ///     [c64::new(5.0, 0.0), c64::new(6.0, 0.0)],
    ///     [c64::new(7.0, 0.0), c64::new(8.0, 0.0)]
    /// ];
    /// 
    /// let test_plan = MatMulPlan::new(lhs.nrows(), rhs.ncols(), lhs.ncols());
    /// test_plan.execute_unchecked(lhs.as_ref(), rhs.as_ref(), out.as_mut());
    /// 
    /// let expected = mat![
    ///     [c64::new(19.0, 0.0), c64::new(22.0, 0.0)],
    ///     [c64::new(43.0, 0.0), c64::new(50.0, 0.0)]
    /// ];
    /// 
    /// assert_eq!(expected, out);
    /// ```
    /// 
    #[inline(always)]
    pub fn execute_unchecked(&self, lhs: MatRef<C>, rhs: MatRef<C>, out: MatMut<C>) {
        let m = lhs.nrows();
        let n = rhs.ncols();
        let k = lhs.ncols();
        let out_col_stride = out.col_stride();

        unsafe {
            self.plan.execute_unchecked(
                m,
                n,
                k,
                out.as_ptr_mut() as _,
                1,
                out_col_stride,
                lhs.as_ptr() as _,
                1,
                lhs.col_stride(),
                rhs.as_ptr() as _,
                1,
                rhs.col_stride(),
                C::zero().into(),
                C::one().into(),
                false,
                false,
            );
        }
    }

    #[inline(always)]
    pub unsafe fn execute_raw_unchecked(
        &self,
        lhs: *const C,
        rhs: *const C,
        out: *mut C,
        dst_rs: isize,
        dst_cs: isize,
        lhs_rs: isize,
        lhs_cs: isize,
        rhs_rs: isize,
        rhs_cs: isize,
    ) {
        self.plan.execute_unchecked(
            self.m,
            self.n,
            self.k,
            out,
            dst_rs,
            dst_cs,
            lhs,
            lhs_rs,
            lhs_cs,
            rhs,
            rhs_rs,
            rhs_cs,
            C::zero(),
            C::one(),
            false,
            false,
        );
    }

    /// Executes the milikernel of the plan, for matrix multiplication followed by addition. 
    /// (`alpha = 1`, `beta = 1`) We do not perform comprehensive checks.
    /// 
    /// # Arguments
    /// 
    /// * `lhs`: The left-hand side matrix to add.
    /// * `rhs`: The right-hand side matrix to add.
    /// * `out`: The output matrix where the result will be stored.
    /// 
    /// # Safety
    /// 
    /// * The matrices must be column-major.
    /// * The dimensions of `out` must be `lhs.nrows() * rhs.nrows()` by `lhs.ncols() * rhs.ncols()`.
    /// 
    /// # Examples
    /// ```
    /// use qudit_core::accel::MatMulPlan;
    /// use qudit_core::matrix::{mat, Mat};
    /// use qudit_core::c64;
    /// 
    /// let mut out = Mat::<c64>::ones(2, 2);
    /// 
    /// let lhs = mat![
    ///     [c64::new(1.0, 0.0), c64::new(2.0, 0.0)],
    ///     [c64::new(3.0, 0.0), c64::new(4.0, 0.0)]
    /// ];
    /// let rhs = mat![
    ///     [c64::new(5.0, 0.0), c64::new(6.0, 0.0)],
    ///     [c64::new(7.0, 0.0), c64::new(8.0, 0.0)]
    /// ];
    /// 
    /// let test_plan = MatMulPlan::new(lhs.nrows(), rhs.ncols(), lhs.ncols());
    /// test_plan.execute_add_unchecked(lhs.as_ref(), rhs.as_ref(), out.as_mut());
    /// 
    /// let expected = mat![
    ///     [c64::new(20.0, 0.0), c64::new(23.0, 0.0)],
    ///     [c64::new(44.0, 0.0), c64::new(51.0, 0.0)]
    /// ];
    /// 
    /// assert_eq!(expected, out);
    /// ```
    ///
    pub fn execute_add_unchecked(&self, lhs: MatRef<C>, rhs: MatRef<C>, out: MatMut<C>) {
        let m = lhs.nrows();
        let n = rhs.ncols();
        let k = lhs.ncols();
        let out_col_stride = out.col_stride();
        
        unsafe {
            self.plan.execute_unchecked(
                m,
                n,
                k,
                out.as_ptr_mut() as _,
                1,
                out_col_stride,
                lhs.as_ptr() as _,
                1,
                lhs.col_stride(),
                rhs.as_ptr() as _,
                1,
                rhs.col_stride(),
                C::one().into(),
                C::one().into(),
                false,
                false,
            );
        }
    }

    #[inline(always)]
    pub unsafe fn execute_add_raw_unchecked(
        &self,
        lhs: *const C,
        rhs: *const C,
        out: *mut C,
        dst_rs: isize,
        dst_cs: isize,
        lhs_rs: isize,
        lhs_cs: isize,
        rhs_rs: isize,
        rhs_cs: isize,
    ) {
        self.plan.execute_unchecked(
            self.m,
            self.n,
            self.k,
            out,
            dst_rs,
            dst_cs,
            lhs,
            lhs_rs,
            lhs_cs,
            rhs,
            rhs_rs,
            rhs_cs,
            C::one(),
            C::one(),
            false,
            false,
        );
    }
}

/// Performs matrix-matrix multiplication. (`alpha = 0`, `beta = 1`) 
/// 
/// # Arguments
/// 
/// * `lhs`: The left-hand side matrix to multiply.
/// * `rhs`: The right-hand side matrix to multiply.
/// * `out`: The output matrix where the result will be stored.
/// 
/// # Safety
/// 
/// * The matrices must be column-major.
/// * The dimensions of `out` must be `lhs.nrows() * rhs.nrows()` by `lhs.ncols() * rhs.ncols()`.
///
/// # Examples
/// ```
/// use qudit_core::accel::matmul_unchecked;
/// use qudit_core::matrix::{mat, Mat};
/// use qudit_core::c64;
/// 
/// let mut out = Mat::<c64>::zeros(2, 2);
/// 
/// let lhs = mat![
///     [c64::new(1.0, 0.0), c64::new(2.0, 0.0)],
///     [c64::new(3.0, 0.0), c64::new(4.0, 0.0)]
/// ];
/// let rhs = mat![
///     [c64::new(5.0, 0.0), c64::new(6.0, 0.0)],
///     [c64::new(7.0, 0.0), c64::new(8.0, 0.0)]
/// ];
/// 
/// matmul_unchecked(lhs.as_ref(), rhs.as_ref(), out.as_mut());
/// 
/// let expected = mat![
///     [c64::new(19.0, 0.0), c64::new(22.0, 0.0)],
///     [c64::new(43.0, 0.0), c64::new(50.0, 0.0)]
/// ];
/// 
/// assert_eq!(expected, out);
/// ```
/// 
#[inline(always)]
pub fn matmul_unchecked<C: ComplexScalar>(lhs: MatRef<C>, rhs: MatRef<C>, out: MatMut<C>) {
    let m = lhs.nrows();
    let n = rhs.ncols();
    let k = lhs.ncols();

    // After the runtime check of C, we explicitly transmute our inputs.
    // This allows type-specific optimizations.
    if is_same::<C, c32>() {
        let plan = Plan::new_colmajor_lhs_and_dst_c32(m, n, k);
        let out: MatMut<c32> = unsafe { std::mem::transmute(out) };
        let rhs: MatRef<c32> = unsafe { std::mem::transmute(rhs) };
        let lhs: MatRef<c32> = unsafe { std::mem::transmute(lhs) };
        let out_col_stride = out.col_stride();
        
        unsafe {
            plan.execute_unchecked(
                m,
                n,
                k,
                out.as_ptr_mut() as _,
                1,
                out_col_stride,
                lhs.as_ptr() as _,
                1,
                lhs.col_stride(),
                rhs.as_ptr() as _,
                1,
                rhs.col_stride(),
                c32::zero().into(),
                c32::one().into(), // TODO: Figure if I can create custom kernels for one/zero alpha/beta
                false,
                false,
            );
        }
    } else {
        let plan = Plan::new_colmajor_lhs_and_dst_c64(m, n, k);
        let out: MatMut<c64> = unsafe { std::mem::transmute(out) };
        let rhs: MatRef<c64> = unsafe { std::mem::transmute(rhs) };
        let lhs: MatRef<c64> = unsafe { std::mem::transmute(lhs) };
        let out_col_stride = out.col_stride();
        
        unsafe {
            plan.execute_unchecked(
                m,
                n,
                k,
                out.as_ptr_mut() as _,
                1,
                out_col_stride,
                lhs.as_ptr() as _,
                1,
                lhs.col_stride(),
                rhs.as_ptr() as _,
                1,
                rhs.col_stride(),
                c64::zero(),
                c64::one(), // TODO: Figure if I can create custom kernels for one/zero alpha/beta
                false,
                false,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{c32, c64};
    use faer::Mat;
    use faer::mat;
    use num_traits::Zero;

    #[test]
    fn test_matmul_unchecked() {
        let m = 2;
        let n = 2;
        let k = 2;

        let mut lhs = Mat::<c32>::zeros(m, k);
        let mut rhs = Mat::<c32>::zeros(k, n);
        let mut out = Mat::<c32>::zeros(m, n);

        for i in 0..m {
            for j in 0..k {
                lhs[(i, j)] = c32::new((i + j) as f32, (i + j) as f32);
            }
        }

        for i in 0..k {
            for j in 0..n {
                rhs[(i, j)] = c32::new((i + j) as f32, (i + j) as f32);
            }
        }

        matmul_unchecked(lhs.as_ref(), rhs.as_ref(), out.as_mut());

        for i in 0..m {
            for j in 0..n {
                let mut sum = c32::zero();
                for l in 0..k {
                    sum += lhs[(i, l)] * rhs[(l, j)];
                }
                assert_eq!(out[(i, j)], sum);
            }
        }
    }
    
    #[test]
    fn matmul_unchecked2() {
        let mut out = Mat::<c64>::zeros(2, 2);

        let lhs = mat![
            [c64::new(1.0, 0.0), c64::new(2.0, 0.0)],
            [c64::new(3.0, 0.0), c64::new(4.0, 0.0)]
        ];
        let rhs = mat![
            [c64::new(5.0, 0.0), c64::new(6.0, 0.0)],
            [c64::new(7.0, 0.0), c64::new(8.0, 0.0)]
        ];

        matmul_unchecked(lhs.as_ref(), rhs.as_ref(), out.as_mut());

        let expected = mat![
            [c64::new(19.0, 0.0), c64::new(22.0, 0.0)],
            [c64::new(43.0, 0.0), c64::new(50.0, 0.0)]
        ];

        assert_eq!(out, expected);
    }
}
