use std::ops::Mul;

// TODO: Add proper documentation to raw methods and add higher level
// functions that call them with the cartesian_match for loop unrolling.
/// Perform element-wise multiplication of two buffers.
pub unsafe fn hadamard_kernel_raw<C: Mul<Output = C> + Copy>(
    nrows: usize,
    ncols: usize,
    dst: *mut C,
    dst_rs: isize,
    dst_cs: isize,
    lhs: *const C,
    lhs_rs: isize,
    lhs_cs: isize,
    rhs: *const C,
    rhs_rs: isize,
    rhs_cs: isize,
) { unsafe {
    let mut current_dst_ptr = dst;
    let mut current_lhs_ptr = lhs;
    let mut current_rhs_ptr = rhs;

    for _i in 0..nrows {
        current_dst_ptr = current_dst_ptr.offset(dst_rs);
        current_lhs_ptr = current_lhs_ptr.offset(lhs_rs);
        current_rhs_ptr = current_rhs_ptr.offset(rhs_rs);

        for _j in 0..ncols {
            *current_dst_ptr = *current_lhs_ptr * *current_rhs_ptr;

            current_dst_ptr = current_dst_ptr.offset(dst_cs);
            current_lhs_ptr = current_lhs_ptr.offset(lhs_cs);
            current_rhs_ptr = current_rhs_ptr.offset(rhs_cs);
        }
    }
}}

/// Perform element-wise multiplication of two buffers and add the result into output.
pub unsafe fn hadamard_kernel_add_raw<C: Mul<Output = C> + Copy>(
    nrows: usize,
    ncols: usize,
    dst: *mut C,
    dst_rs: isize,
    dst_cs: isize,
    lhs: *const C,
    lhs_rs: isize,
    lhs_cs: isize,
    rhs: *const C,
    rhs_rs: isize,
    rhs_cs: isize,
) { unsafe {
    let mut current_dst_ptr = dst;
    let mut current_lhs_ptr = lhs;
    let mut current_rhs_ptr = rhs;

    for _i in 0..nrows {
        current_dst_ptr = current_dst_ptr.offset(dst_rs);
        current_lhs_ptr = current_lhs_ptr.offset(lhs_rs);
        current_rhs_ptr = current_rhs_ptr.offset(rhs_rs);

        for _j in 0..ncols {
            *current_dst_ptr = *current_lhs_ptr * *current_rhs_ptr;

            current_dst_ptr = current_dst_ptr.offset(dst_cs);
            current_lhs_ptr = current_lhs_ptr.offset(lhs_cs);
            current_rhs_ptr = current_rhs_ptr.offset(rhs_cs);
        }
    }
}}
