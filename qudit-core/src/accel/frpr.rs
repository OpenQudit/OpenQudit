use std::collections::VecDeque;
use std::num::Wrapping;

use super::cartesian_match;
use crate::matrix::MatMut;
use crate::matrix::MatRef;

fn __reshape_kernel_0<E: Copy>(
    out: *mut E,
    inp: *const E,
    _dims: &[usize],
    _in_strides: &[isize],
    _out_strides: &[isize],
) {
    unsafe {
        *out = *inp;
    }
}

#[inline(always)]
unsafe fn __reshape_kernel_5_impl<E: Copy>(
    out: *mut E,
    inp: *const E,
    (d0, d1, d2, d3, d4): (usize, usize, usize, usize, usize),
    (is0, is1, is2, is3, is4): (isize, isize, isize, isize, isize),
    (os0, os1, os2, os3, os4): (isize, isize, isize, isize, isize),
) {
    let mut in_offset0 = Wrapping(0isize);
    let mut out_offset0 = Wrapping(0isize);

    for _ in 0..d0 {
        let mut in_offset1 = in_offset0;
        let mut out_offset1 = out_offset0;

        for _ in 0..d1 {
            let mut in_offset2 = in_offset1;
            let mut out_offset2 = out_offset1;

            for _ in 0..d2 {
                let mut in_offset3 = in_offset2;
                let mut out_offset3 = out_offset2;

                for _ in 0..d3 {
                    let mut in_offset4 = in_offset3;
                    let mut out_offset4 = out_offset3;

                    for _ in 0..d4 {
                        unsafe {
                            *out.offset(out_offset4.0) = *inp.offset(in_offset4.0);
                        }
                        in_offset4 += is4;
                        out_offset4 += os4;
                    }

                    in_offset3 += is3;
                    out_offset3 += os3;
                }

                in_offset2 += is2;
                out_offset2 += os2;
            }

            in_offset1 += is1;
            out_offset1 += os1;
        }

        in_offset0 += is0;
        out_offset0 += os0;
    }
}

#[inline(always)]
unsafe fn __reshape_kernel_6_impl<E: Copy>(
    out: *mut E,
    inp: *const E,
    (d0, d1, d2, d3, d4, d5): (usize, usize, usize, usize, usize, usize),
    (is0, is1, is2, is3, is4, is5): (isize, isize, isize, isize, isize, isize),
    (os0, os1, os2, os3, os4, os5): (isize, isize, isize, isize, isize, isize),
) {
    let mut in_offset0 = Wrapping(0isize);
    let mut out_offset0 = Wrapping(0isize);

    for _ in 0..d0 {
        let mut in_offset1 = in_offset0;
        let mut out_offset1 = out_offset0;

        for _ in 0..d1 {
            let mut in_offset2 = in_offset1;
            let mut out_offset2 = out_offset1;

            for _ in 0..d2 {
                let mut in_offset3 = in_offset2;
                let mut out_offset3 = out_offset2;

                for _ in 0..d3 {
                    let mut in_offset4 = in_offset3;
                    let mut out_offset4 = out_offset3;

                    for _ in 0..d4 {
                        let mut in_offset5 = in_offset4;
                        let mut out_offset5 = out_offset4;

                        for _ in 0..d5 {
                            unsafe {
                                *out.offset(out_offset5.0) = *inp.offset(in_offset5.0);
                            }
                            in_offset5 += is5;
                            out_offset5 += os5;
                        }

                        in_offset4 += is4;
                        out_offset4 += os4;
                    }

                    in_offset3 += is3;
                    out_offset3 += os3;
                }

                in_offset2 += is2;
                out_offset2 += os2;
            }

            in_offset1 += is1;
            out_offset1 += os1;
        }

        in_offset0 += is0;
        out_offset0 += os0;
    }
}

#[inline(always)]
unsafe fn __reshape_kernel_2_impl<E: Copy>(
    out: *mut E,
    inp: *const E,
    (d0, d1): (usize,  usize),
    (is0, is1): (isize, isize),
    (os0, os1): (isize, isize),
) {
    let mut in_offset0 = Wrapping(0isize);
    let mut out_offset0 = Wrapping(0isize);

    for _ in 0..d0 {
        let mut in_offset1 = in_offset0;
        let mut out_offset1 = out_offset0;

        for _ in 0..d1 {
            unsafe {
                *out.offset(out_offset1.0) = *inp.offset(in_offset1.0);
            }
            in_offset1 += is1;
            out_offset1 += os1;
        }

        in_offset0 += is0;
        out_offset0 += os0;
    }
}

#[inline(always)]
unsafe fn __reshape_kernel_3_impl<E: Copy>(
    out: *mut E,
    inp: *const E,
    (d0, d1, d2): (usize, usize, usize),
    (is0, is1, is2): (isize, isize, isize),
    (os0, os1, os2): (isize, isize, isize),
) {
    let mut in_offset0 = Wrapping(0isize);
    let mut out_offset0 = Wrapping(0isize);

    for _ in 0..d0 {
        let mut in_offset1 = in_offset0;
        let mut out_offset1 = out_offset0;

        for _ in 0..d1 {
            let mut in_offset2 = in_offset1;
            let mut out_offset2 = out_offset1;

            for _ in 0..d2 {
                unsafe {
                    *out.offset(out_offset2.0) = *inp.offset(in_offset2.0);
                }
                in_offset2 += is2;
                out_offset2 += os2;
            }

            in_offset1 += is1;
            out_offset1 += os1;
        }

        in_offset0 += is0;
        out_offset0 += os0;
    }
}

#[inline(always)]
unsafe fn __reshape_kernel_4_impl<E: Copy>(
    out: *mut E,
    inp: *const E,
    (d0, d1, d2, d3): (usize, usize, usize, usize),
    (is0, is1, is2, is3): (isize, isize, isize, isize),
    (os0, os1, os2, os3): (isize, isize, isize, isize),
) {
    let mut in_offset0 = Wrapping(0isize);
    let mut out_offset0 = Wrapping(0isize);

    for _ in 0..d0 {
        let mut in_offset1 = in_offset0;
        let mut out_offset1 = out_offset0;

        for _ in 0..d1 {
            let mut in_offset2 = in_offset1;
            let mut out_offset2 = out_offset1;

            for _ in 0..d2 {
                let mut in_offset3 = in_offset2;
                let mut out_offset3 = out_offset2;

                for _ in 0..d3 {
                    unsafe {
                        *out.offset(out_offset3.0) = *inp.offset(in_offset3.0);
                    }
                    in_offset3 += is3;
                    out_offset3 += os3;
                }

                in_offset2 += is2;
                out_offset2 += os2;
            }

            in_offset1 += is1;
            out_offset1 += os1;
        }

        in_offset0 += is0;
        out_offset0 += os0;
    }
}

fn __reshape_kernel_3<E: Copy>(
    out: *mut E,
    inp: *const E,
    dims: &[usize],
    in_strides: &[isize],
    out_strides: &[isize],
) {
    let &[d0, d1, d2] = dims else { panic!("") };
    let &[is0, is1, is2] = in_strides else {
        panic!("")
    };
    let &[os0, os1, os2] = out_strides else {
        panic!("")
    };
    let d = (d0, d1, d2);
    let is = (is0, is1, is2);
    let os = (os0, os1, os2);

    unsafe {
        cartesian_match!(
            { __reshape_kernel_3_impl(out, inp, d, is, os) },
            (d0, (d1, (d2, ()))),
            ((2, 3, 4, _), ((2, 3, 4, _), ((2, 3, 4, _), ())))
        );
    }
}

fn __reshape_kernel_2<E: Copy>(
    out: *mut E,
    inp: *const E,
    dims: &[usize],
    in_strides: &[isize],
    out_strides: &[isize],
) {
    let &[d0, d1] = dims else { panic!("") };
    let &[is0, is1] = in_strides else {
        panic!("")
    };
    let &[os0, os1] = out_strides else {
        panic!("")
    };
    let d = (d0, d1);
    let is = (is0, is1);
    let os = (os0, os1);

    unsafe {
        cartesian_match!(
            { __reshape_kernel_2_impl(out, inp, d, is, os) },
            (d0, (d1, ())),
            ((2, 3, 4, _), ((2, 3, 4, _), ()))
        );
    }
}

fn __reshape_kernel_4<E: Copy>(
    out: *mut E,
    inp: *const E,
    dims: &[usize],
    in_strides: &[isize],
    out_strides: &[isize],
) {
    let &[d0, d1, d2, d3] = dims else { panic!("") };
    let &[is0, is1, is2, is3] = in_strides else {
        panic!("")
    };
    let &[os0, os1, os2, os3] = out_strides else {
        panic!("")
    };
    let d = (d0, d1, d2, d3);
    let is = (is0, is1, is2, is3);
    let os = (os0, os1, os2, os3);

    unsafe {
        cartesian_match!(
            { __reshape_kernel_4_impl(out, inp, d, is, os) },
            (d0, (d1, (d2, (d3, ())))),
            (
                (2, 3, 4, _),
                ((2, 3, 4, _), ((2, 3, 4, _), ((2, 3, 4, _), ())))
            )
        );
    }
}

fn __reshape_kernel_5<E: Copy>(
    out: *mut E,
    inp: *const E,
    dims: &[usize],
    in_strides: &[isize],
    out_strides: &[isize],
) {
    let &[d0, d1, d2, d3, d4] = dims else { panic!("") };
    let &[is0, is1, is2, is3, is4] = in_strides else {
        panic!("")
    };
    let &[os0, os1, os2, os3, os4] = out_strides else {
        panic!("")
    };
    let d = (d0, d1, d2, d3, d4);
    let is = (is0, is1, is2, is3, is4);
    let os = (os0, os1, os2, os3, os4);

    unsafe {
        cartesian_match!(
            { __reshape_kernel_5_impl(out, inp, d, is, os) },
            (d0, (d1, (d2, (d3, (d4, ()))))),
            (
                (2, 3, 4, _),
                ((2, 3, 4, _), ((2, 3, 4, _), ((2, 3, 4, _), ((2, 3, 4, _), ()) )))
            )
        );
    }
}

fn __reshape_kernel_6<E: Copy>(
    out: *mut E,
    inp: *const E,
    dims: &[usize],
    in_strides: &[isize],
    out_strides: &[isize],
) {
    let &[d0, d1, d2, d3, d4, d5] = dims else { panic!("") };
    let &[is0, is1, is2, is3, is4, is5] = in_strides else {
        panic!("")
    };
    let &[os0, os1, os2, os3, os4, os5] = out_strides else {
        panic!("")
    };
    let d = (d0, d1, d2, d3, d4, d5);
    let is = (is0, is1, is2, is3, is4, is5);
    let os = (os0, os1, os2, os3, os4, os5);

    unsafe {
        cartesian_match!(
            { __reshape_kernel_6_impl(out, inp, d, is, os) },
            (d0, (d1, (d2, (d3, (d4, (d5, ())))))),
            (
                (2, 3, 4, _),
                (
                    (2, 3, 4, _),
                    ((2, 3, 4, _), ((2, 3, 4, _), ((2, 3, 4, _), ((2, 3, 4, _), ()) )))
                )
            )
        );
    }
}

unsafe fn reshape_outer_kernel<E: Copy>(
    kernel_size: usize,
    inner_kernel: impl Fn(*mut E, *const E, &[usize], &[isize], &[isize]),
    out: *mut E,
    inp: *const E,
    state: &mut [usize],
    in_strides: &[isize],
    out_strides: &[isize],
    dims: &[usize],
) {
    let ndims = dims.len();
    assert!(ndims >= kernel_size);
    if ndims == kernel_size {
        inner_kernel(out, inp, dims, in_strides, out_strides);
        return;
    }

    let mut current_axis = ndims - 1 - kernel_size;
    let mut inp_current_offset = Wrapping(0isize);
    let mut out_current_offset = 0isize;
    'outer: loop {
        inner_kernel(
            out.offset(out_current_offset),
            inp.offset(inp_current_offset.0),
            &dims[ndims - kernel_size..],
            &in_strides[ndims - kernel_size..],
            &out_strides[ndims - kernel_size..],
        );

        state[current_axis] += 1;
        out_current_offset += out_strides[current_axis];
        inp_current_offset += in_strides[current_axis];

        while state[current_axis] == dims[current_axis] {
            if current_axis == 0 {
                break 'outer;
            } else {
                state[current_axis] = 0;
                inp_current_offset -=
                    (dims[current_axis] as isize).wrapping_mul(in_strides[current_axis]);
                out_current_offset -=
                    (dims[current_axis] as isize).wrapping_mul(out_strides[current_axis]);
                state[current_axis - 1] += 1;
                inp_current_offset += in_strides[current_axis - 1];
                out_current_offset += out_strides[current_axis - 1];
            }
            current_axis -= 1;
        }
        current_axis = ndims - 1 - kernel_size;
    }
}

pub fn tensor_fused_reshape_permute_reshape_into_prepare(
    in_shape: &[usize],
    in_strides: &[isize],
    out_shape: &[usize],
    out_strides: &[isize],
    shape: &[usize],
    perm: &[usize],
) -> (Vec<isize>, Vec<isize>, Vec<usize>) {
    let N = in_shape.len();
    assert!(in_strides.len() == N);
    let M = out_shape.len();
    assert!(out_strides.len() == M);
    let K = shape.len();
    assert!(perm.len() == K);
    // Input validation
    
    // Duplicate check: Quadratic check is faster than hashset for most inputs
    for (i, &p) in perm.iter().enumerate() {
        for j in (i + 1)..perm.len() {
            if p == perm[j] {
                panic!("perm must not contain duplicate elements");
            }
        }
    }

    // Dimension equality check
    let tensor_dim = shape.iter().product::<usize>();
    if tensor_dim != in_shape.iter().product::<usize>() {
        panic!("input shape is incompatible with tensor shape");
    }
    if tensor_dim != out_shape.iter().product::<usize>() {
        panic!("output shape is incompatible with tensor shape");
    }

    // Calculate input tensor strides
    let mut tensor_in_strides = vec![0isize; K];
    let mut stride_index = 0usize;
    // let mut dim_accumulator = 1isize;

    // for (dim, suffix_prod) in shape.iter().zip(tensor_in_strides.iter_mut()) {
    //     *suffix_prod = dim_accumulator * in_strides[in_stride_index];
    //     dim_accumulator *= *dim as isize;

    //     if dim_accumulator >= in_shape[in_stride_index] as isize {
    //         in_stride_index += 1;
    //         dim_accumulator = 1;
    //     }

    //     if in_stride_index >= N {
    //         break;
    //     }
    // }
    
    let mut dim_accumulator = in_shape[stride_index] as isize;

    for (dim, suffix_prod) in shape.iter().zip(tensor_in_strides.iter_mut()) {
        dim_accumulator /= *dim as isize;
        *suffix_prod = dim_accumulator * in_strides[stride_index];

        if dim_accumulator == 1 {
            stride_index += 1;
            if stride_index >= N {
                break;
            }
            dim_accumulator = in_shape[stride_index] as isize;
        }
    }

    // Permute input tensor strides
    let mut permuted_input_tensor_strides = vec![0isize; K];
    for (i, dim_index) in perm.iter().enumerate() {
        permuted_input_tensor_strides[i] = tensor_in_strides[*dim_index];
    }

    // Permute shape
    let mut permuted_shape = vec![0usize; K];
    for (i, dim_index) in perm.iter().enumerate() {
        permuted_shape[i] = shape[*dim_index];
    }

    // Calculate output tensor strides
    let mut tensor_out_strides = vec![0isize; K];
    stride_index = 0usize;
    dim_accumulator = out_shape[stride_index] as isize;

    for (dim, suffix_prod) in permuted_shape.iter().zip(tensor_out_strides.iter_mut()) {
        dim_accumulator /= *dim as isize;
        *suffix_prod = dim_accumulator * out_strides[stride_index];

        if dim_accumulator == 1 {
            stride_index += 1;
            if stride_index >= M {
                break;
            }
            dim_accumulator = out_shape[stride_index] as isize;
        }
    }


    // Optimize strides:    
    let candidate_outputs1 = {
        let sorted_out_strides = tensor_out_strides.clone();
        let sorted_perm_in_strides = permuted_input_tensor_strides.clone();
        let sorted_perm_shape = permuted_shape.clone();

        // 2. Going from right group together consecutive groups in
        // sorted_perm_in_strides
        let mut merged_indices = VecDeque::new();
        let mut last_stride = sorted_perm_in_strides[sorted_perm_in_strides.len() - 1];
        let mut group = vec![sorted_perm_in_strides.len() - 1];
        for (i, &s) in sorted_perm_in_strides.iter().rev().skip(1).enumerate() {
            if s == last_stride * sorted_perm_shape[sorted_perm_in_strides.len() - 1 - i] as isize {
                group.push(sorted_perm_in_strides.len() - 2 - i);
            } else {
                merged_indices.push_front(group);
                group = vec![sorted_perm_in_strides.len() - 2 - i];
            }
            last_stride = s;
        }
        merged_indices.push_front(group);

        let mut opt_perm_in_strides = Vec::new();
        let mut opt_out_strides = Vec::new();
        let mut opt_dims = Vec::new();

        for merged_idx_group in merged_indices {
            let min_out_stride = merged_idx_group
                .iter()
                .map(|&i| sorted_out_strides[i])
                .min()
                .unwrap();
            let min_in_stride = merged_idx_group
                .iter()
                .map(|&i| sorted_perm_in_strides[i])
                .min()
                .unwrap();
            let prod_dim = merged_idx_group
                .iter()
                .map(|&i| sorted_perm_shape[i])
                .product::<usize>();
            opt_perm_in_strides.push(min_in_stride);
            opt_out_strides.push(min_out_stride);
            opt_dims.push(prod_dim);
        }

        (opt_perm_in_strides, opt_out_strides, opt_dims)
    };

    let candidate_outputs2 = {
        // 1. Freely sort out_strides (Applying new perm to other arrays):
        let mut out_strides_argsort = (0..K).collect::<Vec<_>>();
        out_strides_argsort.sort_by_key(|&i| -tensor_out_strides[i]);
        let sorted_out_strides = out_strides_argsort
            .iter()
            .map(|&i| tensor_out_strides[i])
            .collect::<Vec<_>>();
        let sorted_perm_in_strides = out_strides_argsort
            .iter()
            .map(|&i| permuted_input_tensor_strides[i])
            .collect::<Vec<_>>();
        let sorted_perm_shape = out_strides_argsort
            .iter()
            .map(|&i| permuted_shape[i])
            .collect::<Vec<_>>();

        // 2. Going from right group together consecutive groups in
        // sorted_perm_in_strides
        let mut merged_indices = VecDeque::new();
        let mut last_stride = sorted_perm_in_strides[sorted_perm_in_strides.len() - 1];
        let mut group = vec![sorted_perm_in_strides.len() - 1];
        for (i, &s) in sorted_perm_in_strides.iter().rev().skip(1).enumerate() {
            if s == last_stride * sorted_perm_shape[sorted_perm_in_strides.len() - 1 - i] as isize {
                group.push(sorted_perm_in_strides.len() - 2 - i);
            } else {
                merged_indices.push_front(group);
                group = vec![sorted_perm_in_strides.len() - 2 - i];
            }
            last_stride = s;
        }
        merged_indices.push_front(group);

        let mut opt_perm_in_strides = Vec::new();
        let mut opt_out_strides = Vec::new();
        let mut opt_dims = Vec::new();

        for merged_idx_group in merged_indices {
            let min_out_stride = merged_idx_group
                .iter()
                .map(|&i| sorted_out_strides[i])
                .min()
                .unwrap();
            let min_in_stride = merged_idx_group
                .iter()
                .map(|&i| sorted_perm_in_strides[i])
                .min()
                .unwrap();
            let prod_dim = merged_idx_group
                .iter()
                .map(|&i| sorted_perm_shape[i])
                .product::<usize>();
            opt_perm_in_strides.push(min_in_stride);
            opt_out_strides.push(min_out_stride);
            opt_dims.push(prod_dim);
        }

        (opt_perm_in_strides, opt_out_strides, opt_dims)
    };
    
    if candidate_outputs2.0.len() < candidate_outputs1.0.len() {
        candidate_outputs2
    } else {
        candidate_outputs1
    }
}

/// Prepare optimized parameters for `fused_reshape_permute_reshape_into_impl`
///
/// Input and output matrices are expected to be contiguous
/// (normal column padding okay) and column-major.
///
/// # Arguments
///
/// * `in_nrows` - Number of rows in the input matrix
/// * `in_ncols` - Number of columns in the input matrix
/// * `in_col_stride` - Stride between columns in the input matrix
/// * `out_nrows` - Number of rows in the output matrix
/// * `out_ncols` - Number of columns in the output matrix
/// * `out_col_stride` - Stride between columns in the output matrix
/// * `shape` - Shape of the intermediate tensor
/// * `perm` - Permutation of the intermediate tensor's shape
///
/// # Returns
///
/// * `opt_perm_in_strides` - Optimized strides for the input in tensor space
/// * `opt_out_strides` - Optimized strides for the output in tensor space
/// * `opt_dims` - Optimized dimensions of the intermediate tensor
///
/// # Panics
///
/// * If `shape` and `perm` are not the same length
/// * If `perm` contain duplicate elements
/// * If the input, output, and intermediate tensor shapes are not compatible
///
/// # See Also
///
/// * [`fused_reshape_permute_reshape_into`] - All-in-one function
/// * [`fused_reshape_permute_reshape_into_impl`] - Low-level implementation
pub fn fused_reshape_permute_reshape_into_prepare(
    in_nrows: usize,
    in_ncols: usize,
    in_col_stride: isize,
    out_nrows: usize,
    out_ncols: usize,
    out_col_stride: isize,
    shape: &[usize],
    perm: &[usize],
) -> (Vec<isize>, Vec<isize>, Vec<usize>) {
    // Input Validation
    if shape.len() != perm.len() {
        panic!("shape and perm must have the same length");
    }

    // Quadratic check is faster than hashset for most inputs
    for (i, &p) in perm.iter().enumerate() {
        for j in (i + 1)..perm.len() {
            if p == perm[j] {
                panic!("perm must not contain duplicate elements");
            }
        }
    }

    // Shape checks
    let tensor_dim = shape.iter().product::<usize>();
    if tensor_dim != in_nrows * in_ncols {
        panic!("input shape is incompatible with tensor shape");
    }
    if tensor_dim != out_nrows * out_ncols {
        panic!("output shape is incompatible with tensor shape");
    }

    // Calculate input tensor strides
    let ndims = shape.len();
    let mut in_strides = vec![0isize; ndims];
    let mut dim_accumulator = 1isize;

    for (dim, suffix_prod) in shape.iter().rev().zip(in_strides.iter_mut().rev()) {
        *suffix_prod = dim_accumulator * in_col_stride;
        dim_accumulator *= *dim as isize;

        if dim_accumulator >= in_ncols as isize {
            break;
        }
    }

    dim_accumulator = in_nrows as isize;

    for (dim, suffix_prod) in shape.iter().zip(in_strides.iter_mut()) {
        if *suffix_prod != 0 {
            break;
        }

        dim_accumulator /= *dim as isize;
        *suffix_prod = dim_accumulator;
    }

    // Calculate output tensor strides
    let mut out_strides = vec![0isize; ndims];
    dim_accumulator = 1;

    let perm_shape = perm.iter().map(|&p| shape[p]).collect::<Vec<_>>();

    for (dim, suffix_prod) in perm_shape.iter().rev().zip(out_strides.iter_mut().rev()) {
        *suffix_prod = dim_accumulator * out_col_stride;
        dim_accumulator *= *dim as isize;

        if dim_accumulator >= out_ncols as isize {
            break;
        }
    }

    dim_accumulator = out_nrows as isize;

    for (dim, suffix_prod) in perm_shape.iter().zip(out_strides.iter_mut()) {
        if *suffix_prod != 0 {
            break;
        }

        dim_accumulator /= *dim as isize;
        *suffix_prod = dim_accumulator;
    }

    // Apply permutation to input strides
    let perm_in_strides = perm
        .iter()
        .map(|&p| in_strides[p] as isize)
        .collect::<Vec<_>>();

    // (perm_in_strides, out_strides, perm_shape)

    // Optimize strides:    
    let candidate_outputs1 = {
        // let sorted_out_strides = out_strides_argsort
        //     .iter()
        //     .map(|&i| out_strides[i])
        //     .collect::<Vec<_>>();
        // let sorted_perm_in_strides = out_strides_argsort
        //     .iter()
        //     .map(|&i| perm_in_strides[i])
        //     .collect::<Vec<_>>();
        // let sorted_perm_shape = out_strides_argsort
        //     .iter()
        //     .map(|&i| perm_shape[i])
        //     .collect::<Vec<_>>();
        let sorted_out_strides = out_strides.clone();
        let sorted_perm_in_strides = perm_in_strides.clone();
        let sorted_perm_shape = perm_shape.clone();

        // 2. Going from right group together consecutive groups in
        // sorted_perm_in_strides
        let mut merged_indices = VecDeque::new();
        let mut last_stride = sorted_perm_in_strides[sorted_perm_in_strides.len() - 1];
        let mut group = vec![sorted_perm_in_strides.len() - 1];
        for (i, &s) in sorted_perm_in_strides.iter().rev().skip(1).enumerate() {
            if s == last_stride * sorted_perm_shape[sorted_perm_in_strides.len() - 1 - i] as isize {
                group.push(sorted_perm_in_strides.len() - 2 - i);
            } else {
                merged_indices.push_front(group);
                group = vec![sorted_perm_in_strides.len() - 2 - i];
            }
            last_stride = s;
        }
        merged_indices.push_front(group);

        let mut opt_perm_in_strides = Vec::new();
        let mut opt_out_strides = Vec::new();
        let mut opt_dims = Vec::new();

        for merged_idx_group in merged_indices {
            let min_out_stride = merged_idx_group
                .iter()
                .map(|&i| sorted_out_strides[i])
                .min()
                .unwrap();
            let min_in_stride = merged_idx_group
                .iter()
                .map(|&i| sorted_perm_in_strides[i])
                .min()
                .unwrap();
            let prod_dim = merged_idx_group
                .iter()
                .map(|&i| sorted_perm_shape[i])
                .product::<usize>();
            opt_perm_in_strides.push(min_in_stride);
            opt_out_strides.push(min_out_stride);
            opt_dims.push(prod_dim);
        }

        (opt_perm_in_strides, opt_out_strides, opt_dims)
    };

    let candidate_outputs2 = {
        // 1. Freely sort out_strides (Applying new perm to other arrays):
        let mut out_strides_argsort = (0..ndims).collect::<Vec<_>>();
        out_strides_argsort.sort_by_key(|&i| -out_strides[i]);
        let sorted_out_strides = out_strides_argsort
            .iter()
            .map(|&i| out_strides[i])
            .collect::<Vec<_>>();
        let sorted_perm_in_strides = out_strides_argsort
            .iter()
            .map(|&i| perm_in_strides[i])
            .collect::<Vec<_>>();
        let sorted_perm_shape = out_strides_argsort
            .iter()
            .map(|&i| perm_shape[i])
            .collect::<Vec<_>>();

        // 2. Going from right group together consecutive groups in
        // sorted_perm_in_strides
        let mut merged_indices = VecDeque::new();
        let mut last_stride = sorted_perm_in_strides[sorted_perm_in_strides.len() - 1];
        let mut group = vec![sorted_perm_in_strides.len() - 1];
        for (i, &s) in sorted_perm_in_strides.iter().rev().skip(1).enumerate() {
            if s == last_stride * sorted_perm_shape[sorted_perm_in_strides.len() - 1 - i] as isize {
                group.push(sorted_perm_in_strides.len() - 2 - i);
            } else {
                merged_indices.push_front(group);
                group = vec![sorted_perm_in_strides.len() - 2 - i];
            }
            last_stride = s;
        }
        merged_indices.push_front(group);

        let mut opt_perm_in_strides = Vec::new();
        let mut opt_out_strides = Vec::new();
        let mut opt_dims = Vec::new();

        for merged_idx_group in merged_indices {
            let min_out_stride = merged_idx_group
                .iter()
                .map(|&i| sorted_out_strides[i])
                .min()
                .unwrap();
            let min_in_stride = merged_idx_group
                .iter()
                .map(|&i| sorted_perm_in_strides[i])
                .min()
                .unwrap();
            let prod_dim = merged_idx_group
                .iter()
                .map(|&i| sorted_perm_shape[i])
                .product::<usize>();
            opt_perm_in_strides.push(min_in_stride);
            opt_out_strides.push(min_out_stride);
            opt_dims.push(prod_dim);
        }

        (opt_perm_in_strides, opt_out_strides, opt_dims)
    };
    
    if candidate_outputs2.0.len() < candidate_outputs1.0.len() {
        candidate_outputs2
    } else {
        candidate_outputs1
    }
}

/// Perform a fused reshape, permute, and reshape operation.
///
/// # Arguments
///
/// * `inp` - Input matrix
/// * `out` - Output matrix
/// * `sorted_perm_in_strides` - Optimized strides for the input in tensor space
/// * `sorted_out_strides` - Optimized strides for the output in tensor space
/// * `sorted_perm_shape` - Optimized dimensions of the intermediate tensor
///
/// # Safety
///
/// * `inp` and `out` must be valid pointers to memory with shapes compatible
///   with the input and output strides
/// * Stride and shape parameters must be computed from
///   [`fused_reshape_permute_reshape_into_prepare`].
///
/// # See Also
///
/// * [`fused_reshape_permute_reshape_into_prepare`] - Prepare optimized parameters
/// * [`fused_reshape_permute_reshape_into`] - Safe wrapper around this function
pub unsafe fn fused_reshape_permute_reshape_into_impl<E: Copy>(
    inp: *const E,
    out: *mut E,
    sorted_perm_in_strides: &[isize],
    sorted_out_strides: &[isize],
    sorted_perm_shape: &[usize],
) {
    // TODO: Investigate PodStack
    let ndims = sorted_perm_in_strides.len();
    let mut state = vec![0usize; ndims]; // TODO: Change to stack/heap vec

    if ndims >= 6 {
        reshape_outer_kernel(
            6,
            __reshape_kernel_6,
            out,
            inp,
            &mut state,
            sorted_perm_in_strides,
            sorted_out_strides,
            sorted_perm_shape,
        );
    } else if ndims == 5 {
        reshape_outer_kernel(
            5,
            __reshape_kernel_5,
            out,
            inp,
            &mut state,
            sorted_perm_in_strides,
            sorted_out_strides,
            sorted_perm_shape,
        );
    } else if ndims == 4 {
        reshape_outer_kernel(
            4,
            __reshape_kernel_4,
            out,
            inp,
            &mut state,
            sorted_perm_in_strides,
            sorted_out_strides,
            sorted_perm_shape,
        );
    } else if ndims == 3 {
        reshape_outer_kernel(
            3,
            __reshape_kernel_3,
            out,
            inp,
            &mut state,
            sorted_perm_in_strides,
            sorted_out_strides,
            sorted_perm_shape,
        );
    } else if ndims == 2 {
        reshape_outer_kernel(
            2,
            __reshape_kernel_2,
            out,
            inp,
            &mut state,
            sorted_perm_in_strides,
            sorted_out_strides,
            sorted_perm_shape,
        );
    } else {
        reshape_outer_kernel(
            0,
            __reshape_kernel_0,
            out,
            inp,
            &mut state,
            sorted_perm_in_strides,
            sorted_out_strides,
            sorted_perm_shape,
        );
    }
}

/// Perform a fused reshape, permute, and reshape operation.
///
/// In Numpy terms, this is equivalent to:
///
/// ```python
/// out = inp.reshape(shape).transpose(perm).reshape(out.shape)
/// ```
///
/// # Arguments
///
/// * `inp` - Input matrix
/// * `shape` - Shape of the intermediate tensor
/// * `perm` - Permutation of the intermediate tensor's shape
/// * `out` - Output matrix
///
/// # See Also
///
/// * [`fused_reshape_permute_reshape_into_prepare`] - Prepare optimized parameters
/// * [`fused_reshape_permute_reshape_into_impl`] - Low-level implementation
pub fn fused_reshape_permute_reshape_into<E: Copy>(
    inp: MatRef<E>,
    shape: &[usize],
    perm: &[usize],
    out: MatMut<E>,
) {
    let (is, os, dims) = fused_reshape_permute_reshape_into_prepare(
        inp.nrows(),
        inp.ncols(),
        inp.col_stride(),
        out.nrows(),
        out.ncols(),
        out.col_stride(),
        shape,
        perm,
    );
    unsafe {
        fused_reshape_permute_reshape_into_impl(inp.as_ptr(), out.as_ptr_mut(), &is, &os, &dims);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Tensor;
    use crate::matrix::{Mat, MatMut};

    #[test]
    fn test_tensor_fused_reshape_permute_reshape() {
        let (a_in, b_in, c_in) = (2, 4, 4);
        let (a_out, b_out, c_out) = (2, 8, 2);
        let intermediate_tensor_shape = [2, 2, 2, 2, 2];
        let intermediate_tensor_transposition = [0, 1, 2, 3, 4];

        let mut tensor_in = Tensor::zeros(&[a_in, b_in, c_in]);
        let mut i = 1.0;
        for a_iter in 0..a_in {
            for b_iter in 0..b_in {
                for c_iter in 0..c_in {
                    *tensor_in.at_mut([a_iter, b_iter, c_iter]) = i;
                    i += 1.0;
                }
            }
        }

        let mut tensor_out = Tensor::zeros(&[a_out, b_out, c_out]);
        
        let in_strides_isize: Vec<isize> = tensor_in.strides().iter().map(|&s| s as isize).collect();
        let out_strides_isize: Vec<isize> = tensor_out.strides().iter().map(|&s| s as isize).collect();
        let (is, os, dim) = tensor_fused_reshape_permute_reshape_into_prepare(
            tensor_in.shape(),
            &in_strides_isize,
            tensor_out.shape(),
            &out_strides_isize,
            &intermediate_tensor_shape,
            &intermediate_tensor_transposition,
        );
        
        unsafe {
            fused_reshape_permute_reshape_into_impl(tensor_in.as_ptr(), tensor_out.as_mut_ptr(), &is, &os, &dim);
        }

        let mut i = 1.0;
        for a_iter in 0..a_out {
            for b_iter in 0..b_out {
                for c_iter in 0..c_out {
                    assert_eq!(*tensor_out.at([a_iter, b_iter, c_iter]), i);
                    i += 1.0;
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "perm must not contain duplicate elements")]
    fn test_tensor_fused_reshape_permute_reshape_into_prepare_duplicate_perm() {
        let in_shape = [2, 3];
        let in_strides = [1, 2];
        let out_shape = [3, 2];
        let out_strides = [1, 3];
        let shape = [2, 3];
        let perm = [0, 0]; // Duplicate element

        tensor_fused_reshape_permute_reshape_into_prepare(
            &in_shape, &in_strides, &out_shape, &out_strides, &shape, &perm,
        );
    }

    #[test]
    #[should_panic(expected = "input shape is incompatible with tensor shape")]
    fn test_tensor_fused_reshape_permute_reshape_into_prepare_incompatible_in_shape() {
        let in_shape = [2, 2]; // Incorrect total size
        let in_strides = [1, 2];
        let out_shape = [3, 2];
        let out_strides = [1, 3];
        let shape = [2, 3];
        let perm = [1, 0];

        tensor_fused_reshape_permute_reshape_into_prepare(
            &in_shape, &in_strides, &out_shape, &out_strides, &shape, &perm,
        );
    }

    #[test]
    #[should_panic(expected = "output shape is incompatible with tensor shape")]
    fn test_tensor_fused_reshape_permute_reshape_into_prepare_incompatible_out_shape() {
        let in_shape = [2, 3];
        let in_strides = [1, 2];
        let out_shape = [2, 2]; // Incorrect total size
        let out_strides = [1, 2];
        let shape = [2, 3];
        let perm = [1, 0];

        tensor_fused_reshape_permute_reshape_into_prepare(
            &in_shape, &in_strides, &out_shape, &out_strides, &shape, &perm,
        );
    }
}
