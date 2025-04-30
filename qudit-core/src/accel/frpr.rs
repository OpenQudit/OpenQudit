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

    // Optimize strides:
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
}

/// Perform a fused reshape, permute, and reshape unit-matrix operation.
///
/// See [`fused_reshape_permute_reshape_into_impl`] for details.
unsafe fn fused_reshape_permute_reshape_into_impl_unit<E: Copy>(
    inp: *const E,
    out: *mut E,
    sorted_perm_in_strides: &[isize],
    sorted_out_strides: &[isize],
    sorted_perm_shape: &[usize],
) {
    // TODO: Investigate PodStack
    let ndims = sorted_perm_in_strides.len();
    let mut state = vec![0usize; ndims]; // TODO: Change to stack/heap vec

    if ndims >= 4 {
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
    inp: MatRef<E>,
    out: MatMut<E>,
    sorted_perm_in_strides: &[isize],
    sorted_out_strides: &[isize],
    sorted_perm_shape: &[usize],
) {
    fused_reshape_permute_reshape_into_impl_unit(
        inp.as_ptr(),
        out.as_ptr_mut(),
        sorted_perm_in_strides,
        sorted_out_strides,
        sorted_perm_shape,
    );
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
        fused_reshape_permute_reshape_into_impl(inp, out, &is, &os, &dims);
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // use crate::matrix::Mat;
    // use crate::memory::*;
    // use crate::c64;

    #[test]
    fn test_frpr_into() {
        // let col_stride_in = calc_col_stride::<c64>(4, 4);
        // let mut memory_in = alloc_zeroed_memory::<c64>(4 * col_stride_in);
        // let inp = unsafe {
        //     let mut matmut: MatMut<c64> = faer::mat::from_raw_parts_mut(
        //         c64::faer_map(
        //             c64::faer_as_mut(&mut memory_in),
        //             #[inline(always)]
        //             |mem| {
        //                 mem.as_ptr() as *mut c64
        //             },
        //         ),
        //         4,
        //         4,
        //         1,
        //         col_stride_in as isize,
        //     );

        //     for i in 0..4 {
        //         for j in 0..4 {
        //             matmut.write(i, j, c64::new((i * 4 + j) as f64, (i * 4 + j) as f64));
        //         }
        //     }

        //     matmut
        // };

        // let col_stride_out = calc_col_stride::<c64>(2, 8);
        // let mut memory_out = alloc_zeroed_memory::<c64>(8 * col_stride_out);
        // let out: MatMut<c64> = unsafe {
        //     faer::mat::from_raw_parts_mut(
        //         c64::faer_map(
        //             c64::faer_as_mut(&mut memory_out),
        //             #[inline(always)]
        //             |mem| {
        //                 mem.as_ptr() as *mut c64
        //             },
        //         ),
        //         2,
        //         8,
        //         1,
        //         col_stride_out as isize,
        //     )
        // };

        // let shape = vec![2, 2, 2, 2];
        // let perm = vec![1, 0, 2, 3];

        // fused_reshape_permute_reshape_into(inp.as_ref(), &shape, &perm, out);
        // let out: MatMut<c64> = unsafe {
        //     faer::mat::from_raw_parts_mut(
        //         c64::faer_map(
        //             c64::faer_as_mut(&mut memory_out),
        //             #[inline(always)]
        //             |mem| {
        //                 mem.as_ptr() as *mut c64
        //             },
        //         ),
        //         2,
        //         8,
        //         1,
        //         col_stride_out as isize,
        //     )
        // };
        // println!("{:?}", out);
    }
}
