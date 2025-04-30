// self.into_shape(shape).permuted_axes(perm).reshape(shape)

use std::collections::VecDeque;
use std::num::Wrapping;

use faer_core::MatMut;
use faer_core::MatRef;
use faer_entity::Entity;

macro_rules! cartesian_match {
    ({ $x: expr }, (), ()) => {
        $x
    };
    (
        { $x: expr },
        ($first_expr: expr, $rest_expr: tt),
        (($($first_pat: pat),* $(,)?), $rest_pat:tt)
    ) => {
        match $first_expr {
            $(
                $first_pat => {
                    cartesian_match!({ $x }, $rest_expr, $rest_pat);
                }
            )*
        }
    };
}

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
                inp_current_offset -= (dims[current_axis] as isize)
                    .wrapping_mul(in_strides[current_axis]);
                out_current_offset -= (dims[current_axis] as isize)
                    .wrapping_mul(out_strides[current_axis]);
                state[current_axis - 1] += 1;
                inp_current_offset += in_strides[current_axis - 1];
                out_current_offset += out_strides[current_axis - 1];
            }
            current_axis -= 1;
        }
        current_axis = ndims - 1 - kernel_size;
    }
}

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
    // TODO: Input validation
    let ndims = shape.len();
    let mut in_strides = vec![0isize; ndims];
    let mut dim_accumulator = 1isize;

    for (dim, suffix_prod) in
        shape.iter().rev().zip(in_strides.iter_mut().rev())
    {
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

    let mut out_strides = vec![0isize; ndims];
    dim_accumulator = 1;

    let perm_shape = perm.iter().map(|&p| shape[p]).collect::<Vec<_>>();

    for (dim, suffix_prod) in
        perm_shape.iter().rev().zip(out_strides.iter_mut().rev())
    {
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

    let perm_in_strides = perm
        .iter()
        .map(|&p| in_strides[p] as isize)
        .collect::<Vec<_>>();

    // reverse argsort out_strides:
    let mut out_strides_argsort = (0..ndims).collect::<Vec<_>>();
    out_strides_argsort.sort_by_key(|&i| -out_strides[i]);

    // apply out_strides_argsort to out_strides, perm_in_strides, and dim:
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

    // Going from right group together consecutive groups in
    // sorted_perm_in_strides
    let mut merged_indices = VecDeque::new();
    let mut last_stride =
        sorted_perm_in_strides[sorted_perm_in_strides.len() - 1];
    let mut group = vec![sorted_perm_in_strides.len() - 1];
    for (i, &s) in sorted_perm_in_strides.iter().rev().skip(1).enumerate() {
        if s == last_stride
            * sorted_perm_shape[sorted_perm_in_strides.len() - 1 - i] as isize
        {
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

pub unsafe fn fused_reshape_permuted_reshape_into_impl_unit<E: Copy>(
    inp: *const E,
    out: *mut E,
    sorted_perm_in_strides: &[isize],
    sorted_out_strides: &[isize],
    sorted_perm_shape: &[usize],
) {
    let ndims = sorted_perm_in_strides.len();
    let mut state = vec![0usize; ndims];

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

pub unsafe fn fused_reshape_permuted_reshape_into_impl<E: Entity>(
    inp: MatRef<E>,
    out: MatMut<E>,
    sorted_perm_in_strides: &[isize],
    sorted_out_strides: &[isize],
    sorted_perm_shape: &[usize],
) {
    E::faer_map(
        E::faer_zip(inp.as_ptr(), out.as_ptr_mut()),
        |(inp_ptr, out_ptr)| {
            fused_reshape_permuted_reshape_into_impl_unit(
                inp_ptr,
                out_ptr,
                sorted_perm_in_strides,
                sorted_out_strides,
                sorted_perm_shape,
            )
        },
    );
}

pub fn fused_reshape_permute_reshape_into<E: Entity>(
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
        fused_reshape_permuted_reshape_into_impl(inp, out, &is, &os, &dims);
    }
}
