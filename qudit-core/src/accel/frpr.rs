//! Functions to efficiently perform a fused reshape, permute, and reshape operation.
//!
//! In Numpy terms, this is equivalent to:
//! ```python
//! out = inp.reshape(shape).transpose(perm).reshape(out.shape)
//! ```

use std::collections::VecDeque;
use std::num::Wrapping;

use super::cartesian_match;
use crate::matrix::MatMut;
use crate::matrix::MatRef;

/// Copies elements of the input to output 0D tensors, even if their strides are different.
/// Effectively provides a reshaped version of the input.
/// 
/// # Arguments
/// 
/// * `out` - Mutable pointer to the output tensor
/// * `inp` - Constant pointer to the input tensor
/// * `_dims` - Dimensions of the input and output tensors
/// * `_in_strides` - Input tensor's strides for each axis
/// * `_out_strides` - Output tensor's strides for each axis
/// 
/// # Safety
///
/// * The dimension and stride values must be valid for the provided input and output tensors.
/// 
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

/// Copies elements of the input to output 5D tensors, even if their strides are different.
/// Effectively provides a reshaped version of the input.
/// 
/// # Arguments
/// 
/// * `out` - Mutable pointer to the output tensor
/// * `inp` - Constant pointer to the input tensor
/// * `(d0, d1, ... d4)` - Dimensions of the input and output tensors
/// * `(is0, is1, ... is4)` - Input tensor's strides for each axis
/// * `(os0, os1, ... os4)` - Output tensor's strides for each axis
/// 
/// # Safety
///
/// * The dimension and stride values must be valid for the provided input and output tensors.
/// 
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

/// Copies elements of the input to output 6D tensors, even if their strides are different.
/// Effectively provides a reshaped version of the input.
/// 
/// 
/// # Arguments
/// 
/// * `out` - Mutable pointer to the output tensor
/// * `inp` - Constant pointer to the input tensor
/// * `(d0, d1, ... d5)` - Dimensions of the input and output tensors
/// * `(is0, is1, ... is5)` - Input tensor's strides for each axis
/// * `(os0, os1, ... os5)` - Output tensor's strides for each axis
/// 
/// # Safety
///
/// * The dimension and stride values must be valid for the provided input and output tensors.
/// 
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

/// Copies elements of the input to output 2D tensors, even if their strides are different.
/// Effectively provides a reshaped version of the input.
/// 
/// # Arguments
/// 
/// * `out` - Mutable pointer to the output tensor
/// * `inp` - Constant pointer to the input tensor
/// * `(d0, d1)` - Dimensions of the input and output tensors
/// * `(is0, is1)` - Input tensor's row and column strides
/// * `(os0, os1)` - Output tensor's row and column strides
/// 
/// # Safety
///
/// * The dimension and stride values must be valid for the provided input and output tensors.
/// 
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

/// Copies elements of the input to output 3D tensors, even if their strides are different.
/// Effectively provides a reshaped version of the input.
/// 
/// # Arguments
/// 
/// * `out` - Mutable pointer to the output tensor
/// * `inp` - Constant pointer to the input tensor
/// * `(d0, d1, d2)` - Dimensions of the input and output tensors
/// * `(is0, is1, is2)` - Input tensor's strides for each axis
/// * `(os0, os1, os2)` - Output tensor's strides for each axis
/// 
/// # Safety
///
/// * The dimension and stride values must be valid for the provided input and output tensors.
/// 
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

/// Copies elements of the input to output 4D tensors, even if their strides are different.
/// Effectively provides a reshaped version of the input.
/// 
/// # Arguments
/// 
/// * `out` - Mutable pointer to the output tensor
/// * `inp` - Constant pointer to the input tensor
/// * `(d0, d1, ... d3)` - Dimensions of the input and output tensors
/// * `(is0, is1, ... is3)` - Input tensor's strides for each axis
/// * `(os0, os1, ... os3)` - Output tensor's strides for each axis
/// 
/// # Safety
///
/// * The dimension and stride values must be valid for the provided input and output tensors.
/// 
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
            ((2, 3, 4, 6, 8, _), ((2, 3, 4, 8, _), ((2, 3, 4, 8, _), ())))
        );
    }
}

/// Copies elements of the input to output 2D tensors, even if their strides are different.
/// Effectively provides a reshaped version of the input.
/// 
/// # Arguments
/// 
/// * `out` - Mutable pointer to the output tensor
/// * `inp` - Constant pointer to the input tensor
/// * `dims` - Reference to an array that holds the
///     dimensions of the input and output tensors
/// * `in_strides` - Input tensor's strides for each axis
/// * `out_strides` - Output tensor's strides for each axis
/// 
/// # Panics
///
/// * If `dims`, `in_strides`, or `out_strides` does not have length 2.
///
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
            ((2, 3, 4, 6, 8, _), ((2, 3, 4, 8, _), ()))
        );
    }
}

/// Copies elements of the input to output 4D tensors, even if their strides are different.
/// Effectively provides a reshaped version of the input.
/// 
/// # Arguments
/// 
/// * `out` - Mutable pointer to the output tensor
/// * `inp` - Constant pointer to the input tensor
/// * `dims` - Reference to an array that holds the
///     dimensions of the input and output tensors
/// * `in_strides` - Input tensor's strides for each axis
/// * `out_strides` - Output tensor's strides for each axis
/// 
/// # Panics
///
/// * If `dims`, `in_strides`, or `out_strides` does not have length 4.
///
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
                (2, 3, 4, 6, 8, _),
                ((2, 3, 4, 8, _), ((2, 3, 4, 8, _), ((2, 3, 4, 8, _), ())))
            )
        );
    }
}

/// Copies elements of the input to output 5D tensors, even if their strides are different.
/// Effectively provides a reshaped version of the input.
/// 
/// # Arguments
/// 
/// * `out` - Mutable pointer to the output tensor
/// * `inp` - Constant pointer to the input tensor
/// * `dims` - Reference to an array that holds the
///     dimensions of the input and output tensors
/// * `in_strides` - Input tensor's strides for each axis
/// * `out_strides` - Output tensor's strides for each axis
/// 
/// # Panics
///
/// * If `dims`, `in_strides`, or `out_strides` does not have length 5.
///
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
                (2, 3, 4, 6, 8, _),
                ((2, 3, 4, 8, _), ((2, 3, 4, 8, _), ((2, 3, 4, 8, _), ((2, 3, 4, 8, _), ()) )))
            )
        );
    }
}

/// Copies elements of the input to output 6D tensors, even if their strides are different.
/// Effectively provides a reshaped version of the input.
/// 
/// # Arguments
/// 
/// * `out` - Mutable pointer to the output tensor
/// * `inp` - Constant pointer to the input tensor
/// * `dims` - Reference to an array that holds the
///     dimensions of the input and output tensors
/// * `in_strides` - Input tensor's strides for each axis
/// * `out_strides` - Output tensor's strides for each axis
/// 
/// # Panics
///
/// * If `dims`, `in_strides`, or `out_strides` does not have length 6.
///
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

/// Copies elements of the input to output tensors, even if their strides are different.
/// Effectively provides a reshaped version of the input.
/// 
/// # Arguments
/// 
/// * `kernel_size` - The number of dimensions the inner kernel will operate on.
/// * `inner_kernel` - A function that performs the actual copying of elements.
/// * `out` - Mutable pointer to the output tensor
/// * `inp` - Constant pointer to the input tensor
/// * `state` - Reference to an array that keeps track of our position in the outer loop.
/// * `in_strides` - Input tensor's strides for each axis
/// * `out_strides` - Output tensor's strides for each axis
/// * `dims` - Dimensions of the input and output tensors
/// 
/// # Panics
/// 
/// * If `kernel_size` is larger than the number of dimensions (or axes) in `dims`.
/// 
/// # Safety
/// 
/// * `inp` and `out` must be valid pointers to memory with shapes compatible
///   with the dimensions and input/output strides.
/// 
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
    let mut inp_current_offset = Wrapping(0isize); //TODO: investigate why only `inp_current_offset` is wrapped
    let mut out_current_offset = 0isize;
    
    'outer: loop {
        // Through multiple iterations in the `outer` loop, apply the inner kernel 
        // to all (inner tensor) elements along the innermost outer axis.
        inner_kernel(
            out.offset(out_current_offset),
            inp.offset(inp_current_offset.0),
            &dims[ndims - kernel_size..],
            &in_strides[ndims - kernel_size..],
            &out_strides[ndims - kernel_size..],
        );
        // Notice `state[current_axis]` goes from 1 ~ `dims[current_axis]`, not 0 ~ `dims[current_axis] - 1`.
        state[current_axis] += 1;
        out_current_offset += out_strides[current_axis];
        inp_current_offset += in_strides[current_axis];

        while state[current_axis] == dims[current_axis] {
            if current_axis == 0 {
                break 'outer;
            } else {
                // Reset the state and offsets for the current axis. We have to do this because we will
                // iterate through this axis again once we complete one instance of the next outer loop.
                state[current_axis] = 0; 
                inp_current_offset -=
                    (dims[current_axis] as isize).wrapping_mul(in_strides[current_axis]);
                out_current_offset -=
                    (dims[current_axis] as isize).wrapping_mul(out_strides[current_axis]);
                
                // Advance our state and offsets for the next outer axis.
                state[current_axis - 1] += 1;
                inp_current_offset += in_strides[current_axis - 1];
                out_current_offset += out_strides[current_axis - 1];
            }
            // Move to the next outer axis. We do this to check if we have iterated through all 
            // (tensor) elements along this outer axis via the while loop condition.
            current_axis -= 1;
        }
        // Reset `current_axis` to the innermost outer axis.
        current_axis = ndims - 1 - kernel_size;
    }
}

fn tensor_fused_reshape_permute_reshape_into_prepare_helper(
    in_shape: &[usize],
    in_strides: &[isize],
    shape: &[usize]
) -> (Vec<isize>, Vec<usize>, Vec<usize>) {
    // The strides of the reshaped input tensor.
    let mut tensor_in_strides: Vec<isize> = Vec::new();
    // The indices correspond to the axes in `tensor_in_strides`. The value corresponds to the axes in `shape`.
    let mut virtual_axes_map: Vec<usize> = Vec::new();
    // Due to non-contiguous axes in the input tensor, we may not be able to read the input tensor according to the user's
    // desired `shape`. We thus work with a `true_shape` that ends up producing the same results as if the input was
    // actually reshaped to `shape`.
    let mut true_shape: Vec<usize> = Vec::new();
    
    // Dummy variables for the following loop
    let mut in_shape_index: usize = 0;
    let mut shape_index: usize = 0;
    let mut accumulator: usize;
    let mut true_shape_accumulator: usize;

    // Determines the strides of the reshaped input tensor.
    // Reshapes via both axes merging and splitting are supported.
    while (in_shape_index < in_shape.len()) && (shape_index < shape.len()) {

        // Shape matches already
        if in_shape[in_shape_index] == shape[shape_index] {
            tensor_in_strides.push(in_strides[in_shape_index]);
            true_shape.push(shape[shape_index]);
            virtual_axes_map.push(shape_index);
            
            in_shape_index += 1;
            shape_index += 1;
        
        // Splitting axes
        } else if in_shape[in_shape_index] > shape[shape_index] {
            accumulator = in_shape[in_shape_index];   
            while accumulator != 1 {
                accumulator /= shape[shape_index];
                
                tensor_in_strides.push(in_strides[in_shape_index] * (accumulator as isize));
                true_shape.push(shape[shape_index]);
                virtual_axes_map.push(shape_index);
                
                shape_index += 1;
            }
            in_shape_index += 1;
        
        // Merging axes (column-wise)
        // This can handle cases where we have a sequence of axes that are and aren't contiguous or column-major.
        // Merging the shape (2, 3, 4, 5, 6) where axes (0, 1, 2) are contiguous, (2, 3) are not, (3, 4) are contiguous,
        // leads to true shape (2*3*4, 5*6).
        } else {
            accumulator = shape[shape_index];
            true_shape_accumulator = 1;
            while accumulator != 1 {
                accumulator /= in_shape[in_shape_index];
                true_shape_accumulator *= in_shape[in_shape_index];
                
                // End of merge
                if accumulator == 1 || 
                // Non-contiguous axes
                in_strides[in_shape_index] != in_strides[in_shape_index + 1] * (in_shape[in_shape_index + 1] as isize) ||
                // Not column-major
                in_strides[in_shape_index] < in_strides[in_shape_index + 1] {
                    tensor_in_strides.push(in_strides[in_shape_index]);
                    true_shape.push(true_shape_accumulator);
                    virtual_axes_map.push(shape_index);

                    true_shape_accumulator = 1;
                }
                in_shape_index += 1;
            }
            shape_index += 1;
        }
    }

    assert_eq!(tensor_in_strides.len(), virtual_axes_map.len(), "Lengths aren't matching up in the helper function!");
    assert_eq!(true_shape.len(), virtual_axes_map.len(), "Lengths aren't matching up in the helper function!");

    return (tensor_in_strides, virtual_axes_map, true_shape);
}

/// Prepare optimized parameters for `fused_reshape_permute_reshape_into_impl`. Specifically,
/// calculates the optimal strides and shape for a tensor that is reshaped, permuted, and reshaped again.
/// 
/// The strides with respect to the memory layout of both the input and output tensors are returned.
/// Optimization involves reducing the number of axes in the permuted tensor by merging contiguous axes.
/// 
/// # Arguments
/// 
/// * `in_shape` - Input tensor shape
/// * `in_strides` - Input tensor strides
/// * `out_shape` - Output tensor shape
/// * `out_strides` - Output tensor strides
/// * `shape` - Shape that the input tensor must be reshaped into
///     before the permutation.
/// * `perm` - Permutation of the axes of the reshaped input tensor.
/// 
/// # Returns
/// 
/// * Optimized strides of the permuted tensor in the memory layout of the input tensor.
/// * Optimized strides of the permuted tensor in the memory layout of the output tensor.
/// * Optimized shape of the permuted tensor.
/// 
/// # Panics
/// 
/// * If `in_shape`, `in_strides`, `out_shape`, `out_strides`, `shape`, or `perm` do not have the expected lengths.
/// * If `perm` contains duplicate elements.
/// * If the number of elements in `shape` does not match the number of elements in `in_shape` or `out_shape`.
/// * If we attempt to merge non-contiguous axes during any of the two reshapes.
/// 
/// # See Also
///
/// * [`fused_reshape_permute_reshape_into`] - All-in-one function for the frpr operation.
/// * [`fused_reshape_permute_reshape_into_impl`] - Low-level implementation for the frpr operation.
/// * [`fused_reshape_permute_reshape_into_prepare`] - A specialized version of this function for 2D tensors.
/// 
#[allow(non_snake_case)]
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
    // println!("in_shape: {:?}", in_shape);
    // println!("in_strides: {:?}", in_strides);
    // println!("out_shape: {:?}", out_shape);
    // println!("out_strides: {:?}", out_strides);
    // println!("shape: {:?}", shape);
    // println!("perm: {:?}", perm);
    
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

    // `tensor_in_strides` - The strides of the reshaped input tensor.
    // `virtual_axes_map_in` - The indices correspond to the axes in `tensor_in_strides`. The value corresponds to the axes in `shape`.
    // `true_shape_in` - The shape that ends up producing the same results as if the input was actually reshaped to `shape`.
    let (tensor_in_strides, virtual_axes_map_in, true_shape_in) = 
    tensor_fused_reshape_permute_reshape_into_prepare_helper(in_shape, in_strides, shape);

    // Permute input tensor strides and shape
    let mut permuted_input_tensor_strides: Vec<isize> = Vec::new();
    let mut permuted_shape: Vec<usize> = Vec::new();
    let mut index_accumulator: usize = 0;
    for &p in perm {
        for index in 0..virtual_axes_map_in.len() {
            if virtual_axes_map_in[index] == p {
                index_accumulator = index;
                break;
            }
        }
        while index_accumulator < virtual_axes_map_in.len() &&
        virtual_axes_map_in[index_accumulator] == p
        {
            permuted_input_tensor_strides.push(tensor_in_strides[index_accumulator]);
            permuted_shape.push(true_shape_in[index_accumulator]);
            index_accumulator += 1;
        }
    }

    // Calculates the strides of the permuted tensor with respect to the output tensor's memory layout.
    // Notice that both `tensor_out_strides` and `permuted_input_tensor_strides` are necessary for a fused operation.
    let (tensor_out_strides, _, true_shape_out) = 
    tensor_fused_reshape_permute_reshape_into_prepare_helper(out_shape, out_strides, &permuted_shape);

    // There exists the possibility that certain axes fail to merge in reshaping the output tensor to the permuted tensor.
    // In this case, it is guranteed that both the input and output can agree on a single shape upon re-calculating
    // the shape and strides of the input. Notice the input only has to split some of its axes (which has no possibility 
    // of failure) to match the output's shape. 
    if true_shape_out != permuted_shape {
        (permuted_input_tensor_strides, _, permuted_shape) = 
        tensor_fused_reshape_permute_reshape_into_prepare_helper(&permuted_shape, &permuted_input_tensor_strides, out_shape);
    }
    assert_eq!(permuted_shape, true_shape_out, "Final input and output tensor shapes don't match!");
  
    // Finds the optimal strides and dimensions for the reshaped and permuted tensor.
    // The output format is `(permuted_input_tensor_strides, tensor_out_strides, permuted_shape)`
    let candidate_outputs1 = {
        // Instead of sorting, leave the strides and shape of the permuted tensor as is.
        let sorted_perm_in_strides = permuted_input_tensor_strides.clone();
        let sorted_out_strides = tensor_out_strides.clone();
        let sorted_perm_shape = permuted_shape.clone();

        // Starting from the inner-most axis, groups together axes of the permuted tensor
        // that are contiguous with respect to both the input and output tensor's memory layout. 
        // Adds all distinct groups to a deque.
        let mut merged_indices = VecDeque::new();
        let mut last_stride_in = sorted_perm_in_strides[sorted_perm_in_strides.len() - 1];
        let mut last_stride_out = sorted_out_strides[sorted_out_strides.len() - 1];
        let mut group = vec![sorted_perm_in_strides.len() - 1];
        for (i, (&is, &os)) in sorted_perm_in_strides.iter().rev().skip(1).zip(sorted_out_strides.iter().rev().skip(1)).enumerate() {
            if is == last_stride_in * sorted_perm_shape[sorted_perm_in_strides.len() - 1 - i] as isize 
            && os == last_stride_out * sorted_perm_shape[sorted_perm_in_strides.len() - 1 - i] as isize { 
                group.push(sorted_perm_in_strides.len() - 2 - i);
            } else {
                merged_indices.push_front(group);
                group = vec![sorted_perm_in_strides.len() - 2 - i];
            }
            last_stride_in = is;
            last_stride_out = os;
        }
        merged_indices.push_front(group);

        // For each group of contiguous axes, finds the dimension and stride in both the 
        // input and output tensor's memory layout, if we were to merge the axes together.
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
        // We sort the axes of the permuted tensor in descending order of their strides in the output 
        // tensor's memory layout. Compared to `candidate_outputs1`, this increases the likelihood that 
        // we find contiguous axes in the output tensor's memory layout upon merging axes below. 
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

        // Starting from the inner-most axis, groups together axes of the permuted tensor
        // that are contiguous with respect to both the input and output tensor's memory layout. 
        // Adds all distinct groups to a deque.
        let mut merged_indices = VecDeque::new();
        let mut last_stride_in = sorted_perm_in_strides[sorted_perm_in_strides.len() - 1];
        let mut last_stride_out = sorted_out_strides[sorted_out_strides.len() - 1];
        let mut group = vec![sorted_perm_in_strides.len() - 1];
        for (i, (&is, &os)) in sorted_perm_in_strides.iter().rev().skip(1).zip(sorted_out_strides.iter().rev().skip(1)).enumerate() {
            if is == last_stride_in * sorted_perm_shape[sorted_perm_in_strides.len() - 1 - i] as isize && os == last_stride_out * sorted_perm_shape[sorted_perm_in_strides.len() - 1 - i] as isize {
                group.push(sorted_perm_in_strides.len() - 2 - i);
            } else {
                merged_indices.push_front(group);
                group = vec![sorted_perm_in_strides.len() - 2 - i];
            }
            last_stride_in = is;
            last_stride_out = os;
        }
        merged_indices.push_front(group);

        // For each group of contiguous axes, finds the dimension and stride in both the 
        // input and output tensor's memory layout, if we were to merge the axes together.
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
    
    // The output with the least amount of non-contiguous axes is returned.
    // Nope: having output contiguous for writes is a net win...
    // if candidate_outputs2.0.len() < candidate_outputs1.0.len() {
        // println!("Candidate2");
        // println!("sorted_perm_in_strides: {:?}", candidate_outputs2.0);
        // println!("sorted_out_strides:     {:?}", candidate_outputs2.1);
        // println!("sorted_perm_shape:      {:?}\n", candidate_outputs2.2);
        candidate_outputs2
    // } else {
    //     // println!("Candidate1");
    //     println!("sorted_perm_in_strides: {:?}", candidate_outputs1.0);
    //     println!("sorted_out_strides:     {:?}", candidate_outputs1.1);
    //     println!("sorted_perm_shape:      {:?}\n", candidate_outputs1.2);
    //     candidate_outputs1
    // }
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
/// * [`fused_reshape_permute_reshape_into`] - All-in-one function for the frpr operation.
/// * [`fused_reshape_permute_reshape_into_impl`] - Low-level implementation for the frpr operation.
/// * [`tensor_fused_reshape_permute_reshape_into_prepare`] - A generalized version of this function 
///     for higher-dimensional input and output tensors.
/// 
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
/// 
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

    // if ndims >= 6 {
    //     reshape_outer_kernel(
    //         6,
    //         __reshape_kernel_6,
    //         out,
    //         inp,
    //         &mut state,
    //         sorted_perm_in_strides,
    //         sorted_out_strides,
    //         sorted_perm_shape,
    //     );
    if ndims == 5 {
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
/// 
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
        fused_reshape_permute_reshape_into_impl(
            inp.as_ptr(), 
            out.as_ptr_mut(), 
            &is, 
            &os,
            &dims
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Tensor;
    use crate::matrix::Mat;
    use crate::accel::frpr::{__reshape_kernel_2_impl, __reshape_kernel_0};
    use crate::c64;

    #[test]
    fn test_tensor_fused_reshape_permute_reshape_into_prepare1() {
        
        // Baseline test
        // Reshape 1 is a split, reshape 2 is a merge.
        // Everything is contiguous.
        {
            let sample1_in_shape = [2*3, 4*5, 6*7];
            let sample1_in_strides = [840, 42, 1];
            let sample1_out_shape = [2*3, 4*5, 6*7];
            let sample1_out_strides = [840, 42, 1];
            let sample1_shape = [2, 3, 4, 5, 2, 3, 7];
            let sample1_perm = [0, 1, 2, 3, 4, 5, 6];
            let result1 = tensor_fused_reshape_permute_reshape_into_prepare(
                &sample1_in_shape,
                &sample1_in_strides,
                &sample1_out_shape,
                &sample1_out_strides,
                &sample1_shape,
                &sample1_perm,
            );
            let expected1 = (
                vec![1],
                vec![1],
                vec![2*3*4*5*6*7],
            );
            assert_eq!(result1, expected1);
        }

        // Merge test (the previous implementation failed this test)
        // Reshape 1 is a merge, reshape 2 is a split. 
        // Everything is contiguous.
        {
            let sample2_in_shape = [2, 3, 4, 5, 2, 3, 7];
            let sample2_in_strides = [3*4*5*2*3*7, 4*5*2*3*7, 5*2*3*7, 2*3*7, 3*7, 7, 1];
            let sample2_out_shape = [2, 3, 4, 5, 2, 3, 7];
            let sample2_out_strides = [3*4*5*2*3*7, 4*5*2*3*7, 5*2*3*7, 2*3*7, 3*7, 7, 1];
            let sample2_shape = [2*3, 4*5, 2*3*7];
            let sample2_perm = [0, 1, 2];
            let result2 = tensor_fused_reshape_permute_reshape_into_prepare(
                &sample2_in_shape,
                &sample2_in_strides,
                &sample2_out_shape,
                &sample2_out_strides,
                &sample2_shape,
                &sample2_perm,
            );
            let expected2 = (
                vec![1],
                vec![1],
                vec![2*3*4*5*2*3*7],
            );
            assert_eq!(result2, expected2);
        }
        
        // Merging contiguous axes, although not all axes are contiguous
        // Non-contiguous test (easy)
        // Reshape 1 is a merge, reshape 2 doesn't do anything.
        {
            let sample3_in_shape = &[2, 3, 20];
            let sample3_in_strides = &[63, 21, 1]; // axes (0, 1) are contiguous. (1, 2) is non-contiguous
            let sample3_out_shape = &[6, 20];
            let sample3_out_strides = &[20, 1]; // all contiguous
            let sample3_shape = &[6, 20];
            let sample3_perm = &[0, 1];

            let result3 = tensor_fused_reshape_permute_reshape_into_prepare(
                sample3_in_shape,
                sample3_in_strides,
                sample3_out_shape,
                sample3_out_strides,
                sample3_shape,
                sample3_perm,
            );

            let expected3 = (
                vec![21, 1],
                vec![20, 1],
                vec![6, 20],
            );
            assert_eq!(result3, expected3);
        }

        // Merging contiguous axes, although not all axes are contiguous
        // Non-contiguous test (harder)
        // Reshape 1 is a merge, reshape 2 is a split.
        {
            let offset_in_1 = 2;
            let offset_in_2 = 3;
            let offset_out_1 = 1;
            let offset_out_2 = 7;

            let sample4_in_shape = [2, 3, 4, 5, 2, 3, 7];
            
            // axes (0, 1), (2, 3), (4, 5, 6) are contiguous. Other pairs are non-contiguous.
            let sample4_in_strides = [ 
                3*(4*5*(2*3*7+offset_in_1)+offset_in_2),
                4*5*(2*3*7+offset_in_1)+offset_in_2, 
                5*(2*3*7+offset_in_1), 
                2*3*7+offset_in_1, 
                3*7, 7, 1];
            
            let sample4_out_shape = [2, 3, 4, 5, 2, 3, 7];
            
            // axes (0, 1), (2, 3), (4, 5, 6) are contiguous. Other pairs are non-contiguous.
            let sample4_out_strides = [ 
                3*(4*5*(2*3*7+offset_out_1)+offset_out_2),
                4*5*(2*3*7+offset_out_1)+offset_out_2, 
                5*(2*3*7+offset_out_1), 
                2*3*7+offset_out_1, 
                3*7, 7, 1];
            
            let sample4_shape = [2*3, 4*5, 2*3*7];
            let sample4_perm = [0, 1, 2];
            
            let result4 = tensor_fused_reshape_permute_reshape_into_prepare(
                &sample4_in_shape,
                &sample4_in_strides,
                &sample4_out_shape,
                &sample4_out_strides,
                &sample4_shape,
                &sample4_perm,
            );
            let expected4 = (
                vec![4*5*(2*3*7+offset_in_1)+offset_in_2, 2*3*7+offset_in_1, 1],
                vec![4*5*(2*3*7+offset_out_1)+offset_out_2, 2*3*7+offset_out_1, 1],
                vec![2*3, 4*5, 2*3*7],
            );
            assert_eq!(result4, expected4);
        }
    
        // Reshape 1 - Merges 2 non-contiguous axes, splits 1 axis
        // Reshape 2 - Logically, does nothing (in reality, needs to split due to the non-contiguous merge)
        {
            let offset_in_1 = 1;

            let sample5_in_shape = [2, 3, 4];
            let sample5_in_strides = [3*4+ offset_in_1, 4, 1]; // has padding

            let sample5_shape = [6, 2, 2];
            let sample5_perm = [2, 0, 1];
            
            let sample5_out_shape = [2, 6, 2];
            let sample5_out_strides = [12, 2, 1]; // all contiguous
            
            let result5 = tensor_fused_reshape_permute_reshape_into_prepare(
                &sample5_in_shape,
                &sample5_in_strides,
                &sample5_out_shape,
                &sample5_out_strides,
                &sample5_shape,
                &sample5_perm,
            );
            // The expected `permute_in_strides` is [1, 3*4+offset_in_1, 4, 1*2].
            // The expected `permute_out_strides` is [12, 6, 2, 1].
            // The expected `permute_shape` is [2, 2, 3, 2].
            // During optimization, axes 2, 3 should be merged.
            let expected5 = (
                vec![1, 3*4+offset_in_1, 1*2],
                vec![12, 6, 1],
                vec![2, 2, 6],
            );
            assert_eq!(result5, expected5);
        }

    }

    #[test]
    fn test_reshape_kernel_2_impl() {
        let mut out = Mat::<c64>::zeros(4, 4);
        let temp = Mat::<c64>::ones(4, 4);
        let inp = temp.transpose();

        unsafe {
            __reshape_kernel_2_impl(
                out.as_ptr_mut(),
                inp.as_ptr(),
                (4, 4),
                (4 as isize, 1 as isize),
                (1 as isize, 4 as isize)
            );
        }

        assert_eq!(out, inp);
    }

    #[test]
    fn test_reshape_outer_kernel () {
        let dims = &[4, 4];
        let strides = &[4, 1];
        let kernel_size = 0;

        let outer_dims_count = dims.len() - kernel_size;
        let mut state = vec![0usize; outer_dims_count];

        let mut out_data = Mat::<c64>::zeros(4, 4);
        let inp_data = Mat::<c64>::ones(4, 4);

        let inner_kernel = __reshape_kernel_0::<c64>;

        unsafe {
            reshape_outer_kernel(
                kernel_size,
                inner_kernel,
                out_data.as_ptr_mut(),
                inp_data.as_ptr(),
                &mut state,
                strides,
                strides,
                dims,
            );
        }

        assert_eq!(inp_data, out_data);
    }

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
    fn test_tensor_fused_reshape_permute_reshape_2() {
        let (a_in, b_in, c_in) = (2, 8, 2);
        let input_strides = [16, 1, 8];
        let (a_out, b_out, c_out) = (2, 4, 4);
        let output_strides = [16, 1, 4];
        let intermediate_tensor_shape = [2, 2, 2, 2, 2];
        let intermediate_tensor_transposition = [0, 4, 1, 2, 3];

        let mut tensor_in = Tensor::zeros_with_strides(&[a_in, b_in, c_in], &input_strides);
        let mut i = 0.0;
        for a_iter in 0..a_in {
            for b_iter in 0..b_in {
                for c_iter in 0..c_in {
                    *tensor_in.at_mut([a_iter, b_iter, c_iter]) = i;
                    i += 1.0;
                }
            }
        }

        println!("{:?}", tensor_in);

        let mut tensor_out = Tensor::zeros_with_strides(&[a_out, b_out, c_out], &output_strides);
        
        let in_strides_isize: Vec<isize> = tensor_in.strides().iter().map(|&s| s as isize).collect();
        println!("in_strides_isize: {:?}", in_strides_isize);
        let out_strides_isize: Vec<isize> = tensor_out.strides().iter().map(|&s| s as isize).collect();
        println!("out_strides_isize: {:?}", out_strides_isize);
        let (is, os, dim) = tensor_fused_reshape_permute_reshape_into_prepare(
            tensor_in.shape(),
            &in_strides_isize,
            tensor_out.shape(),
            &out_strides_isize,
            &intermediate_tensor_shape,
            &intermediate_tensor_transposition,
        );

        println!("is: {:?}, os: {:?}, dim: {:?}", is, os, dim);
        let (cis, cos, cdim) = fused_reshape_permute_reshape_into_prepare(8, 2, 8, 4, 4, 4, &[2, 2, 2, 2], &[3, 0, 1, 2]);
        println!("cis: {:?}, cos: {:?}, cdim: {:?}", cis, cos, cdim);
        
        unsafe {
            fused_reshape_permute_reshape_into_impl(tensor_in.as_ptr(), tensor_out.as_mut_ptr(), &is, &os, &dim);
        }
        println!("tensor_in.data: {:?}", tensor_in.data);
        println!("tensor_out.data: {:?}", tensor_out.data);

        let correct = Tensor::from_slice(&[0.0,2.0,4.0,6.0,8.0,10.0,12.0,14.0,1.0,3.0,5.0,7.0,9.0,11.0,13.0,15.0,16.0,18.0,20.0,22.0,24.0,26.0,28.0,30.0,17.0,19.0,21.0,23.0,25.0,27.0,29.0,31.0], [2, 4, 4]);
        println!("{:?}", tensor_out);
        println!("{:?}", correct);

        for a_iter in 0..a_out {
            for b_iter in 0..b_out {
                for c_iter in 0..c_out {
                    assert_eq!(*tensor_out.at([a_iter, b_iter, c_iter]), *correct.at([a_iter, b_iter, c_iter]));
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
