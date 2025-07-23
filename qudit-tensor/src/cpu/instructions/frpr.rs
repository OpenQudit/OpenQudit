use qudit_core::matrix::{MatMut, MatRef};
use qudit_core::matrix::{SymSqMatMatMut, SymSqMatMatRef};
use qudit_core::matrix::{MatVecMut, MatVecRef};
use qudit_core::accel::{fused_reshape_permute_reshape_into_prepare, tensor_fused_reshape_permute_reshape_into_prepare};
use qudit_core::accel::fused_reshape_permute_reshape_into_impl;
use qudit_core::ComplexScalar;
use qudit_core::memory::MemoryBuffer;
use qudit_expr::{DifferentiationLevel, FUNCTION, GRADIENT, HESSIAN};

use super::super::buffer::SizedTensorBuffer;

pub struct FRPRStruct<C: ComplexScalar> {
    pub len: usize,
    // TODO: Extract 64 to a library level constant (remove magic number)
    pub ins: [isize; 64],
    pub outs: [isize; 64],
    pub dims: [usize; 64],
    pub input: SizedTensorBuffer<C>,
    pub output: SizedTensorBuffer<C>,
}

impl<C: ComplexScalar> FRPRStruct<C> {
    pub fn new(
        input: SizedTensorBuffer<C>,
        shape: &Vec<usize>,
        perm: &Vec<usize>,
        output: SizedTensorBuffer<C>,
        D: DifferentiationLevel,
    ) -> Self {
        assert!(input.nparams() == output.nparams());
    
        let (ins, outs, dims) = if D == FUNCTION {
            tensor_fused_reshape_permute_reshape_into_prepare(
                &input.shape().to_vec(),
                &input.strides().iter().map(|&s| s as isize).collect::<Vec<isize>>(),
                &output.shape().to_vec(),
                &output.strides().iter().map(|&s| s as isize).collect::<Vec<isize>>(),
                shape,
                perm,
            )
        } else if D == GRADIENT {
            // Cool thing: gradient tensors are just tensors, and can have a single FRPR be
            // computed for the transpose operation, rather than a for loop
            // We are really just moving the for loop inside the frpr impl call but this
            // gives prepare a chance to optimize it.
            tensor_fused_reshape_permute_reshape_into_prepare(
                &input.shape().gradient_shape(input.nparams() + 1).to_vec(),
                &input.grad_strides().iter().map(|&s| s as isize).collect::<Vec<isize>>(),
                &output.shape().gradient_shape(input.nparams() + 1).to_vec(),
                &output.grad_strides().iter().map(|&s| s as isize).collect::<Vec<isize>>(),
                &([1 + input.nparams()].iter().chain(shape.iter()).copied().collect::<Vec<usize>>()),
                &([0].into_iter().chain(perm.iter().map(|&p| p + 1)).collect::<Vec<usize>>()),
            )
        } else if D == HESSIAN {
            tensor_fused_reshape_permute_reshape_into_prepare(
                &(input.shape().gradient_shape(input.nparams() + 1) + input.shape().hessian_shape(input.nparams())).to_vec(),
                &input.grad_strides().iter().map(|&s| s as isize).collect::<Vec<isize>>(),
                &(output.shape().gradient_shape(output.nparams() + 1) + output.shape().hessian_shape(output.nparams())).to_vec(),
                &output.grad_strides().iter().map(|&s| s as isize).collect::<Vec<isize>>(),
                &([1 + input.nparams() + (input.nparams() * (input.nparams()+1) / 2)].iter().chain(shape.iter()).copied().collect::<Vec<usize>>()),
                &([0].into_iter().chain(perm.iter().map(|&p| p + 1)).collect::<Vec<usize>>()),
            )
        } else {
            panic!("Invalid differentiation level.");
        };

        let len = ins.len();
        if len > 64 {
            // TODO: Better error message
            panic!("Too many indices in FRPR operaiton!");
        }
        let mut array_ins = [0; 64];
        for (i, v) in ins.iter().enumerate() {
            array_ins[i] = *v;
        }
        let mut array_outs = [0; 64];
        for (i, v) in outs.iter().enumerate() {
            array_outs[i] = *v;
        }
        let mut array_dims = [0; 64];
        for (i, v) in dims.iter().enumerate() {
            array_dims[i] = *v;
        }
        Self {
            len,
            ins: array_ins,
            outs: array_outs,
            dims: array_dims,
            input,
            output,
        }
    }

    #[inline(always)]
    pub unsafe fn evaluate(&self, memory: &mut MemoryBuffer<C>) {
        fused_reshape_permute_reshape_into_impl(
            self.input.as_ptr(memory),
            self.output.as_ptr_mut(memory),
            &self.ins[..self.len],
            &self.outs[..self.len],
            &self.dims[..self.len],
        );
    }

    #[inline(always)]
    pub fn get_output_buffer(&self) -> &SizedTensorBuffer<C> {
        &self.output
    }
}
