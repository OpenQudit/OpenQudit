use faer::{MatMut, MatRef};
use qudit_core::accel::{fused_reshape_permute_reshape_into_prepare, tensor_fused_reshape_permute_reshape_into_prepare};
use qudit_core::accel::fused_reshape_permute_reshape_into_impl;
use qudit_core::ComplexScalar;
use qudit_core::memory::MemoryBuffer;
use qudit_expr::{DifferentiationLevel, FUNCTION, GRADIENT, HESSIAN};

use super::super::buffer::SizedTensorBuffer;

pub struct FRPRStruct<C: ComplexScalar, const D: DifferentiationLevel> {
    pub ins: [Vec<isize>; D],
    pub outs: [Vec<isize>; D],
    pub dims: [Vec<usize>; D],
    pub input: SizedTensorBuffer<C>,
    pub output: SizedTensorBuffer<C>,
}

impl<C: ComplexScalar, const D: DifferentiationLevel> FRPRStruct<C, D> {
    pub fn new(
        input: SizedTensorBuffer<C>,
        shape: &Vec<usize>,
        perm: &Vec<usize>,
        output: SizedTensorBuffer<C>,
    ) -> Self {
        assert!(input.nparams() == output.nparams());

        let mut all_ins = vec![];
        let mut all_outs = vec![];
        let mut all_dims = vec![];

        if D >= FUNCTION {
            let (ins, outs, dims) = tensor_fused_reshape_permute_reshape_into_prepare(
                &input.shape().to_vec(),
                &input.strides().iter().map(|&s| s as isize).collect::<Vec<isize>>(),
                &output.shape().to_vec(),
                &output.strides().iter().map(|&s| s as isize).collect::<Vec<isize>>(),
                shape,
                perm,
            );

            all_ins.push(ins);
            all_outs.push(outs);
            all_dims.push(dims);
        }

        if D >= GRADIENT {
            // Cool thing: gradient tensors are just tensors, and can have a single FRPR be
            // computed for the transpose operation, rather than a for loop
            // We are really just moving the for loop inside the frpr impl call but this
            // gives prepare a chance to optimize it.
            let (ins, outs, dims) = tensor_fused_reshape_permute_reshape_into_prepare(
                &input.shape().gradient_shape(input.nparams() + 1).to_vec(),
                &input.grad_strides().iter().map(|&s| s as isize).collect::<Vec<isize>>(),
                &output.shape().gradient_shape(input.nparams() + 1).to_vec(),
                &output.grad_strides().iter().map(|&s| s as isize).collect::<Vec<isize>>(),
                &([1 + input.nparams()].iter().chain(shape.iter()).copied().collect::<Vec<usize>>()),
                &([0].into_iter().chain(perm.iter().map(|&p| p + 1)).collect::<Vec<usize>>()),
            );

            all_ins.push(ins);
            all_outs.push(outs);
            all_dims.push(dims);
        }

        if D >= HESSIAN {
            let (ins, outs, dims) = tensor_fused_reshape_permute_reshape_into_prepare(
                &(input.shape().gradient_shape(input.nparams() + 1) + input.shape().hessian_shape(input.nparams())).to_vec(),
                &input.grad_strides().iter().map(|&s| s as isize).collect::<Vec<isize>>(),
                &(output.shape().gradient_shape(output.nparams() + 1) + output.shape().hessian_shape(output.nparams())).to_vec(),
                &output.grad_strides().iter().map(|&s| s as isize).collect::<Vec<isize>>(),
                &([1 + input.nparams() + (input.nparams() * (input.nparams()+1) / 2)].iter().chain(shape.iter()).copied().collect::<Vec<usize>>()),
                &([0].into_iter().chain(perm.iter().map(|&p| p + 1)).collect::<Vec<usize>>()),
            );

            all_ins.push(ins);
            all_outs.push(outs);
            all_dims.push(dims);
        }

        let ins = all_ins.try_into().expect("Failed to calculate input strides.");
        let outs = all_outs.try_into().expect("Failed to calculate output strides.");
        let dims = all_dims.try_into().expect("Failed to calculate tensor dimensions.");

        Self {
            ins,
            outs,
            dims,
            input,
            output,
        }
    }

    #[inline(always)]
    pub unsafe fn evaluate<const E: DifferentiationLevel>(&self, memory: &mut MemoryBuffer<C>) {
        fused_reshape_permute_reshape_into_impl(
            self.input.as_ptr(memory),
            self.output.as_ptr_mut(memory),
            &self.ins[E - 1],
            &self.outs[E - 1],
            &self.dims[E - 1],
        );
    }

    #[inline(always)]
    pub fn get_output_buffer(&self) -> &SizedTensorBuffer<C> {
        &self.output
    }
}
