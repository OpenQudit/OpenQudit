use faer::{MatMut, MatRef};
use qudit_core::accel::{fused_reshape_permute_reshape_into_prepare, tensor_fused_reshape_permute_reshape_into_prepare};
use qudit_core::accel::fused_reshape_permute_reshape_into_impl;
use qudit_core::ComplexScalar;
use qudit_core::memory::MemoryBuffer;
use qudit_expr::{DifferentiationLevel, FUNCTION, GRADIENT, HESSIAN};

use super::super::buffer::SizedTensorBuffer;

pub struct TraceStruct<C: ComplexScalar> {
    pub input: SizedTensorBuffer<C>,
    pub output: SizedTensorBuffer<C>,
}

impl<C: ComplexScalar> TraceStruct<C> {
    pub fn new(
        input: SizedTensorBuffer<C>,
        dim_pairs: Vec<(usize, usize)>,
        output: SizedTensorBuffer<C>,
        D: DifferentiationLevel,
    ) -> Self {
        assert!(input.nparams() == output.nparams());
    
        // Trace instruction is extremely likely, if always possible, to
        // optimize out. Leaving as a todo for now to focus on more important
        // things.
        if D == FUNCTION {
            todo!()
        } else if D == GRADIENT {
            todo!()
        } else if D == HESSIAN {
            todo!()
        } else {
            panic!("Invalid differentiation level.");
        };

        Self {
            input,
            output,
        }
    }

    #[inline(always)]
    pub unsafe fn evaluate(&self, memory: &mut MemoryBuffer<C>) {
        todo!()
    }

    #[inline(always)]
    pub fn get_output_buffer(&self) -> &SizedTensorBuffer<C> {
        &self.output
    }
}
