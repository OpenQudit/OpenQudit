#![allow(dead_code)]

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
        _dim_pairs: Vec<(usize, usize)>,
        output: SizedTensorBuffer<C>,
        diff_lvl: DifferentiationLevel,
    ) -> Self {
        assert!(input.nparams() == output.nparams());

        // Trace instruction is extremely likely, if always possible, to
        // optimize out. Leaving as a todo for now to focus on more important
        // things.
        if diff_lvl == FUNCTION {
            todo!()
        } else if diff_lvl == GRADIENT {
            todo!()
        } else if diff_lvl == HESSIAN {
            todo!()
        } else {
            panic!("Invalid differentiation level.");
        };

        // Self {
        //     input,
        //     output,
        // }
    }

    #[inline(always)]
    pub unsafe fn evaluate(&self, _memory: &mut MemoryBuffer<C>) {
        todo!()
    }
}
