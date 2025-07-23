
use qudit_core::{memory::MemoryBuffer, ComplexScalar};
use qudit_expr::UtryFunc; // TODO: Change name

use super::super::buffer::SizedTensorBuffer;


pub struct ConsecutiveParamSingleWriteStruct<C: ComplexScalar> {
    write_fn: UtryFunc<C>,
    pub idx: usize,
    buffer: SizedTensorBuffer<C>,
}

impl<C: ComplexScalar> ConsecutiveParamSingleWriteStruct<C> {
    pub fn new(write_fn: UtryFunc<C>, idx: usize, buffer: SizedTensorBuffer<C>) -> Self {
        Self { write_fn, idx, buffer }
    }

    #[inline(always)]
    pub fn evaluate(&self, params: &[C::R], memory: &mut MemoryBuffer<C>) {
        unsafe {
            let ptr = self.buffer.as_ptr_mut(memory) as *mut C::R;
            (self.write_fn)(params.as_ptr().add(self.idx), ptr);
        }
    }

    #[inline(always)]
    pub fn get_output_buffer(&self) -> &SizedTensorBuffer<C> {
        &self.buffer
    }
}

