use qudit_core::{memory::MemoryBuffer, ComplexScalar};
use qudit_expr::UtryFunc; // TODO: Change name

use super::super::buffer::SizedTensorBuffer;


pub struct SplitParamSingleWriteStruct<C: ComplexScalar> {
    write_fn: UtryFunc<C>,
    buffer: SizedTensorBuffer<C>,
}

impl<C: ComplexScalar> SplitParamSingleWriteStruct<C> {
    pub fn new(write_fn: UtryFunc<C>, buffer: SizedTensorBuffer<C>) -> Self {
        Self { write_fn, buffer }
    }

    #[inline(always)]
    pub fn evaluate(&self, params: &[C::R], memory: &mut MemoryBuffer<C>) {
        unsafe {
            let ptr = self.buffer.as_ptr_mut(memory) as *mut C::R;
            (self.write_fn)(params.as_ptr(), ptr);
        }
    }

    #[inline(always)]
    pub fn get_output_buffer(&self) -> &SizedTensorBuffer<C> {
        &self.buffer
    }
}

