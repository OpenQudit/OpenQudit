use qudit_core::{memory::MemoryBuffer, ComplexScalar};
use qudit_expr::WriteFunc;

use super::super::buffer::SizedTensorBuffer;


pub struct WriteStruct<C: ComplexScalar> {
    write_fn: WriteFunc<C::R>,
    param_map: Vec<u64>,
    output_map: Vec<u64>,
    const_map: Vec<bool>,
    buffer: SizedTensorBuffer<C>,
}

impl<C: ComplexScalar> WriteStruct<C> {
    pub fn new(
        write_fn: WriteFunc<C::R>,
        param_map: Vec<u64>,
        output_map: Vec<u64>,
        const_map: Vec<bool>,
        buffer: SizedTensorBuffer<C>,
    ) -> Self {
        Self {
            write_fn,
            param_map,
            output_map,
            const_map,
            buffer,
        }
    }

    #[inline(always)]
    pub fn evaluate(&self, params: &[C::R], memory: &mut MemoryBuffer<C>) {
        unsafe {
            let ptr = self.buffer.as_ptr_mut(memory) as *mut C::R;
            (self.write_fn)(params.as_ptr(), ptr, self.param_map.as_ptr(), self.output_map.as_ptr(), (self.buffer.unit_memory_size()*2) as u64, self.const_map.as_ptr());
        }
    }

    #[inline(always)]
    pub fn get_output_buffer(&self) -> &SizedTensorBuffer<C> {
        &self.buffer
    }
}

