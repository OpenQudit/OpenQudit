use std::{collections::HashMap, mem::size_of};

// use aligned_vec::CACHELINE_ALIGN;
use faer_entity::Entity;

use crate::sim::qvm::QVMType;

use super::{
    GeneralizedInstruction, MatrixBuffer, SizedMatrixBuffer,
    SpecializedInstruction,
};

#[derive(Clone)]
pub struct Bytecode {
    pub static_code: Vec<GeneralizedInstruction>,
    pub dynamic_code: Vec<GeneralizedInstruction>,
    pub matrix_buffers: Vec<MatrixBuffer>,
    pub merged_buffers: HashMap<usize, usize>,
}

impl Bytecode {
    pub fn print_buffers(&self) {
        println!("Matrix buffers:");
        for (i, buffer) in self.matrix_buffers.iter().enumerate() {
            println!("  {}: {:?}", i, buffer);
        }
    }

    pub fn get_col_stride<C: Entity>(nrows: usize, _ncols: usize) -> isize {
        // TODO: Memory and stuff like umm... cache lines... experiments
        let nrows = nrows as isize;
        // let remainder = ((nrows * size_of::<C::Unit>() as isize) % CACHELINE_ALIGN as isize) / size_of::<C::Unit>() as isize;
        // let remainder = nrows % (CACHELINE_ALIGN as isize / size_of::<C::Unit>() as isize);
        let remainder = nrows % (64 / size_of::<C::Unit>() as isize);
        // println!("nrows: {}, ncols: {}, col_stride: {}", nrows, _ncols, nrows + remainder);
        nrows + remainder
    }

    pub fn specialize<C: Entity>(
        &self,
        ty: QVMType,
    ) -> (
        Vec<SpecializedInstruction>,
        Vec<SpecializedInstruction>,
        usize,
    ) {
        let mut sized_buffers = Vec::new();
        let mut offset = 0;
        for buffer in &self.matrix_buffers {
            let col_stride =
                Self::get_col_stride::<C>(buffer.nrows, buffer.ncols);
            sized_buffers.push(SizedMatrixBuffer {
                offset,
                nrows: buffer.nrows,
                ncols: buffer.ncols,
                col_stride,
                num_params: buffer.num_params,
            });
            let mat_size = col_stride as usize * buffer.ncols;
            offset += mat_size;
            if ty.gradient_capable() {
                offset += mat_size * buffer.num_params;
            }
            if ty.hessian_capable() {
                offset += mat_size
                    * (buffer.num_params * (buffer.num_params + 1))
                    / 2;
            }
        }
        let mut memory_size = offset;
        // println!("Memory size: {}", memory_size);

        // TODO: can be done a lot more efficient
        for (mergee_buffer, merger_buffer) in &self.merged_buffers {
            let mut mergee_size = sized_buffers[*mergee_buffer].ncols
                * sized_buffers[*mergee_buffer].col_stride as usize;
            if ty.gradient_capable() {
                mergee_size +=
                    mergee_size * sized_buffers[*mergee_buffer].num_params;
            }
            if ty.hessian_capable() {
                mergee_size += mergee_size
                    * (sized_buffers[*mergee_buffer].num_params
                        * (sized_buffers[*mergee_buffer].num_params + 1))
                    / 2;
            }

            let offset = sized_buffers[*mergee_buffer].offset;

            for buffer in &mut sized_buffers {
                if buffer.offset >= offset {
                    buffer.offset -= mergee_size;
                }
            }
            sized_buffers[*mergee_buffer].offset =
                sized_buffers[*merger_buffer].offset;
            memory_size -= mergee_size;
        }
        // println!("Post Merged Memory size: {}", memory_size);

        let mut static_out = Vec::new();
        for inst in &self.static_code {
            static_out.push(inst.specialize(&sized_buffers));
        }

        let mut dynamic_out = Vec::new();
        for inst in &self.dynamic_code {
            dynamic_out.push(inst.specialize(&sized_buffers));
        }
        (static_out, dynamic_out, memory_size)
    }
}

impl std::fmt::Debug for Bytecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, ".static\n")?;
        for inst in &self.static_code {
            write!(f, "    {:?}\n", inst)?;
        }
        write!(f, "\n.dynamic\n")?;
        for inst in &self.dynamic_code {
            write!(f, "    {:?}\n", inst)?;
        }
        Ok(())
    }
}
