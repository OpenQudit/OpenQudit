use std::collections::HashMap;

// use aligned_vec::CACHELINE_ALIGN;
// use faer_entity::Entity;

// use crate::sim::qvm::QVMType;

use qudit_core::ComplexScalar;
use qudit_expr::{DifferentiationLevel, Module, ModuleBuilder, UnitaryExpression};

use super::{
    GeneralizedInstruction, MatrixBuffer, SizedMatrixBuffer, SpecializedInstruction,
    // SpecializedInstruction,
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

    pub fn specialize<C: ComplexScalar>(
        &self,
        diff_lvl: DifferentiationLevel,
    ) -> (
        Vec<SpecializedInstruction<C>>,
        Vec<SpecializedInstruction<C>>,
        Module<C>,
        usize,
    ) {
        let mut sized_buffers = Vec::new();
        let mut offset = 0;
        for buffer in &self.matrix_buffers {
            let col_stride =
                qudit_core::memory::calc_col_stride::<C>(buffer.nrows, buffer.ncols);
            let mat_stride = qudit_core::memory::calc_mat_stride::<C>(buffer.nrows, buffer.ncols, col_stride);
            sized_buffers.push(SizedMatrixBuffer {
                offset,
                nrows: buffer.nrows,
                ncols: buffer.ncols,
                col_stride: col_stride as isize,
                mat_stride: mat_stride as isize,
                num_params: buffer.num_params,
            });
            offset += mat_stride;
            if diff_lvl.gradient_capable() {
                offset += mat_stride * buffer.num_params;
            }
            if diff_lvl.hessian_capable() {
                offset += mat_stride
                    * (buffer.num_params * (buffer.num_params + 1))
                    / 2;
            }
        }
        let memory_size = offset;
        // println!("Memory size: {}", memory_size);

        // TODO: can be done a lot more efficient
        // for (mergee_buffer, merger_buffer) in &self.merged_buffers {
        //     let mut mergee_size = sized_buffers[*mergee_buffer].ncols
        //         * sized_buffers[*mergee_buffer].col_stride as usize;
        //     if ty.gradient_capable() {
        //         mergee_size +=
        //             mergee_size * sized_buffers[*mergee_buffer].num_params;
        //     }
        //     if ty.hessian_capable() {
        //         mergee_size += mergee_size
        //             * (sized_buffers[*mergee_buffer].num_params
        //                 * (sized_buffers[*mergee_buffer].num_params + 1))
        //             / 2;
        //     }

        //     let offset = sized_buffers[*mergee_buffer].offset;

        //     for buffer in &mut sized_buffers {
        //         if buffer.offset >= offset {
        //             buffer.offset -= mergee_size;
        //         }
        //     }
        //     sized_buffers[*mergee_buffer].offset =
        //         sized_buffers[*merger_buffer].offset;
        //     memory_size -= mergee_size;
        // }
        // println!("Post Merged Memory size: {}", memory_size);

        let mut builder = ModuleBuilder::new("qvm", diff_lvl);
        for inst in &self.static_code {
            builder = inst.build_expressions(builder);
        }
        for inst in &self.dynamic_code {
            builder = inst.build_expressions(builder);
        }
        // for expr in &self.expression_set {
        //     builder = builder.add_expression(expr.clone());
        // }
        let module = builder.build();

        let mut static_out = Vec::new();
        for inst in &self.static_code {
            static_out.push(inst.specialize(&sized_buffers, &module, diff_lvl));
        }

        let mut dynamic_out = Vec::new();
        for inst in &self.dynamic_code {
            dynamic_out.push(inst.specialize(&sized_buffers, &module, diff_lvl));
        }
        (static_out, dynamic_out, module, memory_size)
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
