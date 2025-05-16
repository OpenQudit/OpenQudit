use std::collections::HashMap;

// use aligned_vec::CACHELINE_ALIGN;
// use faer_entity::Entity;

// use crate::sim::qvm::QVMType;

use qudit_core::{ComplexScalar, ParamIndices};
use qudit_expr::{DifferentiationLevel, Module, ModuleBuilder, TensorExpression, UnitaryExpression};

use super::{
    GeneralizedInstruction, SpecializedInstruction, TensorBuffer
    // SpecializedInstruction,
};

#[derive(Clone)]
pub struct Bytecode {
    pub expressions: Vec<(TensorExpression, Option<ParamIndices>, String)>,
    pub static_code: Vec<GeneralizedInstruction>,
    pub dynamic_code: Vec<GeneralizedInstruction>,
    pub buffers: Vec<TensorBuffer>,
    pub merged_buffers: HashMap<usize, usize>,
}

impl Bytecode {
    pub fn print_buffers(&self) {
        println!("Buffers:");
        for (i, buffer) in self.buffers.iter().enumerate() {
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
        for buffer in &self.buffers {
            let sized_buffer = buffer.specialize::<C>(offset);
            offset += sized_buffer.size(diff_lvl);
            sized_buffers.push(sized_buffer);
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

        for (expr, params, name) in &self.expressions {
            let mut expr_clone = expr.clone();
            expr_clone.name = name.clone();
            match params {
                None => builder = builder.add_tensor_expression(expr_clone),
                Some(params) =>
                    builder = builder.add_tensor_expression_with_param_indices(expr_clone, params.clone())
            };
        }
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
