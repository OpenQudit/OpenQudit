use std::pin::Pin;

use faer::reborrow::ReborrowMut;
use qudit_expr::DifferentiationLevel;
use qudit_expr::GenerationShape;
use qudit_expr::Module;
use qudit_core::TensorShape;
use qudit_expr::ModuleBuilder;


use crate::bytecode::Bytecode;
use crate::cpu::TNVMResult;
use super::buffer::SizedTensorBuffer;
use super::instruction::SpecializedInstruction;

use qudit_core::accel::fused_reshape_permute_reshape_into_impl;
use qudit_core::matrix::MatVecMut;
use qudit_core::matrix::MatVecRef;
use qudit_core::matrix::SymSqMatMatMut;
use qudit_core::matrix::MatMut;
use qudit_core::matrix::MatRef;
use qudit_core::memory::MemoryBuffer;
use qudit_core::memory::alloc_zeroed_memory;
use qudit_core::ComplexScalar;

pub struct TNVM<C: ComplexScalar, const D: DifferentiationLevel> {
    const_instructions: Vec<SpecializedInstruction<C, D>>,
    dynamic_instructions: Vec<SpecializedInstruction<C, D>>,
    #[allow(dead_code)]
    module: Module<C>,
    memory: MemoryBuffer<C>,
}


impl<C: ComplexScalar, const D: DifferentiationLevel> TNVM<C, D> {
    pub fn new(program: &Bytecode) -> Pin<Self> {
        let mut sized_buffers = Vec::new();
        let mut offset = 0;
        for buffer in &program.buffers {
            let sized_buffer = buffer.specialize::<C>(offset, diff_lvl);
            offset += sized_buffer.size(diff_lvl);
            sized_buffers.push(sized_buffer);
        }
        let memory_size = offset;
        // TODO: Explore overlapping buffers to reduce memory usage and increase locality

        let mut builder = ModuleBuilder::new::<D>("tnvm");

        for (expr, params, name) in &program.expressions {
            let mut expr_clone = expr.clone();
            expr_clone.name = name.clone();
            match params {
                None => builder = builder.add_tensor_expression(expr_clone),
                Some(params) =>
                    builder = builder.add_tensor_expression_with_param_indices(expr_clone, params.clone())
            };
        }
        let module = builder.build();

        let mut const_instructions = Vec::new();
        for inst in &program.const_code {
            const_instructions.push(inst.specialize(&sized_buffers, &module, diff_lvl));
        }

        let mut dynamic_instructions = Vec::new();
        for inst in &program.dynamic_code {
            dynamic_instructions.push(inst.specialize(&sized_buffers, &module, diff_lvl));
        }

        let out = Self {
            const_instructions,
            dynamic_instructions,
            module,
            memory: alloc_zeroed_memory::<C>(memory_size),
        };

        // Evaluate const code
        for inst in &out.const_instructions {
            inst.evaluate(&[], &mut out.memory);
        }

        Pin::new(out)
    }

    // TODO: maybe during monomorphization, I can have this return two results for grad, etc to
    // make it easier to work with?
    pub fn evaluate(&mut self, args: &[C::R]) -> TNVMResult<C> {
        for inst in &self.dynamic_instructions {
            inst.evaluate(args, &mut self.memory);
        }
    
        self.get_output_buffer()
    }

    pub fn get_output_buffer(&self) -> TNVMResult<C> { 
        let last_instruction = if self.dynamic_instructions.len() == 0 {
            self.const_instructions.last()
        } else {
            self.dynamic_instructions.last()
        };

        match last_instruction {
            None => panic!("Seriously..."),
            Some(inst) => {
                TNVMResult {
                    memory: &self.memory,
                    buffer: inst.get_output_buffer(),
                }
            }
        }
    }
}

// TODO: TEST: No params in entire circuit, constant everything
