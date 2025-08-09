use std::marker::PhantomPinned;
use std::pin::Pin;

use faer::reborrow::ReborrowMut;
use qudit_expr::DifferentiationLevel;
use qudit_expr::GenerationShape;
use qudit_expr::Module;
use qudit_core::TensorShape;
use qudit_expr::ModuleBuilder;
use qudit_expr::FUNCTION;


use crate::bytecode::Bytecode;
use crate::cpu::TNVMResult;
use super::buffer::SizedTensorBuffer;
use super::instruction::TNVMInstruction;

use qudit_core::accel::fused_reshape_permute_reshape_into_impl;
use qudit_core::matrix::MatVecMut;
use qudit_core::matrix::MatVecRef;
use qudit_core::matrix::SymSqMatMatMut;
use qudit_core::matrix::MatMut;
use qudit_core::matrix::MatRef;
use qudit_core::memory::MemoryBuffer;
use qudit_core::memory::alloc_zeroed_memory;
use qudit_core::ComplexScalar;

pub type PinnedTNVM<C, const D: DifferentiationLevel> = Pin<Box<TNVM<C, D>>>;

pub struct TNVM<C: ComplexScalar, const D: DifferentiationLevel> {
    const_instructions: Vec<TNVMInstruction<C>>,
    dynamic_instructions: Vec<TNVMInstruction<C>>,
    #[allow(dead_code)]
    module: Module<C>,
    memory: MemoryBuffer<C>,
    out_buffer: SizedTensorBuffer<C>,
    _pin: PhantomPinned,
}

impl<C: ComplexScalar, const D: DifferentiationLevel> TNVM<C, D> {
    pub fn new(program: &Bytecode) -> Pin<Box<Self>> {
        let mut sized_buffers = Vec::new();
        let mut offset = 0;
        for (i, buffer) in program.buffers.iter().enumerate() {
            let sized_buffer = if i == program.out_buffer {
                SizedTensorBuffer::contiguous(offset, buffer)
            } else {
                SizedTensorBuffer::new(offset, buffer)
            };
            offset += sized_buffer.memory_size(D);
            sized_buffers.push(sized_buffer);
        }
        let memory_size = offset;
        // println!("ALLOCATING {} bytes for {} units", memory_size * std::mem::size_of::<C>(), memory_size);
        // TODO: Explore overlapping buffers to reduce memory usage and increase locality
        // TODO: Can further optimize FRPR after knowing strides: simple reshapes on continuous
        // buffers can be skipped with the input and output buffer having the same offset
        // but different strides.

        let mut builder: ModuleBuilder<C, D> = ModuleBuilder::new("tnvm");

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
            const_instructions.push(TNVMInstruction::new(inst, &sized_buffers, &module, D));
        }

        let mut dynamic_instructions = Vec::new();
        for inst in &program.dynamic_code {
            dynamic_instructions.push(TNVMInstruction::new(inst, &sized_buffers, &module, D));
        }

        // Get out buffer
        let out_buffer = if dynamic_instructions.len() != 0 {
            dynamic_instructions.last()
                .expect("Just checked length.")
                .get_output_buffer()
                .clone()
        } else if const_instructions.len() != 0 {
            const_instructions.last()
                .expect("Just checked length.")
                .get_output_buffer()
                .clone()
        } else {
            panic!("Cannot build TNVM with zero-length bytecode.");
        };

        let mut out = Self {
            const_instructions,
            dynamic_instructions,
            module,
            memory: alloc_zeroed_memory::<C>(memory_size),
            out_buffer,
            _pin: PhantomPinned,
        };

        // Evaluate const code
        for inst in &out.const_instructions {
            unsafe { inst.evaluate::<FUNCTION>(&[], &mut out.memory) };
        }

        Box::pin(out)
    }

    // TODO: evaluate_into
    
    pub fn evaluate<'a, const E: DifferentiationLevel>(self: &'a mut Pin<Box<Self>>, args: &[C::R]) -> TNVMResult<'a, C> {
        if E > D {
            panic!("Unsafe TNVM evaluation.");
        }

        // Safety: Self is not moved
        unsafe {
            let this = self.as_mut().get_unchecked_mut();

            for inst in &this.dynamic_instructions {
                // Safety: Whole structure of TNVM ensures that the instruction
                // evaluates only on memory it has access to.
                inst.evaluate::<E>(args, &mut this.memory);
            }
       
            // Safety: Projection of const reference from mutable pin. Caller
            // cannot move data from this structure.
            TNVMResult::new(&this.memory, &this.out_buffer)
        }
    }

    pub fn num_params(&self) -> usize {
        self.out_buffer.nparams()
    }

    pub fn out_shape(&self) -> GenerationShape {
        self.out_buffer.shape()
    }
}

