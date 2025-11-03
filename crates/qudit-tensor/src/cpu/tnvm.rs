use std::cell::RefCell;
use std::marker::PhantomData;
use std::marker::PhantomPinned;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;

use faer::reborrow::ReborrowMut;
use qudit_core::RealScalar;
use qudit_expr::DifferentiationLevel;
use qudit_expr::GenerationShape;
use qudit_expr::Module;
use qudit_expr::ModuleBuilder;
use qudit_expr::ExpressionCache;
use qudit_expr::FUNCTION;
use rustc_hash::FxHashMap;


use crate::bytecode::Bytecode;
use crate::cpu::TNVMResult;
use super::buffer::SizedTensorBuffer;
use super::instruction::TNVMInstruction;

use qudit_core::accel::fused_reshape_permute_reshape_into_impl;
use faer::MatMut;
use faer::MatRef;
use qudit_core::memory::MemoryBuffer;
use qudit_core::memory::alloc_zeroed_memory;
use qudit_core::ComplexScalar;

pub type PinnedTNVM<C, const D: DifferentiationLevel> = Pin<Box<TNVM<C, D>>>;

/// Parameters for a TNVM evaluation; tracks constant and variable arguments correctly.
struct ParamBuffer<R: RealScalar> {
    /// Buffer storing the entire parameter vector
    buffer: MemoryBuffer<R>,

    /// Maps variable parameter i to parameter variable_map[i]
    variable_map: Vec<usize>,
}

impl<R: RealScalar> ParamBuffer<R> {
    /// Allocate a new parameter buffer with constant arguments cached
    fn new(num_params: usize, const_map: Option<&FxHashMap<usize, R>>) -> Self {
        let mut buffer = alloc_zeroed_memory(num_params);

        let variable_map = if let Some(const_map) = const_map {
            for (idx, arg) in const_map.iter() {
                buffer[idx] = arg;
            }

            let mut variable_map = Vec::with_capacity(num_params - const_map.len());
            for candidate_var_idx in 0..num_params {
                if !const_map.contains_key(&candidate_var_idx) {
                    variable_map.push(candidate_var_idx);
                }
            }
            
            variable_map
        } else {
            Vec::new()
        };

        Self {
            buffer,
            variable_map,
        }
    }


    /// Places the variable arguments into the buffer
    #[inline(always)]
    fn place_new_var_args(&mut self, var_args: &[R]) {
        debug_assert_eq!(var_args.len(), self.variable_map.len());

        for (arg, idx) in var_args.iter().zip(self.variable_map.iter()) {
            self.buffer[idx] = arg;
        }
    }

    /// Convert the buffer to a slice of arguments
    #[inline(always)]
    fn as_slice(&self) -> &[R] {
        self.buffer.as_slice()
    }
}

pub struct TNVM<C: ComplexScalar, const D: DifferentiationLevel> {
    const_instructions: Vec<TNVMInstruction<C, D>>,
    dynamic_instructions: Vec<TNVMInstruction<C, D>>,
    expressions: Arc<Mutex<ExpressionCache>>,
    memory: MemoryBuffer<C>,
    param_buffer: ParamBuffer<R>,
    out_buffer: SizedTensorBuffer<C>,
    _pin: PhantomPinned,
    // TODO: hold a mutable borrow of the expressions to prevent any uncompiling of it
}

impl<C: ComplexScalar, const D: DifferentiationLevel> TNVM<C, D> {
    pub fn new(program: &Bytecode, const_map: Option<&FxHashMap<usize, C::R>>) -> Pin<Box<Self>> {
        if program.buffers.len() == 0 {
            panic!("Cannot build TNVM with zero-length bytecode.");
        };

        let mut sized_buffers = Vec::with_capacity(program.buffers.len());
        let mut offset = 0;
        let mut out_buffer = None;
        for (i, buffer) in program.buffers.iter().enumerate() {
            let sized_buffer = if i == program.out_buffer {
                let out = SizedTensorBuffer::contiguous(offset, buffer);
                out_buffer = Some(out.clone());
                out
            } else {
                SizedTensorBuffer::new(offset, buffer)
            };
            offset += sized_buffer.memory_size(D);
            sized_buffers.push(sized_buffer);
        }
        let memory_size = offset;
        // println!("ALLOCATING {} bytes for {} units", memory_size * std::mem::size_of::<C>(), memory_size);
        // TODO: Log some of this stuff with proper logging utilities
        // TODO: Explore overlapping buffers to reduce memory usage and increase locality
        // TODO: Can further optimize FRPR after knowing strides: simple reshapes on continuous
        // buffers can be skipped with the input and output buffer having the same offset
        // but different strides.

        let expressions = program.expressions.clone();

        // Ensure that all expressions are prepared up to diff_level D.
        expressions.lock().unwrap().prepare(D);

        // Generate instructions
        let mut const_instructions = Vec::new();
        for inst in &program.const_code {
            const_instructions.push(TNVMInstruction::new(inst, &sized_buffers, expressions.clone()));
        }

        let mut dynamic_instructions = Vec::new();
        for inst in &program.dynamic_code {
            dynamic_instructions.push(TNVMInstruction::new(inst, &sized_buffers, expressions.clone()));
        }

        // Initialize parameter buffer
        let param_buffer = ParamBuffer::new(program.num_params, const_map);

        let mut out = Self {
            const_instructions,
            dynamic_instructions,
            expressions,
            memory: alloc_zeroed_memory::<C>(memory_size),
            param_buffer,
            out_buffer: out_buffer.expect("Error finding output buffer index from bytecode."),
            _pin: PhantomPinned,
        };

        // Evaluate const code
        for inst in &out.const_instructions {
            unsafe { inst.evaluate::<FUNCTION>(&[], &mut out.memory) };
        }

        Box::pin(out)
    }

    // TODO: evaluate_into
    
    /// Evaluate the TNVM with the provided arguments for all variable parameters.
    pub fn evaluate<'a, const E: DifferentiationLevel>(self: &'a mut Pin<Box<Self>>, var_args: &[C::R]) -> TNVMResult<'a, C> {
        if E > D {
            panic!("Unsafe TNVM evaluation.");
        }

        // Safety: Self is not moved
        unsafe {
            let this = self.as_mut().get_unchecked_mut();

            this.param_buffer.place_new_var_args(var_args);

            for inst in &this.dynamic_instructions {
                // Safety: Whole structure of TNVM ensures that the instruction
                // evaluates only on memory it has access to.
                inst.evaluate::<E>(this.param_buffer.as_slice(), &mut this.memory);
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

