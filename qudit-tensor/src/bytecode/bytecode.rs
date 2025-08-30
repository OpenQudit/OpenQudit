use std::{cell::RefCell, collections::HashMap, rc::Rc};

// use aligned_vec::CACHELINE_ALIGN;
// use faer_entity::Entity;

// use crate::sim::qvm::QVMType;

use qudit_core::{ComplexScalar, ParamIndices};
use qudit_expr::{DifferentiationLevel, ExpressionCache, Module, ModuleBuilder, TensorExpression, UnitaryExpression};

use super::{BytecodeInstruction, TensorBuffer};

#[derive(Clone)]
pub struct Bytecode {
    // pub expressions: Vec<(TensorExpression, Option<ParamIndices>, String)>,
    pub expressions: Rc<RefCell<ExpressionCache>>,
    pub const_code: Vec<BytecodeInstruction>,
    pub dynamic_code: Vec<BytecodeInstruction>,
    pub buffers: Vec<TensorBuffer>,
    pub out_buffer: usize,
}

impl Bytecode {
    pub fn print_buffers(&self) {
        println!("Buffers:");
        for (i, buffer) in self.buffers.iter().enumerate() {
            println!("  {}: {:?}", i, buffer);
        }
    }
}

impl std::fmt::Debug for Bytecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.const_code.is_empty() {
            write!(f, ".const\n")?;
            for inst in &self.const_code {
                write!(f, "    {:?}\n", inst)?;
            }
        }
        if !self.dynamic_code.is_empty() {
            write!(f, "\n.dynamic\n")?;
            for inst in &self.dynamic_code {
                write!(f, "    {:?}\n", inst)?;
            }
        }
        Ok(())
    }
}
