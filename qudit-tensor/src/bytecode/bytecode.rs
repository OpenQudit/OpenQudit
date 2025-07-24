use std::collections::HashMap;

// use aligned_vec::CACHELINE_ALIGN;
// use faer_entity::Entity;

// use crate::sim::qvm::QVMType;

use qudit_core::{ComplexScalar, ParamIndices};
use qudit_expr::{DifferentiationLevel, Module, ModuleBuilder, TensorExpression, UnitaryExpression};

use super::{BytecodeInstruction, TensorBuffer};

#[derive(Clone)]
pub struct Bytecode {
    pub expressions: Vec<(TensorExpression, Option<ParamIndices>, String)>,
    pub const_code: Vec<BytecodeInstruction>,
    pub dynamic_code: Vec<BytecodeInstruction>,
    pub buffers: Vec<TensorBuffer>,
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
        write!(f, ".const\n")?;
        for inst in &self.const_code {
            write!(f, "    {:?}\n", inst)?;
        }
        write!(f, "\n.dynamic\n")?;
        for inst in &self.dynamic_code {
            write!(f, "    {:?}\n", inst)?;
        }
        Ok(())
    }
}
