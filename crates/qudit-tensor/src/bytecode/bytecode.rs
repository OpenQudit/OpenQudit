use std::sync::{Arc, Mutex};

// use aligned_vec::CACHELINE_ALIGN;
// use faer_entity::Entity;

// use crate::sim::qvm::QVMType;

use qudit_expr::ExpressionCache;

use super::{BytecodeInstruction, TensorBuffer};

#[derive(Clone)]
pub struct Bytecode {
    // pub expressions: Vec<(TensorExpression, Option<ParamIndices>, String)>,
    pub expressions: Arc<Mutex<ExpressionCache>>,
    pub const_code: Vec<BytecodeInstruction>,
    pub dynamic_code: Vec<BytecodeInstruction>,
    pub buffers: Vec<TensorBuffer>,
    pub out_buffer: usize,
    pub num_params: usize,
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
            writeln!(f, ".const")?;
            for inst in &self.const_code {
                writeln!(f, "    {:?}", inst)?;
            }
        }
        if !self.dynamic_code.is_empty() {
            write!(f, "\n.dynamic\n")?;
            for inst in &self.dynamic_code {
                writeln!(f, "    {:?}", inst)?;
            }
        }
        Ok(())
    }
}
