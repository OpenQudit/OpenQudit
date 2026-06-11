mod id;
mod instruction;

pub use id::InstructionId;
pub use instruction::Instruction;

#[cfg(feature = "python")]
mod reference;

#[cfg(feature = "python")]
pub use reference::PyInstructionReference;
