mod codegen;
mod builtins;
mod module;
mod builder;

use qudit_core::ComplexScalar;
pub type UtryFunc<C> = unsafe extern "C" fn(*const <C as ComplexScalar>::R, *mut <C as ComplexScalar>::R);
pub type UtryGradFunc<C> = unsafe extern "C" fn(*const <C as ComplexScalar>::R, *mut <C as ComplexScalar>::R, *mut <C as ComplexScalar>::R);

pub(self) fn process_name_for_gen(name: &str) -> String {
    name.replace(" ", "_")
        .replace("⊗", "t")
        .replace("†", "d")
        .replace("^", "p")
        .replace("⋅", "x")
}

pub use builder::DifferentiationLevel;
pub use builder::{FUNCTION, GRADIENT, HESSIAN};
// pub use builder::{Function, Gradient, Hessian};
pub use builder::ModuleBuilder;
pub use module::Module;

pub use codegen::CodeGenerator;

