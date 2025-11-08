mod builder;
mod builtins;
#[allow(clippy::module_inception)]
mod codegen;
mod module;

/// Function signature for a JIT-compiled expression
///
/// # Params
///
/// * (*const R): Pointer to input parameter vector
/// * (*mut R): Pointer to output buffer
/// * (*const u64): Pointer to parameter map
/// * (*const u64): Pointer to output map
/// * (u64): offset for each function unit (0*offset = function, 1*offset = first partial grad, ..)
/// * (*const bool): Pointer to constant parameter map
pub type WriteFunc<R> =
    unsafe extern "C" fn(*const R, *mut R, *const u64, *const u64, u64, *const bool);

fn process_name_for_gen(name: &str) -> String {
    name.replace(" ", "_")
        .replace("⊗", "t")
        .replace("†", "d")
        .replace("^", "p")
        .replace("⋅", "x")
}

pub use builder::CompilableUnit;
pub use builder::DifferentiationLevel;
pub use builder::ModuleBuilder;
pub use builder::{FUNCTION, GRADIENT, HESSIAN};
pub use module::Module;

pub use codegen::CodeGenerator;

use qudit_core::RealScalar;

pub struct WriteFuncWithLifeTime<'a, R: RealScalar> {
    #[allow(dead_code)]
    func: WriteFunc<R>,
    _phantom: std::marker::PhantomData<&'a Module<R>>,
}
