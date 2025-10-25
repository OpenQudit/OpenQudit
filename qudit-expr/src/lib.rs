mod qgl;
mod expressions;
mod codegen;
mod analysis;
mod shape;
mod cache;
pub mod index;
pub mod library;

pub use expressions::Constant;
pub use expressions::Expression;
pub use expressions::ComplexExpression;
pub use expressions::NamedExpression;
pub use expressions::BraExpression;
pub use expressions::KetExpression;
pub use expressions::BraSystemExpression;
pub use expressions::KetSystemExpression;
pub use expressions::IsometryExpression;
pub use expressions::UnitaryExpression;
pub use expressions::KrausOperatorsExpression;
pub use expressions::UnitarySystemExpression;
pub use expressions::TensorExpression;
pub use expressions::ExpressionGenerator;
pub use codegen::DifferentiationLevel;
pub use codegen::{FUNCTION, GRADIENT, HESSIAN};
pub use codegen::ModuleBuilder;
pub use codegen::Module;
pub use codegen::WriteFunc;
pub use codegen::WriteFuncWithLifeTime;
pub use codegen::CodeGenerator;
pub use analysis::simplify_matrix_and_matvec;
pub use shape::GenerationShape;
pub use cache::ExpressionCache;
pub use cache::ExpressionId;


////////////////////////////////////////////////////////////////////////
/// Python Module.
////////////////////////////////////////////////////////////////////////
#[cfg(feature = "python")]
pub(crate) mod python {
    use pyo3::prelude::{Bound, PyModule, PyResult, PyAnyMethods, PyModuleMethods};

    /// A trait for objects that can register importables with a PyO3 module.
    pub struct PyExpressionRegistrar {
        /// The registration function
        pub func: fn(parent_module: &Bound<'_, PyModule>) -> PyResult<()>,
    }

    inventory::collect!(PyExpressionRegistrar);

    /// Registers the Circuit submodule with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        let submodule = PyModule::new(parent_module.py(), "expressions")?;

        for registrar in inventory::iter::<PyExpressionRegistrar> {
            (registrar.func)(&submodule)?;
        }

        parent_module.add_submodule(&submodule)?;
        parent_module.py().import("sys")?
            .getattr("modules")?
            .set_item("openqudit.expressions", submodule)?;

        Ok(())
    }

    inventory::submit!(qudit_core::PyRegistrar { func: register });
}
////////////////////////////////////////////////////////////////////////
