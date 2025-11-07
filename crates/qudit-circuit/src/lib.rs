#![warn(missing_docs)]

//! The qudit-circuit package contains the main circuit data structures for the OpenQudit library.

mod circuit;
mod cycle;
mod instruction;
mod operation;
mod param;
mod wire;

pub use circuit::QuditCircuit;
pub use operation::OpCode;
pub use operation::Operation;
pub use param::Argument;
pub use param::ArgumentList;
pub use wire::Wire;
pub use wire::WireList;

////////////////////////////////////////////////////////////////////////
/// Python Module.
////////////////////////////////////////////////////////////////////////
#[cfg(feature = "python")]
pub(crate) mod python {
    use pyo3::prelude::{Bound, PyAnyMethods, PyModule, PyModuleMethods, PyResult};

    /// A trait for objects that can register importables with a PyO3 module.
    pub struct PyCircuitRegistrar {
        /// The registration function
        pub func: fn(parent_module: &Bound<'_, PyModule>) -> PyResult<()>,
    }

    inventory::collect!(PyCircuitRegistrar);

    /// Registers the Circuit submodule with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        let submodule = PyModule::new(parent_module.py(), "circuit")?;

        for registrar in inventory::iter::<PyCircuitRegistrar> {
            (registrar.func)(&submodule)?;
        }

        parent_module.add_submodule(&submodule)?;
        parent_module
            .py()
            .import("sys")?
            .getattr("modules")?
            .set_item("openqudit.circuit", submodule)?;

        Ok(())
    }

    inventory::submit!(qudit_core::PyRegistrar { func: register });
}
////////////////////////////////////////////////////////////////////////
