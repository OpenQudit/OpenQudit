#![warn(missing_docs)]

//! The qudit-circuit package contains the main circuit data structures for the OpenQudit library.

/// Utils module
pub mod utils;

mod operation;
mod wire;
mod cycle;
mod instruction;
mod param;
mod circuit;
// mod compact;
// mod cycle;
// mod cyclelist;
// mod instruction;
// mod iterator;
// mod location;
// mod operation;
// mod param;
// mod point;
// mod subcircuit;

pub use wire::Wire;
pub use wire::WireList;
pub use operation::OpCode;
pub use operation::Operation;
pub use circuit::QuditCircuit;
// pub use subcircuit::Subcircuit;
// pub use location::CircuitLocation;
// pub use param::ParamEntry;
// pub use point::CircuitDitId;
// pub use point::CircuitPoint;
// // pub use instruction::Instruction;
// pub use operation::Operation;
// pub use operation::OperationSet;
// pub use operation::ExpressionOperation;


////////////////////////////////////////////////////////////////////////
/// Python Register Helpers.
////////////////////////////////////////////////////////////////////////
#[cfg(feature = "python")]
pub(crate) mod python {
    use pyo3::prelude::{Bound, PyModule, PyResult, PyAnyMethods, PyModuleMethods};

    /// A trait for objects that can register importables with a PyO3 module.
    pub struct PyCircuitRegistrar {
        /// The registration function
        pub func: fn(parent_module: &Bound<'_, PyModule>) -> PyResult<()>,
    }

    inventory::collect!(PyCircuitRegistrar);

    /// Registers the Circuit submodule with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        let library_submodule = PyModule::new(parent_module.py(), "circuit")?;

        for registrar in inventory::iter::<PyCircuitRegistrar> {
            (registrar.func)(&library_submodule)?;
        }

        parent_module.add_submodule(&library_submodule)?;
        parent_module.py().import("sys")?
            .getattr("modules")?
            .set_item("openqudit.circuit", library_submodule)?;

        Ok(())
    }

    inventory::submit!(qudit_core::PyRegistrar { func: register });
}
////////////////////////////////////////////////////////////////////////
