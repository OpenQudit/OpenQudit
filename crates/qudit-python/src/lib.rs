use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _openqudit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #![allow(unused_imports)]
    use qudit_circuit;
    use qudit_core;
    use qudit_expr;
    use qudit_inst;
    for registrar in inventory::iter::<qudit_core::PyRegistrar> {
        (registrar.func)(m)?;
    }
    Ok(())
}

define_stub_info_gatherer!(stub_info);
