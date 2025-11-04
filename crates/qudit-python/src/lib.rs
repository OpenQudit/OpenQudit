use pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn openqudit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #![allow(unused_imports)]
    use qudit_core;
    use qudit_expr;
    use qudit_circuit;
    use qudit_inst;
    for registrar in inventory::iter::<qudit_core::PyRegistrar> {
        (registrar.func)(m)?;
    }
    Ok(())
}
