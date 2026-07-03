use pyo3::prelude::*;

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

/// Gathers stub info for `cargo run --bin stub_gen`.
///
/// This must live here in the `openqudit` library crate, not in the `gen_stub`
/// binary, so that pyo3-stub-gen's `inventory`-based collection can rely on the
/// same linkage that `_openqudit()` above forces for `qudit_circuit`, `qudit_core`,
/// `qudit_expr`, and `qudit_inst`'s `#[pymethods]`/`#[pyclass]` registrations.
/// Calling `StubInfo::from_project_root` from a separate binary crate silently
/// drops registrations depending on what the linker happens to pull in, per
/// `pyo3_stub_gen::StubInfo::from_project_root`'s own doc comment.
///
/// Uses "openqudit" as the stub module root (not "openqudit._openqudit", which
/// is the maturin wheel name) so that generated stubs match the public API.
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    let project_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    pyo3_stub_gen::StubInfo::from_project_root(
        "openqudit".to_string(),
        project_root,
        true, // mixed layout: generates openqudit/__init__.pyi, openqudit/circuit/__init__.pyi, etc.
        Default::default(),
    )
}
