fn main() -> pyo3_stub_gen::Result<()> {
    // `stub_info()` must live in the `openqudit` library crate (not here in the
    // `gen_stub` binary) because pyo3-stub-gen's `inventory`-based collection
    // only reliably finds `#[pymethods]`/`#[pyclass]` registrations from crates
    // that are actually linked in transitively from the call site. Calling it
    // from this separate binary crate silently drops registrations depending on
    // what the linker happens to pull in. See `pyo3_stub_gen::StubInfo::from_project_root`'s
    // doc comment and `openqudit::stub_info` in `src/lib.rs`.
    let stub = openqudit::stub_info()?;
    stub.generate()?;
    Ok(())
}
