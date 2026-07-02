fn main() -> pyo3_stub_gen::Result<()> {
    // Use "openqudit" as the stub module root (not "openqudit._openqudit" which
    // is the maturin wheel name) so that generated stubs match the public API.
    let project_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    let stub = pyo3_stub_gen::StubInfo::from_project_root(
        "openqudit".to_string(),
        project_root,
        true, // mixed layout: generates openqudit/__init__.pyi, openqudit/circuit/__init__.pyi, etc.
        Default::default(),
    )?;
    stub.generate()?;
    Ok(())
}
