use crate::QuditCircuit;
use crate::Result;

pub trait QuantumLanguageParser {
    fn parse(&self, source: &str) -> Result<QuditCircuit>;

    fn supported_extensions(&self) -> &[&str];
}

mod qasm2;
pub use qasm2::QASM2Parser;

pub fn all_parsers() -> Vec<Box<dyn QuantumLanguageParser>> {
    vec![
        Box::new(QASM2Parser),
    ]
}
