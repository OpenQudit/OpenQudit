use crate::QuditCircuit;
use crate::Result;

pub trait QuantumLanguageParser {
    fn parse(&self, source: &str) -> Result<QuditCircuit>;

    fn supported_extensions(&self) -> &[&str];
}

pub trait QuantumLanguageWriter {
    fn write(&self, circuit: &QuditCircuit) -> Result<String>;

    fn supported_extensions(&self) -> &[&str];
}

mod qasm2;
pub use qasm2::QASM2Parser;
pub use qasm2::QASM2Writer;

pub fn all_parsers() -> Vec<Box<dyn QuantumLanguageParser>> {
    vec![Box::new(QASM2Parser)]
}

pub fn all_writers() -> Vec<Box<dyn QuantumLanguageWriter>> {
    vec![Box::new(QASM2Writer)]
}
