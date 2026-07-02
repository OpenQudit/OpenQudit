use crate::QuditCircuit;
use crate::Result;

/// Parses a quantum language source string into a [`QuditCircuit`].
pub trait QuantumLanguageParser {
    /// Parse `source` and return the resulting circuit, or an error if the
    /// source is invalid.
    fn parse(&self, source: &str) -> Result<QuditCircuit>;

    /// File extensions this parser recognises (e.g. `&["qasm", "qasm2"]`).
    fn supported_extensions(&self) -> &[&str];
}

/// Serializes a [`QuditCircuit`] to a quantum language source string.
pub trait QuantumLanguageWriter {
    /// Serialize `circuit` and return the resulting source string, or an error
    /// if the circuit contains operations the format cannot represent.
    fn write(&self, circuit: &QuditCircuit) -> Result<String>;

    /// File extensions this writer produces (e.g. `&["qasm", "qasm2"]`).
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
