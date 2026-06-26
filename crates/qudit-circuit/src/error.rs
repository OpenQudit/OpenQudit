/// Errors that can occur during circuit operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The qubit or classical wire indices are out of range or otherwise invalid.
    #[error("Invalid wire configuration: {0}")]
    InvalidWires(String),

    /// The number of arguments passed to an operation did not match its signature.
    #[error("Incorrect number of arguments provided. Expected {expected}, received {actual}.")]
    ArgumentListSizeMismatch {
        /// Number of arguments that were provided.
        actual: u64,
        /// Number of arguments the operation requires.
        expected: u64,
    },

    /// A catch-all error for cases not covered by more specific variants.
    #[error("{0}")]
    GenericError(String),

    /// An op-code was referenced that does not exist in the circuit's operation set.
    #[error("Missing operation: {0}")]
    MissingOperation(OpCode),

    /// The number of arguments passed does not match what the operation expects.
    #[error("Incorrect number of arguments. Actual: {0}, Expected: {1}")]
    IncorrectNumberOfArguments(usize, usize),

    /// An argument value is semantically invalid for the operation.
    #[error("Invalid argument: {message}")]
    InvalidArgument {
        /// Description of why the argument is invalid.
        message: String,
    },

    /// A syntax or semantic error encountered while parsing a quantum language file.
    #[error("Error parsing file on line {lineno}: {message}")]
    LanguageError {
        /// Human-readable description of the parse error.
        message: String,
        /// 1-based line number where the error was found.
        lineno: usize,
    },
}

/// Convenience alias so callers can write `Result<T>` instead of `Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(feature = "python")]
use pyo3::{PyErr, exceptions::PyValueError};

use crate::OpCode;

#[cfg(feature = "python")]
impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

impl From<String> for Error {
    fn from(value: String) -> Self {
        Error::GenericError(value)
    }
}
