/// Errors that can occur during circuit operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid wire configuration: {0}")]
    InvalidWires(String),

    #[error("Incorrect number of arguments provided. Expected {expected}, received {actual}.")]
    ArgumentListSizeMismatch { actual: u64, expected: u64 },

    #[error("{0}")]
    GenericError(String),

    #[error("Missing operation: {0}")]
    MissingOperation(OpCode),

    #[error("Incorrect number of arguments. Actual: {0}, Expected: {1}")]
    IncorrectNumberOfArguments(usize, usize),

    #[error("Invalid argument: {message}")]
    InvalidArgument { message: String },

    #[error("Error parsing file on line {lineno}: {message}")]
    LanguageError { message: String, lineno: usize },
}

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
