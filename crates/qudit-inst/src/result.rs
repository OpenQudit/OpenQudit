use qudit_core::{ComplexScalar, RealScalar};

// pub type InstantiationResult = Result<InstantiationOutput, InstantiationError>;

// pub struct InstantiationOutput<R: RealScalar> {
//     params: Vec<R>,
//     func: Option<R>,
//     message: Option<String>,
// }

// pub struct InstantiationError {
//     message: String
// }

/// Result of an instantiation operation.
/// 
/// This struct encapsulates the outcome of an instantiation attempt.
/// It provides information about the success/failure status, and other
/// relevant information. Very little is standardized between instantiaters,
/// so refer to specific instantiaters for more documentation on how
/// the fields are used.
///
/// A successful instantiation is one that was able to run until 
/// 
/// # Status Codes
/// * `0` - Successful termination
/// * `1` - Input cannot be handled by instantiator
/// * `2+` - Instantiator-specific error codes (see relevant documentation)
#[derive(Clone, Debug, PartialEq)]
pub struct InstantiationResult<C: ComplexScalar> {
    /// The instantiated solution's parameters.
    pub params: Option<Vec<C::R>>,

    /// Optional function evaluation at the solution point.
    /// 
    /// If provided, represents instantiation-specific objective function
    /// value at the computed solution. Useful for assessing solution quality.
    /// Consult with instantiater documentation.
    pub fun: Option<C::R>,

    /// Termination status code.
    /// 
    /// - `0`: Successful termination
    /// - `1`: Input cannot be handled by instantiator; see message
    /// - `2+`: Instantiator-specific error codes; see relevant documentation
    pub status: usize,

    /// Optional diagnostic message, providing additional context.
    pub message: Option<String>,
}

impl<C: ComplexScalar> InstantiationResult<C> {
    /// Creates a new `InstantiationResult` with all fields specified.
    /// 
    /// # Arguments
    /// * `params` - Optional solution parameters
    /// * `fun` - Optional function evaluation
    /// * `status` - Status code (0 for success)
    /// * `message` - Optional diagnostic message
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// # use qudit_inst::InstantiationResult;
    /// # use qudit_core::c64;
    /// 
    /// let result = InstantiationResult::<c64>::new(
    ///     Some(vec![1.0, 2.0]),
    ///     Some(0.1),
    ///     0,
    ///     Some("Converged successfully".to_string())
    /// );
    /// ```
    pub fn new(
        params: Option<Vec<C::R>>,
        fun: Option<C::R>,
        status: usize,
        message: Option<String>,
    ) -> Self {
        Self {
            params,
            fun,
            status,
            message,
        }
    }

    /// Creates a successful instantiation result.
    /// 
    /// # Arguments
    /// * `params` - The computed solution parameters
    /// * `fun` - Optional function evaluation
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// # use qudit_inst::InstantiationResult;
    /// # use qudit_core::c64;
    /// 
    /// let result = InstantiationResult::<c64>::success(
    ///     vec![0.5, 1.0, 1.5],
    ///     Some(0.0001)
    /// );
    /// assert!(result.is_success());
    /// ```
    pub fn success(params: Vec<C::R>, fun: Option<C::R>) -> Self {
        Self {
            params: Some(params),
            fun,
            status: 0,
            message: None,
        }
    }

    /// Creates a successful instantiation result with a message.
    /// 
    /// # Arguments
    /// * `params` - The computed solution parameters
    /// * `fun` - Optional function evaluation
    /// * `message` - Success message
    pub fn success_with_message(params: Vec<C::R>, fun: Option<C::R>, message: String) -> Self {
        Self {
            params: Some(params),
            fun,
            status: 0,
            message: Some(message),
        }
    }

    /// Returns true if the instantiation was successful (status == 0).
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// # use qudit_inst::InstantiationResult;
    /// # use qudit_core::c64;
    /// 
    /// let success = InstantiationResult::<c64>::success(vec![1.0], None);
    /// assert!(success.is_success());
    /// ```
    pub fn is_success(&self) -> bool {
        self.status == 0
    }

    /// Returns true if the instantiation failed (status > 0).
    pub fn is_failure(&self) -> bool {
        self.status > 0
    }

    /// Returns true if the result contains solution parameters.
    pub fn has_params(&self) -> bool {
        self.params.is_some()
    }

    /// Returns true if the result contains a function evaluation.
    pub fn has_function_value(&self) -> bool {
        self.fun.is_some()
    }

    /// Returns the number of parameters if available.
    pub fn num_params(&self) -> Option<usize> {
        self.params.as_ref().map(|p| p.len())
    }
}

impl<C: ComplexScalar> std::fmt::Display for InstantiationResult<C> 
where
    C::R: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InstantiationResult {{ status: {}", self.status)?;
        
        if let Some(ref params) = self.params {
            write!(f, ", params: [")?;
            for (i, param) in params.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "{}", param)?;
            }
            write!(f, "]")?;
        }
        
        if let Some(ref fun_val) = self.fun {
            write!(f, ", fun: {}", fun_val)?;
        }
        
        if let Some(ref msg) = self.message {
            write!(f, ", message: \"{}\"", msg)?;
        }
        
        write!(f, " }}")
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use qudit_core::c32;
    use qudit_core::c64;
    use pyo3::prelude::*;
    use pyo3::types::PyList;
    use crate::python::PyInstantiationRegistrar;

    /// Python binding for InstantiationResult using f64 as the real number type.
    /// 
    /// This provides a concrete instantiation of InstantiationResult for use in Python,
    /// using f64 for real numbers (corresponding to Complex64 complex scalars).
    #[pyclass(name = "InstantiationResult", frozen)]
    #[derive(Clone, Debug)]
    pub struct PyInstantiationResult {
        inner: InstantiationResult<c64>
    }

    #[pymethods]
    impl PyInstantiationResult {
        /// Creates a new InstantiationResult.
        /// 
        /// Args:
        ///     params: Optional list of solution parameters
        ///     fun: Optional function evaluation value
        ///     status: Status code (0 for success)
        ///     message: Optional diagnostic message
        /// 
        /// Returns:
        ///     New InstantiationResult instance
        #[new]
        #[pyo3(signature = (params=None, fun=None, status=0, message=None))]
        pub fn new(
            params: Option<Vec<f64>>,
            fun: Option<f64>,
            status: usize,
            message: Option<String>,
        ) -> Self {
            Self {
                inner: InstantiationResult::new(params, fun, status, message)
            }
        }

        /// Creates a successful instantiation result.
        /// 
        /// Args:
        ///     params: The computed solution parameters
        ///     fun: Optional function evaluation
        /// 
        /// Returns:
        ///     Successful InstantiationResult
        #[staticmethod]
        #[pyo3(signature = (params, fun=None))]
        pub fn success(params: Vec<f64>, fun: Option<f64>) -> Self {
            Self {
                inner: InstantiationResult::success(params, fun)
            }
        }

        /// Creates a failed instantiation result.
        /// 
        /// Args:
        ///     status: Error status code (must be > 0)
        ///     message: Optional error message
        /// 
        /// Returns:
        ///     Failed InstantiationResult
        #[staticmethod]
        #[pyo3(signature = (status, message=None))]
        pub fn failure(status: usize, message: Option<String>) -> Self {
            Self {
                inner: InstantiationResult::new(None, None, status, message)
            }
        }

        /// The instantiated solution parameters
        #[getter]
        pub fn params(&self) -> Option<Vec<f64>> {
            self.inner.params.clone()
        }

        /// Optional function evaluation
        #[getter]
        pub fn fun(&self) -> Option<f64> {
            self.inner.fun
        }

        /// Termination status code
        #[getter]
        pub fn status(&self) -> usize {
            self.inner.status
        }

        /// Optional message
        #[getter]
        pub fn message(&self) -> Option<String> {
            self.inner.message.clone()
        }

        /// Returns True if the instantiation was successful.
        pub fn is_success(&self) -> bool {
            self.inner.is_success()
        }

        /// Returns True if the instantiation failed.
        pub fn is_failure(&self) -> bool {
            self.inner.is_failure()
        }

        /// Returns True if the result contains solution parameters.
        pub fn has_params(&self) -> bool {
            self.inner.has_params()
        }

        /// Returns True if the result contains a function evaluation.
        pub fn has_function_value(&self) -> bool {
            self.inner.has_function_value()
        }

        /// Returns the number of parameters if available.
        pub fn num_params(&self) -> Option<usize> {
            self.inner.num_params()
        }

        /// String representation of the result.
        pub fn __repr__(&self) -> String {
            format!("InstantiationResult(status={}, params={:?}, fun={:?}, message={:?})",
                    self.inner.status, self.inner.params, self.inner.fun, self.inner.message)
        }

        /// String representation of the result.
        pub fn __str__(&self) -> String {
            format!("{}", self.inner)
        }
    }


    impl From<InstantiationResult<c32>> for PyInstantiationResult {
        fn from(result: InstantiationResult<c32>) -> Self {
            // Convert c32 result to c64 result
            Self {
                inner: InstantiationResult::new(
                    result.params.map(|p| p.into_iter().map(|x| x as f64).collect()),
                    result.fun.map(|x| x as f64),
                    result.status,
                    result.message,
                )
            }
        }
    }

    impl From<InstantiationResult<c64>> for PyInstantiationResult {
        fn from(result: InstantiationResult<c64>) -> Self {
            Self { inner: result }
        }
    }

    impl From<PyInstantiationResult> for InstantiationResult<c64> {
        fn from(py_result: PyInstantiationResult) -> Self {
            py_result.inner
        }
    }

    impl<'py> IntoPyObject<'py> for InstantiationResult<c64> {
        type Target = PyInstantiationResult;
        type Output = Bound<'py, PyInstantiationResult>;
        type Error = PyErr;
        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            let py_result = PyInstantiationResult::from(self);
            Bound::new(py, py_result)
        }
    }

    impl<'py> IntoPyObject<'py> for InstantiationResult<c32> {
        type Target = PyInstantiationResult;
        type Output = Bound<'py, PyInstantiationResult>;
        type Error = PyErr;
        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            let py_result = PyInstantiationResult::from(self);
            Bound::new(py, py_result)
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for InstantiationResult<c64> {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let py_result: PyInstantiationResult = obj.extract()?;
            Ok(py_result.into())
        }
    }

    /// Registers the InstantiationResult class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyInstantiationResult>()?;
        Ok(())
    }
    inventory::submit!(PyInstantiationRegistrar { func: register });
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use qudit_core::c64;

//     type TestResult = InstantiationResult<c64>;

//     #[test]
//     fn test_new() {
//         let result = TestResult::new(
//             Some(vec![1.0, 2.0, 3.0]),
//             Some(0.5),
//             0,
//             Some("Success".to_string())
//         );
        
//         assert_eq!(result.params, Some(vec![1.0, 2.0, 3.0]));
//         assert_eq!(result.fun, Some(0.5));
//         assert_eq!(result.status, 0);
//         assert_eq!(result.message, Some("Success".to_string()));
//     }

//     #[test]
//     fn test_success() {
//         let result = TestResult::success(vec![1.0, 2.0], Some(0.1));
        
//         assert!(result.is_success());
//         assert!(!result.is_failure());
//         assert!(result.has_params());
//         assert!(result.has_function_value());
//         assert_eq!(result.param_count(), Some(2));
//         assert_eq!(result.status, 0);
//     }

//     #[test]
//     fn test_success_with_message() {
//         let result = TestResult::success_with_message(
//             vec![1.0],
//             None,
//             "Converged in 10 iterations".to_string()
//         );
        
//         assert!(result.is_success());
//         assert_eq!(result.message, Some("Converged in 10 iterations".to_string()));
//     }

//     #[test]
//     fn test_failure() {
//         let result = TestResult::failure(1, Some("Invalid input".to_string()));
        
//         assert!(result.is_failure());
//         assert!(!result.is_success());
//         assert!(!result.has_params());
//         assert!(!result.has_function_value());
//         assert_eq!(result.param_count(), None);
//         assert_eq!(result.status, 1);
//         assert_eq!(result.message, Some("Invalid input".to_string()));
//     }

//     #[test]
//     fn test_into_params() {
//         let params = vec![1.0, 2.0, 3.0];
//         let result = TestResult::success(params.clone(), None);
        
//         assert_eq!(result.into_params(), Some(params));
//     }

//     #[test]
//     fn test_params_ref() {
//         let params = vec![1.0, 2.0, 3.0];
//         let result = TestResult::success(params.clone(), None);
        
//         assert_eq!(result.params_ref(), Some(params.as_slice()));
//     }

//     #[test]
//     fn test_map_params() {
//         let result = TestResult::success(vec![1.0, 2.0], None);
//         let mapped = result.map_params(|params| params.iter().map(|&x| x * 2.0).collect());
        
//         assert_eq!(mapped.params_ref(), Some(&[2.0, 4.0][..]));
//     }

//     #[test]
//     fn test_map_params_no_params() {
//         let result = TestResult::failure(1, None);
//         let mapped = result.map_params(|params| params.iter().map(|&x| x * 2.0).collect());
        
//         assert_eq!(mapped.params_ref(), None);
//     }

//     #[test]
//     fn test_display() {
//         let result = TestResult::success(vec![1.5, 2.5], Some(0.1));
//         let display_str = format!("{}", result);
        
//         assert!(display_str.contains("status: 0"));
//         assert!(display_str.contains("params: [1.5, 2.5]"));
//         assert!(display_str.contains("fun: 0.1"));
//     }

//     #[test]
//     fn test_display_failure() {
//         let result = TestResult::failure(1, Some("Error".to_string()));
//         let display_str = format!("{}", result);
        
//         assert!(display_str.contains("status: 1"));
//         assert!(display_str.contains("message: \"Error\""));
//         assert!(!display_str.contains("params:"));
//         assert!(!display_str.contains("fun:"));
//     }

//     #[test]
//     fn test_default() {
//         let result = TestResult::default();
        
//         assert!(result.is_failure());
//         assert_eq!(result.status, 1);
//         assert!(result.message.is_some());
//     }

//     #[test]
//     fn test_clone_and_debug() {
//         let original = TestResult::success(vec![1.0], None);
//         let cloned = original.clone();
        
//         assert_eq!(original, cloned);
        
//         let debug_str = format!("{:?}", original);
//         assert!(debug_str.contains("InstantiationResult"));
//     }

//     #[cfg(feature = "python")]
//     mod python_tests {
//         use super::super::python::*;

//         #[test]
//         fn test_py_instantiation_result_new() {
//             let result = PyInstantiationResult::new(
//                 Some(vec![1.0, 2.0]),
//                 Some(0.5),
//                 0,
//                 Some("Success".to_string())
//             );
            
//             assert_eq!(result.params, Some(vec![1.0, 2.0]));
//             assert_eq!(result.fun, Some(0.5));
//             assert_eq!(result.status, 0);
//         }

//         #[test]
//         fn test_py_success() {
//             let result = PyInstantiationResult::success(vec![1.0, 2.0], Some(0.1));
            
//             assert!(result.is_success());
//             assert!(!result.is_failure());
//             assert!(result.has_params());
//         }

//         #[test]
//         fn test_py_failure() {
//             let result = PyInstantiationResult::failure(1, Some("Error".to_string()));
            
//             assert!(result.is_failure());
//             assert!(!result.is_success());
//             assert!(!result.has_params());
//         }

//         #[test]
//         fn test_py_conversion_from_rust() {
//             let rust_result = super::TestResult::success(vec![1.0, 2.0], Some(0.5));
//             let py_result: PyInstantiationResult = rust_result.into();
            
//             assert_eq!(py_result.params, Some(vec![1.0, 2.0]));
//             assert_eq!(py_result.fun, Some(0.5));
//             assert_eq!(py_result.status, 0);
//         }

//         #[test]
//         fn test_py_conversion_to_rust() {
//             let py_result = PyInstantiationResult::success(vec![1.0, 2.0], Some(0.5));
//             let rust_result: super::TestResult = py_result.into();
            
//             assert_eq!(rust_result.params, Some(vec![1.0, 2.0]));
//             assert_eq!(rust_result.fun, Some(0.5));
//             assert_eq!(rust_result.status, 0);
//         }
//     }
// }
