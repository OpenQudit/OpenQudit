//! Argument list types for quantum circuit parameters.
//!
//! This module provides the `ArgumentList` struct which represents a collection
//! of `Argument` instances that can be used to parameterize quantum circuits.
//! The list provides methods for extracting parameters, variable names, and
//! substitution expressions from all contained arguments.
//!
//! The module also provides conversions from various Rust collections and Python
//! iterables when the `python` feature is enabled.

use super::Argument;
use super::Parameter;
use qudit_expr::Expression;

/// Represents a list of arguments for quantum circuit parameters.
#[derive(Clone, Debug)]
pub struct ArgumentList {
    entries: Vec<Argument>
}

impl ArgumentList {
    /// Creates a new `ArgumentList` from a vector of arguments.
    pub fn new(entries: Vec<Argument>) -> Self {
        Self { entries }
    }

    /// Returns a reference to the internal vector of arguments.
    pub fn arguments(&self) -> &[Argument] {
        &self.entries
    }

    /// Returns the number of arguments in this list.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether this argument list is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Extracts the parameters from the list of arguments.
    ///
    /// This method aggregates parameters from all arguments in the list,
    /// removing duplicates for named parameters to ensure each unique
    /// parameter variable appears only once in the result.
    ///
    /// Note: one argument can correspond to multiple parameters. For example,
    /// if a user provides `"a*b"` as an argument, this will be managed as two
    /// separate named parameters, `a` and `b`. The expression containing
    /// the multiplication will get folded into the operation.
    ///
    /// # Returns
    /// A vector of `Parameter` instances representing all unique parameters
    /// required by the arguments in this list
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![];
        for argument in self.entries.iter() {
            for param in argument.parameters() {
                if let Parameter::Named(_) = param {
                    if params.contains(&param) {
                        continue;
                    }
                }
                params.push(param);
            }
        }
        params
    }

    /// Gathers the variable names associated with these arguments.
    ///
    /// Note: Non-parameterized arguments are assigned a name `unnamed_i`
    /// where `i` is generated from an internal counter to ensure uniqueness.
    /// Parameterized expression arguments return their actual variable names.
    /// The output of this can be one part of a `substitute_parameters` call on
    /// an expression informing the expression of the new variable names.
    pub fn variables(&self) -> Vec<String> { 
        let mut new_variables = vec![];
        let mut unnamed_counter = 0;

        for argument in self.entries.iter() {
            new_variables.extend(argument.variables(&mut unnamed_counter))
        }

        new_variables
    }

    /// Gathers the expressions associated with these arguments.
    ///
    /// Note: Non-expression arguments are converted to variable expressions
    /// with names `unnamed_i` where `i` is generated from an internal counter.
    /// Expression arguments are returned as-is. The output can be used as
    /// part of a `substitute_parameters` call on an expression informing
    /// the expression what to replace its current variables with.
    pub fn expressions(&self) -> Vec<Expression> {
        let mut new_variables = vec![];
        let mut unnamed_counter = 0;

        for argument in self.entries.iter() {
            new_variables.push(argument.as_substitution_expression(&mut unnamed_counter))
        }

        new_variables
    }


    /// Returns true if any non-simple arguments exist in the list
    ///
    /// A Non-simple argument is one that is not a single constant or single
    /// parameter (named or unnamed). For example, the expression `"a*b"`
    /// requires expression modification and is non-simple. This is because
    /// the multiplication of the two parameters gets folded into the
    /// expression being invoked with this argument list, and as a result,
    /// get's modified to accomodate a multiplication of two parameters in
    /// place of the one in that argument's slot.
    pub fn requires_expression_modification(&self) -> bool {
        self.entries.iter().any(|arg| arg.requires_expression_modification())
    }
}

impl<E: Into<Argument>> From<Vec<E>> for ArgumentList {
    /// Converts a vector of argument-like values into an `ArgumentList`.
    fn from(value: Vec<E>) -> Self {
        Self::new(value.into_iter().map(|e| e.into()).collect())
    }
}

impl<E: Into<Argument>, const N: usize> From<[E; N]> for ArgumentList {
    /// Converts an array of argument-like values into an `ArgumentList`.
    fn from(value: [E; N]) -> Self {
        Self::new(value.into_iter().map(|e| e.into()).collect())
    }
}

impl<E: Into<Argument> + Clone, const N: usize> From<&[E; N]> for ArgumentList {
    /// Converts a reference to an array of argument-like values into an `ArgumentList`.
    fn from(value: &[E; N]) -> Self {
        Self::new(value.into_iter().map(|e| e.clone().into()).collect())
    }
}

#[cfg(feature = "python")]
mod python {
    use super::ArgumentList;
    use super::Argument;
    use pyo3::prelude::*;
    use pyo3::exceptions::{PyTypeError, PyValueError};

    impl<'a, 'py> FromPyObject<'a, 'py> for ArgumentList {
        type Error = PyErr;

        /// Converts a Python object to an `ArgumentList`.
        /// 
        /// # Supported conversions
        /// - Python lists/tuples containing argument-like values → `ArgumentList`
        /// - Single argument-like values → `ArgumentList` with one element
        /// - Empty iterables → Empty `ArgumentList`
        /// 
        /// Each element in the iterable is converted using `Argument::extract_bound()`.
        /// 
        /// # Errors
        /// Returns `PyValueError` if any element cannot be converted to an `Argument`,
        /// or `PyTypeError` if the Python object is not iterable and not convertible to an `Argument`.
        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            // First, try to convert as a single argument
            if let Ok(arg) = Argument::extract(obj) {
                return Ok(ArgumentList::new(vec![arg]));
            }
            
            // If it's not a single argument, try to treat as an iterable (list, tuple, etc.)
            if let Ok(iter) = pyo3::types::PyIterator::from_object(&*obj) {
                let mut arguments = Vec::new();
                for item in iter {
                    let item = item?;
                    match Argument::extract(item.as_borrowed()) {
                        Ok(arg) => arguments.push(arg),
                        Err(e) => return Err(PyValueError::new_err(format!(
                            "Cannot convert item to Argument: {}", e
                        ))),
                    }
                }
                return Ok(ArgumentList::new(arguments));
            }
            
            // If neither worked, return an error
            Err(PyTypeError::new_err(format!(
                "Cannot convert {} to ArgumentList: must be iterable or convertible to Argument", 
                obj.get_type().name()?,
            )))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
        use pyo3::Python;
        #[test]
        fn test_from_py_empty_list() {
            Python::initialize();
            Python::attach(|py| {
                let empty_list = PyList::empty(py);
                let result = ArgumentList::extract_bound(&empty_list).unwrap();
                assert_eq!(result.len(), 0);
                assert!(result.is_empty());
            });
        }
        #[test]
        fn test_from_py_list_of_numbers() {
            Python::initialize();
            Python::attach(|py| {
                let py_list = PyList::new(py, &[1.0, 2.5, 3.14]).unwrap();
                let result = ArgumentList::extract_bound(&py_list).unwrap();
                assert_eq!(result.len(), 3);
                
                let args = result.arguments();
                assert!(matches!(args[0], Argument::Float64(1.0)));
                assert!(matches!(args[1], Argument::Float64(2.5)));
                assert!(matches!(args[2], Argument::Float64(3.14)));
            });
        }
        #[test]
        fn test_from_py_tuple_mixed_types() {
            Python::initialize();
            Python::attach(|py| {
                let one = PyFloat::new(py, 1.5);
                let two = PyString::new(py, "x + 1");
                let none = py.None().into_bound(py);
                let items: Vec<&Bound<'_, PyAny>> = vec![
                    one.as_any(), two.as_any(), &none,
                ];
                let py_tuple = PyTuple::new(py, items).unwrap();
                let result = ArgumentList::extract_bound(&py_tuple).unwrap();
                assert_eq!(result.len(), 3);
                
                let args = result.arguments();
                assert!(matches!(args[0], Argument::Float64(1.5)));
                assert!(matches!(args[1], Argument::Expression(_)));
                assert!(matches!(args[2], Argument::Unspecified));
            });
        }
        #[test]
        fn test_from_py_single_argument() {
            Python::initialize();
            Python::attach(|py| {
                let float_val = PyFloat::new(py, 42.0);
                let result = ArgumentList::extract_bound(&float_val).unwrap();
                assert_eq!(result.len(), 1);
                
                let args = result.arguments();
                assert!(matches!(args[0], Argument::Float64(42.0)));
            });
        }
        #[test]
        fn test_from_py_list_with_expressions() {
            Python::initialize();
            Python::attach(|py| {

                let one = PyString::new(py, "sin(x)");
                let two = PyString::new(py, "a*b + c");
                let three = PyString::new(py, "pi/4");
                let expressions = vec![one.as_any(), two.as_any(), three.as_any()];
                let py_list = PyList::new(py, expressions).unwrap();
                let result = ArgumentList::extract_bound(&py_list).unwrap();
                assert_eq!(result.len(), 3);
                
                let args = result.arguments();
                for arg in args {
                    assert!(matches!(arg, Argument::Expression(_)));
                }
            });
        }
        #[test]
        fn test_from_py_list_with_invalid_item() {
            Python::initialize();
            Python::attach(|py| {
                let one = PyFloat::new(py, 1.0);
                let two = PyDict::new(py);  // This should fail
                let items: Vec<&Bound<'_, PyAny>> = vec![
                    one.as_any(), two.as_any(),
                ];
                let py_list = PyList::new(py, items).unwrap();
                let result = ArgumentList::extract_bound(&py_list);
                assert!(result.is_err());
                let err = result.unwrap_err();
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(err.to_string().contains("Cannot convert item to Argument"));
            });
        }
        #[test]
        fn test_from_py_list_with_complex_expression_rejected() {
            Python::initialize();
            Python::attach(|py| {
                let complex_expr = PyString::new(py, "1 + i");
                let items: Vec<&Bound<'_, PyAny>> = vec![
                    complex_expr.as_any(),  // Complex expression
                ];
                let py_list = PyList::new(py, items).unwrap();
                let result = ArgumentList::extract_bound(&py_list);
                assert!(result.is_err());
                let err = result.unwrap_err();
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            });
        }
        #[test]
        fn test_from_py_list_with_unnamed_rejected() {
            Python::initialize();
            Python::attach(|py| {
                let unnamed_expr = PyString::new(py, "unnamed_var");
                let items: Vec<&Bound<'_, PyAny>> = vec![
                    unnamed_expr.as_any(),
                ];
                let py_list = PyList::new(py, items).unwrap();
                let result = ArgumentList::extract_bound(&py_list);
                assert!(result.is_err());
                let err = result.unwrap_err();
                assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                assert!(err.to_string().contains("Expression arguments cannot contain 'unnamed_'"));
            });
        }

        #[test]
        fn test_from_py_single_string_expression() {
            Python::initialize();
            Python::attach(|py| {
                let string_val = PyString::new(py, "x*y + z");
                let result = ArgumentList::extract_bound(&string_val).unwrap();
                assert_eq!(result.len(), 1);
                
                let args = result.arguments();
                assert!(matches!(args[0], Argument::Expression(_)));
            });
        }

        #[test]
        fn test_parameters_extraction() {
            Python::initialize();
            Python::attach(|py| {
                let one = PyString::new(py, "x");
                let two = PyString::new(py, "x + y");  // x should be deduplicated
                let three = PyFloat::new(py, 1.0);
                let expressions = vec![
                    one.as_any(), two.as_any(), three.as_any()
                ];
                let py_list = PyList::new(py, expressions).unwrap();
                let result = ArgumentList::extract_bound(&py_list).unwrap();
                
                let params = result.parameters();
                // Should have named parameters for x, y and a constant parameter
                assert_eq!(params.len(), 3);
            });
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use qudit_expr::Expression;
    #[test]
    fn test_argumentlist_new() {
        let args = vec![Argument::Float64(1.0), Argument::Float64(2.0)];
        let list = ArgumentList::new(args);
        assert_eq!(list.len(), 2);
        assert!(!list.is_empty());
    }
    #[test]
    fn test_argumentlist_empty() {
        let list = ArgumentList::new(vec![]);
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
    }
    #[test]
    fn test_argumentlist_arguments_access() {
        let args = vec![Argument::Float64(1.0), Argument::Unspecified];
        let list = ArgumentList::new(args);
        let accessed_args = list.arguments();
        assert_eq!(accessed_args.len(), 2);
        assert!(matches!(accessed_args[0], Argument::Float64(1.0)));
        assert!(matches!(accessed_args[1], Argument::Unspecified));
    }
    #[test]
    fn test_parameters_single_argument() {
        let args = vec![Argument::Float64(42.0)];
        let list = ArgumentList::new(args);
        let params = list.parameters();
        assert_eq!(params.len(), 1);
        assert!(matches!(params[0], Parameter::Constant64(42.0)));
    }
    #[test]
    fn test_parameters_deduplication() {
        let x_expr = Argument::try_from("x").unwrap();
        let xy_expr = Argument::try_from("x + y").unwrap();
        let args = vec![x_expr, xy_expr];
        let list = ArgumentList::new(args);
        let params = list.parameters();
        
        // Should have 2 unique named parameters: x and y
        assert_eq!(params.len(), 2);
        let param_names: Vec<String> = params.into_iter().map(|p| match p {
            Parameter::Named(name) => name,
            _ => panic!("Expected Named parameter"),
        }).collect();
        assert!(param_names.contains(&"x".to_string()));
        assert!(param_names.contains(&"y".to_string()));
    }
    #[test]
    fn test_parameters_mixed_types() {
        let args = vec![
            Argument::Float32(1.0),
            Argument::Float64(2.0),
            Argument::Unspecified,
            Argument::try_from("theta").unwrap(),
        ];
        let list = ArgumentList::new(args);
        let params = list.parameters();
        
        assert_eq!(params.len(), 4);
        assert!(matches!(params[0], Parameter::Constant32(1.0)));
        assert!(matches!(params[1], Parameter::Constant64(2.0)));
        assert!(matches!(params[2], Parameter::Indexed));
        assert!(matches!(params[3], Parameter::Named(ref name) if name == "theta"));
    }
    #[test]
    fn test_variables_generation() {
        let args = vec![
            Argument::Float64(1.0),           // Should get unnamed_0
            Argument::try_from("x").unwrap(), // Should keep "x"
            Argument::Unspecified,            // Should get unnamed_1
        ];
        let list = ArgumentList::new(args);
        let vars = list.variables();
        
        assert_eq!(vars.len(), 3);
        assert_eq!(vars[0], "unnamed_0");
        assert_eq!(vars[1], "x");
        assert_eq!(vars[2], "unnamed_1");
    }
    #[test]
    fn test_variables_constant_expression() {
        let constant_expr = Expression::from_float_64(3.14);
        let args = vec![
            Argument::Expression(constant_expr),
            Argument::Float64(2.0),
        ];
        let list = ArgumentList::new(args);
        let vars = list.variables();
        
        assert_eq!(vars.len(), 2);
        assert_eq!(vars[0], "unnamed_0"); // Constant expression gets unnamed
        assert_eq!(vars[1], "unnamed_1"); // Float gets unnamed
    }
    #[test]
    fn test_expressions_generation() {
        let args = vec![
            Argument::Float64(1.0),
            Argument::try_from("x + 1").unwrap(),
        ];
        let list = ArgumentList::new(args.clone());
        let exprs = list.expressions();
        
        assert_eq!(exprs.len(), 2);
        // First should be a variable expression with generated name
        assert_eq!(exprs[0], Expression::Variable("unnamed_0".to_string()));
        // Second should be the original expression
        if let Argument::Expression(original_expr) = &args[1] {
            assert_eq!(exprs[1], *original_expr);
        } else {
            panic!("Expected Expression argument");
        }
    }
    #[test]
    fn test_from_vec() {
        let values = vec![1.0f64, 2.0f64, 3.0f64];
        let list = ArgumentList::from(values);
        assert_eq!(list.len(), 3);
        
        let args = list.arguments();
        assert!(matches!(args[0], Argument::Float64(1.0)));
        assert!(matches!(args[1], Argument::Float64(2.0)));
        assert!(matches!(args[2], Argument::Float64(3.0)));
    }
    #[test]
    fn test_from_array() {
        let values = [1.0f32, 2.0f32];
        let list = ArgumentList::from(values);
        assert_eq!(list.len(), 2);
        
        let args = list.arguments();
        assert!(matches!(args[0], Argument::Float32(1.0)));
        assert!(matches!(args[1], Argument::Float32(2.0)));
    }
    #[test]
    fn test_from_array_ref() {
        let values = [1.0f64, 2.0f64];
        let list = ArgumentList::from(&values);
        assert_eq!(list.len(), 2);
        
        let args = list.arguments();
        assert!(matches!(args[0], Argument::Float64(1.0)));
        assert!(matches!(args[1], Argument::Float64(2.0)));
    }
    #[test]
    fn test_argumentlist_clone() {
        let args = vec![Argument::Float64(42.0), Argument::Unspecified];
        let list1 = ArgumentList::new(args);
        let list2 = list1.clone();
        
        assert_eq!(list1.len(), list2.len());
        assert_eq!(list1.arguments().len(), list2.arguments().len());
    }
    #[test]
    fn test_argumentlist_debug() {
        let args = vec![Argument::Float64(1.0)];
        let list = ArgumentList::new(args);
        let debug_str = format!("{:?}", list);
        assert!(debug_str.contains("ArgumentList"));
        assert!(debug_str.contains("Float64"));
    }
    #[test]
    fn test_complex_multivariate_parameter_extraction() {
        let args = vec![
            Argument::try_from("a*b").unwrap(),      // Variables: a, b
            Argument::try_from("c + a").unwrap(),    // Variables: c, a (a deduplicated)
            Argument::Float64(1.0),                  // Constant parameter
        ];
        let list = ArgumentList::new(args);
        let params = list.parameters();
        
        // Should have 3 named parameters (a, b, c) + 1 constant = 4 total
        assert_eq!(params.len(), 4);
        
        let named_count = params.iter().filter(|p| matches!(p, Parameter::Named(_))).count();
        let constant_count = params.iter().filter(|p| matches!(p, Parameter::Constant64(_))).count();
        
        assert_eq!(named_count, 3); // a, b, c
        assert_eq!(constant_count, 1); // 1.0
    }

    #[test]
    fn test_large_argument_list() {
        let mut args = Vec::new();
        for i in 0..100 {
            args.push(Argument::Float64(i as f64));
        }
        
        let list = ArgumentList::new(args);
        assert_eq!(list.len(), 100);
        assert!(!list.is_empty());
        
        let params = list.parameters();
        assert_eq!(params.len(), 100); // Each float becomes a constant parameter
        
        let vars = list.variables();
        assert_eq!(vars.len(), 100); // Each gets an unnamed variable
        for (i, var) in vars.iter().enumerate() {
            assert_eq!(*var, format!("unnamed_{}", i));
        }
    }
}
