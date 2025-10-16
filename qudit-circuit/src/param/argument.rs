//! Argument types for quantum circuit parameters.
//!
//! This module provides the `Argument` enum which represents different types
//! of parameter values that can be used in quantum circuits. Arguments can be:
//! - Unspecified (an unnamed variable argument)
//! - Concrete numeric values (f32/f64)
//! - Symbolic expressions that can contain variables
//!
//! The module also provides conversions from various Rust and Python types.

use qudit_expr::Expression;
use qudit_expr::ComplexExpression;
use super::Parameter;

/// Represents an argument in a quantum circuit.
///
/// An argument can take several forms depending on how the parameter
/// is specified by the user. This enum provides a unified interface for
/// handling different parameter input formats.
#[derive(Clone, Debug)]
pub enum Argument {
    /// Argument value not yet specified. Will be assigned an indexed parameter.
    Unspecified,
    /// A 32-bit floating point constant value.
    Float32(f32),
    /// A 64-bit floating point constant value.
    Float64(f64),
    /// A symbolic expression that may contain variables and mathematical operations.
    Expression(Expression),
}

impl Argument {
    /// Extracts and flattens internal parameters from the argument.
    ///
    /// This method converts an `Argument` into a vector of `Parameter` instances
    /// describing how this argument should reference a parameter vector. Constant
    /// arguments need a constant parameter, unspecified arguments are represented
    /// by an indexed parameter, and any variables found in an expression become
    /// named parameters.
    ///
    /// # Returns
    /// A vector of `Parameter` instances bound in this argument
    pub fn parameters(&self) -> Vec<Parameter> {
        match self {
            Argument::Unspecified => vec![Parameter::Indexed],
            Argument::Float32(f) => vec![Parameter::Constant32(*f)],
            Argument::Float64(f) => vec![Parameter::Constant64(*f)],
            Argument::Expression(e) => {
                if !e.is_parameterized() {
                    vec![Parameter::ConstantRatio(e.to_constant())]
                } else {
                    e.get_unique_variables()
                        .into_iter()
                        .map(|var| Parameter::Named(var))
                        .collect()
                }
            },
        }
    }

    /// Returns variable names for this argument, generating unique names if needed.
    ///
    /// This method extracts variable names from the argument, using a counter to
    /// generate unique parameter names for constants and unspecified arguments.
    /// For parameterized expressions, it returns the actual variable names.
    ///
    /// # Arguments
    /// * `counter` - A mutable reference to a counter for generating unique names
    ///
    /// # Returns
    /// A vector of variable names associated with this argument
    pub fn variables(&self, counter: &mut usize) -> Vec<String> {
        match self {
            Argument::Expression(e) => {
                if !e.is_parameterized() {
                    let out = vec![String::from(format!("unnamed_{counter}"))];
                    *counter += 1;
                    out
                } else {
                    e.get_unique_variables()
                }
            },
            _ => {
                let out = vec![String::from(format!("unnamed_{counter}"))];
                *counter += 1;
                out
            }
        }
    }

    /// Converts this argument into an expression.
    ///
    /// This method creates an `Expression` that can be used for parameter substitution.
    /// For expression arguments, it returns the expression directly. For other argument
    /// types, it generates a variable expression using the counter.
    ///
    /// # Arguments
    /// * `counter` - A mutable reference to a counter for generating unique variable names
    ///
    /// # Returns
    /// An `Expression` representing this argument for substitution purposes
    ///
    /// If a user specified this argument for a variable in a larger expression, the output
    /// from here would be what that variable get's substituted for.
    pub fn as_substitution_expression(&self, counter: &mut usize) -> Expression {
        match self {
            Argument::Expression(e) => e.clone(),
            _ => {
                let out = Expression::Variable(String::from(format!("unnamed_{counter}")));
                *counter += 1;
                out
            }
        }
    }
}

impl TryFrom<&str> for Argument {
    type Error = &'static str;

    /// Attempts to parse a string slice into an `Argument`.
    ///
    /// This implementation parses the string as a complex expression and extracts
    /// the real part if the expression is purely real.
    ///
    /// # Arguments
    /// * `value` - A string slice containing a mathematical expression
    ///
    /// # Returns
    /// * `Ok(Argument::Expression)` if the string represents a valid real expression
    /// * `Err` if the expression contains complex components
    ///
    /// # Examples
    /// ```dontrun
    /// let arg = Argument::try_from("x + 2.5").unwrap();
    /// let arg2 = Argument::try_from("sin(pi/4)").unwrap();
    /// ```
    fn try_from(value: &str) -> Result<Self, Self::Error> { 
        if value.contains("unnamed_") {
            return Err("Expression arguments cannot contain 'unnamed_'");
        }

        let parsed = ComplexExpression::from_string(value);

        if !parsed.is_real_fast() {
            Result::Err("Unable to handle complex parameter entries currently.")
        } else { 
            Ok(Argument::Expression(parsed.real))
        }
    }
}

impl TryFrom<String> for Argument {
    type Error = &'static str;

    /// Attempts to parse a String into an `Argument`.
    ///
    /// This implementation parses the string as a complex expression and extracts
    /// the real part if the expression is purely real.
    ///
    /// # Arguments
    /// * `value` - A String containing a mathematical expression
    ///
    /// # Returns
    /// * `Ok(Argument::Expression)` if the string represents a valid real expression
    /// * `Err` if the expression contains complex components
    fn try_from(value: String) -> Result<Self, Self::Error> { 
        if value.contains("unnamed_") {
            return Err("Expression arguments cannot contain 'unnamed_'");
        }

        let parsed = ComplexExpression::from_string(value);

        if !parsed.is_real_fast() {
            Result::Err("Unable to handle complex parameter entries currently.")
        } else { 
            Ok(Argument::Expression(parsed.real))
        }
    }
}

impl From<f32> for Argument {
    /// Converts a 32-bit float into an `Argument::Float32`.
    ///
    /// # Arguments
    /// * `value` - A 32-bit floating point number
    ///
    /// # Returns
    /// An `Argument::Float32` containing the given value
    fn from(value: f32) -> Self {
        Argument::Float32(value)
    }
}

impl From<f64> for Argument {
    /// Converts a 64-bit float into an `Argument::Float64`.
    ///
    /// # Arguments
    /// * `value` - A 64-bit floating point number
    ///
    /// # Returns
    /// An `Argument::Float64` containing the given value
    fn from(value: f64) -> Self {
        Argument::Float64(value)
    }
}

impl<T: TryInto<Argument>> TryFrom<Option<T>> for Argument {
    type Error = <T as TryInto<Argument>>::Error;

    /// Converts an `Option<T>` into an `Argument`.
    ///
    /// This implementation handles optional parameters where `None` becomes
    /// `Argument::Unspecified` and `Some(value)` attempts to convert the value
    /// into an `Argument`.
    ///
    /// # Arguments
    /// * `value` - An optional value that can be converted to an `Argument`
    ///
    /// # Returns
    /// * `Ok(Argument::Unspecified)` if the value is `None`
    /// * `Ok(Argument)` if the value converts successfully
    /// * `Err` if the value conversion fails
    fn try_from(value: Option<T>) -> Result<Self, Self::Error> {
        match value {
            None => Ok(Argument::Unspecified),
            Some(v) => v.try_into(),
        }
    }
}

#[cfg(feature = "python")]
mod python {
    use super::Argument;
    use pyo3::prelude::*;
    use pyo3::exceptions::{PyTypeError, PyValueError};

    impl<'py> FromPyObject<'py> for Argument {
        /// Converts a Python object to an `Argument`.
        /// 
        /// # Supported conversions
        /// - `None` → `Argument::Unspecified`
        /// - Python numbers (int/float) → `Argument::Float64`
        /// - Python strings → `Argument::Expression` (if valid)
        /// 
        /// # Errors
        /// Returns `PyValueError` for invalid expression strings or `PyTypeError` for unsupported types.
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            // Handle None -> Unspecified
            if ob.is_none() {
                return Ok(Argument::Unspecified);
            }
            
            // Try extracting as float (handles both Python float and int)
            if let Ok(val) = ob.extract::<f64>() {
                return Ok(Argument::Float64(val));
            }
            
            // Try extracting as string and parsing as expression
            if let Ok(val) = ob.extract::<String>() {
                match Argument::try_from(val) {
                    Ok(entry) => return Ok(entry),
                    Err(e) => return Err(PyValueError::new_err(e)),
                }
            }
            
            // If none of the above worked, return a type error
            Err(PyTypeError::new_err(format!(
                "Cannot convert {} to Argument", 
                ob.get_type().name()?
            )))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use pyo3::types::{PyFloat, PyInt, PyString, PyDict};
        use pyo3::Python;

        #[test]
        fn test_from_py_bound_none() {
            Python::initialize();
            Python::attach(|py| {
                let none = py.None().into_bound(py);
                let result = Argument::extract_bound(&none).unwrap();
                matches!(result, Argument::Unspecified);
            });
        }

        #[test]
        fn test_from_py_bound_int() {
            Python::initialize();
            Python::attach(|py| {
                let int_val = PyInt::new(py, 42);
                let result = Argument::extract_bound(&int_val).unwrap();
                matches!(result, Argument::Float64(42.0));
            });
        }

        #[test]
        fn test_from_py_bound_float() {
            Python::initialize();
            Python::attach(|py| {
                let float_val = PyFloat::new(py, 3.14);
                let result = Argument::extract_bound(&float_val).unwrap();
                matches!(result, Argument::Float64(3.14));
            });
        }

        #[test]
        fn test_from_py_bound_valid_string() {
            Python::initialize();
            Python::attach(|py| {
                let string_val = PyString::new(py, "x + 1");
                let result = Argument::extract_bound(&string_val).unwrap();
                matches!(result, Argument::Expression(_));
            });
        }

        #[test]
        fn test_from_py_bound_invalid_string() {
            Python::initialize();
            Python::attach(|py| {
                let string_val = PyString::new(py, "1 + i");
                let result = Argument::extract_bound(&string_val);
                assert!(result.is_err());
                assert!(result.unwrap_err().is_instance_of::<pyo3::exceptions::PyValueError>(py));
            });
        }

        #[test]
         fn test_from_py_bound_string_with_unnamed_rejected() {
             Python::initialize();
             Python::attach(|py| {
                 let string_val = PyString::new(py, "unnamed_123");
                 let result = Argument::extract_bound(&string_val);
                 assert!(result.is_err());
                 let err = result.unwrap_err();
                 assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                 assert!(err.to_string().contains("Expression arguments cannot contain 'unnamed_'"));
             });
         }

         #[test]
         fn test_from_py_bound_string_with_unnamed_in_expression_rejected() {
             Python::initialize();
             Python::attach(|py| {
                 let string_val = PyString::new(py, "x + unnamed_var");
                 let result = Argument::extract_bound(&string_val);
                 assert!(result.is_err());
                 let err = result.unwrap_err();
                 assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                 assert!(err.to_string().contains("Expression arguments cannot contain 'unnamed_'"));
             });
         }

         #[test]
         fn test_from_py_bound_string_similar_pattern_accepted() {
             Python::initialize();
             Python::attach(|py| {
                 let string_val = PyString::new(py, "named_var");
                 let result = Argument::extract_bound(&string_val);
                 assert!(result.is_ok());
                 matches!(result.unwrap(), Argument::Expression(_));
             });
         }

         #[test]
        fn test_from_py_bound_invalid_type() {
            Python::initialize();
            Python::attach(|py| {
                let dict_val = PyDict::new(py);
                let result = Argument::extract_bound(&dict_val);
                assert!(result.is_err());
                assert!(result.unwrap_err().is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qudit_expr::Expression;

    #[test]
    fn test_argument_unspecified_parameters() {
        let arg = Argument::Unspecified;
        let params = arg.parameters();
        assert_eq!(params.len(), 1);
        assert!(matches!(params[0], Parameter::Indexed));
    }

    #[test]
    fn test_argument_float32_parameters() {
        let arg = Argument::Float32(3.14);
        let params = arg.parameters();
        assert_eq!(params.len(), 1);
        assert!(matches!(params[0], Parameter::Constant32(3.14)));
    }

    #[test]
    fn test_argument_float64_parameters() {
        let arg = Argument::Float64(2.71);
        let params = arg.parameters();
        assert_eq!(params.len(), 1);
        assert!(matches!(params[0], Parameter::Constant64(2.71)));
    }

    #[test]
    fn test_argument_expression_constant_parameters() {
        let expr = Expression::from_float_64(1.5);
        let arg = Argument::Expression(expr);
        let params = arg.parameters();
        assert_eq!(params.len(), 1);
        assert!(matches!(params[0], Parameter::ConstantRatio(_)));
    }

    #[test]
    fn test_argument_expression_parameterized_parameters() {
        let expr = Expression::Variable("x".to_string());
        let arg = Argument::Expression(expr);
        let params = arg.parameters();
        assert_eq!(params.len(), 1);
        assert!(matches!(params[0], Parameter::Named(ref name) if name == "x"));
    }

    #[test]
    fn test_variables_unspecified() {
        let arg = Argument::Unspecified;
        let mut counter = 0;
        let vars = arg.variables(&mut counter);
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0], "unnamed_0");
        assert_eq!(counter, 1);
    }

    #[test]
    fn test_variables_float() {
        let arg = Argument::Float64(42.0);
        let mut counter = 5;
        let vars = arg.variables(&mut counter);
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0], "unnamed_5");
        assert_eq!(counter, 6);
    }

    #[test]
    fn test_variables_expression_constant() {
        let expr = Expression::from_float_64(2.0);
        let arg = Argument::Expression(expr);
        let mut counter = 10;
        let vars = arg.variables(&mut counter);
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0], "unnamed_10");
        assert_eq!(counter, 11);
    }

    #[test]
    fn test_variables_expression_parameterized() {
        let expr = Expression::Variable("theta".to_string());
        let arg = Argument::Expression(expr);
        let mut counter = 0;
        let vars = arg.variables(&mut counter);
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0], "theta");
        assert_eq!(counter, 0); // Counter should not increment for parameterized expressions
    }

    #[test]
    fn test_as_substitution_expression_expression() {
        let original_expr = Expression::Variable("phi".to_string());
        let arg = Argument::Expression(original_expr.clone());
        let mut counter = 0;
        let result = arg.as_substitution_expression(&mut counter);
        assert_eq!(result, original_expr);
        assert_eq!(counter, 0);
    }

    #[test]
    fn test_as_substitution_expression_non_expression() {
        let arg = Argument::Float32(1.0);
        let mut counter = 7;
        let result = arg.as_substitution_expression(&mut counter);
        assert_eq!(result, Expression::Variable("unnamed_7".to_string()));
        assert_eq!(counter, 8);
    }

    #[test]
    fn test_try_from_str_valid() {
        let result = Argument::try_from("x + 2");
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), Argument::Expression(_)));
    }

    #[test]
    fn test_try_from_str_complex_invalid() {
        let result = Argument::try_from("1 + i");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Unable to handle complex parameter entries currently.");
    }

    #[test]
    fn test_try_from_string_valid() {
        let result = Argument::try_from("sin(x)".to_string());
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), Argument::Expression(_)));
    }

    #[test]
    fn test_from_f32() {
        let arg = Argument::from(3.14f32);
        assert!(matches!(arg, Argument::Float32(3.14)));
    }

    #[test]
    fn test_from_f64() {
        let arg = Argument::from(2.718f64);
        assert!(matches!(arg, Argument::Float64(2.718)));
    }

    #[test]
    fn test_try_from_option_none() {
        let result = Argument::try_from(None::<f64>);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), Argument::Unspecified));
    }

    #[test]
    fn test_try_from_option_some() {
        let result = Argument::try_from(Some(1.5f64));
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), Argument::Float64(1.5)));
    }

    #[test]
    fn test_argument_clone() {
        let arg1 = Argument::Float64(42.0);
        let arg2 = arg1.clone();
        assert!(matches!(arg2, Argument::Float64(42.0)));
    }

    #[test]
    fn test_argument_debug() {
        let arg = Argument::Unspecified;
        let debug_str = format!("{:?}", arg);
        assert_eq!(debug_str, "Unspecified");
    }

    #[test]
     fn test_try_from_str_contains_unnamed_rejected() {
         let result = Argument::try_from("unnamed_");
         assert!(result.is_err());
         assert_eq!(result.unwrap_err(), "Expression arguments cannot contain 'unnamed_'");
     }

     #[test]
     fn test_try_from_str_contains_unnamed_with_suffix_rejected() {
         let result = Argument::try_from("unnamed_123");
         assert!(result.is_err());
         assert_eq!(result.unwrap_err(), "Expression arguments cannot contain 'unnamed_'");
     }

     #[test]
     fn test_try_from_str_contains_unnamed_in_expression_rejected() {
         let result = Argument::try_from("x + unnamed_var");
         assert!(result.is_err());
         assert_eq!(result.unwrap_err(), "Expression arguments cannot contain 'unnamed_'");
     }

     #[test]
     fn test_try_from_str_contains_unnamed_in_middle_rejected() {
         let result = Argument::try_from("prefix_unnamed_suffix");
         assert!(result.is_err());
         assert_eq!(result.unwrap_err(), "Expression arguments cannot contain 'unnamed_'");
     }

     #[test]
     fn test_try_from_str_multiple_unnamed_rejected() {
         let result = Argument::try_from("unnamed_1 + unnamed_2");
         assert!(result.is_err());
         assert_eq!(result.unwrap_err(), "Expression arguments cannot contain 'unnamed_'");
     }

     #[test]
     fn test_try_from_string_contains_unnamed_rejected() {
         let result = Argument::try_from("unnamed_".to_string());
         assert!(result.is_err());
         assert_eq!(result.unwrap_err(), "Expression arguments cannot contain 'unnamed_'");
     }

     #[test]
     fn test_try_from_string_contains_unnamed_with_suffix_rejected() {
         let result = Argument::try_from("unnamed_456".to_string());
         assert!(result.is_err());
         assert_eq!(result.unwrap_err(), "Expression arguments cannot contain 'unnamed_'");
     }

     #[test]
     fn test_try_from_string_contains_unnamed_in_expression_rejected() {
         let result = Argument::try_from("sin(unnamed_theta)".to_string());
         assert!(result.is_err());
         assert_eq!(result.unwrap_err(), "Expression arguments cannot contain 'unnamed_'");
     }

     #[test]
     fn test_try_from_str_similar_patterns_accepted() {
         // These should work because they don't contain the exact "unnamed_" pattern
         let result1 = Argument::try_from("named_var");
         assert!(result1.is_ok());

         let result2 = Argument::try_from("unnamedvar"); // no underscore
         assert!(result2.is_ok());

         let result3 = Argument::try_from("unnamed"); // no underscore
         assert!(result3.is_ok());

         let result4 = Argument::try_from("my_unnamed"); // doesn't start with unnamed_
         assert!(result4.is_ok());
     }

     #[test]
     fn test_try_from_string_similar_patterns_accepted() {
         // These should work because they don't contain the exact "unnamed_" pattern
         let result1 = Argument::try_from("named_123".to_string());
         assert!(result1.is_ok());

         let result2 = Argument::try_from("unnamedvariable".to_string());
         assert!(result2.is_ok());

         let result3 = Argument::try_from("var_name".to_string());
         assert!(result3.is_ok());
     }

     #[test]
     fn test_try_from_str_case_sensitive_unnamed_check() {
         // Uppercase variants should be accepted since the check is case-sensitive
         let result1 = Argument::try_from("UNNAMED_var");
         assert!(result1.is_ok());

         let result2 = Argument::try_from("Unnamed_123");
         assert!(result2.is_ok());
     }

     #[test]
    fn test_multivariate_expression_parameters() {
        let result = Argument::try_from("a*c").unwrap();
        if let Argument::Expression(expr) = result {
            let arg = Argument::Expression(expr);
            let params = arg.parameters();
            assert_eq!(params.len(), 2);
            
            // Check that we get Named parameters for both variables
            let param_names: Vec<String> = params.into_iter().map(|p| match p {
                Parameter::Named(name) => name,
                _ => panic!("Expected Named parameter"),
            }).collect();
            
            // Should contain both "a" and "c"
            assert!(param_names.contains(&"a".to_string()));
            assert!(param_names.contains(&"c".to_string()));
        } else {
            panic!("Expected Expression variant");
        }
    }

    #[test]
    fn test_multivariate_expression_variables() {
        let result = Argument::try_from("a*c").unwrap();
        if let Argument::Expression(expr) = result {
            let arg = Argument::Expression(expr);
            let mut counter = 0;
            let vars = arg.variables(&mut counter);
            
            // Should return the actual variable names, not generated ones
            assert_eq!(vars.len(), 2);
            assert!(vars.contains(&"a".to_string()));
            assert!(vars.contains(&"c".to_string()));
            assert_eq!(counter, 0); // Counter should not increment for parameterized expressions
        } else {
            panic!("Expected Expression variant");
        }
    }

    #[test]
    fn test_multivariate_expression_substitution() {
        let result = Argument::try_from("a*c").unwrap();
        if let Argument::Expression(original_expr) = &result {
            let mut counter = 0;
            let substitution_expr = result.as_substitution_expression(&mut counter);
            
            // Should return the same expression for parameterized expressions
            assert_eq!(substitution_expr, *original_expr);
            assert_eq!(counter, 0); // Counter should not increment
        } else {
            panic!("Expected Expression variant");
        }
    }

    #[test]
    fn test_complex_multivariate_expression() {
        let result = Argument::try_from("x + y*z - sin(theta)").unwrap();
        if let Argument::Expression(expr) = result {
            let arg = Argument::Expression(expr);
            let params = arg.parameters();
            
            // Should get Named parameters for all variables
            assert!(params.len() >= 3); // At least x, y, z, theta
            
            let param_names: Vec<String> = params.into_iter().map(|p| match p {
                Parameter::Named(name) => name,
                _ => panic!("Expected Named parameter for complex expression"),
            }).collect();
            
            // Check for expected variables
            assert!(param_names.contains(&"x".to_string()));
            assert!(param_names.contains(&"y".to_string()));
            assert!(param_names.contains(&"z".to_string()));
            assert!(param_names.contains(&"theta".to_string()));
        } else {
            panic!("Expected Expression variant");
        }
    }

    #[test]
    fn test_multivariate_expression_variables_ordering() {
        let result = Argument::try_from("b + a").unwrap(); // Test alphabetical vs appearance order
        if let Argument::Expression(expr) = result {
            let arg = Argument::Expression(expr);
            let mut counter = 0;
            let vars = arg.variables(&mut counter);
            
            assert_eq!(vars.len(), 2);
            assert!(vars.contains(&"a".to_string()));
            assert!(vars.contains(&"b".to_string()));
            assert_eq!(counter, 0);
        } else {
            panic!("Expected Expression variant");
        }
    }
}
