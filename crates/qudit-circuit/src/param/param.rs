use qudit_expr::Constant;

/// Represents different types of parameters that can be used in a quantum circuit.
/// These parameters can be assigned constant values or be dynamic and resolved later.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Parameter {
    /// A parameter with a set 32-bit floating-point value.
    Constant32(f32),

    /// A parameter with a set 64-bit floating-point value.
    Constant64(f64),

    /// A parameter with a set rational value, represented by `qudit_expr::Constant`.
    ConstantRatio(Constant),

    /// Unnamed variable parameter; can be identified only by index.
    Indexed,

    /// Named variable parameter; can be identified by name or index.
    Named(String),
}

impl Parameter {
    pub fn is_constant(&self) -> bool {
        match self {
            Parameter::Constant32(_) => true,
            Parameter::Constant64(_) => true,
            Parameter::ConstantRatio(_) => true,
            Parameter::Indexed => false,
            Parameter::Named(_) => false,
        }
    }
}

#[cfg(feature = "python")]
mod python {
    use super::Parameter;
    use pyo3::prelude::*;
    use pyo3::types::{PyFloat, PyString};
    use num::ToPrimitive;

    impl<'py> IntoPyObject<'py> for Parameter {
        type Target = PyAny;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            match self {
                Parameter::Constant32(value) => Ok(PyFloat::new(py, value as f64).into_any()),
                Parameter::Constant64(value) => Ok(PyFloat::new(py, value).into_any()),
                Parameter::ConstantRatio(constant) => {
                    // Convert rational to float representation
                    let float_val = constant.to_f64().unwrap();
                    Ok(PyFloat::new(py, float_val).into_any())
                }
                Parameter::Indexed => {
                    Ok(py.None().into_bound(py))
                }
                Parameter::Named(name) => {
                    Ok(PyString::new(py, &name).into_any())
                }
            }
        }
    }

    impl<'py> FromPyObject<'py> for Parameter {
        fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
            // Check for None first (represents indexed parameters)
            if obj.is_none() {
                return Ok(Parameter::Indexed);
            }

            // Try to extract as a float first (most common case)
            if let Ok(float_val) = obj.extract::<f64>() {
                return Ok(Parameter::Constant64(float_val));
            }

            // Try to extract as a string (for named parameters passed as plain strings)
            if let Ok(string_val) = obj.extract::<String>() {
                return Ok(Parameter::Named(string_val));
            }

            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Cannot convert Python object to Parameter"
            ))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use pyo3::types::PyFloat;
        use qudit_expr::Constant;

        #[test]
        fn test_into_pyobject_constant32() {
            Python::initialize();
            Python::attach(|py| {
                let param = Parameter::Constant32(3.14f32);
                let py_obj = param.into_pyobject(py).unwrap();
                let value: f64 = py_obj.extract().unwrap();
                assert!((value - 3.14f64).abs() < 1e-6);
            });
        }

        #[test]
        fn test_into_pyobject_constant64() {
            Python::initialize();
            Python::attach(|py| {
                let param = Parameter::Constant64(2.71828);
                let py_obj = param.into_pyobject(py).unwrap();
                let value: f64 = py_obj.extract().unwrap();
                assert!((value - 2.71828).abs() < 1e-10);
            });
        }

        #[test]
        fn test_into_pyobject_constant_ratio() {
            Python::initialize();
            Python::attach(|py| {
                let constant = Constant::new(1.into(), 2.into()); // 0.5
                let param = Parameter::ConstantRatio(constant);
                let py_obj = param.into_pyobject(py).unwrap();
                let value: f64 = py_obj.extract().unwrap();
                assert!((value - 0.5).abs() < 1e-10);
            });
        }

        #[test]
        fn test_into_pyobject_indexed() {
            Python::initialize();
            Python::attach(|py| {
                let param = Parameter::Indexed;
                let py_obj = param.into_pyobject(py).unwrap();
                assert!(py_obj.is_none());
            });
        }

        #[test]
        fn test_into_pyobject_named() {
            Python::initialize();
            Python::attach(|py| {
                let param = Parameter::Named("theta".to_string());
                let py_obj = param.into_pyobject(py).unwrap();
                let value: String = py_obj.extract().unwrap();
                assert_eq!(value, "theta");
            });
        }

        #[test]
        fn test_from_pyobject_none() {
            Python::initialize();
            Python::attach(|py| {
                let py_none = py.None();
                let param: Parameter = py_none.extract(py).unwrap();
                assert_eq!(param, Parameter::Indexed);
            });
        }

        #[test]
        fn test_from_pyobject_float() {
            Python::initialize();
            Python::attach(|py| {
                let py_float = PyFloat::new(py, 1.234);
                let param: Parameter = py_float.extract().unwrap();
                if let Parameter::Constant64(value) = param {
                    assert!((value - 1.234).abs() < 1e-10);
                } else {
                    panic!("Expected Parameter::Constant64, got {:?}", param);
                }
            });
        }

        #[test]
        fn test_from_pyobject_string() {
            Python::initialize();
            Python::attach(|py| {
                let py_str = PyString::new(py, "alpha");
                let param: Parameter = py_str.extract().unwrap();
                if let Parameter::Named(name) = param {
                    assert_eq!(name, "alpha");
                } else {
                    panic!("Expected Parameter::Named, got {:?}", param);
                }
            });
        }

        #[test]
        fn test_from_pyobject_invalid_type() {
            Python::initialize();
            Python::attach(|py| {
                let py_list = pyo3::types::PyList::empty(py);
                let result: PyResult<Parameter> = py_list.extract();
                assert!(result.is_err());
                let err = result.unwrap_err();
                assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            });
        }

        #[test]
        fn test_roundtrip_conversion() {
            Python::initialize();
            Python::attach(|py| {
                // Test all parameter types for roundtrip conversion
                let params = vec![
                    Parameter::Constant32(3.14f32),
                    Parameter::Constant64(2.71828),
                    Parameter::ConstantRatio(Constant::new(1.into(), 3.into())),
                    Parameter::Indexed,
                    Parameter::Named("beta".to_string()),
                ];

                for original_param in params {
                    let py_obj = original_param.clone().into_pyobject(py).unwrap();
                    let converted_param: Parameter = py_obj.extract().unwrap();
                    
                    match (&original_param, &converted_param) {
                        (Parameter::Constant32(a), Parameter::Constant64(b)) => {
                            // f32 -> f64 conversion expected
                            assert!((*a as f64 - *b).abs() < 1e-6);
                        }
                        (Parameter::ConstantRatio(_), Parameter::Constant64(_)) => {
                            // Ratio -> f64 conversion expected
                            // Just verify it's the correct type
                        }
                        _ => {
                            assert_eq!(original_param, converted_param);
                        }
                    }
                }
            });
        }
    }
}
