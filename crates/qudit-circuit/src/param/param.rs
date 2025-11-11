use num::ToPrimitive;
use qudit_core::RealScalar;
use qudit_expr::Constant;

/// Either assigned or un-assigned parameters in a quantum circuit.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Parameter {
    /// A parameter with a set 32-bit floating-point value.
    Assigned32(f32),

    /// A parameter with a set 64-bit floating-point value.
    Assigned64(f64),

    /// A parameter with a set rational value, represented by `qudit_expr::Constant`.
    AssignedRatio(Constant),

    /// An unassigned variable parameter.
    Unassigned,
}

impl Parameter {
    pub fn extract_float<R: RealScalar>(&self) -> Option<R> {
        match self {
            Self::Assigned32(val) => Some(R::from32(*val)),
            Self::Assigned64(val) => Some(R::from64(*val)),
            Self::AssignedRatio(val) => val.to_f64().map(|f| R::from64(f)),
            Self::Unassigned => None,
        }
    }

    pub fn is_assigned(&self) -> bool {
        !matches!(self, Parameter::Unassigned)
    }
}

#[cfg(feature = "python")]
mod python {
    use super::Parameter;
    use num::ToPrimitive;
    use pyo3::prelude::*;
    use pyo3::types::PyFloat;

    impl<'py> IntoPyObject<'py> for Parameter {
        type Target = PyAny;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            match self {
                Parameter::Assigned32(value) => Ok(PyFloat::new(py, value as f64).into_any()),
                Parameter::Assigned64(value) => Ok(PyFloat::new(py, value).into_any()),
                Parameter::AssignedRatio(constant) => {
                    // Convert rational to float representation
                    let float_val = constant.to_f64().unwrap();
                    Ok(PyFloat::new(py, float_val).into_any())
                }
                Parameter::Unassigned => Ok(py.None().into_bound(py)),
            }
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for Parameter {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            if obj.is_none() {
                return Ok(Parameter::Unassigned);
            }

            if let Ok(float_val) = obj.extract::<f64>() {
                return Ok(Parameter::Assigned64(float_val));
            }

            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Cannot convert Python object to Parameter",
            ))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use pyo3::types::PyFloat;
        use qudit_expr::Constant;

        #[test]
        fn test_into_pyobject_assigned32() {
            Python::initialize();
            Python::attach(|py| {
                let param = Parameter::Assigned32(3.14f32);
                let py_obj = param.into_pyobject(py).unwrap();
                let value: f64 = py_obj.extract().unwrap();
                assert!((value - 3.14f64).abs() < 1e-6);
            });
        }

        #[test]
        fn test_into_pyobject_assigned64() {
            Python::initialize();
            Python::attach(|py| {
                let param = Parameter::Assigned64(2.71828);
                let py_obj = param.into_pyobject(py).unwrap();
                let value: f64 = py_obj.extract().unwrap();
                assert!((value - 2.71828).abs() < 1e-10);
            });
        }

        #[test]
        fn test_into_pyobject_assigned_ratio() {
            Python::initialize();
            Python::attach(|py| {
                let constant = Constant::new(1.into(), 2.into()); // 0.5
                let param = Parameter::AssignedRatio(constant);
                let py_obj = param.into_pyobject(py).unwrap();
                let value: f64 = py_obj.extract().unwrap();
                assert!((value - 0.5).abs() < 1e-10);
            });
        }

        #[test]
        fn test_into_pyobject_unassigned() {
            Python::initialize();
            Python::attach(|py| {
                let param = Parameter::Unassigned;
                let py_obj = param.into_pyobject(py).unwrap();
                assert!(py_obj.is_none());
            });
        }

        #[test]
        fn test_from_pyobject_none() {
            Python::initialize();
            Python::attach(|py| {
                let py_none = py.None();
                let param: Parameter = py_none.extract(py).unwrap();
                assert_eq!(param, Parameter::Unassigned);
            });
        }

        #[test]
        fn test_from_pyobject_float() {
            Python::initialize();
            Python::attach(|py| {
                let py_float = PyFloat::new(py, 1.234);
                let param: Parameter = py_float.extract().unwrap();
                if let Parameter::Assigned64(value) = param {
                    assert!((value - 1.234).abs() < 1e-10);
                } else {
                    panic!("Expected Parameter::Assigned64, got {:?}", param);
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
                    Parameter::Assigned32(3.14f32),
                    Parameter::Assigned64(2.71828),
                    Parameter::AssignedRatio(Constant::new(1.into(), 3.into())),
                    Parameter::Unassigned,
                ];

                for original_param in params {
                    let py_obj = original_param.clone().into_pyobject(py).unwrap();
                    let converted_param: Parameter = py_obj.extract().unwrap();

                    match (&original_param, &converted_param) {
                        (Parameter::Assigned32(a), Parameter::Assigned64(b)) => {
                            // f32 -> f64 conversion expected
                            assert!((*a as f64 - *b).abs() < 1e-6);
                        }
                        (Parameter::AssignedRatio(_), Parameter::Assigned64(b)) => {
                            // Ratio -> f64 conversion expected
                            assert!((*b - (1.0/3.0)).abs() < 1e-6)
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
