mod argument;
mod list;
mod param;
mod vector;

pub use argument::Argument;
pub use list::ArgumentList;
pub use param::Parameter;
pub use vector::ParameterVector;

#[derive(Clone, PartialEq)]
pub enum NameOrParameter {
    Name(String),
    Parameter(Parameter),
}

#[derive(Clone, PartialEq)]
pub enum Value {
    F32(f32),
    F64(f64),
    Ratio(qudit_expr::Constant),
}

impl From<Value> for Parameter {
    fn from(value: Value) -> Self {
        match value {
            Value::F32(f) => Parameter::Assigned32(f),
            Value::F64(f) => Parameter::Assigned64(f),
            Value::Ratio(f) => Parameter::AssignedRatio(f),
        }
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Value::F32(value)
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::F64(value)
    }
}

impl From<qudit_expr::Constant> for Value {
    fn from(value: qudit_expr::Constant) -> Self {
        Value::Ratio(value)
    }
}

#[cfg(feature = "python")]
mod python {
    use super::Value;
    use pyo3::exceptions::PyTypeError;
    use pyo3::types::PyFloat;
    use pyo3::prelude::*;
    use num::ToPrimitive;

    impl<'a, 'py> FromPyObject<'a, 'py> for Value {
        type Error = PyErr;
        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            if let Ok(val) = obj.extract::<f64>() {
                return Ok(Value::F64(val));
            }

            // TODO: extract String and parse as expression into ratio

            // If none of the above worked, return a type error
            Err(PyTypeError::new_err(format!(
                "Cannot convert {} to Parameter Value",
                obj.get_type().name()?
            )))
        }
    }

    impl<'py> IntoPyObject<'py> for Value {
        type Target = PyAny;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            match self {
                Value::F32(value) => Ok(PyFloat::new(py, value as f64).into_any()),
                Value::F64(value) => Ok(PyFloat::new(py, value).into_any()),
                Value::Ratio(constant) => {
                    // Convert rational to float representation
                    let float_val = constant.to_f64().unwrap();
                    Ok(PyFloat::new(py, float_val).into_any())
                }
            }
        }
    }
}
