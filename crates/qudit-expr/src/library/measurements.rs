use crate::BraSystemExpression;

/// A terminating z-basis measurement for a qudit.
#[cfg_attr(feature = "python", pyo3::pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (radix = 2)))]
pub fn ZMeasurement(radix: usize) -> BraSystemExpression {
    let proto = format!("ZMeasurement<{}>()", radix);
    let mut body = "".to_string();
    body += "[";
    for i in 0..radix {
        body += "[[";
        for j in 0..radix {
            if i == j {
                body += "1,";
            } else {
                body += "0,";
            }
        }
        body += "]],";
    }
    body += "]";

    BraSystemExpression::new(proto + "{" + &body + "}")
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use crate::python::PyExpressionRegistrar;
    use pyo3::prelude::*;

    /// Registers the measurement library with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_function(wrap_pyfunction!(ZMeasurement, parent_module)?)?;
        Ok(())
    }
    inventory::submit!(PyExpressionRegistrar { func: register });
}
