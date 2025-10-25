use crate::KetExpression;

/// A terminating z-basis measurement for a qudit.
#[cfg_attr(feature = "python", pyo3::pyfunction)]
#[cfg_attr(feature = "python", pyo3(signature = (radix = 2)))]
pub fn ZeroState(radix: usize) -> KetExpression {
    let proto = format!("Zero<{}>()", radix);
    let mut body = "".to_string();
    body += "[";
    for i in 0..radix {
        body += "[";
        if i == 0 {
            body += "1,";
        } else {
            body += "0,";
        }
        body += "],";
    }
    body += "]";

    KetExpression::new(proto + "{" + &body + "}")
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;
    use crate::python::PyExpressionRegistrar;

    /// Registers the measurement library with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_function(wrap_pyfunction!(ZeroState, parent_module)?)?;
        Ok(())
    }
    inventory::submit!(PyExpressionRegistrar { func: register });
}
