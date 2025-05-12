use pyo3::prelude::*;
use qudit_expr::UnitaryExpression;
use numpy::PyArray2;

#[pyclass]
struct PyUnitaryExpression {
    expr: UnitaryExpression,
}

#[pymethods]
impl PyUnitaryExpression {
    #[new]
    fn new(expr: String) -> Self {
        Self {
            expr: UnitaryExpression::from(expr),
        }
    }

    fn evaluate(&self, args: Vec<f64>) -> PyResult<PyArray2<f64>> {
        self.expr.eval(&args)
            .map(|result| {
                let gil = Python::acquire_gil();
                let py = gil.python();
                let shape = (result.len(), result[0].len());
                let array = PyArray2::zeros(py, shape, false);
                for (i, row) in result.iter().enumerate() {
                    for (j, &value) in row.iter().enumerate() {
                        array[[i, j]] = value;
                    }
                }
                array
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))
    }
}


/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn openqudit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUnitaryExpression>()?;
    Ok(())
}
