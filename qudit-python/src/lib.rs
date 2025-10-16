use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
// use qudit_circuit::CircuitLocation;
// use qudit_circuit::ExpressionOperation;
// use qudit_circuit::Operation;
// use qudit_circuit::ParamEntry;
// use qudit_circuit::QuditCircuit;
use qudit_expr::UnitaryExpression;
use numpy::PyArray2;
use numpy::PyArrayMethods;
use ndarray::ArrayViewMut2;
use qudit_core::QuditSystem;
use qudit_core::c64;

// #[pyclass]
// #[pyo3(name = "UnitaryExpression")]
// struct PyUnitaryExpression {
//     expr: UnitaryExpression,
// }

// #[pymethods]
// impl PyUnitaryExpression {
//     #[new]
//     fn new(expr: String) -> Self {
//         Self {
//             expr: UnitaryExpression::new(expr),
//         }
//     }

//     #[pyo3(signature = (*args))]
//     fn __call__<'py>(&self, args: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyArray2<c64>>> {
//         let py = args.py();
//         let args: Vec<f64> = args.extract()?;
//         let unitary = self.expr.eval(&args);
//         let py_array: Bound<'py, PyArray2<c64>> = PyArray2::zeros(py, (unitary.dimension(), unitary.dimension()), false);

//         {
//             let mut readwrite = py_array.readwrite();
//             let mut py_array_view: ArrayViewMut2<c64> = readwrite.as_array_mut();

//             for (j, col) in unitary.col_iter().enumerate() {
//                 for (i, val) in col.iter().enumerate() {
//                     py_array_view[[i, j]] = *val;
//                 }
//             }
//         }

//         Ok(py_array)
//     }

//     fn num_params(&self) -> usize {
//         self.expr.num_params()
//     }
// }

// impl From<UnitaryExpression> for PyUnitaryExpression {
//     fn from(value: UnitaryExpression) -> Self {
//         PyUnitaryExpression {
//             expr: value,
//         }
//     }
// }

// /// Helper function to parse a Python object that can be
// /// either an integer or an iterable of integers.
// fn parse_int_or_iterable<'py>(input: &Bound<'py, PyAny>) -> PyResult<Vec<usize>> {
//     // First, try to extract the input as a single integer.
//     if let Ok(val) = input.extract::<usize>() {
//         // If successful, create a Vec of that length filled with the value 2.
//         Ok(vec![2; val])
//     } else {
//         // If it's not an integer, try to treat it as an iterable.
//         // This will raise a TypeError if the object is not iterable
//         // or its elements are not integers.
//         pyo3::types::PyIterator::from_object(input)?
//             .map(|item| item?.extract::<usize>())
//             .collect::<PyResult<Vec<usize>>>()
//     }
// }

// #[pyclass(unsendable)]
// #[pyo3(name = "QuditCircuit")]
// struct PyQuditCircuit {
//     circuit: QuditCircuit
// }

// #[pymethods]
// impl PyQuditCircuit {
//     /// Creates a new QuditCircuit instance.
//     ///
//     /// Args:
//     ///     qudits (int | Iterable[int]): An integer specifying number of qudits 
//     ///         or an iterable of qudit radices.
//     ///     dits (int | Iterable[int] | None): An integer, an iterable, or None.
//     ///         Defaults to None, which results in no classical dits.
//     #[new]
//     #[pyo3(signature = (qudits, dits = None))]
//     fn new<'py>(qudits: &Bound<'py, PyAny>, dits: Option<&Bound<'py, PyAny>>) -> PyResult<PyQuditCircuit> {
//         let qudits_vec = parse_int_or_iterable(qudits)?;

//         let dits_vec = match dits {
//             Some(pyany) => parse_int_or_iterable(pyany)?,
//             None => Vec::new(),
//         };

//         Ok(PyQuditCircuit {
//             circuit: QuditCircuit::new(qudits_vec, dits_vec),
//         })
//     }

//     // --- Properties ---

//     /// Returns the number of parameters in the circuit.
//     #[getter]
//     fn num_params(&self) -> PyResult<usize> {
//         Ok(self.circuit.num_params())
//     }

//     /// Returns the number of operations in the circuit.
//     #[getter]
//     fn num_operations(&self) -> PyResult<usize> {
//         Ok(self.circuit.num_operations())
//     }

//     /// Returns the number of cycles in the circuit.
//     #[getter]
//     fn num_cycles(&self) -> PyResult<usize> {
//         Ok(self.circuit.num_cycles())
//     }

//     #[getter]
//     fn is_empty(&self) -> PyResult<bool> {
//         Ok(self.circuit.is_empty())
//     }

//     #[getter]
//     fn active_qudits(&self) -> PyResult<Vec<usize>> {
//         Ok(self.circuit.active_qudits())
//     }

//     #[getter]
//     fn active_dits(&self) -> PyResult<Vec<usize>> {
//         Ok(self.circuit.active_dits())
//     }

//     // @property def params(self) -> Parameters?
//     // @property def coupling_graph(self) -> CouplingGraph?
//     // @property def gate_set(self) -> GateSet?

//     // Metrics
//     //
//     // def depth(self, *, filter: Optional[Callable] = None, recursive: bool = True) -> int:
//     // def parallelism(self) -> float:
//     // def gate_counts(self, *, filter: Optional[Callable] = None, recursive: bool = True) -> int:
//     //
//     // Qudit Methods
//     //
//     // def append_qudit(self, radix: int = 2) -> None:
//     // def extend_qudits(self, radixes: Iterable[int]) -> None:
//     // def insert_qudit(self, qudit_index: int, radix: int = 2) -> None:
//     // def pop_qudit(self, qudit_index: int) -> None:
//     // def is_qudit_in_range(self, qudit_index: int) -> bool:
//     // def is_qudit_idle(self, qudit_index: int) -> bool:
//     // def renumber_qudits(self, qudit_permutation: Iterable[int]) -> None:
//     //
//     // Cycle Methods
//     //
//     // def pop_cycle(self, cycle_index: int) -> None?
//     //
//     // DAG Methods:
//     //
//     // @property def front(self) -> set[CircuitPoint]:
//     // @property def rear(self) -> set[CircuitPoint]:
//     // @property def first_on(self) -> CircuitPoint | None:
//     // @property def last_on(self) -> CircuitPoint | None:
//     // def next(self, current: CircuitPointLike | CircuitRegionLike, /) -> set[CircuitPoint]:
//     // def prev(self, current: CircuitPointLike | CircuitRegionLike, /) -> set[CircuitPoint]:
//     //
//     // At Methods:
//     //
//     // def get_operation(self, point: CircuitPointLike) -> Operation:
//     // 
//     // def point(
//     //     self,
//     //     op: Operation | Gate,
//     //     start: CircuitPointLike = (0, 0),
//     //     end: CircuitPointLike | None = None,
//     // ) -> CircuitPoint:
//     //
//     // def append(self, op: Operation) -> int:

//     #[pyo3(signature = (op, loc, params = None))]
//     pub fn append<'py>(
//         &mut self,
//         op: &Bound<'py, PyAny>,
//         loc: &Bound<'py, PyAny>,
//         params: Option<&Bound<'py, PyAny>>,
//     ) -> PyResult<()> {

//         // 1. Parse 'op' as either PyUnitaryExpression or PyQuditCircuit
//         let parsed_op = if let Ok(unitary_expr) = op.extract::<PyRef<PyUnitaryExpression>>() {
//             Operation::Expression(ExpressionOperation::UnitaryGate(unitary_expr.expr.clone()))
//         } else {
//             return Err(PyTypeError::new_err(
//                 "Unrecognized operation type.",
//             ));
//         };

//         let num_qudits = parsed_op.num_qudits();

//         // 2. Parse 'loc' as an int, iterable of ints, or tuple of iterables
//         let parsed_loc = if let Ok(single_loc) = loc.extract::<usize>() {
//             CircuitLocation::pure(&[single_loc])
//         } else if let Ok(tuple) = loc.downcast::<PyTuple>() {
//             if tuple.len() == 2 {
//                 let item0 = tuple.get_item(0)?;
//                 let item1 = tuple.get_item(1)?;

//                 // Try parsing as (iterable, iterable)
//                 if let (Ok(vec0), Ok(vec1)) = (item0.extract::<Vec<usize>>(), item1.extract::<Vec<usize>>()) {
//                     CircuitLocation::new(vec0, vec1)
//                 } 
//                 // Else, try parsing as (int, int)
//                 else if let (Ok(int0), Ok(int1)) = (item0.extract::<usize>(), item1.extract::<usize>()) {
//                     if num_qudits.is_some() && num_qudits == Some(1) {
//                         CircuitLocation::new(vec![int0], vec![int1])
//                     } else {
//                         CircuitLocation::pure(&[int0, int1])
//                     }
//                 } else {
//                     return Err(PyTypeError::new_err("A 2-element 'loc' tuple must contain (int, int) or (iterable, iterable)"));
//                 }
//             } else {
//                 let mut qudit_register = tuple.extract::<Vec<usize>>()?;
//                 let dit_register = qudit_register.split_off(num_qudits.unwrap());
//                 CircuitLocation::new(qudit_register, dit_register)
//             }
//         } else if let Ok(list_loc) = loc.extract::<Vec<usize>>() {
//             let mut qudit_register = list_loc.clone();
//             let dit_register = qudit_register.split_off(num_qudits.unwrap());
//             CircuitLocation::new(qudit_register, dit_register)
//         } else {
//             return Err(PyTypeError::new_err(
//                 "Argument 'loc' must be an int, an iterable of ints, or a tuple of two iterables of ints",
//             ));
//         };

//         let parsed_params: PyResult<Option<Vec<ParamEntry>>> = match params {
//             // Case 1: The user provided the `params` argument.
//             Some(p) => {
//                 let iter = pyo3::types::PyIterator::from_object(p)?;
//                 let result_vec = iter.map(|item_result| {
//                     let item = item_result?;

//                     if item.is_none() {
//                         // CORRECT way to check for Python's `None`.
//                         Ok(ParamEntry::new_indexed())
//                     } else if let Ok(float_val) = item.extract::<f64>() {
//                         Ok(ParamEntry::new_constant(float_val))
//                     } else if let Ok(str_val) = item.extract::<String>() {
//                         Ok(ParamEntry::new(str_val))
//                     } else {
//                         Err(PyTypeError::new_err(
//                             "All elements in `params` must be a float, string, or None.",
//                         ))
//                     }
//                 }).collect::<PyResult<Vec<ParamEntry>>>()?;

//                 Ok(Some(result_vec))
//             }
//             // Case 2: The user did not provide `params`.
//             None => Ok(None),
//         };

//         self.circuit.append(parsed_op, parsed_loc, parsed_params?);
//         Ok(())
//     }

//     // def append_gate
//     // def append_circuit
//     // def extend
//     // def insert
//     // def insert_gate
//     // def insert_circuit
//     // def remove(op)
//     // def remove_all
//     // def count
//     // def pop(point)
//     // def batch_pop(points)
//     // def replace(point, op)
//     // def replace_all(op, op)
//     // def batch_replace(points, ops)
//     // def replace_gate
//     // def replace_with_circuit(... as circuit_gate = False)
//     //
//     // But like also, initializations, barriers, measurements, kraus ops, classical controls
//     //
//     // Movement/Ownership
//     //
//     // def copy()
//     // def become()
//     // def clear()
//     //
//     // Parameter Methods?
//     //
//     // def un-constant?
//     // def freeze
//     // Specified vs Constant appends are funky, A user should edit their expression
//     // before hand if they want to hardcode a constant into it, otherwise, circuits
//     // should track constant parameters, whether they are added as specified or const.
//     //
//     // User's should have ability to safely `def freeze(self, parameter_index: int, value: Constant)` and
//     // `def thaw(self, parameter_index: int)`
//     //
//     // For this, might need to add a new type of Parameter => Frozen({value: Constant, name:
//     // Option<String>}), when a named parameter is frozen, the name is stored for future thaws
//     // Also, if I add an expression with the same parameter, it should also be frozen.
//     // 
//     // Advanced Algorithms
//     //
//     // def compress()
//     // def surround() // Need to seriously think about subcircuits/regions/grouping
//     // def invert()
//     // def evaluate(args)
//     // def evaluate_gradient(args)
//     // def evaluate_hessian(args)
//     // def instantiate(...)
//     //
//     // dunder methods
//     //
//     // __getitem__
//     // __iter__
//     // __reversed__
//     // __contains__
//     // __len__
//     // __invert__
//     // __eq__
//     // __ne__
//     // __add__
//     // __mul__
//     // __radd__
//     // __iadd__
//     // __imul__
//     // __str__
//     // __repr__
//     // operations/operations_with_cycles
//     //
//     // IO
//     //
//     // save
//     // to
//     // from_file
//     // from_unitary
//     // __reduce__
//     // rebuild_circuit
// }

// use qudit_expr::ExpressionGenerator;
// use qudit_gates::XGate;
// use qudit_gates::ParameterizedUnitary;
// use qudit_core::QuditRadices;

// #[pyfunction]
// #[pyo3(name = "XGate")]
// #[pyo3(signature = (radix = 2))]
// fn pyxgate(radix: usize) -> PyUnitaryExpression {
//     XGate::new(radix).generate_expression().into()
// }

// #[pyfunction]
// #[pyo3(name = "ParameterizedUnitary")]
// #[pyo3(signature = (radices = QuditRadices::new(&[2])))]
// fn pyugate(radices: QuditRadices) -> PyUnitaryExpression {
//     ParameterizedUnitary::new(radices).generate_expression().into()
// }

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn openqudit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use qudit_core;
    use qudit_circuit;
    for registrar in inventory::iter::<qudit_core::PyRegistrar> {
        println!("Registering a func");
        (registrar.func)(m)?;
    }
  
    // m.add_class::<PyUnitaryExpression>()?;
    // m.add_class::<PyQuditCircuit>()?;

    // let library_submodule = PyModule::new(m.py(), "library")?;
    // library_submodule.add_function(wrap_pyfunction!(pyxgate, &library_submodule)?)?;
    // library_submodule.add_function(wrap_pyfunction!(pyugate, &library_submodule)?)?;
    // m.add_submodule(&library_submodule)?;
    // m.py().import("sys")?.getattr("modules")?.set_item("openqudit.library", library_submodule)?;

    Ok(())
}
