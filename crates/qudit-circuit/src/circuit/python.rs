use super::QuditCircuit;
use crate::cycle::CycleIndex;
use crate::instruction::{Instruction, InstructionId};
use crate::operation::OpCode;
use crate::operation::Operation;
use crate::param::ArgumentList;
use crate::wire::Wire;
use crate::wire::WireList;
use qudit_core::QuditSystem;
use qudit_core::array::Tensor;
use std::collections::HashMap;

use crate::instruction::PyInstructionReference;
use crate::python::PyCircuitRegistrar;
use numpy::PyArray3;
use numpy::PyArrayMethods;
use numpy::ndarray::ArrayViewMut3;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use qudit_core::c64;

#[pyclass]
#[pyo3(name = "ParameterVector")]
pub struct PyParameterVector {
    circuit_ref: Py<PyQuditCircuit>,
}

#[pymethods]
impl PyParameterVector {
    fn assign_all<'py>(&mut self, py: Python<'py>, values: Vec<crate::param::Value>) {
        self.circuit_ref
            .bind(py)
            .borrow_mut()
            .circuit
            .params
            .assign_all(values);
    }
}

#[pyclass]
#[pyo3(name = "QuditCircuitIterator")]
struct PyQuditCircuitIterator {
    circuit_ref: Py<PyQuditCircuit>,
    cycle_index: CycleIndex,
    inner_ids: Vec<crate::cycle::InstId>,
    inner_index: usize,
}

impl PyQuditCircuitIterator {
    fn new<'py>(circuit_ref: Py<PyQuditCircuit>, py: Python<'py>) -> PyQuditCircuitIterator {
        let circuit = &circuit_ref.bind(py).borrow().circuit;
        let initial_inner_ids = if circuit.is_empty() {
            Vec::new()
        } else {
            circuit.cycles[0].id_iter().collect()
        };
        PyQuditCircuitIterator {
            circuit_ref,
            cycle_index: CycleIndex(0),
            inner_ids: initial_inner_ids,
            inner_index: 0,
        }
    }
}

#[pymethods]
impl PyQuditCircuitIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyInstructionReference>> {
        let py = slf.py();
        let circuit = &slf.circuit_ref.bind(py).borrow().circuit;

        while slf.inner_index >= slf.inner_ids.len() {
            slf.cycle_index += 1_usize;

            if slf.cycle_index >= circuit.num_cycles().into() {
                return Ok(None);
            }

            slf.inner_index = 0;
            slf.inner_ids.clear();
            let cycle_index = slf.cycle_index;
            slf.inner_ids.extend(circuit.cycles[cycle_index].id_iter());
        }

        let cycle_id = circuit.cycles.index_to_id(slf.cycle_index);
        let inner_id = slf.inner_ids[slf.inner_index];
        slf.inner_index += 1;
        Ok(Some(PyInstructionReference::new(
            InstructionId::new(cycle_id, inner_id),
            slf.circuit_ref.clone_ref(py),
        )))
    }
}

/// Helper function to parse a Python object that can be
/// either an integer or an iterable of integers.
fn parse_int_or_iterable<'py>(input: &Bound<'py, PyAny>) -> PyResult<Vec<usize>> {
    // First, try to extract the input as a single integer.
    if let Ok(val) = input.extract::<usize>() {
        // If successful, create a Vec of that length filled with the value 2.
        Ok(vec![2; val])
    } else {
        // If it's not an integer, try to treat it as an iterable.
        // This will raise a TypeError if the object is not iterable
        // or its elements are not integers.
        pyo3::types::PyIterator::from_object(input)?
            .map(|item| item?.extract::<usize>())
            .collect::<PyResult<Vec<usize>>>()
    }
}

#[pyclass(from_py_object, module = "openqudit.circuit")]
#[pyo3(name = "QuditCircuit")]
#[derive(Clone)]
pub struct PyQuditCircuit {
    pub(crate) circuit: QuditCircuit,
}

#[pymethods]
impl PyQuditCircuit {
    /// Creates a new QuditCircuit instance.
    ///
    /// Args:
    ///     qudits (int | Iterable[int]): An integer specifying number of qudits
    ///         or an iterable of qudit radices.
    ///     dits (int | Iterable[int] | None): An integer, an iterable, or None.
    ///         Defaults to None, which results in no classical dits.
    #[new]
    #[pyo3(signature = (qudits, dits = None))]
    fn new<'py>(
        qudits: &Bound<'py, PyAny>,
        dits: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PyQuditCircuit> {
        let qudits_vec = parse_int_or_iterable(qudits)?;

        let dits_vec = match dits {
            Some(pyany) => parse_int_or_iterable(pyany)?,
            None => Vec::new(),
        };

        Ok(PyQuditCircuit {
            circuit: QuditCircuit::new(qudits_vec, dits_vec),
        })
    }

    fn clone(&self) -> PyQuditCircuit {
        Clone::clone(self)
    }

    // --- Properties ---
    //
    /// Returns the number of qudits in the circuit.
    #[getter]
    fn num_qudits(&self) -> PyResult<usize> {
        Ok(self.circuit.num_qudits())
    }

    /// Returns the number of parameters in the circuit.
    #[getter]
    fn num_params(&self) -> PyResult<usize> {
        Ok(self.circuit.num_params())
    }

    /// Returns the number of operations in the circuit.
    #[getter]
    fn num_operations(&self) -> PyResult<usize> {
        Ok(self.circuit.num_operations())
    }

    /// Returns the number of cycles in the circuit.
    #[getter]
    fn num_cycles(&self) -> PyResult<usize> {
        Ok(self.circuit.num_cycles())
    }

    #[getter]
    fn is_empty(&self) -> PyResult<bool> {
        Ok(self.circuit.is_empty())
    }

    #[getter]
    fn active_qudits(&self) -> PyResult<Vec<usize>> {
        Ok(self.circuit.active_qudits())
    }

    #[getter]
    fn active_dits(&self) -> PyResult<Vec<usize>> {
        Ok(self.circuit.active_dits())
    }

    #[getter]
    fn params(slf: PyRef<'_, Self>) -> PyResult<PyParameterVector> {
        let param_vec = PyParameterVector {
            circuit_ref: slf.into(),
        };

        Ok(param_vec)
    }

    // @property def coupling_graph(self) -> CouplingGraph?
    // @property def gate_set(self) -> GateSet?

    // Metrics
    //
    // def depth(self, *, filter: Optional[Callable] = None, recursive: bool = True) -> int:
    // def parallelism(self) -> float:
    // def gate_counts(self, *, filter: Optional[Callable] = None, recursive: bool = True) -> int:
    //
    // Qudit Methods
    //
    // def append_qudit(self, radix: int = 2) -> None:
    // def extend_qudits(self, radixes: Iterable[int]) -> None:
    // def insert_qudit(self, qudit_index: int, radix: int = 2) -> None:
    // def pop_qudit(self, qudit_index: int) -> None:
    // def is_qudit_in_range(self, qudit_index: int) -> bool:
    // def is_qudit_idle(self, qudit_index: int) -> bool:
    // def renumber_qudits(self, qudit_permutation: Iterable[int]) -> None:
    //
    // Cycle Methods
    //
    // def pop_cycle(self, cycle_index: int) -> None?
    //

    // DAG Methods:
    #[getter]
    fn front(&self) -> HashMap<Wire, InstructionId> {
        self.circuit.front()
    }

    #[getter]
    fn rear(&self) -> HashMap<Wire, InstructionId> {
        self.circuit.rear()
    }

    fn first_on(&self, wire: Wire) -> Option<InstructionId> {
        self.circuit.first_on(wire)
    }

    fn last_on(&self, wire: Wire) -> Option<InstructionId> {
        self.circuit.last_on(wire)
    }

    fn next(&self, inst_id: InstructionId) -> HashMap<Wire, InstructionId> {
        self.circuit.next(inst_id)
    }

    fn prev(&self, inst_id: InstructionId) -> HashMap<Wire, InstructionId> {
        self.circuit.prev(inst_id)
    }

    // At Methods:
    //
    // def get_operation(self, point: CircuitPointLike) -> Operation:
    //
    // def point(
    //     self,
    //     op: Operation | Gate,
    //     start: CircuitPointLike = (0, 0),
    //     end: CircuitPointLike | None = None,
    // ) -> CircuitPoint:
    //
    // def append(self, op: Operation) -> int:

    /// Returns the Kraus operators of the circuit as a NumPy array.
    #[pyo3(signature = (args = None))]
    pub fn kraus_ops<'py>(
        &self,
        py: Python<'py>,
        args: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray3<c64>>> {
        let rust_args: Vec<f64> = match args {
            Some(py_args) => py_args.extract()?,
            None => {
                if self.circuit.num_unassigned_params() != 0 {
                    return Err(PyTypeError::new_err(
                        "Circuit has unassigned parameters, but no arguments were provided to kraus_ops.",
                    ));
                }
                Vec::new()
            }
        };

        // Call the underlying Rust method
        let tensor: Tensor<c64, 3> = self.circuit.kraus_ops(&rust_args);
        let shape = tensor.dims();

        let py_array: Bound<'py, PyArray3<c64>> = PyArray3::zeros(py, *shape, false);

        {
            let mut readwrite = py_array.readwrite();
            let mut py_array_view: ArrayViewMut3<c64> = readwrite.as_array_mut();

            for k in 0..shape[0] {
                let kraus_op = tensor.subtensor_ref(k);
                for (j, col) in kraus_op.col_iter().enumerate() {
                    for (i, val) in col.iter().enumerate() {
                        py_array_view[[k, i, j]] = *val;
                    }
                }
            }
        }

        Ok(py_array)
    }

    // fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PyQuditCircuitIterator> {
    //     let py = slf.py();
    //     Ok(PyQuditCircuitIterator::new(slf.into(), py))
    // }
    #[pyo3(signature = (op, loc, args = None))]
    pub fn append<'py>(
        slf: Bound<'py, Self>,
        op: Operation,
        loc: &Bound<'py, PyAny>,
        args: Option<ArgumentList>,
    ) -> PyResult<PyInstructionReference> {
        let num_qudits = op.num_qudits();

        // 2. Parse 'loc' as an int, iterable of ints, or tuple of iterables
        let parsed_loc = if let Ok(single_loc) = loc.extract::<usize>() {
            WireList::pure([single_loc])
        } else if let Ok(tuple) = loc.cast::<PyTuple>() {
            if tuple.len() == 2 {
                let item0 = tuple.get_item(0)?;
                let item1 = tuple.get_item(1)?;

                // Try parsing as (iterable, iterable)
                if let (Ok(vec0), Ok(vec1)) =
                    (item0.extract::<Vec<usize>>(), item1.extract::<Vec<usize>>())
                {
                    WireList::new(vec0, vec1)
                }
                // Else, try parsing as (int, int)
                else if let (Ok(int0), Ok(int1)) =
                    (item0.extract::<usize>(), item1.extract::<usize>())
                {
                    if num_qudits.is_some() && num_qudits == Some(1) {
                        WireList::new(vec![int0], vec![int1])
                    } else {
                        WireList::pure([int0, int1])
                    }
                } else {
                    return Err(PyTypeError::new_err(
                        "A 2-element 'loc' tuple must contain (int, int) or (iterable, iterable)",
                    ));
                }
            } else {
                let mut qudit_register = tuple.extract::<Vec<usize>>()?;
                let dit_register = qudit_register.split_off(num_qudits.unwrap());
                WireList::new(qudit_register, dit_register)
            }
        } else if let Ok(list_loc) = loc.extract::<Vec<usize>>() {
            let mut qudit_register = list_loc.clone();
            let dit_register = qudit_register.split_off(num_qudits.unwrap());
            WireList::new(qudit_register, dit_register)
        } else {
            return Err(PyTypeError::new_err(
                "Argument 'loc' must be an int, an iterable of ints, or a tuple of two iterables of ints",
            ));
        };

        let inst_id = slf.borrow_mut().circuit.append(op, parsed_loc, args)?;
        Ok(PyInstructionReference::new(inst_id, slf.unbind()))
    }

    pub fn cache(&mut self, op: Operation) -> OpCode {
        self.circuit.cache_operation(op)
    }

    // def append_gate
    // def append_circuit
    // def extend
    // def insert
    // def insert_gate
    // def insert_circuit
    // def remove(op)
    fn remove(&mut self, inst_id: InstructionId) -> Option<Instruction> {
        self.circuit.remove(inst_id)
    }
    // def remove_all
    fn count(&self, op_code: OpCode) -> usize {
        self.circuit.count(op_code)
    }
    // def pop(point)
    // def batch_pop(points)
    // def replace(point, op)
    // def replace_all(op, op)
    // def batch_replace(points, ops)
    // def replace_gate
    // def replace_with_circuit(... as circuit_gate = False)
    //
    // But like also, initializations, barriers, measurements, kraus ops, classical controls
    //
    // Movement/Ownership
    //
    // def copy()
    // def become()
    // def clear()
    //
    // Parameter Methods?
    // def un-constant?
    // def freeze
    // Specified vs Constant appends are funky, A user should edit their expression
    // before hand if they want to hardcode a constant into it, otherwise, circuits
    // should track constant parameters, whether they are added as specified or const.
    //
    // User's should have ability to safely `def freeze(self, parameter_index: int, value: Constant)` and
    // `def thaw(self, parameter_index: int)`
    //
    // For this, might need to add a new type of Parameter => Frozen({value: Constant, name:
    // Option<String>}), when a named parameter is frozen, the name is stored for future thaws
    // Also, if I add an expression with the same parameter, it should also be frozen.

    // def group(*instructions) -> InstructionReference:
    //      instructions: List[Union[Into<InstructionId> | InstructionReference]] |
    //      List[List[Union<Into<InstructionId> | InstructionReference]] | Into<CircuitRegion>

    // Advanced Algorithms
    //
    // def compress()
    // def surround() // Need to seriously think about subcircuits/regions/grouping
    // def invert()
    // def evaluate(args)
    // def evaluate_gradient(args)
    // def evaluate_hessian(args)
    // def instantiate(...)
    //
    // dunder methods
    //
    // __getitem__
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PyQuditCircuitIterator> {
        let py = slf.py();
        Ok(PyQuditCircuitIterator::new(slf.into(), py))
    }
    // __reversed__
    // __contains__
    // __len__
    // __invert__
    // __eq__
    // __ne__
    // __add__
    // __mul__
    // __radd__
    // __iadd__
    // __imul__
    // __str__
    // __repr__
    // operations/operations_with_cycles
    //
    // IO
    //
    // save
    // to

    /// Load a circuit from a file.  The format is inferred from the file
    /// extension (e.g. `.qasm` or `.qasm2` selects the QASM 2.0 parser).
    #[staticmethod]
    pub fn load(_py: Python<'_>, path: String) -> PyResult<PyQuditCircuit> {
        let circuit = QuditCircuit::load(path.as_str())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PyQuditCircuit { circuit })
    }

    /// Parse a circuit from a string.
    ///
    /// Args:
    ///     source: The source string to parse.
    ///     format: The format name or file extension to use (default ``"qasm2"``).
    #[staticmethod]
    #[pyo3(signature = (source, *, format = "qasm2"))]
    pub fn loads(source: String, format: &str) -> PyResult<PyQuditCircuit> {
        let circuit = QuditCircuit::loads(&source, format)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyQuditCircuit { circuit })
    }

    /// Write the circuit to a file.  The format is inferred from the file
    /// extension (e.g. `.qasm` or `.qasm2` selects the QASM 2.0 writer).
    pub fn dump(&self, path: String) -> PyResult<()> {
        self.circuit
            .dump(path.as_str())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Serialise the circuit to a string.
    ///
    /// Args:
    ///     format: The format name or file extension to use (default ``"qasm2"``).
    #[pyo3(signature = (*, format = "qasm2"))]
    pub fn dumps(&self, format: &str) -> PyResult<String> {
        self.circuit
            .saves(format)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // from_unitary
    // __reduce__
    // rebuild_circuit

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.circuit = postcard::from_bytes(state.as_bytes())
            .map_err(|e| PyTypeError::new_err(format!("Failed to deserialize circuit: {e}")))?;
        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes: Vec<u8> = postcard::to_allocvec(&self.circuit)
            .map_err(|e| PyTypeError::new_err(format!("Failed to serialize circuit: {e}")))?;
        Ok(PyBytes::new(py, &bytes))
    }

    pub fn __getnewargs__(&self) -> PyResult<(Vec<usize>,)> {
        // Dummy args: __setstate__ fully restores the circuit, so we just need
        // any valid value for the qudits argument of __new__.
        Ok((vec![2],))
    }
}

impl<'py> IntoPyObject<'py> for QuditCircuit {
    type Target = <PyQuditCircuit as IntoPyObject<'py>>::Target;
    type Output = <PyQuditCircuit as IntoPyObject<'py>>::Output;
    type Error = <PyQuditCircuit as IntoPyObject<'py>>::Error;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        PyQuditCircuit::from(self).into_pyobject(py)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for QuditCircuit {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py_circuit_ref = obj.cast::<PyQuditCircuit>()?;
        Ok(py_circuit_ref.borrow().circuit.clone())
    }
}

impl From<QuditCircuit> for PyQuditCircuit {
    fn from(value: QuditCircuit) -> Self {
        PyQuditCircuit { circuit: value }
    }
}

impl From<PyQuditCircuit> for QuditCircuit {
    fn from(value: PyQuditCircuit) -> Self {
        value.circuit
    }
}

/// Registers the QuditCircuit class with the Python module.
fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    parent_module.add_class::<PyQuditCircuit>()?;
    Ok(())
}
inventory::submit!(PyCircuitRegistrar { func: register });
