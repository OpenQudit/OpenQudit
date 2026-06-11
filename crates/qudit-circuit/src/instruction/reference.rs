use pyo3::prelude::*;
use super::InstructionId;
use crate::cycle::{CycleId, CycleIndex};
use crate::wire::WireList;
use crate::circuit::PyQuditCircuit;
use pyo3::exceptions::PyRuntimeError;

#[pyclass]
#[pyo3(name = "InstructionReference")]
pub struct PyInstructionReference {
    inst_id: InstructionId,
    circuit_ref: Py<PyQuditCircuit>,
}

impl PyInstructionReference {
    pub fn new(inst_id: InstructionId, circuit_ref: Py<PyQuditCircuit>) -> Self {
        Self { inst_id, circuit_ref }
    }
}

#[pymethods]
impl PyInstructionReference {
    fn name<'py>(&self, py: Python<'py>) -> PyResult<String> {
        let circuit = &self.circuit_ref.bind(py).borrow().circuit;
        match circuit.get(self.inst_id) {
            Some(inst) => Ok(circuit
                .operations()
                .name(inst.op_code())
                .replace("_subbed", "")),
            None => Err(PyRuntimeError::new_err("Invalid instruction reference.")),
        }
    }

    fn wires<'py>(&self, py: Python<'py>) -> PyResult<WireList> {
        let circuit = &self.circuit_ref.bind(py).borrow().circuit;
        match circuit.get(self.inst_id) {
            Some(inst) => Ok(inst.wires()),
            None => Err(PyRuntimeError::new_err("Invalid instruction reference.")),
        }
    }

    fn params<'py>(&self, py: Python<'py>) -> PyResult<Vec<crate::param::Parameter>> {
        let circuit = &self.circuit_ref.bind(py).borrow().circuit;
        let params = circuit.params();
        match circuit.get(self.inst_id) {
            Some(inst) => {
                let param_indices = params.convert_ids_to_indices(inst.params());
                Ok(param_indices
                    .iter()
                    .map(|i| params[i].clone())
                    .collect())
            }
            None => Err(PyRuntimeError::new_err("Invalid Instruction reference.")),
        }
    }

    fn cycle_id(&self) -> CycleId {
        self.inst_id.cycle()
    }

    fn cycle_index<'py>(&self, py: Python<'py>) -> PyResult<CycleIndex> {
        let circuit = &self.circuit_ref.bind(py).borrow().circuit;
        match circuit.cycle_id_to_index(self.inst_id.cycle()) {
            Some(cycle_index) => Ok(cycle_index),
            None => Err(PyRuntimeError::new_err("Invalid instruction reference.")),
        }
    }

    fn num_qudits<'py>(&self, py: Python<'py>) -> PyResult<usize> {
        self.wires(py).map(|w| w.get_num_qudits())
    }

    fn num_dits<'py>(&self, py: Python<'py>) -> PyResult<usize> {
        self.wires(py).map(|w| w.get_num_dits())
    }
}
