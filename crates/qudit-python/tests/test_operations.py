from openqudit.circuit import QuditCircuit
from openqudit.expressions import HGate


def test_nested_circuit():
    qc1 = QuditCircuit(1)
    qc1.append(HGate(), 0)
    qc2 = QuditCircuit(1)
    qc1.append(qc2, 0)


def test_double_nested_circuit():
    qc1 = QuditCircuit(1)
    qc1.append(HGate(), 0)
    qc2 = QuditCircuit(1)
    qc2.append(qc1, 0)
    qc3 = QuditCircuit(1)
    qc3.append(qc2, 0)


def test_operation_name():
    qc = QuditCircuit(1)
    qc.append(HGate(), 0)
    assert next(iter(qc)).name() == "H"
