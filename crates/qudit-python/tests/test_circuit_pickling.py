"""Tests for QuditCircuit pickle/unpickle (serialization round-trips)."""

import copy
import io
import pickle

import numpy as np
import pytest

from openqudit.circuit import QuditCircuit
from openqudit.expressions import HGate, U3Gate, XGate, ZGate, RZGate


@pytest.fixture
def u3():
    return U3Gate()


# --- Structural preservation ---

def test_empty_circuit_roundtrip():
    c = QuditCircuit(3)
    c2 = pickle.loads(pickle.dumps(c))
    assert c2.num_qudits == 3
    assert c2.num_cycles == c.num_cycles
    assert c2.num_params == 0


def test_qudit_count_preserved(u3):
    for n in [1, 2, 4]:
        c = QuditCircuit(n)
        c2 = pickle.loads(pickle.dumps(c))
        assert c2.num_qudits == n


def test_param_count_preserved(u3):
    c = QuditCircuit(1)
    c.append(u3, 0, [1.0, 2.0, 3.0])
    c.append(u3, 0, [0.1, 0.2, 0.3])
    c2 = pickle.loads(pickle.dumps(c))
    assert c2.num_params == c.num_params  # 6


def test_operation_count_preserved(u3):
    H, X = HGate(), XGate()
    c = QuditCircuit(2)
    c.append(H, 0)
    c.append(X, 1)
    c.append(u3, 0, [1.0, 2.0, 3.0])
    c2 = pickle.loads(pickle.dumps(c))
    assert c2.num_operations == c.num_operations  # 3


def test_cycle_count_preserved(u3):
    c = QuditCircuit(1)
    c.append(u3, 0, [0.1, 0.2, 0.3])
    c.append(u3, 0, [0.4, 0.5, 0.6])
    c2 = pickle.loads(pickle.dumps(c))
    assert c2.num_cycles == c.num_cycles


def test_qutrit_circuit_roundtrip():
    c = QuditCircuit([3])  # single qutrit
    c2 = pickle.loads(pickle.dumps(c))
    assert c2.num_qudits == 1
    assert c2.num_cycles == c.num_cycles


# --- Unitary correctness ---

def test_single_qubit_unitary_preserved(u3):
    params = [1.1, 2.2, 3.3]
    c = QuditCircuit(1)
    c.append(u3, 0, params)
    c2 = pickle.loads(pickle.dumps(c))
    assert np.allclose(c.kraus_ops(params), c2.kraus_ops(params))


def test_two_qubit_unitary_preserved():
    H, X = HGate(), XGate()
    c = QuditCircuit(2)
    c.append(H, 0)
    c.append(X, 1)
    c2 = pickle.loads(pickle.dumps(c))
    assert np.allclose(c.kraus_ops(), c2.kraus_ops())


def test_multi_gate_circuit_unitary_preserved(u3):
    H, Z = HGate(), ZGate()
    params = [0.5, 1.0, 1.5]
    c = QuditCircuit(1)
    c.append(H, 0)
    c.append(u3, 0, params)
    c.append(Z, 0)
    c2 = pickle.loads(pickle.dumps(c))
    assert np.allclose(c.kraus_ops(params), c2.kraus_ops(params))


def test_parameterized_gate_evaluates_correctly(u3):
    params = [0.3, 0.6, 0.9]
    c = QuditCircuit(1)
    c.append(u3, 0, params)
    c2 = pickle.loads(pickle.dumps(c))
    # Evaluate at a different point than what was appended with
    eval_params = [1.0, 2.0, 3.0]
    assert np.allclose(c.kraus_ops(eval_params), c2.kraus_ops(eval_params))


# --- Independence ---

def test_copy_is_independent(u3):
    c = QuditCircuit(1)
    c.append(u3, 0, [0.1, 0.2, 0.3])
    c2 = pickle.loads(pickle.dumps(c))

    cycles_before = c2.num_cycles
    c.append(u3, 0, [0.4, 0.5, 0.6])  # mutate original

    assert c2.num_cycles == cycles_before  # copy unaffected


# --- Interfaces ---

def test_deepcopy(u3):
    c = QuditCircuit(2)
    c.append(u3, 0, [1.0, 2.0, 3.0])
    c2 = copy.deepcopy(c)
    assert c2.num_params == c.num_params
    assert c2.num_qudits == c.num_qudits


def test_bytesio_roundtrip(u3):
    params = [0.7, 1.4, 2.1]
    c = QuditCircuit(1)
    c.append(u3, 0, params)

    buf = io.BytesIO()
    pickle.dump(c, buf)
    buf.seek(0)
    c2 = pickle.load(buf)

    assert c2.num_params == c.num_params
    assert np.allclose(c.kraus_ops(params), c2.kraus_ops(params))


def test_pickle_highest_protocol(u3):
    params = [1.0, 2.0, 3.0]
    c = QuditCircuit(1)
    c.append(u3, 0, params)
    c2 = pickle.loads(pickle.dumps(c, protocol=pickle.HIGHEST_PROTOCOL))
    assert c2.num_params == c.num_params
    assert np.allclose(c.kraus_ops(params), c2.kraus_ops(params))


def test_double_roundtrip(u3):
    params = [0.2, 0.4, 0.6]
    c = QuditCircuit(1)
    c.append(u3, 0, params)
    c2 = pickle.loads(pickle.dumps(pickle.loads(pickle.dumps(c))))
    assert c2.num_params == c.num_params
    assert np.allclose(c.kraus_ops(params), c2.kraus_ops(params))
