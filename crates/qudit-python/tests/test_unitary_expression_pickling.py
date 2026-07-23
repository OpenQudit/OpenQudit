"""Tests for UnitaryExpression pickle/unpickle (serialization round-trips)."""

import copy
import io
import pickle

import numpy as np
import pytest

from openqudit.expressions import HGate, RZGate, U3Gate, UnitaryExpression, XGate, ZGate


@pytest.fixture
def u3():
    return U3Gate()


# --- Structural preservation ---


def test_unparameterized_gate_roundtrip():
    h = HGate()
    h2 = pickle.loads(pickle.dumps(h))
    assert h2.name() == h.name()
    assert h2.radices() == h.radices()
    assert h2.dimension() == h.dimension()
    assert h2.num_params() == h.num_params()  # 0


def test_param_count_preserved(u3):
    u3_2 = pickle.loads(pickle.dumps(u3))
    assert u3_2.num_params() == u3.num_params()  # 3


def test_radices_preserved(u3):
    u3_2 = pickle.loads(pickle.dumps(u3))
    assert u3_2.radices() == u3.radices()


def test_name_preserved(u3):
    u3_2 = pickle.loads(pickle.dumps(u3))
    assert u3_2.name() == u3.name()


def test_identity_roundtrip():
    u = UnitaryExpression.identity("my_identity", [2, 3])
    u2 = pickle.loads(pickle.dumps(u))
    assert u2.name() == u.name()
    assert u2.radices() == u.radices()
    assert u2.dimension() == u.dimension()
    assert np.allclose(u(), u2())


# --- Unitary correctness ---


def test_unparameterized_unitary_preserved():
    h = HGate()
    h2 = pickle.loads(pickle.dumps(h))
    assert np.allclose(h(), h2())


def test_parameterized_unitary_preserved(u3):
    params = [1.1, 2.2, 3.3]
    u3_2 = pickle.loads(pickle.dumps(u3))
    assert np.allclose(u3(*params), u3_2(*params))


def test_parameterized_gate_evaluates_correctly_at_new_point(u3):
    # Evaluate at a different point than any it was pickled with;
    # __setstate__ must fully restore the symbolic expression, not
    # just a cached evaluation.
    u3_2 = pickle.loads(pickle.dumps(u3))
    eval_params = [0.4, 0.8, 1.2]
    assert np.allclose(u3(*eval_params), u3_2(*eval_params))


@pytest.mark.parametrize("gate_factory", [HGate, XGate, ZGate])
def test_single_qubit_gates_roundtrip(gate_factory):
    g = gate_factory()
    g2 = pickle.loads(pickle.dumps(g))
    assert np.allclose(g(), g2())


def test_composed_expression_roundtrip():
    combined = HGate().dot(XGate())
    combined2 = pickle.loads(pickle.dumps(combined))
    assert np.allclose(combined(), combined2())


def test_tensor_product_expression_roundtrip():
    combined = HGate().otimes(XGate())
    combined2 = pickle.loads(pickle.dumps(combined))
    assert combined2.radices() == combined.radices()
    assert np.allclose(combined(), combined2())


def test_transposed_expression_roundtrip():
    rz = RZGate()
    rz.transpose()
    rz2 = pickle.loads(pickle.dumps(rz))
    params = [0.5]
    assert np.allclose(rz(*params), rz2(*params))


def test_daggered_expression_roundtrip():
    rz = RZGate()
    rz.dagger()
    rz2 = pickle.loads(pickle.dumps(rz))
    params = [0.5]
    assert np.allclose(rz(*params), rz2(*params))


# --- Independence ---


def test_mutation_after_pickling_is_independent(u3):
    u3_2 = pickle.loads(pickle.dumps(u3))
    u3.transpose()  # mutate the original in place
    params = [0.3, 0.6, 0.9]
    assert not np.allclose(u3(*params), u3_2(*params))


# --- Interfaces ---


def test_deepcopy(u3):
    u3_2 = copy.deepcopy(u3)
    params = [1.0, 2.0, 3.0]
    assert u3_2.num_params() == u3.num_params()
    assert np.allclose(u3(*params), u3_2(*params))


def test_bytesio_roundtrip(u3):
    params = [0.7, 1.4, 2.1]

    buf = io.BytesIO()
    pickle.dump(u3, buf)
    buf.seek(0)
    u3_2 = pickle.load(buf)

    assert u3_2.num_params() == u3.num_params()
    assert np.allclose(u3(*params), u3_2(*params))


def test_pickle_highest_protocol(u3):
    params = [1.0, 2.0, 3.0]
    u3_2 = pickle.loads(pickle.dumps(u3, protocol=pickle.HIGHEST_PROTOCOL))
    assert u3_2.num_params() == u3.num_params()
    assert np.allclose(u3(*params), u3_2(*params))


def test_double_roundtrip(u3):
    params = [0.2, 0.4, 0.6]
    u3_2 = pickle.loads(pickle.dumps(pickle.loads(pickle.dumps(u3))))
    assert u3_2.num_params() == u3.num_params()
    assert np.allclose(u3(*params), u3_2(*params))
