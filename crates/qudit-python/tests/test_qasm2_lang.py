"""Tests for QASM 2.0 load/dump on QuditCircuit.

Mirrors the coverage of bqskit/tests/ir/lang but uses the qudit Python API:
  QuditCircuit.loads(source, *, format="qasm2") -> QuditCircuit
  QuditCircuit.load(path) -> QuditCircuit
  circuit.dumps(*, format="qasm2") -> str
  circuit.dump(path)

NOTE on num_operations: num_operations() always returns 0 for circuits built
by the parser (known limitation — intern_operation bypasses the counter).
Use num_cycles as a proxy for "how many layers of gates were scheduled".
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from openqudit.circuit import QuditCircuit
from openqudit.expressions import (
    HGate,
    RXGate,
    RYGate,
    RZGate,
    SwapGate,
    U3Gate,
    XGate,
    YGate,
    ZGate,
)

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def loads(body: str) -> QuditCircuit:
    return QuditCircuit.loads(HEADER + body)


def unitary(body: str) -> np.ndarray:
    """Return the unitary matrix of a circuit given its QASM body."""
    c = loads(body)
    ops = c.kraus_ops()
    assert len(ops) == 1, f"Expected unitary circuit, got {len(ops)} Kraus ops"
    return np.array(ops[0])


def roundtrip(body: str) -> tuple[QuditCircuit, QuditCircuit, str]:
    """Parse, serialise, re-parse.  Returns (orig, reparsed, emitted_qasm)."""
    orig = loads(body)
    emitted = orig.dumps()
    reparsed = QuditCircuit.loads(emitted)
    return orig, reparsed, emitted


# ---------------------------------------------------------------------------
# Parse: register declarations
# ---------------------------------------------------------------------------


class TestRegisters:
    def test_single_qubit_register(self) -> None:
        c = loads("qreg q[1];\n")
        assert c.num_qudits == 1

    def test_multi_qubit_register(self) -> None:
        c = loads("qreg q[4];\n")
        assert c.num_qudits == 4

    def test_multiple_quantum_registers(self) -> None:
        c = loads("qreg q[2];\nqreg r[3];\n")
        assert c.num_qudits == 5

    def test_classical_register_does_not_add_qudits(self) -> None:
        c = loads("qreg q[2];\ncreg c[2];\n")
        assert c.num_qudits == 2

    def test_empty_circuit_zero_cycles(self) -> None:
        c = loads("qreg q[2];\n")
        assert c.num_cycles == 0


# ---------------------------------------------------------------------------
# Parse: constant (non-parametric) gates
# ---------------------------------------------------------------------------


class TestConstantGates:
    def test_x_gate(self) -> None:
        c = loads("qreg q[1];\nx q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_y_gate(self) -> None:
        c = loads("qreg q[1];\ny q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_z_gate(self) -> None:
        c = loads("qreg q[1];\nz q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_h_gate(self) -> None:
        c = loads("qreg q[1];\nh q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_s_gate(self) -> None:
        c = loads("qreg q[1];\ns q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_t_gate(self) -> None:
        c = loads("qreg q[1];\nt q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_cx_gate(self) -> None:
        c = loads("qreg q[2];\ncx q[0], q[1];\n")
        assert c.num_qudits == 2
        assert c.num_cycles == 1

    def test_cz_gate(self) -> None:
        c = loads("qreg q[2];\ncz q[0], q[1];\n")
        assert c.num_qudits == 2
        assert c.num_cycles == 1

    def test_cy_gate(self) -> None:
        c = loads("qreg q[2];\ncy q[0], q[1];\n")
        assert c.num_qudits == 2
        assert c.num_cycles == 1

    def test_swap_gate(self) -> None:
        c = loads("qreg q[2];\nswap q[0], q[1];\n")
        assert c.num_qudits == 2
        assert c.num_cycles == 1

    def test_ccx_gate(self) -> None:
        c = loads("qreg q[3];\nccx q[0], q[1], q[2];\n")
        assert c.num_qudits == 3
        assert c.num_cycles == 1

    def test_c3x_gate(self) -> None:
        c = loads("qreg q[4];\nc3x q[0], q[1], q[2], q[3];\n")
        assert c.num_qudits == 4
        assert c.num_cycles == 1

    def test_c4x_gate(self) -> None:
        c = loads("qreg q[5];\nc4x q[0], q[1], q[2], q[3], q[4];\n")
        assert c.num_qudits == 5
        assert c.num_cycles == 1

    def test_id_gate(self) -> None:
        c = loads("qreg q[1];\nid q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_sdg_gate_accepted(self) -> None:
        # sdg is accepted (may be canonicalised to s in output)
        c = loads("qreg q[1];\nsdg q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_tdg_gate_accepted(self) -> None:
        c = loads("qreg q[1];\ntdg q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1


# ---------------------------------------------------------------------------
# Parse: parametric gates
# ---------------------------------------------------------------------------


class TestParametricGates:
    def test_rx(self) -> None:
        c = loads("qreg q[1];\nrx(1.5707963267948966) q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_ry(self) -> None:
        c = loads("qreg q[1];\nry(1.5707963267948966) q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_rz(self) -> None:
        c = loads("qreg q[1];\nrz(1.5707963267948966) q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_u1(self) -> None:
        c = loads("qreg q[1];\nu1(0.5) q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_u2(self) -> None:
        c = loads("qreg q[1];\nu2(0.5, 1.0) q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_u3(self) -> None:
        c = loads("qreg q[1];\nu3(0.5, 1.0, 1.5) q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_p_gate(self) -> None:
        c = loads("qreg q[1];\np(0.5) q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_pi_parameter(self) -> None:
        c = loads("qreg q[1];\nrz(pi) q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_negative_parameter(self) -> None:
        c = loads("qreg q[1];\nrz(-1.5707963267948966) q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_pi_fraction_parameter(self) -> None:
        c = loads("qreg q[1];\nrz(pi/4) q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    @pytest.mark.parametrize(
        "angles",
        [
            [-0.1, 0.2, -0.3],
            [1.0, 2.0, 3.0],
            [3.141592653589793, -2.718281828459045, 1.4142135623730951],
        ],
    )
    def test_u3_decimal_angles(self, angles: list[float]) -> None:
        src = f"qreg q[1];\nu3({angles[0]}, {angles[1]}, {angles[2]}) q[0];\n"
        c = loads(src)
        assert c.num_qudits == 1
        assert c.num_cycles == 1


# ---------------------------------------------------------------------------
# Parse: measure and reset
# ---------------------------------------------------------------------------


class TestMeasureAndReset:
    def test_measure_single(self) -> None:
        c = loads("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_measure_register(self) -> None:
        c = loads("qreg q[2];\ncreg c[2];\nmeasure q -> c;\n")
        assert c.num_qudits == 2
        assert c.num_cycles == 1

    def test_measure_mid_circuit(self) -> None:
        body = "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\nx q[0];\n"
        c = loads(body)
        assert c.num_qudits == 1
        assert c.num_cycles == 3

    @pytest.mark.xfail(reason="reset not yet supported by the qudit QASM2 parser")
    def test_reset_single(self) -> None:
        c = loads("qreg q[1];\nreset q[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    @pytest.mark.xfail(reason="reset not yet supported by the qudit QASM2 parser")
    def test_reset_register(self) -> None:
        c = loads("qreg q[2];\nreset q;\n")
        assert c.num_qudits == 2

    @pytest.mark.xfail(reason="reset not yet supported by the qudit QASM2 parser")
    def test_reset_mid_circuit(self) -> None:
        body = "qreg q[1];\nx q[0];\nreset q[0];\nx q[0];\n"
        c = loads(body)
        assert c.num_qudits == 1
        assert c.num_cycles == 3


# ---------------------------------------------------------------------------
# Parse: comments
# ---------------------------------------------------------------------------


class TestComments:
    def test_comment_before_header(self) -> None:
        src = '// comment\nOPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nx q[0];\n'
        c = QuditCircuit.loads(src)
        assert c.num_qudits == 1

    def test_comment_after_gates(self) -> None:
        body = "qreg q[1];\nx q[0];\n// trailing comment\n"
        c = loads(body)
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_comment_between_gates(self) -> None:
        body = "qreg q[1];\nx q[0];\n// middle comment\nh q[0];\n"
        c = loads(body)
        assert c.num_qudits == 1
        assert c.num_cycles == 2


# ---------------------------------------------------------------------------
# Serialise: exact output for well-known gates
# ---------------------------------------------------------------------------


class TestSerialise:
    def test_header(self) -> None:
        c = loads("qreg q[1];\n")
        out = c.dumps()
        assert out.startswith('OPENQASM 2.0;\ninclude "qelib1.inc";\n')

    def test_h_roundtrip_exact(self) -> None:
        body = "qreg q[1];\nh q[0];\n"
        c = loads(body)
        assert c.dumps() == HEADER + body

    def test_cx_roundtrip_exact(self) -> None:
        body = "qreg q[2];\ncx q[0], q[1];\n"
        c = loads(body)
        assert c.dumps() == HEADER + body

    def test_x_roundtrip_exact(self) -> None:
        body = "qreg q[1];\nx q[0];\n"
        c = loads(body)
        assert c.dumps() == HEADER + body

    def test_s_roundtrip_exact(self) -> None:
        body = "qreg q[1];\ns q[0];\n"
        c = loads(body)
        assert c.dumps() == HEADER + body

    def test_t_roundtrip_exact(self) -> None:
        body = "qreg q[1];\nt q[0];\n"
        c = loads(body)
        assert c.dumps() == HEADER + body

    def test_ccx_roundtrip_exact(self) -> None:
        body = "qreg q[3];\nccx q[0], q[1], q[2];\n"
        c = loads(body)
        assert c.dumps() == HEADER + body

    def test_c3x_roundtrip_exact(self) -> None:
        body = "qreg q[4];\nc3x q[0], q[1], q[2], q[3];\n"
        c = loads(body)
        assert c.dumps() == HEADER + body

    def test_c4x_roundtrip_exact(self) -> None:
        body = "qreg q[5];\nc4x q[0], q[1], q[2], q[3], q[4];\n"
        c = loads(body)
        assert c.dumps() == HEADER + body

    def test_swap_roundtrip_exact(self) -> None:
        body = "qreg q[2];\nswap q[0], q[1];\n"
        c = loads(body)
        assert c.dumps() == HEADER + body

    def test_cz_roundtrip_exact(self) -> None:
        body = "qreg q[2];\ncz q[0], q[1];\n"
        c = loads(body)
        assert c.dumps() == HEADER + body

    def test_cy_roundtrip_exact(self) -> None:
        body = "qreg q[2];\ncy q[0], q[1];\n"
        c = loads(body)
        assert c.dumps() == HEADER + body

    def test_rz_pi_fraction_emitted(self) -> None:
        # rz(pi/2) parsed from decimal should serialise as rz(pi/2)
        body = "qreg q[1];\nrz(1.5707963267948966) q[0];\n"
        c = loads(body)
        assert c.dumps() == HEADER + "qreg q[1];\nrz(pi/2) q[0];\n"

    def test_multi_qubit_register_serialised(self) -> None:
        body = "qreg q[3];\nh q[0];\nh q[1];\nh q[2];\n"
        c = loads(body)
        out = c.dumps()
        assert "qreg q[3];" in out
        assert out.count("h q[") == 3

    def test_measure_roundtrip_exact(self) -> None:
        body = "qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n"
        c = loads(body)
        assert c.dumps() == HEADER + body

    @pytest.mark.xfail(reason="reset not yet supported by the qudit QASM2 parser")
    def test_reset_roundtrip_exact(self) -> None:
        body = "qreg q[1];\nreset q[0];\n"
        c = loads(body)
        assert c.dumps() == HEADER + body

    def test_multiple_qregs_serialised(self) -> None:
        body = "qreg q[2];\nqreg r[1];\ncx q[0], r[0];\n"
        c = loads(body)
        out = c.dumps()
        assert c.num_qudits == 3
        # re-parse should give same circuit shape
        c2 = QuditCircuit.loads(out)
        assert c2.num_qudits == 3
        assert c2.num_cycles == 1


# ---------------------------------------------------------------------------
# Roundtrip: structural consistency after parse -> dump -> parse
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_bell_circuit(self) -> None:
        body = "qreg q[2];\nh q[0];\ncx q[0], q[1];\n"
        orig, rep, _ = roundtrip(body)
        assert rep.num_qudits == orig.num_qudits
        assert rep.num_cycles == orig.num_cycles

    def test_ghz_three_qubits(self) -> None:
        body = "qreg q[3];\nh q[0];\ncx q[0], q[1];\ncx q[0], q[2];\n"
        orig, rep, _ = roundtrip(body)
        assert rep.num_qudits == orig.num_qudits
        assert rep.num_cycles == orig.num_cycles

    def test_parametric_gate(self) -> None:
        body = "qreg q[1];\nrz(pi/4) q[0];\n"
        orig, rep, _ = roundtrip(body)
        assert rep.num_qudits == orig.num_qudits
        assert rep.num_cycles == orig.num_cycles

    def test_measure_then_gate(self) -> None:
        body = "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n"
        orig, rep, _ = roundtrip(body)
        assert rep.num_qudits == orig.num_qudits

    def test_multi_gate_circuit(self) -> None:
        body = "qreg q[2];\ncx q[0], q[1];\nx q[0];\nh q[1];\n"
        orig, rep, _ = roundtrip(body)
        assert rep.num_qudits == orig.num_qudits
        assert rep.num_cycles == orig.num_cycles

    def test_deeply_sequential_circuit(self) -> None:
        gates = "".join(f"h q[0];\n" for _ in range(5))
        body = "qreg q[1];\n" + gates
        orig, rep, _ = roundtrip(body)
        assert rep.num_qudits == orig.num_qudits
        assert rep.num_cycles == orig.num_cycles

    def test_ccx_roundtrip(self) -> None:
        body = "qreg q[3];\nccx q[0], q[1], q[2];\n"
        orig, rep, _ = roundtrip(body)
        assert rep.num_qudits == orig.num_qudits
        assert rep.num_cycles == orig.num_cycles


# ---------------------------------------------------------------------------
# File I/O: load / dump
# ---------------------------------------------------------------------------


class TestFileIO:
    def test_dump_then_load(self) -> None:
        body = "qreg q[2];\nh q[0];\ncx q[0], q[1];\n"
        orig = loads(body)
        with tempfile.NamedTemporaryFile(suffix=".qasm", delete=False) as f:
            path = f.name
        try:
            orig.dump(path)
            loaded = QuditCircuit.load(path)
            assert loaded.num_qudits == orig.num_qudits
            assert loaded.num_cycles == orig.num_cycles
        finally:
            os.unlink(path)

    def test_dump_content_matches_dumps(self) -> None:
        body = "qreg q[2];\ncx q[0], q[1];\n"
        c = loads(body)
        with tempfile.NamedTemporaryFile(suffix=".qasm", delete=False) as f:
            path = f.name
        try:
            c.dump(path)
            with open(path) as f:
                file_content = f.read()
            assert file_content == c.dumps()
        finally:
            os.unlink(path)

    def test_load_missing_file_raises_os_error(self) -> None:
        with pytest.raises(OSError):
            QuditCircuit.load("/nonexistent/path/circuit.qasm")

    def test_dump_infers_format_from_extension(self) -> None:
        body = "qreg q[1];\nh q[0];\n"
        c = loads(body)
        with tempfile.NamedTemporaryFile(suffix=".qasm2", delete=False) as f:
            path = f.name
        try:
            c.dump(path)
            c2 = QuditCircuit.load(path)
            assert c2.num_qudits == 1
            assert c2.num_cycles == 1
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_invalid_qasm_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            QuditCircuit.loads("this is not valid qasm")

    def test_unknown_format_raises_value_error(self) -> None:
        body = "qreg q[1];\nh q[0];\n"
        c = loads(body)
        with pytest.raises(ValueError):
            c.dumps(format="unknown_format")

    def test_loads_unknown_format_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            QuditCircuit.loads(HEADER + "qreg q[1];\n", format="xyz")

    def test_non_standard_version_accepted(self) -> None:
        # Parser does not validate version numbers
        src = 'OPENQASM 3.0;\ninclude "qelib1.inc";\nqreg q[1];\n'
        c = QuditCircuit.loads(src)
        assert c.num_qudits == 1

    def test_missing_semicolon_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            QuditCircuit.loads(HEADER + "qreg q[1]\nh q[0];\n")


# ---------------------------------------------------------------------------
# Unitary equivalence: parsed circuit matches known gate matrices
# ---------------------------------------------------------------------------


ATOL = 1e-7


class TestUnitary:
    def test_x_gate_unitary(self) -> None:
        assert np.allclose(unitary("qreg q[1];\nx q[0];\n"), XGate()(), atol=ATOL)

    def test_y_gate_unitary(self) -> None:
        assert np.allclose(unitary("qreg q[1];\ny q[0];\n"), YGate()(), atol=ATOL)

    def test_z_gate_unitary(self) -> None:
        assert np.allclose(unitary("qreg q[1];\nz q[0];\n"), ZGate()(), atol=ATOL)

    def test_h_gate_unitary(self) -> None:
        assert np.allclose(unitary("qreg q[1];\nh q[0];\n"), HGate()(), atol=ATOL)

    def test_rz_pi_over_2_unitary(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nrz(pi/2) q[0];\n"),
            RZGate()(np.pi / 2),
            atol=ATOL,
        )

    def test_rx_pi_over_2_unitary(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nrx(pi/2) q[0];\n"),
            RXGate()(np.pi / 2),
            atol=ATOL,
        )

    def test_ry_pi_over_2_unitary(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nry(pi/2) q[0];\n"),
            RYGate()(np.pi / 2),
            atol=ATOL,
        )

    def test_u3_unitary(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nu3(0.5, 1.0, 1.5) q[0];\n"),
            U3Gate()(0.5, 1.0, 1.5),
            atol=ATOL,
        )

    def test_swap_unitary(self) -> None:
        assert np.allclose(
            unitary("qreg q[2];\nswap q[0], q[1];\n"),
            SwapGate()(),
            atol=ATOL,
        )

    @pytest.mark.parametrize("angles", [
        [-0.1, 0.2, -0.3],
        [1.0, 2.0, 3.0],
        [3.141592653589793, -2.718281828459045, 1.4142135623730951],
    ])
    def test_u3_parametric_unitary(self, angles: list[float]) -> None:
        body = f"qreg q[1];\nu3({angles[0]}, {angles[1]}, {angles[2]}) q[0];\n"
        assert np.allclose(unitary(body), U3Gate()(*angles), atol=ATOL)

    def test_bell_roundtrip_unitary(self) -> None:
        body = "qreg q[2];\nh q[0];\ncx q[0], q[1];\n"
        c = loads(body)
        c2 = QuditCircuit.loads(c.dumps())
        u1 = np.array(c.kraus_ops()[0])
        u2 = np.array(c2.kraus_ops()[0])
        assert np.allclose(u1, u2, atol=ATOL)

    def test_ghz_roundtrip_unitary(self) -> None:
        body = "qreg q[3];\nh q[0];\ncx q[0], q[1];\ncx q[0], q[2];\n"
        c = loads(body)
        c2 = QuditCircuit.loads(c.dumps())
        u1 = np.array(c.kraus_ops()[0])
        u2 = np.array(c2.kraus_ops()[0])
        assert np.allclose(u1, u2, atol=ATOL)

    def test_parametric_roundtrip_unitary(self) -> None:
        body = "qreg q[1];\nrz(pi/4) q[0];\n"
        c = loads(body)
        c2 = QuditCircuit.loads(c.dumps())
        u1 = np.array(c.kraus_ops()[0])
        u2 = np.array(c2.kraus_ops()[0])
        assert np.allclose(u1, u2, atol=ATOL)

    def test_multi_gate_roundtrip_unitary(self) -> None:
        body = "qreg q[2];\ncx q[0], q[1];\nrz(pi/3) q[0];\nh q[1];\n"
        c = loads(body)
        c2 = QuditCircuit.loads(c.dumps())
        u1 = np.array(c.kraus_ops()[0])
        u2 = np.array(c2.kraus_ops()[0])
        assert np.allclose(u1, u2, atol=ATOL)
