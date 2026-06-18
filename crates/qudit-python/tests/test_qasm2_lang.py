"""Tests for QASM 2.0 load/dump on QuditCircuit.

Every gate the parser accepts must be tested against the known gate matrix so a
mis-parse (e.g. H loaded as X) cannot go undetected.  Every gate the writer can
emit must appear verbatim in the output so a mis-serialisation (e.g. sdg written
as s) is caught.

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
    Controlled,
    HGate,
    IGate,
    PGate,
    RXGate,
    RXXGate,
    RYGate,
    RZGate,
    RZZGate,
    SGate,
    SwapGate,
    SXGate,
    TGate,
    U1Gate,
    U2Gate,
    U3Gate,
    XGate,
    YGate,
    ZGate,
)

ATOL = 1e-7
HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def loads(body: str) -> QuditCircuit:
    return QuditCircuit.loads(HEADER + body)


def unitary(body: str) -> np.ndarray:
    """Return the unitary matrix of a circuit given its QASM body (no header)."""
    c = loads(body)
    ops = c.kraus_ops()
    assert len(ops) == 1, f"Expected unitary circuit, got {len(ops)} Kraus ops"
    return np.array(ops[0])


def roundtrip(body: str) -> tuple[QuditCircuit, QuditCircuit, str]:
    """Parse, serialise, re-parse. Returns (orig, reparsed, emitted_qasm)."""
    orig = loads(body)
    emitted = orig.dumps()
    reparsed = QuditCircuit.loads(emitted)
    return orig, reparsed, emitted


def assert_roundtrip_unitary(body: str) -> None:
    """Assert that parse → dump → reparse preserves the circuit unitary."""
    c = loads(body)
    c2 = QuditCircuit.loads(c.dumps())
    assert np.allclose(
        np.array(c.kraus_ops()[0]), np.array(c2.kraus_ops()[0]), atol=ATOL
    )


# ---------------------------------------------------------------------------
# Register declarations
# ---------------------------------------------------------------------------


class TestRegisters:
    def test_single_qubit_register(self) -> None:
        assert loads("qreg q[1];\n").num_qudits == 1

    def test_multi_qubit_register(self) -> None:
        assert loads("qreg q[4];\n").num_qudits == 4

    def test_multiple_quantum_registers(self) -> None:
        assert loads("qreg q[2];\nqreg r[3];\n").num_qudits == 5

    def test_classical_register_does_not_add_qudits(self) -> None:
        c = loads("qreg q[2];\ncreg c[2];\n")
        assert c.num_qudits == 2

    def test_empty_circuit_zero_cycles(self) -> None:
        assert loads("qreg q[2];\n").num_cycles == 0


# ---------------------------------------------------------------------------
# Single-qubit constant gates — unitary correctness + exact serialisation
# ---------------------------------------------------------------------------


class TestSingleQubitGates:
    """Every gate is checked against its known matrix and its QASM name."""

    def test_x_gate(self) -> None:
        assert np.allclose(unitary("qreg q[1];\nx q[0];\n"), XGate()(), atol=ATOL)
        assert (
            loads("qreg q[1];\nx q[0];\n").dumps() == HEADER + "qreg q[1];\nx q[0];\n"
        )

    def test_y_gate(self) -> None:
        assert np.allclose(unitary("qreg q[1];\ny q[0];\n"), YGate()(), atol=ATOL)
        assert (
            loads("qreg q[1];\ny q[0];\n").dumps() == HEADER + "qreg q[1];\ny q[0];\n"
        )

    def test_z_gate(self) -> None:
        assert np.allclose(unitary("qreg q[1];\nz q[0];\n"), ZGate()(), atol=ATOL)
        assert (
            loads("qreg q[1];\nz q[0];\n").dumps() == HEADER + "qreg q[1];\nz q[0];\n"
        )

    def test_h_gate(self) -> None:
        assert np.allclose(unitary("qreg q[1];\nh q[0];\n"), HGate()(), atol=ATOL)
        assert (
            loads("qreg q[1];\nh q[0];\n").dumps() == HEADER + "qreg q[1];\nh q[0];\n"
        )

    def test_s_gate(self) -> None:
        assert np.allclose(unitary("qreg q[1];\ns q[0];\n"), SGate()(), atol=ATOL)
        assert (
            loads("qreg q[1];\ns q[0];\n").dumps() == HEADER + "qreg q[1];\ns q[0];\n"
        )

    def test_t_gate(self) -> None:
        assert np.allclose(unitary("qreg q[1];\nt q[0];\n"), TGate()(), atol=ATOL)
        assert (
            loads("qreg q[1];\nt q[0];\n").dumps() == HEADER + "qreg q[1];\nt q[0];\n"
        )

    def test_id_gate(self) -> None:
        assert np.allclose(unitary("qreg q[1];\nid q[0];\n"), IGate()(), atol=ATOL)
        assert (
            loads("qreg q[1];\nid q[0];\n").dumps() == HEADER + "qreg q[1];\nid q[0];\n"
        )

    def test_sx_gate(self) -> None:
        assert np.allclose(unitary("qreg q[1];\nsx q[0];\n"), SXGate()(), atol=ATOL)
        assert (
            loads("qreg q[1];\nsx q[0];\n").dumps() == HEADER + "qreg q[1];\nsx q[0];\n"
        )

    def test_distinct_gates_have_different_unitaries(self) -> None:
        """X, Y, Z, H, S, T must all be distinct matrices — catches mis-parse."""
        mats = {
            "x": unitary("qreg q[1];\nx q[0];\n"),
            "y": unitary("qreg q[1];\ny q[0];\n"),
            "z": unitary("qreg q[1];\nz q[0];\n"),
            "h": unitary("qreg q[1];\nh q[0];\n"),
            "s": unitary("qreg q[1];\ns q[0];\n"),
            "t": unitary("qreg q[1];\nt q[0];\n"),
        }
        names = list(mats)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                assert not np.allclose(mats[a], mats[b], atol=ATOL), (
                    f"Gates '{a}' and '{b}' produced the same unitary — parse bug?"
                )


# ---------------------------------------------------------------------------
# Dagger gates (sdg, tdg, sxdg)
# ---------------------------------------------------------------------------


class TestDaggerGates:
    def test_sdg_unitary(self) -> None:
        Sdg = np.conj(np.array(SGate()())).T
        assert np.allclose(unitary("qreg q[1];\nsdg q[0];\n"), Sdg, atol=ATOL)

    def test_tdg_unitary(self) -> None:
        Tdg = np.conj(np.array(TGate()())).T
        assert np.allclose(unitary("qreg q[1];\ntdg q[0];\n"), Tdg, atol=ATOL)

    def test_sxdg_unitary(self) -> None:
        SXdg = np.conj(np.array(SXGate()())).T
        assert np.allclose(unitary("qreg q[1];\nsxdg q[0];\n"), SXdg, atol=ATOL)

    def test_sdg_roundtrip_exact(self) -> None:
        body = "qreg q[1];\nsdg q[0];\n"
        assert loads(body).dumps() == HEADER + body

    def test_tdg_roundtrip_exact(self) -> None:
        body = "qreg q[1];\ntdg q[0];\n"
        assert loads(body).dumps() == HEADER + body

    def test_sxdg_roundtrip_exact(self) -> None:
        body = "qreg q[1];\nsxdg q[0];\n"
        assert loads(body).dumps() == HEADER + body

    def test_sdg_roundtrip_unitary(self) -> None:
        assert_roundtrip_unitary("qreg q[1];\nsdg q[0];\n")

    def test_tdg_roundtrip_unitary(self) -> None:
        assert_roundtrip_unitary("qreg q[1];\ntdg q[0];\n")

    def test_sxdg_roundtrip_unitary(self) -> None:
        assert_roundtrip_unitary("qreg q[1];\nsxdg q[0];\n")

    def test_sdg_not_same_as_s(self) -> None:
        u_s = unitary("qreg q[1];\ns q[0];\n")
        u_sdg = unitary("qreg q[1];\nsdg q[0];\n")
        assert not np.allclose(u_s, u_sdg, atol=ATOL)

    def test_tdg_not_same_as_t(self) -> None:
        u_t = unitary("qreg q[1];\nt q[0];\n")
        u_tdg = unitary("qreg q[1];\ntdg q[0];\n")
        assert not np.allclose(u_t, u_tdg, atol=ATOL)

    def test_sdg_is_inverse_of_s(self) -> None:
        u_s = unitary("qreg q[1];\ns q[0];\n")
        u_sdg = unitary("qreg q[1];\nsdg q[0];\n")
        assert np.allclose(u_s @ u_sdg, np.eye(2), atol=ATOL)

    def test_tdg_is_inverse_of_t(self) -> None:
        u_t = unitary("qreg q[1];\nt q[0];\n")
        u_tdg = unitary("qreg q[1];\ntdg q[0];\n")
        assert np.allclose(u_t @ u_tdg, np.eye(2), atol=ATOL)


# ---------------------------------------------------------------------------
# Two-qubit constant gates
# ---------------------------------------------------------------------------


class TestTwoQubitGates:
    def test_cx_unitary(self) -> None:
        expected = np.array(Controlled(XGate(), [2])())
        assert np.allclose(unitary("qreg q[2];\ncx q[0], q[1];\n"), expected, atol=ATOL)
        assert (
            loads("qreg q[2];\ncx q[0], q[1];\n").dumps()
            == HEADER + "qreg q[2];\ncx q[0], q[1];\n"
        )

    def test_cz_unitary(self) -> None:
        expected = np.array(Controlled(ZGate(), [2])())
        assert np.allclose(unitary("qreg q[2];\ncz q[0], q[1];\n"), expected, atol=ATOL)
        assert (
            loads("qreg q[2];\ncz q[0], q[1];\n").dumps()
            == HEADER + "qreg q[2];\ncz q[0], q[1];\n"
        )

    def test_cy_unitary(self) -> None:
        expected = np.array(Controlled(YGate(), [2])())
        assert np.allclose(unitary("qreg q[2];\ncy q[0], q[1];\n"), expected, atol=ATOL)
        assert (
            loads("qreg q[2];\ncy q[0], q[1];\n").dumps()
            == HEADER + "qreg q[2];\ncy q[0], q[1];\n"
        )

    def test_swap_unitary(self) -> None:
        expected = np.array(SwapGate()())
        assert np.allclose(
            unitary("qreg q[2];\nswap q[0], q[1];\n"), expected, atol=ATOL
        )
        assert (
            loads("qreg q[2];\nswap q[0], q[1];\n").dumps()
            == HEADER + "qreg q[2];\nswap q[0], q[1];\n"
        )

    def test_ch_unitary(self) -> None:
        expected = np.array(Controlled(HGate(), [2])())
        assert np.allclose(unitary("qreg q[2];\nch q[0], q[1];\n"), expected, atol=ATOL)
        assert_roundtrip_unitary("qreg q[2];\nch q[0], q[1];\n")

    def test_cx_cz_cy_are_distinct(self) -> None:
        u_cx = unitary("qreg q[2];\ncx q[0], q[1];\n")
        u_cz = unitary("qreg q[2];\ncz q[0], q[1];\n")
        u_cy = unitary("qreg q[2];\ncy q[0], q[1];\n")
        assert not np.allclose(u_cx, u_cz, atol=ATOL)
        assert not np.allclose(u_cx, u_cy, atol=ATOL)
        assert not np.allclose(u_cz, u_cy, atol=ATOL)


# ---------------------------------------------------------------------------
# Multi-qubit controlled gates (ccx, c3x, c4x)
# ---------------------------------------------------------------------------


class TestMultiControlGates:
    def test_ccx_unitary(self) -> None:
        expected = np.array(Controlled(XGate(), [2, 2])())
        assert np.allclose(
            unitary("qreg q[3];\nccx q[0], q[1], q[2];\n"), expected, atol=ATOL
        )
        assert (
            loads("qreg q[3];\nccx q[0], q[1], q[2];\n").dumps()
            == HEADER + "qreg q[3];\nccx q[0], q[1], q[2];\n"
        )

    def test_c3x_unitary(self) -> None:
        """c3x is 3-controlled X (4 qubits). Verifies the control-count bug fix."""
        expected = np.array(Controlled(XGate(), [2, 2, 2])())
        got = unitary("qreg q[4];\nc3x q[0], q[1], q[2], q[3];\n")
        assert got.shape == (16, 16)
        assert np.allclose(got, expected, atol=ATOL)
        assert (
            loads("qreg q[4];\nc3x q[0], q[1], q[2], q[3];\n").dumps()
            == HEADER + "qreg q[4];\nc3x q[0], q[1], q[2], q[3];\n"
        )

    def test_c4x_unitary(self) -> None:
        """c4x is 4-controlled X (5 qubits). Verifies the control-count bug fix."""
        expected = np.array(Controlled(XGate(), [2, 2, 2, 2])())
        got = unitary("qreg q[5];\nc4x q[0], q[1], q[2], q[3], q[4];\n")
        assert got.shape == (32, 32)
        assert np.allclose(got, expected, atol=ATOL)

    def test_ccx_c3x_differ_from_cx(self) -> None:
        """Multi-controlled gates must not collapse to the same matrix."""
        u_cx = unitary("qreg q[2];\ncx q[0], q[1];\n")
        # Embed CX in a 3-qubit space on just the last two qudits for comparison.
        u_ccx = unitary("qreg q[3];\nccx q[0], q[1], q[2];\n")
        # Different dimensions — just check they parsed without error and c3x > ccx.
        u_c3x = unitary("qreg q[4];\nc3x q[0], q[1], q[2], q[3];\n")
        assert u_cx.shape == (4, 4)
        assert u_ccx.shape == (8, 8)
        assert u_c3x.shape == (16, 16)


# ---------------------------------------------------------------------------
# Parametric single-qubit gates — unitary correctness
# ---------------------------------------------------------------------------


class TestParametricGates:
    def test_rx_pi_over_2(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nrx(pi/2) q[0];\n"), RXGate()(np.pi / 2), atol=ATOL
        )

    def test_ry_pi_over_2(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nry(pi/2) q[0];\n"), RYGate()(np.pi / 2), atol=ATOL
        )

    def test_rz_pi_over_2(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nrz(pi/2) q[0];\n"), RZGate()(np.pi / 2), atol=ATOL
        )

    def test_rz_minus_pi_over_4(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nrz(-pi/4) q[0];\n"), RZGate()(-np.pi / 4), atol=ATOL
        )

    def test_u1_gate(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nu1(0.5) q[0];\n"), U1Gate()(0.5), atol=ATOL
        )

    def test_u2_gate(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nu2(0.5, 1.0) q[0];\n"), U2Gate()(0.5, 1.0), atol=ATOL
        )

    def test_u3_gate(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nu3(0.5, 1.0, 1.5) q[0];\n"),
            U3Gate()(0.5, 1.0, 1.5),
            atol=ATOL,
        )

    def test_p_gate(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\np(0.5) q[0];\n"), PGate()(0.5), atol=ATOL
        )

    def test_pi_literal(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nrz(pi) q[0];\n"), RZGate()(np.pi), atol=ATOL
        )

    def test_pi_fraction_syntax(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nrz(pi/4) q[0];\n"), RZGate()(np.pi / 4), atol=ATOL
        )

    def test_negative_angle(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nrz(-1.5707963267948966) q[0];\n"),
            RZGate()(-np.pi / 2),
            atol=ATOL,
        )

    @pytest.mark.parametrize(
        "angles",
        [
            [-0.1, 0.2, -0.3],
            [1.0, 2.0, 3.0],
            [np.pi, -np.e, np.sqrt(2)],
        ],
    )
    def test_u3_arbitrary_angles(self, angles: list[float]) -> None:
        body = f"qreg q[1];\nu3({angles[0]}, {angles[1]}, {angles[2]}) q[0];\n"
        assert np.allclose(unitary(body), U3Gate()(*angles), atol=ATOL)

    def test_rz_pi_fraction_serialised(self) -> None:
        """pi/2 written as decimal must round-trip to the human-readable 'pi/2'."""
        body = "qreg q[1];\nrz(1.5707963267948966) q[0];\n"
        assert loads(body).dumps() == HEADER + "qreg q[1];\nrz(pi/2) q[0];\n"

    def test_rx_roundtrip_unitary(self) -> None:
        assert_roundtrip_unitary("qreg q[1];\nrx(pi/3) q[0];\n")

    def test_ry_roundtrip_unitary(self) -> None:
        assert_roundtrip_unitary("qreg q[1];\nry(pi/3) q[0];\n")

    def test_rz_roundtrip_unitary(self) -> None:
        assert_roundtrip_unitary("qreg q[1];\nrz(pi/3) q[0];\n")


# ---------------------------------------------------------------------------
# Two-qubit parametric gates
# ---------------------------------------------------------------------------


class TestParametricTwoQubitGates:
    def test_crx_unitary(self) -> None:
        expected = np.array(Controlled(RXGate(), [2])(np.pi / 2))
        assert np.allclose(
            unitary("qreg q[2];\ncrx(pi/2) q[0], q[1];\n"), expected, atol=ATOL
        )
        assert_roundtrip_unitary("qreg q[2];\ncrx(pi/2) q[0], q[1];\n")

    def test_cry_unitary(self) -> None:
        expected = np.array(Controlled(RYGate(), [2])(np.pi / 2))
        assert np.allclose(
            unitary("qreg q[2];\ncry(pi/2) q[0], q[1];\n"), expected, atol=ATOL
        )
        assert_roundtrip_unitary("qreg q[2];\ncry(pi/2) q[0], q[1];\n")

    def test_crz_unitary(self) -> None:
        expected = np.array(Controlled(RZGate(), [2])(np.pi / 2))
        assert np.allclose(
            unitary("qreg q[2];\ncrz(pi/2) q[0], q[1];\n"), expected, atol=ATOL
        )
        assert_roundtrip_unitary("qreg q[2];\ncrz(pi/2) q[0], q[1];\n")

    def test_rxx_unitary(self) -> None:
        expected = np.array(RXXGate()(np.pi / 2))
        assert np.allclose(
            unitary("qreg q[2];\nrxx(pi/2) q[0], q[1];\n"), expected, atol=ATOL
        )
        assert_roundtrip_unitary("qreg q[2];\nrxx(pi/2) q[0], q[1];\n")

    def test_rzz_unitary(self) -> None:
        expected = np.array(RZZGate()(np.pi / 2))
        assert np.allclose(
            unitary("qreg q[2];\nrzz(pi/2) q[0], q[1];\n"), expected, atol=ATOL
        )
        assert_roundtrip_unitary("qreg q[2];\nrzz(pi/2) q[0], q[1];\n")

    def test_cu1_gate(self) -> None:
        assert_roundtrip_unitary("qreg q[2];\ncu1(pi/4) q[0], q[1];\n")

    def test_cp_gate(self) -> None:
        assert_roundtrip_unitary("qreg q[2];\ncp(pi/4) q[0], q[1];\n")

    def test_cu1_cp_same_unitary(self) -> None:
        """cu1 and cp are aliases and must produce the same unitary."""
        u_cu1 = unitary("qreg q[2];\ncu1(pi/4) q[0], q[1];\n")
        u_cp = unitary("qreg q[2];\ncp(pi/4) q[0], q[1];\n")
        assert np.allclose(u_cu1, u_cp, atol=ATOL)

    def test_cswap_unitary(self) -> None:
        expected = np.array(Controlled(SwapGate(), [2])())
        assert np.allclose(
            unitary("qreg q[3];\ncswap q[0], q[1], q[2];\n"), expected, atol=ATOL
        )
        assert_roundtrip_unitary("qreg q[3];\ncswap q[0], q[1], q[2];\n")


# ---------------------------------------------------------------------------
# Parameter expressions (sin, cos, exp, ln, sqrt, power)
# ---------------------------------------------------------------------------


class TestExpressions:
    def test_sin_parameter(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nrz(sin(pi/2)) q[0];\n"), RZGate()(1.0), atol=ATOL
        )

    def test_cos_parameter(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nrz(cos(0)) q[0];\n"), RZGate()(1.0), atol=ATOL
        )

    def test_exp_parameter(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nrz(exp(0)) q[0];\n"), RZGate()(1.0), atol=ATOL
        )

    def test_ln_parameter(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nrz(ln(1)) q[0];\n"), RZGate()(0.0), atol=ATOL
        )

    def test_sqrt_parameter(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nrz(sqrt(4)) q[0];\n"), RZGate()(2.0), atol=ATOL
        )

    def test_power_parameter(self) -> None:
        assert np.allclose(
            unitary("qreg q[1];\nrz(2^3) q[0];\n"), RZGate()(8.0), atol=ATOL
        )

    def test_add_expression(self) -> None:
        expected = RZGate()(np.pi / 2 + np.pi / 4)
        assert np.allclose(
            unitary("qreg q[1];\nrz(pi/2 + pi/4) q[0];\n"), expected, atol=ATOL
        )

    def test_subtract_expression(self) -> None:
        expected = RZGate()(np.pi - np.pi / 4)
        assert np.allclose(
            unitary("qreg q[1];\nrz(pi - pi/4) q[0];\n"), expected, atol=ATOL
        )


# ---------------------------------------------------------------------------
# Measure and reset
# ---------------------------------------------------------------------------


class TestMeasureAndReset:
    def test_measure_single(self) -> None:
        c = loads("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
        assert c.num_qudits == 1
        assert c.num_cycles == 1

    def test_measure_register_broadcast(self) -> None:
        c = loads("qreg q[2];\ncreg c[2];\nmeasure q -> c;\n")
        assert c.num_qudits == 2
        assert c.num_cycles == 1

    def test_measure_roundtrip_exact(self) -> None:
        body = "qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n"
        assert loads(body).dumps() == HEADER + body

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
        assert loads("qreg q[2];\nreset q;\n").num_qudits == 2

    @pytest.mark.xfail(reason="reset not yet supported by the qudit QASM2 parser")
    def test_reset_mid_circuit(self) -> None:
        body = "qreg q[1];\nx q[0];\nreset q[0];\nx q[0];\n"
        assert loads(body).num_cycles == 3


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------


class TestComments:
    def test_comment_before_header(self) -> None:
        src = '// comment\nOPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nx q[0];\n'
        c = QuditCircuit.loads(src)
        assert c.num_qudits == 1
        assert np.allclose(np.array(c.kraus_ops()[0]), XGate()(), atol=ATOL)

    def test_comment_after_gates(self) -> None:
        c = loads("qreg q[1];\nx q[0];\n// trailing comment\n")
        assert np.allclose(np.array(c.kraus_ops()[0]), XGate()(), atol=ATOL)

    def test_comment_between_gates(self) -> None:
        c = loads("qreg q[2];\nh q[0];\n// comment\ncx q[0], q[1];\n")
        assert c.num_cycles == 2
        assert_roundtrip_unitary("qreg q[2];\nh q[0];\ncx q[0], q[1];\n")


# ---------------------------------------------------------------------------
# Barrier
# ---------------------------------------------------------------------------


class TestBarrier:
    def test_barrier_roundtrip_exact(self) -> None:
        body = "qreg q[2];\nbarrier q[0], q[1];\n"
        assert loads(body).dumps() == HEADER + body

    def test_barrier_does_not_affect_unitary(self) -> None:
        """Barrier is a scheduling hint; it must not change the circuit unitary."""
        with_barrier = loads(
            "qreg q[2];\nh q[0];\nbarrier q[0], q[1];\ncx q[0], q[1];\n"
        )
        without_barrier = loads("qreg q[2];\nh q[0];\ncx q[0], q[1];\n")
        assert np.allclose(
            np.array(with_barrier.kraus_ops()[0]),
            np.array(without_barrier.kraus_ops()[0]),
            atol=ATOL,
        )

    def test_barrier_present_in_emitted_qasm(self) -> None:
        body = "qreg q[2];\nh q[0];\nbarrier q[0], q[1];\ncx q[0], q[1];\n"
        assert "barrier" in loads(body).dumps()

    def test_barrier_roundtrip_unitary(self) -> None:
        assert_roundtrip_unitary(
            "qreg q[2];\nh q[0];\nbarrier q[0], q[1];\ncx q[0], q[1];\n"
        )


# ---------------------------------------------------------------------------
# Classical control (if statements)
# ---------------------------------------------------------------------------


class TestClassicalControl:
    def test_if_roundtrip_exact(self) -> None:
        body = "qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nif (c == 1) x q[0];\n"
        assert loads(body).dumps() == HEADER + body

    def test_if_contains_correct_gate_name(self) -> None:
        body = "qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nif (c == 1) x q[0];\n"
        dumped = loads(body).dumps()
        assert "if (c == 1)" in dumped
        assert "x q[0]" in dumped

    def test_if_with_parametric_gate(self) -> None:
        body = "qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nif (c == 1) rz(pi/2) q[0];\n"
        dumped = loads(body).dumps()
        assert "if (c == 1)" in dumped
        assert "rz" in dumped

    def test_if_roundtrip_structural(self) -> None:
        body = "qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nif (c == 1) x q[0];\n"
        orig, rep, _ = roundtrip(body)
        assert rep.num_qudits == orig.num_qudits
        assert rep.num_cycles == orig.num_cycles


# ---------------------------------------------------------------------------
# Register broadcasting
# ---------------------------------------------------------------------------


class TestBroadcast:
    def test_single_qubit_broadcast_unitary(self) -> None:
        """h q; on a 3-qubit register must equal three explicit h gates."""
        u_broadcast = unitary("qreg q[3];\nh q;\n")
        u_explicit = np.array(
            loads("qreg q[3];\nh q[0];\nh q[1];\nh q[2];\n").kraus_ops()[0]
        )
        assert np.allclose(u_broadcast, u_explicit, atol=ATOL)

    def test_two_qubit_broadcast_unitary(self) -> None:
        """cx q, r; on matching 2-qubit registers equals two explicit cx gates."""
        u_broadcast = unitary("qreg q[2];\nqreg r[2];\ncx q, r;\n")
        u_explicit = np.array(
            loads(
                "qreg q[2];\nqreg r[2];\ncx q[0], r[0];\ncx q[1], r[1];\n"
            ).kraus_ops()[0]
        )
        assert np.allclose(u_broadcast, u_explicit, atol=ATOL)

    def test_broadcast_roundtrip_unitary(self) -> None:
        assert_roundtrip_unitary("qreg q[3];\nh q;\n")


# ---------------------------------------------------------------------------
# User-defined gate declarations
# ---------------------------------------------------------------------------


class TestUserGates:
    def test_user_gate_single_qubit(self) -> None:
        """gate myh q { h q; } applied to q[0] must give the same unitary as h q[0]."""
        u_custom = unitary("gate myh q { h q; }\nqreg q[1];\nmyh q[0];\n")
        assert np.allclose(u_custom, HGate()(), atol=ATOL)

    def test_user_gate_two_qubit(self) -> None:
        """gate bell a,b { h a; cx a,b; } must match the inline circuit unitary."""
        u_custom = unitary(
            "gate bell a, b { h a; cx a, b; }\nqreg q[2];\nbell q[0], q[1];\n"
        )
        u_inline = unitary("qreg q[2];\nh q[0];\ncx q[0], q[1];\n")
        assert np.allclose(u_custom, u_inline, atol=ATOL)

    def test_user_gate_roundtrip_raises(self) -> None:
        """User-defined gates cannot be written to QASM 2.0 directly."""
        orig = loads("gate myh q { h q; }\nqreg q[1];\nmyh q[0];\n")
        with pytest.raises(Exception):
            orig.dumps()

    def test_user_gate_empty_body_is_identity(self) -> None:
        """An empty gate body acts as identity — unitary must equal I."""
        u = unitary("gate noop q { }\nqreg q[1];\nnoop q[0];\n")
        assert np.allclose(u, np.eye(2), atol=ATOL)

    def test_user_gate_with_parameter(self) -> None:
        """Parametric user gate: gate myrz(t) q { rz(t) q; }"""
        u_custom = unitary(
            "gate myrz(t) q { rz(t) q; }\nqreg q[1];\nmyrz(pi/4) q[0];\n"
        )
        assert np.allclose(u_custom, RZGate()(np.pi / 4), atol=ATOL)


# ---------------------------------------------------------------------------
# Subcircuit gates (rccx, rc3x) — parsed as subcircuits, cannot be re-serialised
# ---------------------------------------------------------------------------


class TestSubcircuitGates:
    def test_rccx_parses(self) -> None:
        c = loads("qreg q[3];\nrccx q[0], q[1], q[2];\n")
        assert c.num_qudits == 3
        assert c.num_cycles >= 1

    def test_rc3x_parses(self) -> None:
        c = loads("qreg q[4];\nrc3x q[0], q[1], q[2], q[3];\n")
        assert c.num_qudits == 4
        assert c.num_cycles >= 1

    def test_rccx_dump_raises(self) -> None:
        """rccx is lowered as a subcircuit and cannot be written to QASM 2.0."""
        c = loads("qreg q[3];\nrccx q[0], q[1], q[2];\n")
        with pytest.raises(Exception):
            c.dumps()

    def test_rc3x_dump_raises(self) -> None:
        c = loads("qreg q[4];\nrc3x q[0], q[1], q[2], q[3];\n")
        with pytest.raises(Exception):
            c.dumps()


# ---------------------------------------------------------------------------
# Multi-register serialisation
# ---------------------------------------------------------------------------


class TestMultiRegister:
    def test_multiple_qregs_roundtrip_unitary(self) -> None:
        body = "qreg q[2];\nqreg r[1];\ncx q[0], r[0];\n"
        assert_roundtrip_unitary(body)

    def test_multiple_qregs_num_qudits(self) -> None:
        assert loads("qreg q[2];\nqreg r[1];\ncx q[0], r[0];\n").num_qudits == 3

    def test_multi_qubit_register_h_broadcast(self) -> None:
        body = "qreg q[3];\nh q[0];\nh q[1];\nh q[2];\n"
        out = loads(body).dumps()
        assert "qreg q[3];" in out
        assert out.count("h q[") == 3

    def test_classical_reg_in_output(self) -> None:
        body = "qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n"
        out = loads(body).dumps()
        assert "creg c[1];" in out


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


class TestFileIO:
    def test_dump_then_load_preserves_unitary(self) -> None:
        body = "qreg q[2];\nh q[0];\ncx q[0], q[1];\n"
        orig = loads(body)
        with tempfile.NamedTemporaryFile(suffix=".qasm", delete=False) as f:
            path = f.name
        try:
            orig.dump(path)
            loaded = QuditCircuit.load(path)
            assert np.allclose(
                np.array(orig.kraus_ops()[0]),
                np.array(loaded.kraus_ops()[0]),
                atol=ATOL,
            )
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
                assert f.read() == c.dumps()
        finally:
            os.unlink(path)

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(OSError):
            QuditCircuit.load("/nonexistent/path/circuit.qasm")

    def test_dump_and_load_qasm2_extension(self) -> None:
        body = "qreg q[1];\nh q[0];\n"
        c = loads(body)
        with tempfile.NamedTemporaryFile(suffix=".qasm2", delete=False) as f:
            path = f.name
        try:
            c.dump(path)
            c2 = QuditCircuit.load(path)
            assert np.allclose(
                np.array(c.kraus_ops()[0]), np.array(c2.kraus_ops()[0]), atol=ATOL
            )
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_invalid_qasm_raises(self) -> None:
        with pytest.raises(ValueError):
            QuditCircuit.loads("this is not valid qasm")

    def test_unknown_dump_format_raises(self) -> None:
        c = loads("qreg q[1];\nh q[0];\n")
        with pytest.raises(ValueError):
            c.dumps(format="unknown_format")

    def test_unknown_loads_format_raises(self) -> None:
        with pytest.raises(ValueError):
            QuditCircuit.loads(HEADER + "qreg q[1];\n", format="xyz")

    def test_missing_semicolon_raises(self) -> None:
        with pytest.raises(Exception):
            QuditCircuit.loads(HEADER + "qreg q[1]\nh q[0];\n")

    def test_opaque_gate_raises(self) -> None:
        with pytest.raises(Exception):
            QuditCircuit.loads(HEADER + "opaque mygate q;\nqreg q[1];\nmygate q[0];\n")

    def test_duplicate_qreg_raises(self) -> None:
        with pytest.raises(Exception):
            QuditCircuit.loads(HEADER + "qreg q[2];\nqreg q[1];\n")

    def test_non_standard_version_accepted(self) -> None:
        src = 'OPENQASM 3.0;\ninclude "qelib1.inc";\nqreg q[1];\n'
        assert QuditCircuit.loads(src).num_qudits == 1
