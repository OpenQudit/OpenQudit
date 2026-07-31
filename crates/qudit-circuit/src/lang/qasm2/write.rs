use num::ToPrimitive;
use qudit_core::{ClassicalSystem, QuditSystem};
use qudit_expr::{UnitaryExpression, UnitarySystemExpression};

use crate::operation::{DirectiveOperation, ExpressionOperation};
use crate::param::Parameter;
use crate::{QuditCircuit, Result};

/// Strips all trailing `_subbed` suffixes that `substitute_parameters` appends.
fn strip_subbed(name: &str) -> &str {
    let mut s = name;
    while let Some(stripped) = s.strip_suffix("_subbed") {
        s = stripped;
    }
    s
}

fn no_qasm_equivalent_error(name: &str, n_qudits: usize, n_params: usize) -> crate::Error {
    crate::Error::LanguageError {
        message: format!(
            "Gate '{}' with {} qubit(s) and {} parameter(s) has no QASM 2.0 equivalent. \
             QASM 2.0 only supports standard qelib1.inc gates and a fixed set of controlled/dagger variants.",
            strip_subbed(name),
            n_qudits,
            n_params,
        ),
        lineno: 0,
    }
}

/// Looks up the QASM 2.0 gate name for a plain unitary expression, erroring
/// with a diagnostic message if it has no fixed `qelib1.inc` equivalent.
fn qasm2_gate_name(expr: &UnitaryExpression) -> Result<&'static str> {
    expr.qasm_name().ok_or_else(|| {
        no_qasm_equivalent_error(expr.name(), expr.radices().len(), expr.num_params())
    })
}

/// Looks up the QASM 2.0 gate name for a classically-controlled unitary
/// system expression, erroring with a diagnostic message if it has no fixed
/// `qelib1.inc` equivalent.
fn qasm2_system_gate_name(expr: &UnitarySystemExpression) -> Result<&'static str> {
    expr.qasm_name()
        .ok_or_else(|| no_qasm_equivalent_error(expr.name(), expr.num_qudits(), expr.num_params()))
}

fn format_float(v: f64) -> String {
    let pi = std::f64::consts::PI;
    let ratio = v / pi;

    // Detect simple multiples/fractions of pi up to ±8*pi/8.
    for num in 0i64..=8 {
        for den in 1i64..=8 {
            let approx = num as f64 / den as f64;
            if (ratio - approx).abs() < 1e-9 {
                return if num == 0 {
                    "0".to_string()
                } else if den == 1 {
                    if num == 1 {
                        "pi".to_string()
                    } else {
                        format!("{}*pi", num)
                    }
                } else {
                    if num == 1 {
                        format!("pi/{}", den)
                    } else {
                        format!("{}*pi/{}", num, den)
                    }
                };
            }
            let neg = -(num as f64) / den as f64;
            if num != 0 && (ratio - neg).abs() < 1e-9 {
                return if den == 1 {
                    if num == 1 {
                        "-pi".to_string()
                    } else {
                        format!("-{}*pi", num)
                    }
                } else {
                    if num == 1 {
                        format!("-pi/{}", den)
                    } else {
                        format!("-{}*pi/{}", num, den)
                    }
                };
            }
        }
    }

    // Fall back to full double precision.
    format!("{:.17e}", v)
}

fn format_param(param: &Parameter) -> String {
    match param {
        Parameter::Assigned64(v) => format_float(*v),
        Parameter::Assigned32(v) => format_float(*v as f64),
        Parameter::AssignedRatio(c) => c
            .to_f64()
            .map(format_float)
            .unwrap_or_else(|| "0".to_string()),
        // Unassigned (symbolic) parameters cannot be serialised to QASM 2.0.
        // Emit "0" as a safe fallback; callers should assign all parameters first.
        Parameter::Unassigned => "0".to_string(),
    }
}

pub(super) fn write_qasm(circuit: &QuditCircuit) -> Result<String> {
    // QASM 2.0 is qubit/bit only.
    if !circuit.is_qubit_only() {
        return Err(crate::Error::LanguageError {
            message: "QASM 2.0 only supports qubits (radix 2); circuit contains higher-dimensional qudits".into(),
            lineno: 0,
        });
    }
    if !circuit.is_bit_only() {
        return Err(crate::Error::LanguageError {
            message: "QASM 2.0 only supports classical bits (radix 2); circuit contains higher-dimensional dits".into(),
            lineno: 0,
        });
    }

    let nq = circuit.num_qudits();
    let nc = circuit.num_dits();

    let mut out = String::new();
    out.push_str("OPENQASM 2.0;\n");
    out.push_str("include \"qelib1.inc\";\n");

    if nq > 0 {
        out.push_str(&format!("qreg q[{}];\n", nq));
    }
    if nc > 0 {
        out.push_str(&format!("creg c[{}];\n", nc));
    }

    for inst in circuit.iter_sorted() {
        let op_code = inst.op_code();
        let wires = inst.wires();
        let param_ids = inst.params();

        let qwires: Vec<usize> = wires.qudits().collect();
        let cwires: Vec<usize> = wires.dits().collect();

        let param_indices = circuit.params().convert_ids_to_indices(param_ids);
        let param_strs: Vec<String> = param_indices
            .iter()
            .map(|idx| format_param(&circuit.params()[idx]))
            .collect();

        let operation =
            circuit
                .operations()
                .get(op_code)
                .ok_or_else(|| crate::Error::LanguageError {
                    message: format!("Unknown operation code {:?} in circuit", op_code),
                    lineno: 0,
                })?;

        match &operation {
            crate::Operation::Expression(ExpressionOperation::UnitaryGate(expr)) => {
                let gate = qasm2_gate_name(expr)?;
                let qargs: Vec<String> = qwires.iter().map(|&i| format!("q[{}]", i)).collect();
                if param_strs.is_empty() {
                    out.push_str(&format!("{} {};\n", gate, qargs.join(", ")));
                } else {
                    out.push_str(&format!(
                        "{}({}) {};\n",
                        gate,
                        param_strs.join(","),
                        qargs.join(", ")
                    ));
                }
            }

            crate::Operation::Expression(ExpressionOperation::TerminatingMeasurement(_)) => {
                for (&q, &c) in qwires.iter().zip(cwires.iter()) {
                    out.push_str(&format!("measure q[{}] -> c[{}];\n", q, c));
                }
            }

            crate::Operation::Expression(ExpressionOperation::ClassicallyControlledUnitary(
                expr,
            )) => {
                let gate = qasm2_system_gate_name(expr)?;
                // The current lowering always activates at the highest level for each
                // control bit (i.e. all bits == 1), so value = 2^n - 1.
                let value: u64 = if cwires.is_empty() {
                    1
                } else {
                    (1u64 << cwires.len()) - 1
                };
                let qargs: Vec<String> = qwires.iter().map(|&i| format!("q[{}]", i)).collect();
                let gate_str = if param_strs.is_empty() {
                    format!("{} {}", gate, qargs.join(", "))
                } else {
                    format!("{}({}) {}", gate, param_strs.join(","), qargs.join(", "))
                };
                out.push_str(&format!("if (c == {}) {};\n", value, gate_str));
            }

            crate::Operation::Directive(DirectiveOperation::Barrier) => {
                let qargs: Vec<String> = qwires.iter().map(|&i| format!("q[{}]", i)).collect();
                out.push_str(&format!("barrier {};\n", qargs.join(", ")));
            }

            crate::Operation::Subcircuit(_) => {
                return Err(crate::Error::LanguageError {
                    message: "Subcircuit operations cannot be written to QASM 2.0 directly. \
                         Flatten the circuit first."
                        .into(),
                    lineno: 0,
                });
            }

            op => {
                return Err(crate::Error::LanguageError {
                    message: format!("Unsupported operation type for QASM 2.0 output: {:?}", op),
                    lineno: 0,
                });
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::{QASM2Parser, QASM2Writer};
    use crate::QuditCircuit;
    use crate::lang::{QuantumLanguageParser, QuantumLanguageWriter};

    const HEADER: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n";

    fn prog(body: &str) -> String {
        format!("{HEADER}{body}")
    }

    fn parse(src: &str) -> QuditCircuit {
        QASM2Parser
            .parse(src)
            .unwrap_or_else(|e| panic!("parse failed: {e}"))
    }

    fn qasm_lines(circuit: &QuditCircuit) -> Vec<String> {
        QASM2Writer
            .write(circuit)
            .unwrap()
            .lines()
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn write_bell_circuit() {
        let src = prog("qreg q[2];\nh q[0];\ncx q[0], q[1];\n");
        assert_eq!(
            qasm_lines(&parse(&src)),
            vec![
                "OPENQASM 2.0;",
                r#"include "qelib1.inc";"#,
                "qreg q[2];",
                "h q[0];",
                "cx q[0], q[1];",
            ]
        );
    }

    #[test]
    fn write_measurement_circuit() {
        let src = prog(
            "qreg q[2];\ncreg c[2];\n\
             h q[0];\ncx q[0], q[1];\n\
             measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        );
        assert_eq!(
            qasm_lines(&parse(&src)),
            vec![
                "OPENQASM 2.0;",
                r#"include "qelib1.inc";"#,
                "qreg q[2];",
                "creg c[2];",
                "h q[0];",
                "cx q[0], q[1];",
                "measure q[0] -> c[0];",
                "measure q[1] -> c[1];",
            ]
        );
    }

    #[test]
    fn write_barrier() {
        let src = prog("qreg q[2];\nh q[0];\nbarrier q[0], q[1];\ncx q[0], q[1];\n");
        assert_eq!(
            qasm_lines(&parse(&src)),
            vec![
                "OPENQASM 2.0;",
                r#"include "qelib1.inc";"#,
                "qreg q[2];",
                "h q[0];",
                "barrier q[0], q[1];",
                "cx q[0], q[1];",
            ]
        );
    }

    #[test]
    fn write_param_formats() {
        // Sequential gates on one qubit force deterministic ordering.
        // Covers: pi, pi/2, 3*pi/4, -pi/4, and 0.
        let src = prog(
            "qreg q[1];\n\
             rz(pi) q[0];\nrx(pi/2) q[0];\nrz(3*pi/4) q[0];\nrx(-pi/4) q[0];\nrx(0) q[0];\n",
        );
        assert_eq!(
            qasm_lines(&parse(&src)),
            vec![
                "OPENQASM 2.0;",
                r#"include "qelib1.inc";"#,
                "qreg q[1];",
                "rz(pi) q[0];",
                "rx(pi/2) q[0];",
                "rz(3*pi/4) q[0];",
                "rx(-pi/4) q[0];",
                "rx(0) q[0];",
            ]
        );
    }

    #[test]
    fn write_normalizes_multiple_registers() {
        // Multiple named qregs/cregs collapse to a single q[N]/c[N] register;
        // qudit and dit indices are preserved relative to insertion order.
        let src = prog(
            "qreg a[2];\nqreg b[1];\ncreg r[1];\ncreg s[2];\n\
             h a[0];\ncx a[0], b[0];\nmeasure b[0] -> s[0];\n",
        );
        assert_eq!(
            qasm_lines(&parse(&src)),
            vec![
                "OPENQASM 2.0;",
                r#"include "qelib1.inc";"#,
                "qreg q[3];",
                "creg c[3];",
                "h q[0];",
                "cx q[0], q[2];",
                "measure q[2] -> c[1];",
            ]
        );
    }

    #[test]
    fn write_classically_controlled_gate() {
        let src = prog("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nif (c == 1) x q[0];\n");
        assert_eq!(
            qasm_lines(&parse(&src)),
            vec![
                "OPENQASM 2.0;",
                r#"include "qelib1.inc";"#,
                "qreg q[1];",
                "creg c[1];",
                "measure q[0] -> c[0];",
                "if (c == 1) x q[0];",
            ]
        );
    }

    #[test]
    fn write_classically_controlled_parametric_gate() {
        let src = prog("qreg q[1];\ncreg c[1];\nif (c == 1) rz(pi/2) q[0];\n");
        assert_eq!(
            qasm_lines(&parse(&src)),
            vec![
                "OPENQASM 2.0;",
                r#"include "qelib1.inc";"#,
                "qreg q[1];",
                "creg c[1];",
                "if (c == 1) rz(pi/2) q[0];",
            ]
        );
    }

    #[test]
    fn write_single_qubit_constant_gates() {
        let src = prog(
            "qreg q[1];\n\
             x q[0];\ny q[0];\nz q[0];\nh q[0];\ns q[0];\nt q[0];\n\
             id q[0];\nsx q[0];\nsdg q[0];\ntdg q[0];\nsxdg q[0];\n",
        );
        assert_eq!(
            qasm_lines(&parse(&src)),
            vec![
                "OPENQASM 2.0;",
                r#"include "qelib1.inc";"#,
                "qreg q[1];",
                "x q[0];",
                "y q[0];",
                "z q[0];",
                "h q[0];",
                "s q[0];",
                "t q[0];",
                "id q[0];",
                "sx q[0];",
                "sdg q[0];",
                "tdg q[0];",
                "sxdg q[0];",
            ]
        );
    }

    #[test]
    fn write_two_qubit_constant_gates() {
        let src = prog(
            "qreg q[2];\n\
             cx q[0], q[1];\ncz q[0], q[1];\ncy q[0], q[1];\nswap q[0], q[1];\n\
             ch q[0], q[1];\ncsx q[0], q[1];\n",
        );
        assert_eq!(
            qasm_lines(&parse(&src)),
            vec![
                "OPENQASM 2.0;",
                r#"include "qelib1.inc";"#,
                "qreg q[2];",
                "cx q[0], q[1];",
                "cz q[0], q[1];",
                "cy q[0], q[1];",
                "swap q[0], q[1];",
                "ch q[0], q[1];",
                "csx q[0], q[1];",
            ]
        );
    }

    #[test]
    fn write_parametric_two_qubit_gates() {
        let src = prog(
            "qreg q[2];\n\
             crx(pi/2) q[0], q[1];\ncry(pi/2) q[0], q[1];\ncrz(pi/2) q[0], q[1];\n\
             cu1(pi/4) q[0], q[1];\ncp(pi/4) q[0], q[1];\n\
             rxx(pi/4) q[0], q[1];\nrzz(pi/4) q[0], q[1];\n",
        );
        assert_eq!(
            qasm_lines(&parse(&src)),
            vec![
                "OPENQASM 2.0;",
                r#"include "qelib1.inc";"#,
                "qreg q[2];",
                "crx(pi/2) q[0], q[1];",
                "cry(pi/2) q[0], q[1];",
                "crz(pi/2) q[0], q[1];",
                "cu1(pi/4) q[0], q[1];",
                "cp(pi/4) q[0], q[1];",
                "rxx(pi/4) q[0], q[1];",
                "rzz(pi/4) q[0], q[1];",
            ]
        );
    }

    #[test]
    fn write_multi_qubit_gates() {
        // All gates share wires through q[0..2], forcing deterministic order.
        let src = prog(
            "qreg q[5];\n\
             ccx q[0], q[1], q[2];\ncswap q[0], q[1], q[2];\n\
             c3x q[0], q[1], q[2], q[3];\nc4x q[0], q[1], q[2], q[3], q[4];\n",
        );
        assert_eq!(
            qasm_lines(&parse(&src)),
            vec![
                "OPENQASM 2.0;",
                r#"include "qelib1.inc";"#,
                "qreg q[5];",
                "ccx q[0], q[1], q[2];",
                "cswap q[0], q[1], q[2];",
                "c3x q[0], q[1], q[2], q[3];",
                "c4x q[0], q[1], q[2], q[3], q[4];",
            ]
        );
    }

    #[test]
    fn write_decimal_rounds_to_pi_fraction() {
        let src = prog("qreg q[1];\nrz(1.5707963267948966) q[0];\n");
        assert_eq!(
            qasm_lines(&parse(&src)),
            vec![
                "OPENQASM 2.0;",
                r#"include "qelib1.inc";"#,
                "qreg q[1];",
                "rz(pi/2) q[0];",
            ]
        );
    }

    #[test]
    fn write_rejects_qutrit_circuit() {
        assert!(QASM2Writer.write(&QuditCircuit::pure([3usize])).is_err());
    }

    #[test]
    fn write_rejects_subcircuit() {
        let src = prog("gate bell a, b { h a; cx a, b; }\nqreg q[2];\nbell q[0], q[1];\n");
        assert!(QASM2Writer.write(&parse(&src)).is_err());
    }
}
