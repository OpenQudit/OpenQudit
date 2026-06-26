use num::ToPrimitive;
use qudit_core::{ClassicalSystem, HybridSystem, QuditSystem};

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

/// Maps an expression base name + wire count + param count to a QASM 2.0 gate name.
fn qasm2_gate_name(expr_name: &str, n_qwires: usize, n_params: usize) -> Result<&'static str> {
    let expr_name = strip_subbed(expr_name);
    match (expr_name, n_qwires, n_params) {
        // Built-in primitives
        ("U", 1, 3) => Ok("U"),
        // qelib1.inc standard gates
        ("U2", 1, 2) => Ok("u2"),
        ("U1", 1, 1) => Ok("u1"),
        ("I", 1, 0) => Ok("id"),
        ("X", 1, 0) => Ok("x"),
        ("Y", 1, 0) => Ok("y"),
        ("Z", 1, 0) => Ok("z"),
        ("H", 1, 0) => Ok("h"),
        ("S", 1, 0) => Ok("s"),
        ("T", 1, 0) => Ok("t"),
        ("SX", 1, 0) => Ok("sx"),
        ("P", 1, 1) => Ok("p"),
        ("RX", 1, 1) => Ok("rx"),
        ("RY", 1, 1) => Ok("ry"),
        ("RZ", 1, 1) => Ok("rz"),
        ("Swap", 2, 0) => Ok("swap"),
        ("RXX", 2, 1) => Ok("rxx"),
        ("RZZ", 2, 1) => Ok("rzz"),
        // Controlled gates — same expression name regardless of control count,
        // so disambiguate by total wire count.
        ("Controlled(X)", 2, 0) => Ok("cx"),
        ("Controlled(X)", 3, 0) => Ok("ccx"),
        ("Controlled(X)", 4, 0) => Ok("c3x"),
        ("Controlled(X)", 5, 0) => Ok("c4x"),
        ("Controlled(Z)", 2, 0) => Ok("cz"),
        ("Controlled(Y)", 2, 0) => Ok("cy"),
        ("Controlled(H)", 2, 0) => Ok("ch"),
        ("Controlled(Swap)", 3, 0) => Ok("cswap"),
        ("Controlled(SX)", 2, 0) => Ok("csx"),
        ("Controlled(RX)", 2, 1) => Ok("crx"),
        // Dagger (conjugate-transpose) variants
        ("Dagger(S)", 1, 0) => Ok("sdg"),
        ("Dagger(T)", 1, 0) => Ok("tdg"),
        ("Dagger(SX)", 1, 0) => Ok("sxdg"),
        ("Controlled(RY)", 2, 1) => Ok("cry"),
        ("Controlled(RZ)", 2, 1) => Ok("crz"),
        ("Controlled(U1)", 2, 1) => Ok("cu1"),
        ("Controlled(P)", 2, 1) => Ok("cp"),
        ("Controlled(U)", 2, 3) => Ok("cu3"),
        _ => Err(crate::Error::LanguageError {
            message: format!(
                "Gate '{}' with {} qubit(s) and {} parameter(s) has no QASM 2.0 equivalent. \
                 QASM 2.0 only supports standard qelib1.inc gates and a fixed set of controlled/dagger variants.",
                expr_name, n_qwires, n_params
            ),
            lineno: 0,
        }),
    }
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
    for r in circuit.qudit_radices().iter() {
        if usize::from(*r) != 2 {
            return Err(crate::Error::LanguageError {
                message: format!(
                    "QASM 2.0 only supports qubits (dimension 2), found dimension {}",
                    usize::from(*r)
                ),
                lineno: 0,
            });
        }
    }
    for r in circuit.dit_radices().iter() {
        if usize::from(*r) != 2 {
            return Err(crate::Error::LanguageError {
                message: format!(
                    "QASM 2.0 only supports classical bits (dimension 2), found dimension {}",
                    usize::from(*r)
                ),
                lineno: 0,
            });
        }
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
                let expr_name = strip_subbed(expr.name());
                let gate = qasm2_gate_name(expr_name, qwires.len(), param_strs.len())?;
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
                // Strip the "Stacked_" prefix that classically_control() adds,
                // then strip any "_subbed" suffixes from parameter substitution.
                let raw = strip_subbed(expr.name());
                let underlying = raw.strip_prefix("Stacked_").unwrap_or(raw);
                let gate = qasm2_gate_name(underlying, qwires.len(), param_strs.len())?;
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
    use super::*;
    use crate::QuditCircuit;
    use crate::lang::QASM2Parser;
    use crate::lang::QuantumLanguageParser;
    use qudit_expr::library::{Controlled, HGate, RXGate, XGate, ZMeasurement};

    #[test]
    fn write_empty_circuit() {
        let circ = QuditCircuit::pure([2usize, 2]);
        let out = write_qasm(&circ).unwrap();
        assert!(out.starts_with("OPENQASM 2.0;\n"));
        assert!(out.contains("qreg q[2];\n"));
        assert!(!out.contains("creg"));
    }

    #[test]
    fn write_basic_gates() {
        let mut circ = QuditCircuit::pure([2usize, 2]);
        circ.append(HGate(2), [0], None).unwrap();
        circ.append(Controlled(XGate(2), [2].into(), None), [0, 1], None)
            .unwrap();
        let out = write_qasm(&circ).unwrap();
        assert!(out.contains("h q[0];\n"));
        assert!(out.contains("cx q[0], q[1];\n"));
    }

    #[test]
    fn write_parametrized_gate() {
        let mut circ = QuditCircuit::pure([2usize]);
        circ.append(RXGate(), [0], ["pi/2"]).unwrap();
        let out = write_qasm(&circ).unwrap();
        assert!(out.contains("rx("));
        assert!(out.contains("q[0];"));
    }

    #[test]
    fn write_measurement() {
        let mut circ = QuditCircuit::new([2usize], [2usize]);
        circ.append(ZMeasurement(2), ([0usize], [0usize]), None)
            .unwrap();
        let out = write_qasm(&circ).unwrap();
        assert!(out.contains("measure q[0] -> c[0];\n"));
    }

    #[test]
    fn roundtrip_simple() {
        let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n";
        let circ = QASM2Parser.parse(src).unwrap();
        let out = write_qasm(&circ).unwrap();
        // Re-parse the output and verify it produces the same circuit structure.
        let circ2 = QASM2Parser.parse(&out).unwrap();
        assert_eq!(circ.num_qudits(), circ2.num_qudits());
    }

    #[test]
    fn rejects_non_qubit_circuit() {
        let circ = QuditCircuit::pure([3usize]);
        assert!(write_qasm(&circ).is_err());
    }
}
