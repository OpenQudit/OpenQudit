// ============================================================================
// 4. Lowering Passes
// ============================================================================

use std::collections::HashMap;

use qudit_expr::library::ClassicallyControlled;
use qudit_expr::library::Controlled;
use qudit_expr::library::Dagger;
use qudit_expr::library::HGate;
use qudit_expr::library::IGate;
use qudit_expr::library::PGate;
use qudit_expr::library::RXGate;
use qudit_expr::library::RXXGate;
use qudit_expr::library::RYGate;
use qudit_expr::library::RZGate;
use qudit_expr::library::RZZGate;
use qudit_expr::library::SGate;
use qudit_expr::library::SXGate;
use qudit_expr::library::SwapGate;
use qudit_expr::library::TGate;
use qudit_expr::library::U1Gate;
use qudit_expr::library::U2Gate;
use qudit_expr::library::U3Gate;
use qudit_expr::library::XGate;
use qudit_expr::library::YGate;
use qudit_expr::library::ZGate;
use qudit_expr::library::ZMeasurement;

use crate::operation::ExpressionOperation;
use crate::param::ArgumentList;
use crate::Operation;
use crate::QuditCircuit;
use crate::Result;

use super::ast::{
    Argument, BinaryOperator, Expr, GateOp, QASMGateDecl, QASMProgram, QASMStatement, Qop,
    UnaryOperator, Uop,
};
use super::parser::parse_qasm_body;

enum GateBody {
    Circ(QuditCircuit),
    Op(Operation),
}

fn resolve_stmts(
    stmts: Vec<super::ast::QASMParsedStatement>,
) -> Result<Vec<super::ast::QASMParsedStatement>> {
    let mut out = Vec::new();
    for stmt in stmts {
        if let QASMStatement::Include(path) = stmt.kind {
            if path == "qelib1.inc" {
                // If the standard library file is present on disk, load it;
                // otherwise its gates are available implicitly and we skip it.
                match std::fs::read_to_string(&path) {
                    Ok(source) => {
                        let included = parse_qasm_body(&source)?;
                        out.extend(resolve_stmts(included)?);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => {
                        return Err(crate::Error::LanguageError {
                            message: format!("failed to read include '{}': {}", path, e),
                            lineno: stmt.line,
                        });
                    }
                }
            } else {
                let source =
                    std::fs::read_to_string(&path).map_err(|e| crate::Error::LanguageError {
                        message: format!("failed to read include '{}': {}", path, e),
                        lineno: stmt.line,
                    })?;
                let included = parse_qasm_body(&source)?;
                out.extend(resolve_stmts(included)?);
            }
        } else {
            out.push(stmt);
        }
    }
    Ok(out)
}

/// Recursively resolves `include` statements in a parsed program.
///
/// - `include "qelib1.inc";` is silently dropped: its gates are always
///   available implicitly.
/// - All other includes are read from disk relative to the process working
///   directory, parsed, and their statements spliced in at the include site.
///   Nested includes are resolved depth-first.
fn resolve_includes(program: QASMProgram) -> Result<QASMProgram> {
    let statements = resolve_stmts(program.statements)?;
    Ok(QASMProgram {
        version: program.version,
        statements,
    })
}

fn build_default_gate_table() -> HashMap<String, GateBody> {
    let mut gate_table = HashMap::new();
    // Default QASM gates
    gate_table.insert("U".into(), GateBody::Op(U3Gate().into()));
    gate_table.insert(
        "CX".into(),
        GateBody::Op(Controlled(XGate(2), [2].into(), None).into()),
    );

    // qelib1.inc gates
    gate_table.insert("u3".into(), GateBody::Op(U3Gate().into()));
    gate_table.insert("u2".into(), GateBody::Op(U2Gate().into()));
    gate_table.insert("u1".into(), GateBody::Op(U1Gate().into()));
    gate_table.insert(
        "cx".into(),
        GateBody::Op(Controlled(XGate(2), [2].into(), None).into()),
    );
    gate_table.insert("id".into(), GateBody::Op(IGate(2).into()));
    gate_table.insert("u0".into(), GateBody::Op(IGate(2).into()));
    gate_table.insert("u".into(), GateBody::Op(U3Gate().into()));
    gate_table.insert("p".into(), GateBody::Op(PGate(2).into()));
    gate_table.insert("x".into(), GateBody::Op(XGate(2).into()));
    gate_table.insert("y".into(), GateBody::Op(YGate(2).into()));
    gate_table.insert("z".into(), GateBody::Op(ZGate(2).into()));
    gate_table.insert("h".into(), GateBody::Op(HGate(2).into()));
    gate_table.insert("s".into(), GateBody::Op(SGate(2).into()));
    gate_table.insert("sdg".into(), GateBody::Op(Dagger(SGate(2)).into()));
    gate_table.insert("t".into(), GateBody::Op(TGate(2).into()));
    gate_table.insert("tdg".into(), GateBody::Op(Dagger(TGate(2)).into()));
    gate_table.insert("rx".into(), GateBody::Op(RXGate().into()));
    gate_table.insert("ry".into(), GateBody::Op(RYGate().into()));
    gate_table.insert("rz".into(), GateBody::Op(RZGate().into()));
    gate_table.insert("sx".into(), GateBody::Op(SXGate().into()));
    gate_table.insert("sxdg".into(), GateBody::Op(Dagger(SXGate()).into()));
    gate_table.insert("swap".into(), GateBody::Op(SwapGate(2).into()));
    gate_table.insert(
        "cz".into(),
        GateBody::Op(Controlled(ZGate(2), [2].into(), None).into()),
    );
    gate_table.insert(
        "cy".into(),
        GateBody::Op(Controlled(YGate(2), [2].into(), None).into()),
    );
    gate_table.insert(
        "ch".into(),
        GateBody::Op(Controlled(HGate(2), [2].into(), None).into()),
    );
    gate_table.insert(
        "ccx".into(),
        GateBody::Op(Controlled(XGate(2), [2, 2].into(), None).into()),
    );
    gate_table.insert(
        "cswap".into(),
        GateBody::Op(Controlled(SwapGate(2), [2].into(), None).into()),
    );
    gate_table.insert(
        "crx".into(),
        GateBody::Op(Controlled(RXGate(), [2].into(), None).into()),
    );
    gate_table.insert(
        "cry".into(),
        GateBody::Op(Controlled(RYGate(), [2].into(), None).into()),
    );
    gate_table.insert(
        "crz".into(),
        GateBody::Op(Controlled(RZGate(), [2].into(), None).into()),
    );
    gate_table.insert(
        "cu1".into(),
        GateBody::Op(Controlled(U1Gate(), [2].into(), None).into()),
    );
    gate_table.insert(
        "cp".into(),
        GateBody::Op(Controlled(PGate(2), [2].into(), None).into()),
    );
    gate_table.insert(
        "cu3".into(),
        GateBody::Op(Controlled(U3Gate(), [2].into(), None).into()),
    );
    gate_table.insert(
        "csx".into(),
        GateBody::Op(Controlled(SXGate(), [2].into(), None).into()),
    );
    gate_table.insert(
        "cu".into(),
        GateBody::Op(Controlled(U3Gate(), [2].into(), None).into()),
    );
    gate_table.insert("rxx".into(), GateBody::Op(RXXGate().into()));
    gate_table.insert("rzz".into(), GateBody::Op(RZZGate().into()));
    gate_table.insert(
        "c3x".into(),
        GateBody::Op(Controlled(XGate(2), [2, 2, 2].into(), None).into()),
    );
    gate_table.insert(
        "c4x".into(),
        GateBody::Op(Controlled(XGate(2), [2, 2, 2, 2].into(), None).into()),
    );
    gate_table.insert(
        "c3sqrtx".into(),
        GateBody::Op(Controlled(SXGate(), [2, 2].into(), None).into()),
    );

    // rccx
    let cx = Controlled(XGate(2), [2].into(), None);
    const PI: f64 = std::f64::consts::PI;
    const PI4: f64 = PI / 4.0;
    use crate::param::ArgumentList;
    let mut rccx_circuit = QuditCircuit::pure([2, 2, 2]);
    rccx_circuit.append(U2Gate(), [2], ArgumentList::from([0.0f64, PI]));
    rccx_circuit.append(U1Gate(), [2], ArgumentList::from([PI4]));
    rccx_circuit.append(cx.clone(), [1, 2], None);
    rccx_circuit.append(U1Gate(), [2], ArgumentList::from([-PI4]));
    rccx_circuit.append(cx.clone(), [0, 2], None);
    rccx_circuit.append(U1Gate(), [2], ArgumentList::from([PI4]));
    rccx_circuit.append(cx.clone(), [1, 2], None);
    rccx_circuit.append(U1Gate(), [2], ArgumentList::from([-PI4]));
    rccx_circuit.append(U2Gate(), [2], ArgumentList::from([0.0f64, PI]));
    gate_table.insert("rccx".into(), GateBody::Circ(rccx_circuit));

    // rc3x
    let mut rc3x_circuit = QuditCircuit::pure([2, 2, 2, 2]);
    rc3x_circuit.append(U2Gate(), [3], ArgumentList::from([0.0f64, PI]));
    rc3x_circuit.append(U1Gate(), [3], ArgumentList::from([PI4]));
    rc3x_circuit.append(cx.clone(), [2, 3], None);
    rc3x_circuit.append(U1Gate(), [3], ArgumentList::from([-PI4]));
    rc3x_circuit.append(U2Gate(), [3], ArgumentList::from([0.0f64, PI]));
    rc3x_circuit.append(cx.clone(), [0, 3], None);
    rc3x_circuit.append(U1Gate(), [3], ArgumentList::from([PI4]));
    rc3x_circuit.append(cx.clone(), [1, 3], None);
    rc3x_circuit.append(U1Gate(), [3], ArgumentList::from([-PI4]));
    rc3x_circuit.append(cx.clone(), [0, 3], None);
    rc3x_circuit.append(U1Gate(), [3], ArgumentList::from([PI4]));
    rc3x_circuit.append(cx.clone(), [1, 3], None);
    rc3x_circuit.append(U1Gate(), [3], ArgumentList::from([-PI4]));
    rc3x_circuit.append(U2Gate(), [3], ArgumentList::from([0.0f64, PI]));
    rc3x_circuit.append(U1Gate(), [3], ArgumentList::from([PI4]));
    rc3x_circuit.append(cx.clone(), [2, 3], None);
    rc3x_circuit.append(U1Gate(), [3], ArgumentList::from([-PI4]));
    rc3x_circuit.append(U2Gate(), [3], ArgumentList::from([0.0f64, PI]));
    gate_table.insert("rc3x".into(), GateBody::Circ(rc3x_circuit));

    // Directives
    gate_table.insert(
        "barrier".into(),
        GateBody::Op(Operation::Directive(
            crate::operation::DirectiveOperation::Barrier,
        )),
    );
    gate_table.insert(
        "Barrier".into(),
        GateBody::Op(Operation::Directive(
            crate::operation::DirectiveOperation::Barrier,
        )),
    );

    gate_table
}

/// Converts a QASM2 [`Expr`] AST node to a [`qudit_expr::Expression`] directly,
/// without any string serialisation.  Only called for parameterised expressions
/// (constants are handled by [`eval_const_expr`] before this is reached).
fn expr_to_qexpr(expr: &Expr) -> Result<qudit_expr::Expression> {
    use qudit_expr::Expression as QExpr;
    match expr {
        Expr::Real(v) => Ok(QExpr::from_float_64(*v)),
        Expr::Integer(n) => Ok(QExpr::from_float_64(*n as f64)),
        Expr::Pi => Ok(QExpr::Pi),
        Expr::Id(s) => Ok(QExpr::Variable(s.clone())),
        Expr::UnaryOp(op, inner) => {
            let iq = expr_to_qexpr(inner)?;
            match op {
                UnaryOperator::Negate => Ok(QExpr::Neg(Box::new(iq))),
                UnaryOperator::Sqrt => Ok(QExpr::Sqrt(Box::new(iq))),
                UnaryOperator::Sin => Ok(QExpr::Sin(Box::new(iq))),
                UnaryOperator::Cos => Ok(QExpr::Cos(Box::new(iq))),
                UnaryOperator::Tan => {
                    let iq2 = expr_to_qexpr(inner)?;
                    Ok(QExpr::Div(
                        Box::new(QExpr::Sin(Box::new(iq))),
                        Box::new(QExpr::Cos(Box::new(iq2))),
                    ))
                }
                UnaryOperator::Exp | UnaryOperator::Ln => {
                    if let Some(v) = eval_const_expr(expr) {
                        Ok(QExpr::from_float_64(v))
                    } else {
                        Err(crate::Error::LanguageError {
                            message: format!(
                                "parameterised {:?} is not supported in expressions",
                                op
                            ),
                            lineno: 0,
                        })
                    }
                }
            }
        }
        Expr::BinaryOp(l, op, r) => {
            let lq = expr_to_qexpr(l)?;
            let rq = expr_to_qexpr(r)?;
            match op {
                BinaryOperator::Plus => Ok(QExpr::Add(Box::new(lq), Box::new(rq))),
                BinaryOperator::Minus => Ok(QExpr::Sub(Box::new(lq), Box::new(rq))),
                BinaryOperator::Multiply => Ok(QExpr::Mul(Box::new(lq), Box::new(rq))),
                BinaryOperator::Divide => Ok(QExpr::Div(Box::new(lq), Box::new(rq))),
                BinaryOperator::Power => Ok(QExpr::Pow(Box::new(lq), Box::new(rq))),
            }
        }
    }
}

/// Evaluates a QASM2 expression to an `f64` when it is a compile-time constant
/// (contains no variable references).  Returns `None` for parameterised exprs.
fn eval_const_expr(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Real(v) => Some(*v),
        Expr::Integer(n) => Some(*n as f64),
        Expr::Pi => Some(std::f64::consts::PI),
        Expr::Id(_) => None,
        Expr::UnaryOp(op, inner) => {
            let v = eval_const_expr(inner)?;
            Some(match op {
                UnaryOperator::Negate => -v,
                UnaryOperator::Sin => v.sin(),
                UnaryOperator::Cos => v.cos(),
                UnaryOperator::Tan => v.tan(),
                UnaryOperator::Exp => v.exp(),
                UnaryOperator::Ln => v.ln(),
                UnaryOperator::Sqrt => v.sqrt(),
            })
        }
        Expr::BinaryOp(l, op, r) => {
            let lv = eval_const_expr(l)?;
            let rv = eval_const_expr(r)?;
            Some(match op {
                BinaryOperator::Plus => lv + rv,
                BinaryOperator::Minus => lv - rv,
                BinaryOperator::Multiply => lv * rv,
                BinaryOperator::Divide => lv / rv,
                BinaryOperator::Power => lv.powf(rv),
            })
        }
    }
}

fn expr_to_param_argument(expr: &Expr) -> Result<crate::param::Argument> {
    use qudit_expr::Expression as QExpr;
    match expr {
        Expr::Real(v) => Ok(crate::param::Argument::Float64(*v)),
        Expr::Integer(n) => Ok(crate::param::Argument::Float64(*n as f64)),
        Expr::Pi => Ok(crate::param::Argument::Expression(QExpr::Pi)),
        Expr::Id(s) => Ok(crate::param::Argument::Expression(QExpr::Variable(
            s.clone(),
        ))),
        Expr::UnaryOp(_, _) | Expr::BinaryOp(_, _, _) => {
            if let Some(v) = eval_const_expr(expr) {
                return Ok(crate::param::Argument::Float64(v));
            }
            Ok(crate::param::Argument::Expression(expr_to_qexpr(expr)?))
        }
    }
}

/// Resolves a gate-body [`Argument`] to its 0-based qubit index
/// formal-argument-name → index map built from the enclosing `gate` declaration.
fn resolve_gate_decl_qarg_idx(
    arg: &Argument,
    qarg_index: &HashMap<&str, usize>,
    gate_name: &str,
    line: usize,
) -> Result<usize> {
    match arg {
        Argument::Register(name) => {
            qarg_index
                .get(name.as_str())
                .copied()
                .ok_or_else(|| crate::Error::LanguageError {
                    message: format!("unknown qubit argument '{}' in gate '{}'", name, gate_name),
                    lineno: line,
                })
        }
        Argument::Bit(name, _) => Err(crate::Error::LanguageError {
            message: format!(
                "indexed qubit '{}[...]' is not allowed inside a gate body",
                name
            ),
            lineno: line,
        }),
    }
}

/// Resolves a [`QASMGateDecl`] into a [`QuditCircuit`] by lowering each
/// [`GateOp`] in its body against the gate operations already present in `table`.
///
/// All qubits are assumed to be dimension-2 (qubit) operands.
fn resolve_gate_decl(
    decl: &QASMGateDecl,
    table: &HashMap<String, GateBody>,
    line: usize,
) -> Result<QuditCircuit> {
    let qarg_index: HashMap<&str, usize> = decl
        .qargs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let mut circuit = QuditCircuit::pure(vec![2usize; decl.qargs.len()]);

    for gate_op in &decl.body {
        let (op_name, indices, param_args): (&str, Vec<usize>, ArgumentList) = match gate_op {
            GateOp::Uop(Uop::U {
                theta,
                phi,
                lambda,
                target,
            }) => {
                let idx = resolve_gate_decl_qarg_idx(target, &qarg_index, &decl.name, line)?;
                let params = ArgumentList::new(vec![
                    expr_to_param_argument(theta)?,
                    expr_to_param_argument(phi)?,
                    expr_to_param_argument(lambda)?,
                ]);
                ("U", vec![idx], params)
            }
            GateOp::Uop(Uop::CX { control, target }) => {
                let ctrl = resolve_gate_decl_qarg_idx(control, &qarg_index, &decl.name, line)?;
                let tgt = resolve_gate_decl_qarg_idx(target, &qarg_index, &decl.name, line)?;
                ("CX", vec![ctrl, tgt], ArgumentList::new(vec![]))
            }
            GateOp::Uop(Uop::Custom {
                name,
                params: exprs,
                args: qargs,
            }) => {
                let indices = qargs
                    .iter()
                    .map(|a| resolve_gate_decl_qarg_idx(a, &qarg_index, &decl.name, line))
                    .collect::<Result<Vec<_>>>()?;
                let effective_params = if name == "u0" {
                    // u0(gamma) is a global-phase identity; gamma is ignored.
                    vec![]
                } else {
                    exprs
                        .iter()
                        .map(expr_to_param_argument)
                        .collect::<Result<Vec<_>>>()?
                };
                (name.as_str(), indices, ArgumentList::new(effective_params))
            }
            GateOp::Barrier(qargs) => {
                let indices = qargs
                    .iter()
                    .map(|a| resolve_gate_decl_qarg_idx(a, &qarg_index, &decl.name, line))
                    .collect::<Result<Vec<_>>>()?;
                ("barrier", indices, ArgumentList::new(vec![]))
            }
        };

        let gate_body = table
            .get(op_name)
            .ok_or_else(|| crate::Error::LanguageError {
                message: format!(
                    "unknown gate '{}' used in definition of '{}'",
                    op_name, decl.name
                ),
                lineno: line,
            })?;

        match gate_body {
            GateBody::Op(op) => circuit.append(op.clone(), indices, param_args)?,
            GateBody::Circ(circ) => circuit.append(circ.clone(), indices, param_args)?,
        };
    }

    Ok(circuit)
}

/// Builds a complete gate-name → [`GateBody`] table for `program`.
///
/// Starts from the built-in default table (U, CX, and all qelib1.inc gates)
/// and then processes every [`QASMStatement::GateDecl`] and
/// [`QASMStatement::OpaqueDecl`] in order, so that gates may only reference
/// names that were declared earlier (matching the OPENQASM 2.0 spec).
fn resolve_gate_table(program: &QASMProgram) -> Result<HashMap<String, GateBody>> {
    let mut table = build_default_gate_table();

    for stmt in &program.statements {
        match &stmt.kind {
            QASMStatement::GateDecl(g) => {
                if table.contains_key(&g.name) {
                    return Err(crate::Error::LanguageError {
                        message: format!("gate '{}' is already defined", g.name),
                        lineno: stmt.line,
                    });
                }
                let circuit = resolve_gate_decl(g, &table, stmt.line)?;
                table.insert(g.name.clone(), GateBody::Circ(circuit));
            }
            QASMStatement::OpaqueDecl { .. } => {
                return Err(crate::Error::LanguageError {
                    message: "Opaque gate definitions are not supported.".into(),
                    lineno: stmt.line,
                });
            }
            _ => {}
        }
    }

    Ok(table)
}

/// Maps a QASM register name to its `(start_index, size)` within the
/// `QuditCircuit`'s linear qubit or clbit space.
type ArgTable = HashMap<String, (usize, usize)>;

/// Walks `program` collecting all `qreg` and `creg` declarations and assigns
/// each register a contiguous slice of linear qubit / clbit indices.
///
/// Duplicate register names produce an error with the offending line number.
/// Returns `(qreg_table, creg_table)`.
fn resolve_registers(program: &QASMProgram) -> Result<(ArgTable, ArgTable)> {
    let mut qregs: ArgTable = HashMap::new();
    let mut cregs: ArgTable = HashMap::new();
    let mut next_qubit: usize = 0;
    let mut next_clbit: usize = 0;

    for stmt in &program.statements {
        match &stmt.kind {
            QASMStatement::QReg(name, size) => {
                if qregs.contains_key(name) {
                    return Err(crate::Error::LanguageError {
                        message: format!("quantum register '{}' is already declared", name),
                        lineno: stmt.line,
                    });
                }
                qregs.insert(name.clone(), (next_qubit, *size));
                next_qubit += size;
            }
            QASMStatement::CReg(name, size) => {
                if cregs.contains_key(name) {
                    return Err(crate::Error::LanguageError {
                        message: format!("classical register '{}' is already declared", name),
                        lineno: stmt.line,
                    });
                }
                cregs.insert(name.clone(), (next_clbit, *size));
                next_clbit += size;
            }
            _ => {}
        }
    }

    Ok((qregs, cregs))
}

/// Resolves a QASM [`Argument`] to a list of linear circuit bit indices.
///
/// - `id[n]` → `[start + n]`
/// - `id`    → `[start, start+1, …, start+size-1]`
fn resolve_arg(arg: &Argument, table: &ArgTable, line: usize) -> Result<Vec<usize>> {
    match arg {
        Argument::Bit(name, idx) => {
            let &(start, size) = table.get(name).ok_or_else(|| crate::Error::LanguageError {
                message: format!("unknown register '{}'", name),
                lineno: line,
            })?;
            if *idx >= size {
                return Err(crate::Error::LanguageError {
                    message: format!(
                        "index {} out of bounds for register '{}[{}]'",
                        idx, name, size
                    ),
                    lineno: line,
                });
            }
            Ok(vec![start + idx])
        }
        Argument::Register(name) => {
            let &(start, size) = table.get(name).ok_or_else(|| crate::Error::LanguageError {
                message: format!("unknown register '{}'", name),
                lineno: line,
            })?;
            Ok((start..start + size).collect())
        }
    }
}

/// Expands a gate's argument list into one or more per-qubit index lists,
/// implementing QASM 2.0 register broadcasting.
///
/// All register arguments must have the same size; single-bit arguments are
/// replicated across every broadcast step.
fn expand_gate_arguments(
    args: &[Argument],
    qreg_table: &ArgTable,
    line: usize,
) -> Result<Vec<Vec<usize>>> {
    let resolved: Vec<Vec<usize>> = args
        .iter()
        .map(|a| resolve_arg(a, qreg_table, line))
        .collect::<Result<_>>()?;

    let broadcast_size = resolved.iter().map(|v| v.len()).max().unwrap_or(1);

    for v in &resolved {
        if v.len() > 1 && v.len() != broadcast_size {
            return Err(crate::Error::LanguageError {
                message: "register sizes do not match in broadcast gate application".into(),
                lineno: line,
            });
        }
    }

    Ok((0..broadcast_size)
        .map(|i| {
            resolved
                .iter()
                .map(|v| if v.len() == 1 { v[0] } else { v[i] })
                .collect()
        })
        .collect())
}

/// Lowers a [`Uop`] to one or more `circuit.append` calls,
/// handling register broadcasting.
fn lower_uop(
    uop: &Uop,
    circuit: &mut QuditCircuit,
    gate_table: &HashMap<String, GateBody>,
    qreg_table: &ArgTable,
    line: usize,
) -> Result<()> {
    let (op_name, qasm_args, params): (&str, Vec<Argument>, ArgumentList) = match uop {
        Uop::U {
            theta,
            phi,
            lambda,
            target,
        } => {
            let params = ArgumentList::new(vec![
                expr_to_param_argument(theta)?,
                expr_to_param_argument(phi)?,
                expr_to_param_argument(lambda)?,
            ]);
            ("U", vec![target.clone()], params)
        }
        Uop::CX { control, target } => (
            "CX",
            vec![control.clone(), target.clone()],
            ArgumentList::new(vec![]),
        ),
        Uop::Custom {
            name,
            params: exprs,
            args,
        } => {
            let effective_params = if name == "u0" {
                // u0(gamma) is defined as U(0,0,0) in qelib1.inc: gamma is a
                // global-phase argument that has no physical effect and is
                // intentionally ignored.  IGate (its gate-table entry) takes 0
                // parameters, so we drop the single QASM argument here.
                vec![]
            } else {
                exprs
                    .iter()
                    .map(expr_to_param_argument)
                    .collect::<Result<Vec<_>>>()?
            };
            (name.as_str(), args.clone(), ArgumentList::new(effective_params))
        }
    };

    let gate_body = gate_table
        .get(op_name)
        .ok_or_else(|| crate::Error::LanguageError {
            message: format!("unknown gate '{}'", op_name),
            lineno: line,
        })?;

    for indices in expand_gate_arguments(&qasm_args, qreg_table, line)? {
        match gate_body {
            GateBody::Op(op) => circuit.append(op.clone(), indices, params.clone())?,
            GateBody::Circ(circ) => circuit.append(circ.clone(), indices, params.clone())?,
        };
    }

    Ok(())
}

/// Final lowering pass: walks `program` and appends all quantum operations
/// to a freshly created [`QuditCircuit`].
fn lower_program(
    program: &QASMProgram,
    gate_table: &HashMap<String, GateBody>,
    qreg_table: &ArgTable,
    creg_table: &ArgTable,
) -> Result<QuditCircuit> {
    let total_qubits: usize = qreg_table.values().map(|&(_, size)| size).sum();
    let total_clbits: usize = creg_table.values().map(|&(_, size)| size).sum();
    let mut circuit = QuditCircuit::new(vec![2usize; total_qubits], vec![2usize; total_clbits]);

    for stmt in &program.statements {
        match &stmt.kind {
            QASMStatement::Qop(qop) => match qop {
                Qop::Uop(uop) => {
                    lower_uop(uop, &mut circuit, gate_table, qreg_table, stmt.line)?;
                }
                Qop::Measure(src, dst) => {
                    let qubit_indices = resolve_arg(src, qreg_table, stmt.line)?;
                    let clbit_indices = resolve_arg(dst, creg_table, stmt.line)?;
                    if qubit_indices.len() != clbit_indices.len() {
                        return Err(crate::Error::LanguageError {
                            message: format!(
                                "measurement size mismatch: {} qubit(s) vs {} classical bit(s)",
                                qubit_indices.len(),
                                clbit_indices.len(),
                            ),
                            lineno: stmt.line,
                        });
                    }
                    for (q, c) in qubit_indices.into_iter().zip(clbit_indices) {
                        circuit.append(
                            ZMeasurement(2),
                            (vec![q], vec![c]),
                            ArgumentList::new(vec![]),
                        )?;
                    }
                }
                Qop::Reset(_arg) => {
                    return Err(crate::Error::LanguageError {
                        message: "reset is not supported".into(),
                        lineno: stmt.line,
                    });
                }
            },
            QASMStatement::If { creg, value, op } => {
                let &(creg_start, creg_size) =
                    creg_table
                        .get(creg)
                        .ok_or_else(|| crate::Error::LanguageError {
                            message: format!("unknown classical register '{}'", creg),
                            lineno: stmt.line,
                        })?;
                let clbit_indices: Vec<usize> = (creg_start..creg_start + creg_size).collect();

                let Qop::Uop(uop) = op else {
                    return Err(crate::Error::LanguageError {
                        message: "only unitary gate operations are supported inside if statements"
                            .into(),
                        lineno: stmt.line,
                    });
                };

                let (op_name, qasm_args, params): (&str, Vec<Argument>, ArgumentList) = match uop {
                    Uop::U {
                        theta,
                        phi,
                        lambda,
                        target,
                    } => {
                        let params = ArgumentList::new(vec![
                            expr_to_param_argument(theta)?,
                            expr_to_param_argument(phi)?,
                            expr_to_param_argument(lambda)?,
                        ]);
                        ("U", vec![target.clone()], params)
                    }
                    Uop::CX { control, target } => (
                        "CX",
                        vec![control.clone(), target.clone()],
                        ArgumentList::new(vec![]),
                    ),
                    Uop::Custom {
                        name,
                        params: exprs,
                        args,
                    } => {
                        let params = ArgumentList::new(
                            exprs
                                .iter()
                                .map(expr_to_param_argument)
                                .collect::<Result<Vec<_>>>()?,
                        );
                        (name.as_str(), args.clone(), params)
                    }
                };

                let gate_body =
                    gate_table
                        .get(op_name)
                        .ok_or_else(|| crate::Error::LanguageError {
                            message: format!("unknown gate '{}'", op_name),
                            lineno: stmt.line,
                        })?;

                for qubit_indices in expand_gate_arguments(&qasm_args, qreg_table, stmt.line)? {
                    match gate_body {
                        GateBody::Op(op) => {
                            if let Operation::Expression(expr_op) = op {
                                if let ExpressionOperation::UnitaryGate(u_expr) = expr_op {
                                    // ClassicallyControlled in qudit-expr currently only
                                    // supports single-qubit target gates; multi-qubit
                                    // targets hit a tensor-dimension mismatch.
                                    if qubit_indices.len() != 1 {
                                        return Err(crate::Error::LanguageError {
                                            message: format!(
                                                "Gate '{}' has {} target qubits; classically \
                                                 controlled gates are currently limited to \
                                                 single-qubit targets.",
                                                op_name,
                                                qubit_indices.len()
                                            ),
                                            lineno: stmt.line,
                                        });
                                    }
                                    // Assuming all qubits are dimension 2 for target radices.
                                    let target_radices: Vec<usize> =
                                        qubit_indices.iter().map(|_| 2usize).collect();
                                    let wrapped: Operation = ClassicallyControlled(
                                        u_expr.clone(),
                                        target_radices.into(),
                                        None,
                                    )
                                    .into();
                                    circuit.append(
                                        wrapped,
                                        (qubit_indices, clbit_indices.clone()),
                                        params.clone(),
                                    )?;
                                } else {
                                    return Err(crate::Error::LanguageError {
                                        message: format!(
                                            "Gate '{}' cannot be classically controlled currently.",
                                            op_name
                                        ),
                                        lineno: stmt.line,
                                    });
                                }
                            } else {
                                return Err(crate::Error::LanguageError {
                                    message: format!(
                                        "Gate '{}' cannot be classically controlled currently.",
                                        op_name
                                    ),
                                    lineno: stmt.line,
                                });
                            }
                        }
                        GateBody::Circ(_circ) => {
                            return Err(crate::Error::LanguageError {
                                message: format!(
                                    "Gate '{}' cannot be classically controlled currently.",
                                    op_name
                                ),
                                lineno: stmt.line,
                            });
                        }
                    };
                }
            }
            QASMStatement::Barrier(args) => {
                // A barrier spans all its arguments simultaneously — collect
                // all qubit indices into a single flat list.
                let indices: Vec<usize> = args
                    .iter()
                    .map(|a| resolve_arg(a, qreg_table, stmt.line))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                let barrier_body = gate_table
                    .get("barrier")
                    .expect("barrier is always present in the gate table");
                if let GateBody::Op(op) = barrier_body {
                    circuit.append(op.clone(), indices, ArgumentList::new(vec![]))?;
                }
            }
            // Consumed by earlier passes.
            QASMStatement::QReg(..)
            | QASMStatement::CReg(..)
            | QASMStatement::GateDecl(..)
            | QASMStatement::OpaqueDecl { .. }
            | QASMStatement::Include(..) => {}
        }
    }

    Ok(circuit)
}

pub(super) fn lower_qasm(ast: QASMProgram) -> Result<QuditCircuit> {
    let ast = resolve_includes(ast)?;
    let gate_table = resolve_gate_table(&ast)?;
    let (qreg_table, creg_table) = resolve_registers(&ast)?;
    lower_program(&ast, &gate_table, &qreg_table, &creg_table)
}
