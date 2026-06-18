/// Parses an OpenQASM 2.0 program into a `QuditCircuit`.
///
/// # OpenQASM 2.0 Grammar (Backus-Naur Form)
///
/// ```text
/// <mainprogram> ::= OPENQASM <real>; <program>
///
/// <program>     ::= <statement>
///                 | <program> <statement>
///
/// <statement>   ::= <decl>
///                 | <gatedecl> <goplist> }
///                 | <gatedecl> }
///                 | opaque <id> <idlist> ;
///                 | opaque <id> ( ) <idlist> ;
///                 | opaque <id> ( <idlist> ) <idlist> ;
///                 | <qop>
///                 | if ( <id> == <nninteger> ) <qop>
///                 | include " <id> " ;
///                 | barrier <anylist> ;
///
/// <decl>        ::= qreg <id> [ <nninteger> ] ;
///                 | creg <id> [ <nninteger> ] ;
///
/// <gatedecl>    ::= gate <id> <idlist> {
///                 | gate <id> ( ) <idlist> {
///                 | gate <id> ( <idlist> ) <idlist> {
///
/// <goplist>     ::= <uop>
///                 | barrier <idlist> ;
///                 | <goplist> <uop>
///                 | <goplist> barrier <idlist> ;
///
/// <qop>         ::= <uop>
///                 | measure <argument> -> <argument> ;
///                 | reset <argument> ;
///
/// <uop>         ::= U ( <explist> ) <argument> ;
///                 | CX <argument> , <argument> ;
///                 | <id> <anylist> ;
///                 | <id> () <anylist> ;
///                 | <id> ( <explist> ) <anylist> ;
///
/// <anylist>     ::= <idlist>
///                 | <mixedlist>
///
/// <idlist>      ::= <id>
///                 | <idlist> , <id>
///
/// <mixedlist>   ::= <id> [ <nninteger> ]
///                 | <mixedlist> , <id>
///                 | <mixedlist> , <id> [ <nninteger> ]
///                 | <idlist> , <id> [ <nninteger> ]
///
/// <argument>    ::= <id>
///                 | <id> [ <nninteger> ]
///
/// <explist>     ::= <exp>
///                 | <explist> , <exp>
///
/// <exp>         ::= <real>
///                 | <nninteger>
///                 | pi
///                 | <id>
///                 | <exp> + <exp>
///                 | <exp> - <exp>
///                 | <exp> * <exp>
///                 | <exp> / <exp>
///                 | - <exp>
///                 | <exp> ^ <exp>
///                 | ( <exp> )
///                 | <unaryop> ( <exp> )
///
/// <unaryop>     ::= sin | cos | tan | exp | ln | sqrt
/// ```
///
/// # Lexical Tokens (regular expressions):
///
/// ```text
/// <id>          ::= [a-z][A-Za-z0-9_]*
/// <real>        ::= ([0-9]+\.[0-9]*|[0-9]*\.[0-9]+)([eE][-+]?[0-9]+)?
/// <nninteger>   ::= [1-9]+[0-9]*|0
/// ```
///
/// # Implementation Notes
///
/// This lexer intentionally accepts the following super-sets of the strict grammar:
/// - `<nninteger>`: leading zeros (e.g., `007`) are accepted and parsed as their decimal value.
/// - `<real>`: bare exponent notation without a decimal point (e.g., `1e5`) is accepted.
/// - `<id>`: identifiers may start with any alphabetic character or `_`, not just `[a-z]`.
mod ast;
mod lexer;
mod lower;
mod parser;
mod write;

use super::QuantumLanguageParser;
use super::QuantumLanguageWriter;
use crate::QuditCircuit;
use crate::Result;

// ============================================================================
// 5. Public API
// ============================================================================

pub struct QASM2Parser;

impl QuantumLanguageParser for QASM2Parser {
    fn parse(&self, source: &str) -> Result<QuditCircuit> {
        let ast = parser::parse_qasm_program(source)?;
        lower::lower_qasm(ast)
    }

    fn supported_extensions(&self) -> &[&str] {
        &["qasm", "qasm2"]
    }
}

pub struct QASM2Writer;

impl QuantumLanguageWriter for QASM2Writer {
    fn write(&self, circuit: &QuditCircuit) -> Result<String> {
        write::write_qasm(circuit)
    }

    fn supported_extensions(&self) -> &[&str] {
        &["qasm", "qasm2"]
    }
}

#[cfg(test)]
mod tests {
    use super::{QASM2Parser, QASM2Writer};
    use crate::lang::{QuantumLanguageParser, QuantumLanguageWriter};
    use qudit_core::{ClassicalSystem, QuditSystem};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn parse(src: &str) -> crate::QuditCircuit {
        QASM2Parser
            .parse(src)
            .unwrap_or_else(|e| panic!("parse failed: {e}"))
    }

    fn parse_err(src: &str) -> String {
        match QASM2Parser.parse(src) {
            Ok(_) => panic!("expected parse to fail but it succeeded"),
            Err(e) => e.to_string(),
        }
    }

    fn roundtrip(src: &str) -> crate::QuditCircuit {
        let circ = parse(src);
        let out = QASM2Writer
            .write(&circ)
            .unwrap_or_else(|e| panic!("write failed: {e}"));
        parse(&out)
    }

    // Full QASM2 preamble used in most tests.
    const HEADER: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n";

    fn prog(body: &str) -> String {
        format!("{HEADER}{body}")
    }

    // -----------------------------------------------------------------------
    // Parse happy-path: registers
    // -----------------------------------------------------------------------

    #[test]
    fn parse_empty_program() {
        let circ = parse(&prog(""));
        assert_eq!(circ.num_qudits(), 0);
        assert_eq!(circ.num_dits(), 0);
        assert_eq!(circ.iter_sorted().count(), 0);
    }

    #[test]
    fn parse_qreg_only() {
        let circ = parse(&prog("qreg q[3];\n"));
        assert_eq!(circ.num_qudits(), 3);
        assert_eq!(circ.num_dits(), 0);
    }

    #[test]
    fn parse_creg_only() {
        let circ = parse(&prog("creg c[2];\n"));
        assert_eq!(circ.num_qudits(), 0);
        assert_eq!(circ.num_dits(), 2);
    }

    #[test]
    fn parse_multiple_registers() {
        let circ = parse(&prog("qreg a[2];\nqreg b[3];\ncreg c[2];\n"));
        assert_eq!(circ.num_qudits(), 5);
        assert_eq!(circ.num_dits(), 2);
    }

    // -----------------------------------------------------------------------
    // Parse happy-path: single-qubit gates
    // -----------------------------------------------------------------------

    #[test]
    fn parse_standard_single_qubit_gates() {
        let src = prog(
            "qreg q[1];\n\
             h q[0];\nx q[0];\ny q[0];\nz q[0];\n\
             s q[0];\nt q[0];\nsx q[0];\nid q[0];\n",
        );
        let circ = parse(&src);
        assert_eq!(circ.num_qudits(), 1);
        assert_eq!(circ.iter_sorted().count(), 8);
    }

    #[test]
    fn parse_sdg_tdg_sxdg() {
        let src = prog("qreg q[1];\nsdg q[0];\ntdg q[0];\nsxdg q[0];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 3);
    }

    #[test]
    fn parse_parametrized_single_qubit_gates() {
        let src = prog(
            "qreg q[1];\n\
             rx(pi/2) q[0];\nry(pi/4) q[0];\nrz(pi) q[0];\n\
             p(pi/3) q[0];\nu1(pi/6) q[0];\nu2(0,pi) q[0];\n",
        );
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 6);
    }

    #[test]
    fn parse_u_gate_with_expressions() {
        let src = prog("qreg q[1];\nU(pi/2, 0, pi) q[0];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_negative_param() {
        let src = prog("qreg q[1];\nrx(-pi/4) q[0];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_complex_param_expression() {
        let src = prog("qreg q[1];\nrz(2*pi/3 + 0.5) q[0];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    // -----------------------------------------------------------------------
    // Parse happy-path: two-qubit gates
    // -----------------------------------------------------------------------

    #[test]
    fn parse_cx_gate() {
        let src = prog("qreg q[2];\ncx q[0], q[1];\n");
        let circ = parse(&src);
        assert_eq!(circ.num_qudits(), 2);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_cx_uppercase() {
        let src = prog("qreg q[2];\nCX q[0], q[1];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_swap_gate() {
        let src = prog("qreg q[2];\nswap q[0], q[1];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_cz_cy_ch_gates() {
        let src = prog("qreg q[2];\ncz q[0], q[1];\ncy q[0], q[1];\nch q[0], q[1];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 3);
    }

    #[test]
    fn parse_crx_cry_crz() {
        let src =
            prog("qreg q[2];\ncrx(pi/2) q[0], q[1];\ncry(pi/4) q[0], q[1];\ncrz(pi) q[0], q[1];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 3);
    }

    #[test]
    fn parse_ccx_gate() {
        let src = prog("qreg q[3];\nccx q[0], q[1], q[2];\n");
        let circ = parse(&src);
        assert_eq!(circ.num_qudits(), 3);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_cswap_gate() {
        let src = prog("qreg q[3];\ncswap q[0], q[1], q[2];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    // -----------------------------------------------------------------------
    // Parse happy-path: measurement and barrier
    // -----------------------------------------------------------------------

    #[test]
    fn parse_measure_single() {
        let src = prog("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n");
        let circ = parse(&src);
        assert_eq!(circ.num_qudits(), 1);
        assert_eq!(circ.num_dits(), 1);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_measure_broadcast() {
        let src = prog("qreg q[3];\ncreg c[3];\nmeasure q -> c;\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 3);
    }

    #[test]
    fn parse_barrier() {
        let src = prog("qreg q[3];\nh q[0];\nbarrier q[0], q[1], q[2];\nh q[1];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 3);
    }

    // -----------------------------------------------------------------------
    // Parse happy-path: classical control
    // -----------------------------------------------------------------------

    #[test]
    fn parse_if_statement() {
        let src = prog("qreg q[1];\ncreg c[1];\nif (c == 1) x q[0];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    // -----------------------------------------------------------------------
    // Parse happy-path: register broadcasting
    // -----------------------------------------------------------------------

    #[test]
    fn parse_broadcast_full_register() {
        let src = prog("qreg q[3];\nh q;\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 3);
    }

    #[test]
    fn parse_broadcast_cx_one_vs_register() {
        // cx q[0], r; broadcasts over r
        let src = prog("qreg q[1];\nqreg r[3];\ncx q[0], r;\n");
        let circ = parse(&src);
        assert_eq!(circ.num_qudits(), 4);
        assert_eq!(circ.iter_sorted().count(), 3);
    }

    // -----------------------------------------------------------------------
    // Parse happy-path: custom gate declarations
    // -----------------------------------------------------------------------

    #[test]
    fn parse_custom_gate_no_params() {
        let src = prog(
            "gate bell a, b { h a; cx a, b; }\n\
             qreg q[2];\nbell q[0], q[1];\n",
        );
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_custom_gate_with_params() {
        let src = prog(
            "gate myrot(theta) q { rx(theta) q; }\n\
             qreg q[1];\nmyrot(pi/4) q[0];\n",
        );
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_custom_gate_calling_another() {
        let src = prog(
            "gate bell a, b { h a; cx a, b; }\n\
             gate double_bell a, b, c, d { bell a, b; bell c, d; }\n\
             qreg q[4];\ndouble_bell q[0], q[1], q[2], q[3];\n",
        );
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    // -----------------------------------------------------------------------
    // Parse happy-path: include
    // -----------------------------------------------------------------------

    #[test]
    fn parse_include_qelib1_is_accepted() {
        // include "qelib1.inc" must not error even when the file is absent
        let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nh q[0];\n";
        let circ = parse(src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn error_missing_version_header() {
        let err = parse_err("qreg q[1];\nh q[0];\n");
        assert!(
            err.contains("OPENQASM"),
            "expected OPENQASM mention, got: {err}"
        );
    }

    // Note: the parser stores but does not validate the version field, so
    // non-2.0 version strings are accepted without error.
    #[test]
    fn parse_non_standard_version_is_accepted() {
        let circ = parse("OPENQASM 3.0;\nqreg q[1];\n");
        assert_eq!(circ.num_qudits(), 1);
    }

    #[test]
    fn error_unknown_gate() {
        let err = parse_err(&prog("qreg q[1];\nfoo q[0];\n"));
        assert!(
            err.contains("foo"),
            "expected gate name in error, got: {err}"
        );
    }

    #[test]
    fn error_duplicate_qreg() {
        let err = parse_err(&prog("qreg q[2];\nqreg q[2];\n"));
        assert!(
            err.contains("already declared") || err.contains("already defined"),
            "{err}"
        );
    }

    #[test]
    fn error_index_out_of_bounds() {
        let err = parse_err(&prog("qreg q[2];\nh q[5];\n"));
        assert!(
            err.contains("out of bounds") || err.contains("out-of-bounds"),
            "{err}"
        );
    }

    #[test]
    fn error_unknown_register() {
        let err = parse_err(&prog("h r[0];\n"));
        assert!(err.contains("r") || err.contains("unknown"), "{err}");
    }

    #[test]
    fn error_reset_not_supported() {
        let err = parse_err(&prog("qreg q[1];\nreset q[0];\n"));
        assert!(err.contains("reset"), "{err}");
    }

    #[test]
    fn error_opaque_not_supported() {
        let err = parse_err(&prog("opaque mygate(a) q;\n"));
        assert!(err.contains("paque"), "{err}");
    }

    #[test]
    fn error_gate_redefinition() {
        let err = parse_err(&prog(
            "gate mygate a { h a; }\ngate mygate a { x a; }\nqreg q[1];\nmygate q[0];\n",
        ));
        assert!(
            err.contains("mygate") || err.contains("already defined"),
            "{err}"
        );
    }

    #[test]
    fn error_broadcast_size_mismatch() {
        // cx between two registers of different sizes — neither is 1 — must fail
        let err = parse_err(&prog("qreg a[2];\nqreg b[3];\ncx a, b;\n"));
        assert!(
            err.contains("broadcast") || err.contains("mismatch"),
            "{err}"
        );
    }

    // -----------------------------------------------------------------------
    // Roundtrips: write then re-parse preserves structure
    // -----------------------------------------------------------------------

    #[test]
    fn roundtrip_bell_circuit() {
        let src = prog("qreg q[2];\nh q[0];\ncx q[0], q[1];\n");
        let orig = parse(&src);
        let out = QASM2Writer.write(&orig).unwrap();
        let circ2 = parse(&out);
        assert_eq!(circ2.num_qudits(), orig.num_qudits());
        assert_eq!(circ2.num_dits(), orig.num_dits());
    }

    #[test]
    fn roundtrip_measurement() {
        let src = prog("qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\nmeasure q -> c;\n");
        let orig = parse(&src);
        let out = QASM2Writer.write(&orig).unwrap();
        let circ2 = parse(&out);
        assert_eq!(circ2.num_qudits(), orig.num_qudits());
        assert_eq!(circ2.num_dits(), orig.num_dits());
        assert_eq!(circ2.num_operations(), orig.num_operations());
    }

    #[test]
    fn roundtrip_parametrized_gates() {
        let src = prog("qreg q[1];\nrx(pi/2) q[0];\nrz(-pi/4) q[0];\n");
        let orig = parse(&src);
        let out = QASM2Writer.write(&orig).unwrap();
        let circ2 = parse(&out);
        assert_eq!(circ2.num_operations(), orig.num_operations());
    }

    #[test]
    fn roundtrip_barrier() {
        let src = prog("qreg q[2];\nh q[0];\nbarrier q[0], q[1];\ncx q[0], q[1];\n");
        let orig = parse(&src);
        let out = QASM2Writer.write(&orig).unwrap();
        let circ2 = parse(&out);
        assert_eq!(circ2.num_operations(), orig.num_operations());
    }

    // -----------------------------------------------------------------------
    // Additional standard gates: u0 / u / u3 aliases
    // -----------------------------------------------------------------------

    #[test]
    fn parse_u0_u_u3_aliases() {
        let src = prog(
            "qreg q[1];\n\
             u0(pi) q[0];\n\
             u(pi/2, 0, pi) q[0];\n\
             u3(pi/2, 0, pi) q[0];\n",
        );
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 3);
    }

    // -----------------------------------------------------------------------
    // Additional standard gates: controlled parametrized
    // -----------------------------------------------------------------------

    #[test]
    fn parse_cu1_cp_gates() {
        let src = prog("qreg q[2];\ncu1(pi/4) q[0], q[1];\ncp(pi/4) q[0], q[1];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 2);
    }

    #[test]
    fn parse_cu3_cu_gates() {
        let src = prog("qreg q[2];\ncu3(pi/2, 0, pi) q[0], q[1];\ncu(pi/2, 0, pi) q[0], q[1];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 2);
    }

    #[test]
    fn parse_csx_gate() {
        let src = prog("qreg q[2];\ncsx q[0], q[1];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    // -----------------------------------------------------------------------
    // Additional standard gates: two-qubit entangling
    // -----------------------------------------------------------------------

    #[test]
    fn parse_rxx_rzz_gates() {
        let src = prog("qreg q[2];\nrxx(pi/4) q[0], q[1];\nrzz(pi/4) q[0], q[1];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 2);
    }

    // -----------------------------------------------------------------------
    // Additional standard gates: multi-controlled X
    //
    // In QASM 2.0 / qelib1.inc:
    //   c3x    = 3-controlled-X → 4 qubits (3 controls + 1 target)
    //   c4x    = 4-controlled-X → 5 qubits (4 controls + 1 target)
    //   c3sqrtx = 3-controlled-SX → 4 qubits
    // -----------------------------------------------------------------------

    #[test]
    fn parse_c3x_gate() {
        let src = prog("qreg q[4];\nc3x q[0], q[1], q[2], q[3];\n");
        let circ = parse(&src);
        assert_eq!(circ.num_qudits(), 4);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_c4x_gate() {
        let src = prog("qreg q[5];\nc4x q[0], q[1], q[2], q[3], q[4];\n");
        let circ = parse(&src);
        assert_eq!(circ.num_qudits(), 5);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_c3sqrtx_gate() {
        let src = prog("qreg q[4];\nc3sqrtx q[0], q[1], q[2], q[3];\n");
        let circ = parse(&src);
        assert_eq!(circ.num_qudits(), 4);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    // -----------------------------------------------------------------------
    // Additional standard gates: circuit-backed rccx / rc3x
    // -----------------------------------------------------------------------

    #[test]
    fn parse_rccx_gate() {
        let src = prog("qreg q[3];\nrccx q[0], q[1], q[2];\n");
        let circ = parse(&src);
        assert_eq!(circ.num_qudits(), 3);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_rc3x_gate() {
        let src = prog("qreg q[4];\nrc3x q[0], q[1], q[2], q[3];\n");
        let circ = parse(&src);
        assert_eq!(circ.num_qudits(), 4);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    // -----------------------------------------------------------------------
    // Expression functions: sin, cos, tan, exp, ln, sqrt, power
    // -----------------------------------------------------------------------

    #[test]
    fn parse_trig_functions_in_params() {
        let src = prog(
            "qreg q[1];\n\
             rx(sin(pi/2)) q[0];\n\
             ry(cos(0)) q[0];\n\
             rz(tan(pi/4)) q[0];\n",
        );
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 3);
    }

    #[test]
    fn parse_exp_ln_sqrt_in_params() {
        let src = prog(
            "qreg q[1];\n\
             rx(exp(0)) q[0];\n\
             ry(ln(1)) q[0];\n\
             rz(sqrt(2)) q[0];\n",
        );
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 3);
    }

    #[test]
    fn parse_power_expression() {
        let src = prog("qreg q[1];\nrx(2^3) q[0];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    // -----------------------------------------------------------------------
    // Lexer features: comments and number formats
    // -----------------------------------------------------------------------

    #[test]
    fn parse_with_line_comments() {
        let src = prog(
            "// This is a comment\n\
             qreg q[2]; // inline comment\n\
             h q[0]; // apply H\n\
             // another comment\n\
             cx q[0], q[1];\n",
        );
        let circ = parse(&src);
        assert_eq!(circ.num_qudits(), 2);
        assert_eq!(circ.iter_sorted().count(), 2);
    }

    #[test]
    fn parse_leading_zero_integer() {
        // The lexer accepts leading zeros (super-set of spec)
        let src = prog("qreg q[007];\n");
        let circ = parse(&src);
        assert_eq!(circ.num_qudits(), 7);
    }

    #[test]
    fn parse_bare_exponent_float() {
        // The lexer accepts bare exponent notation like 1e0
        let src = prog("qreg q[1];\nrx(1e0) q[0];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_version_integer_is_accepted() {
        // OPENQASM 2; (integer, not float) should parse without error
        let circ = parse("OPENQASM 2;\nqreg q[1];\n");
        assert_eq!(circ.num_qudits(), 1);
    }

    // -----------------------------------------------------------------------
    // Gate body features
    // -----------------------------------------------------------------------

    #[test]
    fn parse_gate_empty_body() {
        // A gate with an empty body is valid — it is a no-op subcircuit
        let src = prog("gate noop a {}\nqreg q[1];\nnoop q[0];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_gate_empty_param_list() {
        // Explicit empty param list `gate foo () a { ... }` is legal
        let src = prog("gate myid () q { id q; }\nqreg q[1];\nmyid q[0];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_gate_with_barrier_in_body() {
        let src = prog(
            "gate mygate a, b { h a; barrier a, b; cx a, b; }\n\
             qreg q[2];\nmygate q[0], q[1];\n",
        );
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_gate_u_inside_body() {
        // Primitive U gate usable inside a gate body
        let src = prog(
            "gate myu(t, p, l) q { U(t, p, l) q; }\n\
             qreg q[1];\nmyu(pi/2, 0, pi) q[0];\n",
        );
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_gate_cx_inside_body() {
        // Primitive CX gate usable inside a gate body
        let src = prog(
            "gate mycx a, b { CX a, b; }\n\
             qreg q[2];\nmycx q[0], q[1];\n",
        );
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    // -----------------------------------------------------------------------
    // Classical control: additional if-statement cases
    // -----------------------------------------------------------------------

    #[test]
    fn parse_if_value_zero() {
        let src = prog("qreg q[1];\ncreg c[1];\nif (c == 0) x q[0];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    #[test]
    fn parse_if_with_cx_gate() {
        // Classically controlling a 2-qubit gate triggers a tensor-dimension
        // mismatch in qudit-expr's ClassicallyControlledUnitary constructor.
        // This is a known limitation: only 1-qubit gates can be used in `if`
        // statements currently.
        let src = prog("qreg q[2];\ncreg c[1];\nif (c == 1) cx q[0], q[1];\n");
        assert!(
            QASM2Parser.parse(&src).is_err(),
            "classically-controlled 2-qubit gate should error (known limitation)"
        );
    }

    #[test]
    fn parse_if_with_parametrized_gate() {
        let src = prog("qreg q[1];\ncreg c[1];\nif (c == 1) rx(pi/2) q[0];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    // -----------------------------------------------------------------------
    // Register semantics: index correctness
    // -----------------------------------------------------------------------

    #[test]
    fn multiple_qregs_gate_on_second_register() {
        // q[2] occupies qubits 0..1; r[3] occupies qubits 2..4
        let src = prog("qreg q[2];\nqreg r[3];\nh r[0];\nh r[1];\nh r[2];\n");
        let circ = parse(&src);
        assert_eq!(circ.num_qudits(), 5);
        assert_eq!(circ.iter_sorted().count(), 3);
    }

    #[test]
    fn measure_into_second_creg() {
        // a[2] occupies classical bits 0..1; b[2] occupies classical bits 2..3
        let src = prog(
            "qreg q[2];\ncreg a[2];\ncreg b[2];\n\
             measure q[0] -> b[0];\nmeasure q[1] -> b[1];\n",
        );
        let circ = parse(&src);
        assert_eq!(circ.num_qudits(), 2);
        assert_eq!(circ.num_dits(), 4);
        assert_eq!(circ.iter_sorted().count(), 2);
    }

    // -----------------------------------------------------------------------
    // Barrier variations
    // -----------------------------------------------------------------------

    #[test]
    fn parse_barrier_full_register_name() {
        // `barrier q;` should expand to all qubits in q as one barrier op
        let src = prog("qreg q[3];\nh q[0];\nbarrier q;\nh q[1];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 3); // h, barrier, h
    }

    #[test]
    fn parse_barrier_spans_multiple_registers() {
        let src = prog("qreg a[2];\nqreg b[2];\nbarrier a[0], b[0];\n");
        let circ = parse(&src);
        assert_eq!(circ.iter_sorted().count(), 1);
    }

    // -----------------------------------------------------------------------
    // Additional error cases
    // -----------------------------------------------------------------------

    #[test]
    fn error_duplicate_creg() {
        let err = parse_err(&prog("creg c[2];\ncreg c[2];\n"));
        assert!(
            err.contains("already declared") || err.contains("already defined"),
            "{err}"
        );
    }

    #[test]
    fn error_measure_size_mismatch() {
        // 2 qubits measured into 1 classical bit → error
        let err = parse_err(&prog("qreg q[2];\ncreg c[1];\nmeasure q -> c;\n"));
        assert!(err.contains("mismatch") || err.contains("size"), "{err}");
    }

    #[test]
    fn error_u_gate_wrong_param_count() {
        let err = parse_err(&prog("qreg q[1];\nU(pi/2, 0) q[0];\n"));
        assert!(err.contains("3") || err.contains("parameter"), "{err}");
    }

    #[test]
    fn error_unknown_creg_in_if() {
        let err = parse_err(&prog("qreg q[1];\nif (nosuch == 1) x q[0];\n"));
        assert!(err.contains("nosuch") || err.contains("unknown"), "{err}");
    }

    #[test]
    fn error_if_with_measure() {
        // `measure` is a Qop::Measure, not a Uop — lowering must reject it inside if
        let err = parse_err(&prog(
            "qreg q[1];\ncreg c[1];\ncreg d[1];\nif (c == 1) measure q[0] -> d[0];\n",
        ));
        assert!(
            err.contains("unitary") || err.contains("measure") || err.contains("support"),
            "{err}"
        );
    }

    #[test]
    fn error_if_with_circuit_backed_gate() {
        // bell is a GateBody::Circ — lowering must reject it inside if
        let err = parse_err(&prog(
            "gate bell a, b { h a; cx a, b; }\n\
             qreg q[2];\ncreg c[1];\nif (c == 1) bell q[0], q[1];\n",
        ));
        assert!(
            err.contains("classically") || err.contains("cannot") || err.contains("support"),
            "{err}"
        );
    }

    #[test]
    fn error_qreg_used_as_creg_in_measure() {
        // Using a qreg name as measure destination must fail
        let err = parse_err(&prog("qreg q[1];\nqreg r[1];\nmeasure q[0] -> r[0];\n"));
        assert!(err.contains("r") || err.contains("unknown"), "{err}");
    }

    #[test]
    fn error_creg_used_as_qubit() {
        // Using a creg name where a qubit register is expected must fail
        let err = parse_err(&prog("creg c[1];\nh c[0];\n"));
        assert!(err.contains("c") || err.contains("unknown"), "{err}");
    }

    #[test]
    fn error_indexed_qubit_in_gate_body() {
        // Indexed argument `a[0]` inside a gate body is forbidden by the spec
        let err = parse_err(&prog("gate foo a { h a[0]; }\nqreg q[1];\nfoo q[0];\n"));
        assert!(
            err.contains("indexed") || err.contains("not allowed") || err.contains("["),
            "{err}"
        );
    }

    #[test]
    fn error_redefine_builtin_gate() {
        // Redefining a built-in qelib1 gate must error
        let err = parse_err(&prog("gate h a { x a; }\nqreg q[1];\nh q[0];\n"));
        assert!(
            err.contains("already defined") || err.contains("h"),
            "{err}"
        );
    }

    #[test]
    fn error_unknown_gate_in_gate_body() {
        let err = parse_err(&prog("gate foo a { nogate a; }\nqreg q[1];\nfoo q[0];\n"));
        assert!(err.contains("nogate") || err.contains("unknown"), "{err}");
    }

    #[test]
    fn error_creg_index_out_of_bounds() {
        let err = parse_err(&prog("qreg q[1];\ncreg c[2];\nmeasure q[0] -> c[5];\n"));
        assert!(
            err.contains("out of bounds") || err.contains("out-of-bounds"),
            "{err}"
        );
    }

    // -----------------------------------------------------------------------
    // Roundtrips: additional gate types
    // -----------------------------------------------------------------------

    #[test]
    fn roundtrip_sdg_tdg_sxdg() {
        let src = prog("qreg q[1];\nsdg q[0];\ntdg q[0];\nsxdg q[0];\n");
        let circ2 = roundtrip(&src);
        assert_eq!(circ2.iter_sorted().count(), 3);
    }

    #[test]
    fn roundtrip_crx_cry_crz() {
        let src =
            prog("qreg q[2];\ncrx(pi/2) q[0], q[1];\ncry(pi/4) q[0], q[1];\ncrz(pi) q[0], q[1];\n");
        let circ2 = roundtrip(&src);
        assert_eq!(circ2.iter_sorted().count(), 3);
    }

    #[test]
    fn roundtrip_cu1_cp() {
        let src = prog("qreg q[2];\ncu1(pi/4) q[0], q[1];\ncp(pi/8) q[0], q[1];\n");
        let circ2 = roundtrip(&src);
        assert_eq!(circ2.iter_sorted().count(), 2);
    }

    #[test]
    fn roundtrip_ccx() {
        let src = prog("qreg q[3];\nccx q[0], q[1], q[2];\n");
        let circ2 = roundtrip(&src);
        assert_eq!(circ2.iter_sorted().count(), 1);
    }

    #[test]
    fn roundtrip_cswap() {
        let src = prog("qreg q[3];\ncswap q[0], q[1], q[2];\n");
        let circ2 = roundtrip(&src);
        assert_eq!(circ2.iter_sorted().count(), 1);
    }

    #[test]
    fn roundtrip_rxx_rzz() {
        let src = prog("qreg q[2];\nrxx(pi/4) q[0], q[1];\nrzz(pi/8) q[0], q[1];\n");
        let circ2 = roundtrip(&src);
        assert_eq!(circ2.iter_sorted().count(), 2);
    }

    #[test]
    fn roundtrip_c3x() {
        // c3x (3-controlled X, 4 qubits) should survive write→re-parse unchanged
        let src = prog("qreg q[4];\nc3x q[0], q[1], q[2], q[3];\n");
        let orig = parse(&src);
        let out = QASM2Writer.write(&orig).unwrap();
        let circ2 = parse(&out);
        assert_eq!(circ2.num_qudits(), orig.num_qudits());
        assert_eq!(circ2.iter_sorted().count(), orig.iter_sorted().count());
    }

    #[test]
    fn roundtrip_c4x() {
        let src = prog("qreg q[5];\nc4x q[0], q[1], q[2], q[3], q[4];\n");
        let orig = parse(&src);
        let out = QASM2Writer.write(&orig).unwrap();
        let circ2 = parse(&out);
        assert_eq!(circ2.num_qudits(), orig.num_qudits());
        assert_eq!(circ2.iter_sorted().count(), orig.iter_sorted().count());
    }

    #[test]
    fn roundtrip_if_statement() {
        let src = prog("qreg q[1];\ncreg c[1];\nif (c == 1) x q[0];\n");
        let orig = parse(&src);
        let out = QASM2Writer.write(&orig).unwrap();
        assert!(out.contains("if"), "expected 'if' in written output: {out}");
        let circ2 = parse(&out);
        assert_eq!(circ2.iter_sorted().count(), orig.iter_sorted().count());
    }

    #[test]
    fn roundtrip_full_bell_with_measurement() {
        let src = prog(
            "qreg q[2];\ncreg c[2];\n\
             h q[0];\ncx q[0], q[1];\n\
             measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        );
        let orig = parse(&src);
        let out = QASM2Writer.write(&orig).unwrap();
        let circ2 = parse(&out);
        assert_eq!(circ2.num_qudits(), orig.num_qudits());
        assert_eq!(circ2.num_dits(), orig.num_dits());
        assert_eq!(circ2.iter_sorted().count(), orig.iter_sorted().count());
    }

    #[test]
    fn roundtrip_multi_register_collapses_to_single() {
        // Writer always emits one 'q' and one 'c'; roundtrip must preserve counts.
        let src = prog(
            "qreg a[2];\nqreg b[1];\ncreg r[1];\ncreg s[2];\n\
             h a[0];\ncx a[0], b[0];\nmeasure b[0] -> s[0];\n",
        );
        let orig = parse(&src);
        let out = QASM2Writer.write(&orig).unwrap();
        let circ2 = parse(&out);
        assert_eq!(circ2.num_qudits(), orig.num_qudits());
        assert_eq!(circ2.num_dits(), orig.num_dits());
        assert_eq!(circ2.iter_sorted().count(), orig.iter_sorted().count());
    }

    // -----------------------------------------------------------------------
    // Writer-specific tests (format and failure modes)
    // -----------------------------------------------------------------------

    #[test]
    fn write_output_starts_with_header() {
        let src = prog("qreg q[1];\nh q[0];\n");
        let circ = parse(&src);
        let out = QASM2Writer.write(&circ).unwrap();
        assert!(out.starts_with("OPENQASM 2.0;\n"), "bad header: {out}");
        assert!(
            out.contains("include \"qelib1.inc\";\n"),
            "missing include: {out}"
        );
    }

    #[test]
    fn write_normalizes_to_single_qreg() {
        // Multiple source qregs collapse to a single 'q[N]' in written output.
        let src = prog("qreg a[2];\nqreg b[3];\nh a[0];\n");
        let circ = parse(&src);
        let out = QASM2Writer.write(&circ).unwrap();
        assert!(out.contains("qreg q[5]"), "expected qreg q[5] in: {out}");
        assert!(out.contains("h q[0]"), "expected h q[0] in: {out}");
    }

    #[test]
    fn write_param_pi_fractions_formatted_correctly() {
        let src = prog("qreg q[1];\nrx(pi/2) q[0];\nrz(3*pi/4) q[0];\n");
        let circ = parse(&src);
        let out = QASM2Writer.write(&circ).unwrap();
        assert!(out.contains("pi/2"), "expected 'pi/2' in: {out}");
        assert!(out.contains("3*pi/4"), "expected '3*pi/4' in: {out}");
    }

    #[test]
    fn write_param_negative_pi_formatted_correctly() {
        let src = prog("qreg q[1];\nrx(-pi/4) q[0];\n");
        let circ = parse(&src);
        let out = QASM2Writer.write(&circ).unwrap();
        assert!(out.contains("-pi/4"), "expected '-pi/4' in: {out}");
    }

    #[test]
    fn write_param_zero_formatted_correctly() {
        let src = prog("qreg q[1];\nrx(0) q[0];\n");
        let circ = parse(&src);
        let out = QASM2Writer.write(&circ).unwrap();
        assert!(out.contains("rx(0)"), "expected 'rx(0)' in: {out}");
    }

    #[test]
    fn write_param_pi_formatted_correctly() {
        let src = prog("qreg q[1];\nrz(pi) q[0];\n");
        let circ = parse(&src);
        let out = QASM2Writer.write(&circ).unwrap();
        assert!(out.contains("rz(pi)"), "expected 'rz(pi)' in: {out}");
    }

    #[test]
    fn write_arbitrary_float_param() {
        // 1.23456789 is not a simple pi fraction → falls back to scientific notation
        let src = prog("qreg q[1];\nrx(1.23456789) q[0];\n");
        let circ = parse(&src);
        let out = QASM2Writer.write(&circ).unwrap();
        assert!(out.contains("rx("), "expected 'rx(' in: {out}");
        assert!(out.contains("q[0]"), "expected 'q[0]' in: {out}");
    }

    #[test]
    fn write_subcircuit_fails() {
        // Custom gates produce subcircuit operations which the writer cannot serialize
        let src = prog(
            "gate bell a, b { h a; cx a, b; }\n\
             qreg q[2];\nbell q[0], q[1];\n",
        );
        let circ = parse(&src);
        assert!(
            QASM2Writer.write(&circ).is_err(),
            "expected write to fail for circuit containing a subcircuit operation"
        );
    }

    #[test]
    fn write_if_statement_contains_if_keyword() {
        let src = prog("qreg q[1];\ncreg c[1];\nif (c == 1) x q[0];\n");
        let circ = parse(&src);
        let out = QASM2Writer.write(&circ).unwrap();
        assert!(out.contains("if (c =="), "expected 'if (c ==' in: {out}");
    }
}
