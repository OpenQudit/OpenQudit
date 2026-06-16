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
}
