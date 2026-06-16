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

use super::QuantumLanguageParser;
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
