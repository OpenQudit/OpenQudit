//! Qudit Gate Language (QGL) compiler.
//!
//! This module provides a JIT compiler for QGL, a domain-specific
//! language for describing unitary expressions, built on LLVM.
//!
//! # Grammar
//!
//! The grammar for QGL is as follows:
//!
//! ```text
//!
//! (* Start Symbol *)
//!
//! qobj ::= identifier radices? '(' varlist ')' '{' expression '}'
//!
//! radices ::= '<' constlist '>'
//!
//! (* Expression Grammar *)
//!
//! tensor      ::= '[' matrix (',' matrix)* ','? ']' ;
//!
//! matrix      ::= '[' row (',' row)* ','? ']' ;
//!
//! row         ::= '[' exprlist ']' ;
//!
//! exprlist    ::= (expression (',' expression)* ','?)? ;  
//!
//! expression  ::= term (('+' | '-') term)* ;
//!
//! term        ::= '~'* factor (('*' | '/') factor)* ;
//!
//! factor      ::= primary ('^' primary)* ;
//!
//! primary     ::= variable
//!              | constant
//!              | function_call
//!              | row
//!              | matrix
//!              | tensor
//!              | '(' expression ')' ;
//!
//! variable    ::= identifier ;
//!
//! varlist     ::= (variable (',' variable)* ','?)? ;
//!
//! constlist   ::= (constant (',' constant)* ','?)? ;
//!
//! constant    ::= digit+ (('.' digit+)?) ;
//!
//! function_call ::= identifier '(' exprlist ')' ;
//!
//! identifier  ::= letter (letter | digit)* ;
//!
//! digit       ::= '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' ;
//!
//! letter      ::= 'a'..'z' | 'A'..'Z' | '_' | greek_letters ;
//! ```
//!
//! Single-line comments starting with `#` are ignored. Certain greek letters
//! and special characters are parsed as their associated constants, such
//! as `π` for pi and `e` for Euler's number. The full list of recognized
//! constants is as follows:
//!
//! - `π` (pi)
//! - `e` (Euler's number)
//! - `i` (imaginary unit)
//!
//! # Examples
//!
//! A simple Hadamard gate definition:
//!
//! ```qgl
//! utry H() {
//!     [
//!         [ 1, 1 ],
//!         [ 1, ~1 ]
//!     ] / sqrt(2)
//! }
//! ```
//!
//! A qutrit phase gate:
//!
//! ```qgl
//! utry P<3>(θ0, θ1) {
//!     [
//!         [ 1, 0,       0 ],
//!         [ 0, e^(iθ0), 0 ],
//!         [ 0, 0, e^(iθ1) ],
//!     ]
//! }
//! ```

mod expr;
pub(crate) mod lexer;
mod parser;

pub use expr::Expression;
pub use expr::ParsedDefinition;

pub fn parse_scalar(input: &str) -> Result<Expression, String> {
    let mut parser = parser::Parser::new(input, true);
    let result = parser.parse_scalar();

    let scalar = match result {
        Ok(scalar) => scalar,
        Err(e) => return Err(e.error),
    };

    Ok(scalar)
}

pub fn parse_qobj(input: &str) -> Result<ParsedDefinition, String> {
    let mut parser = parser::Parser::new(input, false);
    let qdef_result = parser.parse();

    let qdef = match qdef_result {
        Ok(qdef) => qdef,
        Err(e) => return Err(e.error),
    };

    if qdef.len() != 1 {
        return Err(format!(
            "Expected exactly one qobj definition, found {}",
            qdef.len()
        ));
    }

    let qdef = qdef.into_iter().next().unwrap();

    Ok(qdef)
}

// pub fn parse_unitary(input: &str) -> Result<UnitaryDefinition, String> {
//     let mut parser = parser::Parser::new(input);
//     let udef_result = parser.parse();

//     let udef = match udef_result {
//         Ok(udef) => udef,
//         Err(e) => return Err(e.error),
//     };

//     if udef.len() != 1 {
//         return Err(format!(
//             "Expected exactly one unitary definition, found {}",
//             udef.len()
//         ));
//     }
//     let udef = udef.into_iter().next().unwrap();

//     Ok(udef)
// }
