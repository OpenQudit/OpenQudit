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
use crate::Operation;
use crate::param::ArgumentList;
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

use crate::QuditCircuit;
use crate::Result;

use std::collections::HashMap;
use std::iter::Peekable;
use std::str::Chars;

use super::QuantumLanguageParser;

// ============================================================================
// 1. AST Definitions
// ============================================================================

#[derive(Debug, PartialEq)]
pub struct QASMProgram {
    version: f64,
    statements: Vec<QASMParsedStatement>,
}

/// A parsed statement together with the source line it starts on.
#[derive(Debug, PartialEq)]
pub struct QASMParsedStatement {
    pub line: usize,
    pub kind: QASMStatement,
}

#[derive(Debug, PartialEq)]
pub struct QASMGateDecl {
    pub name: String,
    pub params: Vec<String>,
    pub qargs: Vec<String>,
    pub body: Vec<GateOp>,
}

#[derive(Debug, PartialEq)]
pub enum QASMStatement {
    QReg(String, usize),
    CReg(String, usize),
    Include(String),
    GateDecl(QASMGateDecl),
    OpaqueDecl {
        name: String,
        params: Vec<String>,
        qargs: Vec<String>,
    },
    Qop(Qop),
    If {
        creg: String,
        value: usize,
        op: Qop,
    },
    Barrier(Vec<Argument>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum GateOp {
    Uop(Uop),
    Barrier(Vec<Argument>),
}

#[derive(Debug, PartialEq)]
pub enum Qop {
    Uop(Uop),
    Measure(Argument, Argument),
    Reset(Argument),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Uop {
    U {
        theta: Expr,
        phi: Expr,
        lambda: Expr,
        target: Argument,
    },
    CX {
        control: Argument,
        target: Argument,
    },
    Custom {
        name: String,
        params: Vec<Expr>,
        args: Vec<Argument>,
    },
}

#[derive(Debug, PartialEq, Clone)]
pub enum Argument {
    Register(String),
    Bit(String, usize),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Real(f64),
    Integer(usize),
    Pi,
    Id(String),
    BinaryOp(Box<Expr>, BinaryOperator, Box<Expr>), // (+, -, *, /)
    UnaryOp(UnaryOperator, Box<Expr>),             // sin, cos, etc.
}

#[derive(Debug, PartialEq, Clone)]
pub enum BinaryOperator {
    Plus,
    Minus,
    Multiply,
    Divide,
    Power,
}

#[derive(Debug, PartialEq, Clone)]
pub enum UnaryOperator {
    Negate,
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
}

// ============================================================================
// 2. Lexer
// ============================================================================

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    // Keywords
    OpenQasm,
    Include,
    Qreg,
    Creg,
    Gate,
    Opaque,
    If,
    Barrier,
    Measure,
    Reset,
    U,
    CX,
    Pi,

    // Logic & Control
    Arrow, // ->
    EqEq,  // ==
    
    // Binary Operators
    Plus, Minus, Star, Slash, Caret,
    
    // Unary Math Functions
    Sin, Cos, Tan, Exp, Ln, Sqrt,

    // Data
    Ident(String),
    StringLit(String),
    Int(usize),
    Real(f64),
    
    // Punctuation: (, ), [, ], {, }, ;, ,
    Punct(char), 
    
    Eof,
}

pub struct Lexer<'a> {
    chars: Peekable<Chars<'a>>,
    line: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Lexer { chars: source.chars().peekable(), line: 1 }
    }

    /// Advances the iterator and tracks newlines for error reporting.
    fn advance(&mut self) -> Option<char> {
        let c = self.chars.next();
        if c == Some('\n') {
            self.line += 1;
        }
        c
    }

    fn lex_error(&self, message: impl Into<String>) -> crate::Error {
        crate::Error::LanguageError { message: message.into(), lineno: self.line }
    }

    fn skip_whitespace(&mut self) {
        while self.chars.peek().is_some_and(|&c| c.is_whitespace()) {
            self.advance();
        }
    }

    pub fn next_token(&mut self) -> Result<Token> {
        // Use a loop so if we consume a comment, we can restart the token search
        loop {
            self.skip_whitespace();

            let c = match self.advance() {
                Some(c) => c,
                None => return Ok(Token::Eof),
            };

            match c {
                '/' => {
                    if self.chars.peek().is_some_and(|&nc| nc == '/') {
                        // Comment: skip to end of line and restart loop
                        while self.chars.peek().is_some_and(|&nc| nc != '\n') {
                            self.advance();
                        }
                        continue;
                    } else {
                        // Division operator
                        return Ok(Token::Slash);
                    }
                }

                ';' | ',' | '(' | ')' | '[' | ']' | '{' | '}' => return Ok(Token::Punct(c)),
                '+' => return Ok(Token::Plus),
                '*' => return Ok(Token::Star),
                '^' => return Ok(Token::Caret),
                '-' => {
                    if self.chars.peek().is_some_and(|&nc| nc == '>') {
                        self.advance();
                        return Ok(Token::Arrow);
                    } else {
                        return Ok(Token::Minus);
                    }
                }
                '=' => {
                    if self.chars.peek().is_some_and(|&nc| nc == '=') {
                        self.advance();
                        return Ok(Token::EqEq);
                    } else {
                        return Err(self.lex_error("expected '==' but found single '='"));
                    }
                }
                '"' => {
                    let mut s = String::new();
                    loop {
                        match self.advance() {
                            Some('"') => break,
                            Some(nc)  => s.push(nc),
                            None => return Err(self.lex_error("unterminated string literal")),
                        }
                    }
                    return Ok(Token::StringLit(s));
                }
                _ if c.is_alphabetic() || c == '_' => {
                    let mut s = String::new();
                    s.push(c);
                    while self.chars.peek().is_some_and(|&nc| nc.is_alphanumeric() || nc == '_') {
                        // peek() confirmed Some, so advance() is infallible here
                        s.push(self.advance().expect("peek guaranteed Some"));
                    }
                    return Ok(match s.as_str() {
                        "OPENQASM" => Token::OpenQasm,
                        "include"  => Token::Include,
                        "qreg"     => Token::Qreg,
                        "creg"     => Token::Creg,
                        "gate"     => Token::Gate,
                        "opaque"   => Token::Opaque,
                        "if"       => Token::If,
                        "barrier"  => Token::Barrier,
                        "measure"  => Token::Measure,
                        "reset"    => Token::Reset,
                        "U"        => Token::U,
                        "CX"       => Token::CX,
                        "pi"       => Token::Pi,
                        "sin"      => Token::Sin,
                        "cos"      => Token::Cos,
                        "tan"      => Token::Tan,
                        "exp"      => Token::Exp,
                        "ln"       => Token::Ln,
                        "sqrt"     => Token::Sqrt,
                        _          => Token::Ident(s),
                    });
                }
                _ if c.is_ascii_digit() || c == '.' => {
                    let mut s = String::new();
                    s.push(c);
                    let mut is_real = c == '.';

                    while let Some(&nc) = self.chars.peek() {
                        if nc.is_ascii_digit() || nc == '.' || nc == 'e' || nc == 'E' ||
                           ((nc == '+' || nc == '-') && (s.ends_with('e') || s.ends_with('E')))
                        {
                            if nc == '.' || nc == 'e' || nc == 'E' { is_real = true; }
                            // peek() confirmed Some, so advance() is infallible here
                            s.push(self.advance().expect("peek guaranteed Some"));
                        } else {
                            break;
                        }
                    }

                    if is_real {
                        let val = s.parse::<f64>().map_err(|e| {
                            self.lex_error(format!("invalid float literal '{}': {}", s, e))
                        })?;
                        return Ok(Token::Real(val));
                    } else {
                        let val = s.parse::<usize>().map_err(|e| {
                            self.lex_error(format!("invalid integer literal '{}': {}", s, e))
                        })?;
                        return Ok(Token::Int(val));
                    }
                }
                _ => return Err(self.lex_error(format!("unexpected character '{}'", c))),
            }
        }
    }
}

// ============================================================================
// 3. Recursive Descent Parser
// ============================================================================

struct Parser<'a> {
    lexer: Lexer<'a>,
    /// One-token lookahead buffer.
    peeked: Option<Token>,
}

impl<'a> Parser<'a> {
    fn new(source: &'a str) -> Self {
        Parser { lexer: Lexer::new(source), peeked: None }
    }

    /// Returns a clone of the next token without consuming it.
    fn peek(&mut self) -> Result<Token> {
        if self.peeked.is_none() {
            self.peeked = Some(self.lexer.next_token()?);
        }
        Ok(self.peeked.as_ref().unwrap().clone())
    }

    /// Consumes and returns the next token.
    fn advance(&mut self) -> Result<Token> {
        match self.peeked.take() {
            Some(t) => Ok(t),
            None => self.lexer.next_token(),
        }
    }

    fn parse_error(&self, message: impl Into<String>) -> crate::Error {
        crate::Error::LanguageError { message: message.into(), lineno: self.lexer.line }
    }

    fn expect_punct(&mut self, ch: char) -> Result<()> {
        match self.advance()? {
            Token::Punct(c) if c == ch => Ok(()),
            t => Err(self.parse_error(format!("expected '{}', got {:?}", ch, t))),
        }
    }

    fn expect_ident(&mut self) -> Result<String> {
        match self.advance()? {
            Token::Ident(s) => Ok(s),
            t => Err(self.parse_error(format!("expected identifier, got {:?}", t))),
        }
    }

    fn expect_int(&mut self) -> Result<usize> {
        match self.advance()? {
            Token::Int(n) => Ok(n),
            t => Err(self.parse_error(format!("expected integer, got {:?}", t))),
        }
    }

    // -----------------------------------------------------------------------
    // Top-level: Entry
    // -----------------------------------------------------------------------

    /// `<mainprogram> ::= OPENQASM <real> ; <program>`
    fn parse_program(&mut self) -> Result<QASMProgram> {
        match self.advance()? {
            Token::OpenQasm => {}
            t => return Err(self.parse_error(format!("expected 'OPENQASM', got {:?}", t))),
        }
        let version = match self.advance()? {
            Token::Real(v) => v,
            Token::Int(n)  => n as f64,
            t => return Err(self.parse_error(format!("expected version number, got {:?}", t))),
        };
        self.expect_punct(';')?;
        let statements = self.parse_body()?;
        Ok(QASMProgram { version, statements })
    }

    /// `<program> ::= <statement>*`
    ///
    /// Parses a sequence of statements without a leading `OPENQASM` header.
    /// This is the entry point used when parsing included files, which are
    /// not required to carry their own version declaration.
    fn parse_body(&mut self) -> Result<Vec<QASMParsedStatement>> {
        let mut statements = Vec::new();
        while self.peek()? != Token::Eof {
            statements.push(self.parse_statement()?);
        }
        Ok(statements)
    }

    // -----------------------------------------------------------------------
    // Statements
    // -----------------------------------------------------------------------

    fn parse_statement(&mut self) -> Result<QASMParsedStatement> {
        // Load the lookahead token first so that lexer.line already reflects
        // the line on which this statement begins.
        self.peek()?;
        let line = self.lexer.line;
        let kind = match self.peek()? {
            Token::Qreg | Token::Creg => self.parse_decl()?,
            Token::Gate               => self.parse_gate_decl_stmt()?,
            Token::Opaque             => self.parse_opaque()?,
            Token::If                 => self.parse_if()?,
            Token::Include            => self.parse_include()?,
            Token::Barrier => {
                self.advance()?;
                let args = self.parse_anylist()?;
                self.expect_punct(';')?;
                QASMStatement::Barrier(args)
            }
            _ => QASMStatement::Qop(self.parse_qop()?),
        };
        Ok(QASMParsedStatement { line, kind })
    }

    /// `<decl> ::= qreg <id> [ <nninteger> ] ; | creg <id> [ <nninteger> ] ;`
    fn parse_decl(&mut self) -> Result<QASMStatement> {
        let is_qreg = matches!(self.advance()?, Token::Qreg);
        let name    = self.expect_ident()?;
        self.expect_punct('[')?;
        let size = self.expect_int()?;
        self.expect_punct(']')?;
        self.expect_punct(';')?;
        if is_qreg { Ok(QASMStatement::QReg(name, size)) } else { Ok(QASMStatement::CReg(name, size)) }
    }

    /// `<gatedecl> <goplist> } | <gatedecl> }`
    fn parse_gate_decl_stmt(&mut self) -> Result<QASMStatement> {
        self.advance()?; // consume 'gate'
        let name   = self.expect_ident()?;
        let params = self.parse_optional_param_idlist()?;
        let qargs  = self.parse_idlist()?;
        self.expect_punct('{')?;
        let body = if self.peek()? == Token::Punct('}') { vec![] } else { self.parse_goplist()? };
        self.expect_punct('}')?;
        Ok(QASMStatement::GateDecl(QASMGateDecl { name, params, qargs, body }))
    }

    /// `opaque <id> <idlist> ; | opaque <id> () <idlist> ; | opaque <id> (<idlist>) <idlist> ;`
    fn parse_opaque(&mut self) -> Result<QASMStatement> {
        self.advance()?; // consume 'opaque'
        let name   = self.expect_ident()?;
        let params = self.parse_optional_param_idlist()?;
        let qargs  = self.parse_idlist()?;
        self.expect_punct(';')?;
        Ok(QASMStatement::OpaqueDecl { name, params, qargs })
    }

    /// Parses an optional `( <idlist> )` or `()` parameter list, returning the identifiers.
    /// If there is no opening `(`, returns an empty `Vec`.
    fn parse_optional_param_idlist(&mut self) -> Result<Vec<String>> {
        if self.peek()? != Token::Punct('(') {
            return Ok(vec![]);
        }
        self.advance()?; // consume '('
        let params = if self.peek()? == Token::Punct(')') { vec![] } else { self.parse_idlist()? };
        self.expect_punct(')')?;
        Ok(params)
    }

    /// `if ( <id> == <nninteger> ) <qop>`
    fn parse_if(&mut self) -> Result<QASMStatement> {
        self.advance()?; // consume 'if'
        self.expect_punct('(')?;
        let creg = self.expect_ident()?;
        match self.advance()? {
            Token::EqEq => {}
            t => return Err(self.parse_error(format!("expected '==', got {:?}", t))),
        }
        let value = self.expect_int()?;
        self.expect_punct(')')?;
        let op = self.parse_qop()?;
        Ok(QASMStatement::If { creg, value, op })
    }

    /// `include " <string> " ;`
    fn parse_include(&mut self) -> Result<QASMStatement> {
        self.advance()?; // consume 'include'
        match self.advance()? {
            Token::StringLit(s) => {
                self.expect_punct(';')?;
                Ok(QASMStatement::Include(s))
            }
            t => Err(self.parse_error(format!("expected string literal after 'include', got {:?}", t))),
        }
    }

    // -----------------------------------------------------------------------
    // Gate body
    // -----------------------------------------------------------------------

    /// `<goplist> ::= (<uop> | barrier <idlist> ;)*`
    fn parse_goplist(&mut self) -> Result<Vec<GateOp>> {
        let mut ops = Vec::new();
        loop {
            match self.peek()? {
                Token::Punct('}') | Token::Eof => break,
                Token::Barrier => {
                    self.advance()?;
                    let args = self.parse_idlist()?
                        .into_iter()
                        .map(Argument::Register)
                        .collect();
                    self.expect_punct(';')?;
                    ops.push(GateOp::Barrier(args));
                }
                _ => ops.push(GateOp::Uop(self.parse_uop()?)),
            }
        }
        Ok(ops)
    }

    // -----------------------------------------------------------------------
    // Quantum operations
    // -----------------------------------------------------------------------

    /// `<qop> ::= <uop> | measure <argument> -> <argument> ; | reset <argument> ;`
    fn parse_qop(&mut self) -> Result<Qop> {
        match self.peek()? {
            Token::Measure => {
                self.advance()?;
                let src = self.parse_argument()?;
                match self.advance()? {
                    Token::Arrow => {}
                    t => return Err(self.parse_error(format!("expected '->', got {:?}", t))),
                }
                let dst = self.parse_argument()?;
                self.expect_punct(';')?;
                Ok(Qop::Measure(src, dst))
            }
            Token::Reset => {
                self.advance()?;
                let arg = self.parse_argument()?;
                self.expect_punct(';')?;
                Ok(Qop::Reset(arg))
            }
            _ => Ok(Qop::Uop(self.parse_uop()?)),
        }
    }

    /// `<uop> ::= U(<explist>) <argument> ;`
    ///          `| CX <argument> , <argument> ;`
    ///          `| <id> [(<explist>)] <anylist> ;`
    fn parse_uop(&mut self) -> Result<Uop> {
        match self.peek()? {
            Token::U => {
                self.advance()?;
                self.expect_punct('(')?;
                let mut params = self.parse_explist()?;
                if params.len() != 3 {
                    return Err(self.parse_error(
                        format!("U gate requires exactly 3 parameters, got {}", params.len()),
                    ));
                }
                self.expect_punct(')')?;
                let lambda = params.pop().unwrap();
                let phi    = params.pop().unwrap();
                let theta  = params.pop().unwrap();
                let target = self.parse_argument()?;
                self.expect_punct(';')?;
                Ok(Uop::U { theta, phi, lambda, target })
            }
            Token::CX => {
                self.advance()?;
                let control = self.parse_argument()?;
                self.expect_punct(',')?;
                let target  = self.parse_argument()?;
                self.expect_punct(';')?;
                Ok(Uop::CX { control, target })
            }
            Token::Ident(_) => {
                let name   = self.expect_ident()?;
                let params = self.parse_optional_param_explist()?;
                let args   = self.parse_anylist()?;
                self.expect_punct(';')?;
                Ok(Uop::Custom { name, params, args })
            }
            t => Err(self.parse_error(format!("expected gate operation (U, CX, or identifier), got {:?}", t))),
        }
    }

    /// Parses an optional `( <explist> )` or `()` expression parameter list.
    /// If there is no opening `(`, returns an empty `Vec`.
    fn parse_optional_param_explist(&mut self) -> Result<Vec<Expr>> {
        if self.peek()? != Token::Punct('(') {
            return Ok(vec![]);
        }
        self.advance()?; // consume '('
        let params = if self.peek()? == Token::Punct(')') { vec![] } else { self.parse_explist()? };
        self.expect_punct(')')?;
        Ok(params)
    }

    // -----------------------------------------------------------------------
    // Lists
    // -----------------------------------------------------------------------

    /// `<anylist> ::= <idlist> | <mixedlist>`
    ///
    /// Parsed uniformly: a comma-separated list of `<id>` or `<id> [ <nninteger> ]`.
    fn parse_anylist(&mut self) -> Result<Vec<Argument>> {
        let mut args = vec![self.parse_argument()?];
        while self.peek()? == Token::Punct(',') {
            self.advance()?;
            args.push(self.parse_argument()?);
        }
        Ok(args)
    }

    /// `<idlist> ::= <id> | <idlist> , <id>`
    fn parse_idlist(&mut self) -> Result<Vec<String>> {
        let mut ids = vec![self.expect_ident()?];
        while self.peek()? == Token::Punct(',') {
            // Peek ahead to distinguish `id,` from what follows in other contexts.
            // We only consume the comma if the next token after it is an identifier
            // without a following `[`, to avoid over-consuming in anylist/argument
            // contexts.  For idlist the spec guarantees the next item is a plain id.
            self.advance()?; // consume ','
            ids.push(self.expect_ident()?);
        }
        Ok(ids)
    }

    /// `<argument> ::= <id> | <id> [ <nninteger> ]`
    fn parse_argument(&mut self) -> Result<Argument> {
        let id = self.expect_ident()?;
        if self.peek()? == Token::Punct('[') {
            self.advance()?;
            let idx = self.expect_int()?;
            self.expect_punct(']')?;
            Ok(Argument::Bit(id, idx))
        } else {
            Ok(Argument::Register(id))
        }
    }

    // -----------------------------------------------------------------------
    // Expressions
    // -----------------------------------------------------------------------

    /// `<explist> ::= <exp> | <explist> , <exp>`
    fn parse_explist(&mut self) -> Result<Vec<Expr>> {
        let mut exps = vec![self.parse_exp()?];
        while self.peek()? == Token::Punct(',') {
            self.advance()?;
            exps.push(self.parse_exp()?);
        }
        Ok(exps)
    }

    /// `<exp> ::= <term> (('+' | '-') <term>)*`  (left-associative, lowest precedence)
    fn parse_exp(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_term()?;
        loop {
            let op = match self.peek()? {
                Token::Plus  => BinaryOperator::Plus,
                Token::Minus => BinaryOperator::Minus,
                _            => break,
            };
            self.advance()?;
            lhs = Expr::BinaryOp(Box::new(lhs), op, Box::new(self.parse_term()?));
        }
        Ok(lhs)
    }

    /// `<term> ::= <power> (('*' | '/') <power>)*`  (left-associative)
    fn parse_term(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_power()?;
        loop {
            let op = match self.peek()? {
                Token::Star  => BinaryOperator::Multiply,
                Token::Slash => BinaryOperator::Divide,
                _            => break,
            };
            self.advance()?;
            lhs = Expr::BinaryOp(Box::new(lhs), op, Box::new(self.parse_power()?));
        }
        Ok(lhs)
    }

    /// `<power> ::= <atom> ('^' <power>)?`  (right-associative, highest binary precedence)
    fn parse_power(&mut self) -> Result<Expr> {
        let base = self.parse_exp_atom()?;
        if self.peek()? == Token::Caret {
            self.advance()?;
            Ok(Expr::BinaryOp(Box::new(base), BinaryOperator::Power, Box::new(self.parse_power()?)))
        } else {
            Ok(base)
        }
    }

    /// Parses an atomic expression (no binary operators at this level).
    fn parse_exp_atom(&mut self) -> Result<Expr> {
        match self.peek()? {
            // Unary minus: `-<atom>` (binds tightly, same as any unary prefix)
            Token::Minus => {
                self.advance()?;
                let inner = self.parse_exp_atom()?;
                Ok(Expr::UnaryOp(UnaryOperator::Negate, Box::new(inner)))
            }
            // Grouped expression
            Token::Punct('(') => {
                self.advance()?;
                let inner = self.parse_exp()?;
                self.expect_punct(')')?;
                Ok(inner)
            }
            Token::Real(v) => {
                self.advance()?;
                Ok(Expr::Real(v))
            }
            Token::Int(n) => {
                self.advance()?;
                Ok(Expr::Integer(n))
            }
            Token::Pi => {
                self.advance()?;
                Ok(Expr::Pi)
            }
            Token::Ident(s) => {
                self.advance()?;
                Ok(Expr::Id(s))
            }
            // Named unary math functions: `<unaryop> ( <exp> )`
            Token::Sin | Token::Cos | Token::Tan
            | Token::Exp | Token::Ln | Token::Sqrt => {
                let op = match self.advance()? {
                    Token::Sin  => UnaryOperator::Sin,
                    Token::Cos  => UnaryOperator::Cos,
                    Token::Tan  => UnaryOperator::Tan,
                    Token::Exp  => UnaryOperator::Exp,
                    Token::Ln   => UnaryOperator::Ln,
                    Token::Sqrt => UnaryOperator::Sqrt,
                    _           => unreachable!(),
                };
                self.expect_punct('(')?;
                let inner = self.parse_exp()?;
                self.expect_punct(')')?;
                Ok(Expr::UnaryOp(op, Box::new(inner)))
            }
            t => Err(self.parse_error(format!("expected expression atom, got {:?}", t))),
        }
    }
}

// ============================================================================
// 4. Lowering Passes
// ============================================================================

fn resolve_stmts(stmts: Vec<QASMParsedStatement>) -> Result<Vec<QASMParsedStatement>> {
    let mut out = Vec::new();
    for stmt in stmts {
        if let QASMStatement::Include(path) = stmt.kind {
            if path == "qelib1.inc" {
                // If the standard library file is present on disk, load it;
                // otherwise its gates are available implicitly and we skip it.
                match std::fs::read_to_string(&path) {
                    Ok(source) => {
                        let included = Parser::new(&source).parse_body()?;
                        out.extend(resolve_stmts(included)?);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(crate::Error::LanguageError {
                        message: format!("failed to read include '{}': {}", path, e),
                        lineno: stmt.line,
                    }),
                }
            } else {
                let source = std::fs::read_to_string(&path).map_err(|e| {
                    crate::Error::LanguageError {
                        message: format!("failed to read include '{}': {}", path, e),
                        lineno: stmt.line,
                    }
                })?;
                let included = Parser::new(&source).parse_body()?;
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
    Ok(QASMProgram { version: program.version, statements })
}

enum GateBody {
    Circ(QuditCircuit),
    Op(Operation),
}


fn build_default_gate_table() -> HashMap<String, GateBody> {
    let mut gate_table = HashMap::new();
    // Default QASM gates
    gate_table.insert("U".into(), GateBody::Op(U3Gate().into()));
    gate_table.insert("CX".into(), GateBody::Op(Controlled(XGate(2), [2].into(), None).into()));

    // qelib1.inc gates
    gate_table.insert("u3".into(), GateBody::Op(U3Gate().into()));
    gate_table.insert("u2".into(), GateBody::Op(U2Gate().into()));
    gate_table.insert("u1".into(), GateBody::Op(U1Gate().into()));
    gate_table.insert("cx".into(), GateBody::Op(Controlled(XGate(2), [2].into(), None).into()));
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
    gate_table.insert("cz".into(), GateBody::Op(Controlled(ZGate(2), [2].into(), None).into()));
    gate_table.insert("cy".into(), GateBody::Op(Controlled(YGate(2), [2].into(), None).into()));
    gate_table.insert("ch".into(), GateBody::Op(Controlled(HGate(2), [2].into(), None).into()));
    gate_table.insert("ccx".into(), GateBody::Op(Controlled(XGate(2), [2, 2].into(), None).into()));
    gate_table.insert("cswap".into(), GateBody::Op(Controlled(SwapGate(2), [2].into(), None).into()));
    gate_table.insert("crx".into(), GateBody::Op(Controlled(RXGate(), [2].into(), None).into()));
    gate_table.insert("cry".into(), GateBody::Op(Controlled(RYGate(), [2].into(), None).into()));
    gate_table.insert("crz".into(), GateBody::Op(Controlled(RZGate(), [2].into(), None).into()));
    gate_table.insert("cu1".into(), GateBody::Op(Controlled(U1Gate(), [2].into(), None).into()));
    gate_table.insert("cp".into(), GateBody::Op(Controlled(PGate(2), [2].into(), None).into()));
    gate_table.insert("cu3".into(), GateBody::Op(Controlled(U3Gate(), [2].into(), None).into()));
    gate_table.insert("csx".into(), GateBody::Op(Controlled(SXGate(), [2].into(), None).into()));
    gate_table.insert("cu".into(), GateBody::Op(Controlled(U3Gate(), [2].into(), None).into()));
    gate_table.insert("rxx".into(), GateBody::Op(RXXGate().into()));
    gate_table.insert("rzz".into(), GateBody::Op(RZZGate().into()));
    gate_table.insert("c3x".into(), GateBody::Op(Controlled(XGate(2), [2, 2].into(), None).into()));
    gate_table.insert("c4x".into(), GateBody::Op(Controlled(XGate(2), [2, 2, 2].into(), None).into()));
    gate_table.insert("c3sqrtx".into(), GateBody::Op(Controlled(SXGate(), [2, 2].into(), None).into()));

    // rccx
    let cx = Controlled(XGate(2), [2].into(), None);
    let mut rccx_circuit = QuditCircuit::pure([2, 2, 2]);
    rccx_circuit.append(U2Gate(), [2], ["0", "pi"]);
    rccx_circuit.append(U1Gate(), [2], ["pi/4"]);
    rccx_circuit.append(cx.clone(), [1, 2], None);
    rccx_circuit.append(U1Gate(), [2], ["-pi/4"]);
    rccx_circuit.append(cx.clone(), [0, 2], None);
    rccx_circuit.append(U1Gate(), [2], ["pi/4"]);
    rccx_circuit.append(cx.clone(), [1, 2], None);
    rccx_circuit.append(U1Gate(), [2], ["-pi/4"]);
    rccx_circuit.append(U2Gate(), [2], ["0", "pi"]);
    gate_table.insert("rccx".into(), GateBody::Circ(rccx_circuit));

    // rc3x
    let mut rc3x_circuit = QuditCircuit::pure([2, 2, 2, 2]);
    rc3x_circuit.append(U2Gate(), [3], ["0", "pi"]);
    rc3x_circuit.append(U1Gate(), [3], ["pi/4"]);
    rc3x_circuit.append(cx.clone(), [2, 3], None);
    rc3x_circuit.append(U1Gate(), [3], ["-pi/4"]);
    rc3x_circuit.append(U2Gate(), [3], ["0", "pi"]);
    rc3x_circuit.append(cx.clone(), [0, 3], None);
    rc3x_circuit.append(U1Gate(), [3], ["pi/4"]);
    rc3x_circuit.append(cx.clone(), [1, 3], None);
    rc3x_circuit.append(U1Gate(), [3], ["-pi/4"]);
    rc3x_circuit.append(cx.clone(), [0, 3], None);
    rc3x_circuit.append(U1Gate(), [3], ["pi/4"]);
    rc3x_circuit.append(cx.clone(), [1, 3], None);
    rc3x_circuit.append(U1Gate(), [3], ["-pi/4"]);
    rc3x_circuit.append(U2Gate(), [3], ["0", "pi"]);
    rc3x_circuit.append(U1Gate(), [3], ["pi/4"]);
    rc3x_circuit.append(cx.clone(), [2, 3], None);
    rc3x_circuit.append(U1Gate(), [3], ["-pi/4"]);
    rc3x_circuit.append(U2Gate(), [3], ["0", "pi"]);
    gate_table.insert("rc3x".into(), GateBody::Circ(rc3x_circuit));

    // Directives
    gate_table.insert("barrier".into(), GateBody::Op(Operation::Directive(crate::operation::DirectiveOperation::Barrier)));
    gate_table.insert("Barrier".into(), GateBody::Op(Operation::Directive(crate::operation::DirectiveOperation::Barrier)));

    // Return
    gate_table
}

/// Serialises an [`Expr`] node to a string. Used internally as a bridge for
/// compound expressions where the `qudit_expr` builder API is not directly
/// accessible. Prefer [`expr_to_param_argument`] for all call sites.
fn expr_to_string(expr: &Expr) -> String {
    match expr {
        Expr::Real(v)    => v.to_string(),
        Expr::Integer(n) => n.to_string(),
        Expr::Pi         => "pi".into(),
        Expr::Id(s)      => s.clone(),
        Expr::BinaryOp(l, op, r) => {
            let sym = match op {
                BinaryOperator::Plus     => "+",
                BinaryOperator::Minus    => "-",
                BinaryOperator::Multiply => "*",
                BinaryOperator::Divide   => "/",
                BinaryOperator::Power    => "^",
            };
            format!("({} {} {})", expr_to_string(l), sym, expr_to_string(r))
        }
        Expr::UnaryOp(op, inner) => match op {
            UnaryOperator::Negate => format!("(-{})",         expr_to_string(inner)),
            UnaryOperator::Sin    => format!("sin({})",       expr_to_string(inner)),
            UnaryOperator::Cos    => format!("cos({})",       expr_to_string(inner)),
            UnaryOperator::Tan    => format!("tan({})",       expr_to_string(inner)),
            UnaryOperator::Exp    => format!("exp({})",       expr_to_string(inner)),
            UnaryOperator::Ln     => format!("ln({})",        expr_to_string(inner)),
            UnaryOperator::Sqrt   => format!("sqrt({})",      expr_to_string(inner)),
        },
    }
}

/// Converts a QASM2 [`Expr`] AST node directly into a [`crate::param::Argument`],
/// bypassing any string-serialisation round-trip.
///
/// Leaf nodes (`Real`, `Integer`, `Pi`, `Id`) are constructed without going
/// through the parser. Compound expressions (`BinaryOp`, `UnaryOp`) fall back
/// to the string bridge since the `qudit_expr::Expression` builder API is not
/// exposed at this level.
fn expr_to_param_argument(expr: &Expr) -> Result<crate::param::Argument> {
    use qudit_expr::Expression as QExpr;
    match expr {
        Expr::Real(v)    => Ok(crate::param::Argument::Float64(*v)),
        Expr::Integer(n) => Ok(crate::param::Argument::Float64(*n as f64)),
        Expr::Pi         => Ok(crate::param::Argument::Expression(QExpr::Pi)),
        Expr::Id(s)      => Ok(crate::param::Argument::Expression(QExpr::Variable(s.clone()))),
        compound => crate::param::Argument::try_from(expr_to_string(compound)).map_err(|e| {
            crate::Error::LanguageError {
                message: format!("invalid parameter expression: {}", e),
                lineno: 0,
            }
        }),
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
            qarg_index.get(name.as_str()).copied().ok_or_else(|| crate::Error::LanguageError {
                message: format!("unknown qubit argument '{}' in gate '{}'", name, gate_name),
                lineno: line,
            })
        }
        Argument::Bit(name, _) => Err(crate::Error::LanguageError {
            message: format!("indexed qubit '{}[...]' is not allowed inside a gate body", name),
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
    let qarg_index: HashMap<&str, usize> =
        decl.qargs.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();

    let mut circuit = QuditCircuit::pure(vec![2usize; decl.qargs.len()]);

    for gate_op in &decl.body {
        let (op_name, indices, param_args): (&str, Vec<usize>, ArgumentList) = match gate_op {
            GateOp::Uop(Uop::U { theta, phi, lambda, target }) => {
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
                let tgt  = resolve_gate_decl_qarg_idx(target,  &qarg_index, &decl.name, line)?;
                ("CX", vec![ctrl, tgt], ArgumentList::new(vec![]))
            }
            GateOp::Uop(Uop::Custom { name, params: exprs, args: qargs }) => {
                let indices = qargs.iter()
                    .map(|a| resolve_gate_decl_qarg_idx(a, &qarg_index, &decl.name, line))
                    .collect::<Result<Vec<_>>>()?;
                let params = ArgumentList::new(
                    exprs.iter().map(expr_to_param_argument).collect::<Result<Vec<_>>>()?)
                ;
                (name.as_str(), indices, params)
            }
            GateOp::Barrier(qargs) => {
                let indices = qargs.iter()
                    .map(|a| resolve_gate_decl_qarg_idx(a, &qarg_index, &decl.name, line))
                    .collect::<Result<Vec<_>>>()?;
                ("barrier", indices, ArgumentList::new(vec![]))
            }
        };

        let gate_body = table.get(op_name).ok_or_else(|| crate::Error::LanguageError {
            message: format!("unknown gate '{}' used in definition of '{}'", op_name, decl.name),
            lineno: line,
        })?;

        match gate_body {
            GateBody::Op(op)     => circuit.append(op.clone(),   indices, param_args)?,
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
                    message: format!("index {} out of bounds for register '{}[{}]'", idx, name, size),
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
    let resolved: Vec<Vec<usize>> = args.iter()
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
        .map(|i| resolved.iter().map(|v| if v.len() == 1 { v[0] } else { v[i] }).collect())
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
        Uop::U { theta, phi, lambda, target } => {
            let params = ArgumentList::new(vec![
                expr_to_param_argument(theta)?,
                expr_to_param_argument(phi)?,
                expr_to_param_argument(lambda)?,
            ]);
            ("U", vec![target.clone()], params)
        }
        Uop::CX { control, target } => {
            ("CX", vec![control.clone(), target.clone()], ArgumentList::new(vec![]))
        }
        Uop::Custom { name, params: exprs, args } => {
            let params = ArgumentList::new(
                exprs.iter().map(expr_to_param_argument).collect::<Result<Vec<_>>>()?)
            ;
            (name.as_str(), args.clone(), params)
        }
    };

    let gate_body = gate_table.get(op_name).ok_or_else(|| crate::Error::LanguageError {
        message: format!("unknown gate '{}'", op_name),
        lineno: line,
    })?;

    for indices in expand_gate_arguments(&qasm_args, qreg_table, line)? {
        match gate_body {
            GateBody::Op(op)     => circuit.append(op.clone(),  indices, params.clone())?,
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
                                qubit_indices.len(), clbit_indices.len(),
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
                let &(creg_start, creg_size) = creg_table.get(creg).ok_or_else(|| {
                    crate::Error::LanguageError {
                        message: format!("unknown classical register '{}'", creg),
                        lineno: stmt.line,
                    }
                })?;
                let clbit_indices: Vec<usize> = (creg_start..creg_start + creg_size).collect();

                let Qop::Uop(uop) = op else {
                    return Err(crate::Error::LanguageError {
                        message: "only unitary gate operations are supported inside if statements".into(),
                        lineno: stmt.line,
                    });
                };

                let (op_name, qasm_args, params): (&str, Vec<Argument>, ArgumentList) = match uop {
                    Uop::U { theta, phi, lambda, target } => {
                        let params = ArgumentList::new(vec![
                            expr_to_param_argument(theta)?,
                            expr_to_param_argument(phi)?,
                            expr_to_param_argument(lambda)?,
                        ]);
                        ("U", vec![target.clone()], params)
                    }
                    Uop::CX { control, target } => {
                        ("CX", vec![control.clone(), target.clone()], ArgumentList::new(vec![]))
                    }
                    Uop::Custom { name, params: exprs, args } => {
                        let params = ArgumentList::new(
                            exprs.iter().map(expr_to_param_argument).collect::<Result<Vec<_>>>()?)
                        ;
                        (name.as_str(), args.clone(), params)
                    }
                };

                let gate_body = gate_table.get(op_name).ok_or_else(|| crate::Error::LanguageError {
                    message: format!("unknown gate '{}'", op_name),
                    lineno: stmt.line,
                })?;

                for qubit_indices in expand_gate_arguments(&qasm_args, qreg_table, stmt.line)? {
                    match gate_body {
                        GateBody::Op(op) => {
                            if let Operation::Expression(expr_op) = op {
                                if let ExpressionOperation::UnitaryGate(u_expr) = expr_op {
                                    // Assuming all qubits are dimension 2 for target radices.
                                    // The ClassicallyControlled gate expects radices for its target qubits
                                    // in the format Option<Vec<Vec<usize>>> for multi-qudit support.
                                    let target_radices: Vec<usize> = qubit_indices.iter().map(|_| 2usize).collect();
                                    let wrapped: Operation = ClassicallyControlled(
                                        u_expr.clone(),
                                        target_radices.into(),
                                        None,
                                    ).into();
                                    circuit.append(wrapped, (qubit_indices, clbit_indices.clone()), params.clone())?;
                                } else {
                                    return Err(crate::Error::LanguageError {
                                        message: format!("Gate '{}' cannot be classically controlled currently.", op_name),
                                        lineno: stmt.line,
                                    });
                                }
                            } else {
                                return Err(crate::Error::LanguageError {
                                    message: format!("Gate '{}' cannot be classically controlled currently.", op_name),
                                    lineno: stmt.line,
                                });
                            }
                        }
                        GateBody::Circ(circ) => return Err(crate::Error::LanguageError {
                            message: format!("Gate '{}' cannot be classically controlled currently.", op_name),
                            lineno: stmt.line,
                        }),
                    };
                }
            }
            QASMStatement::Barrier(args) => {
                // A barrier spans all its arguments simultaneously — collect
                // all qubit indices into a single flat list.
                let indices: Vec<usize> = args.iter()
                    .map(|a| resolve_arg(a, qreg_table, stmt.line))
                    .collect::<Result<Vec<_>>>()?                    
                    .into_iter().flatten().collect();
                let barrier_body = gate_table.get("barrier")
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

// ============================================================================
// 5. Public API
// ============================================================================

pub struct QASM2Parser;

impl QuantumLanguageParser for QASM2Parser {
    fn parse(&self, source: &str) -> Result<QuditCircuit> {
        let ast = Parser::new(source).parse_program()?;
        let ast = resolve_includes(ast)?;
        let gate_table = resolve_gate_table(&ast)?;
        let (qreg_table, creg_table) = resolve_registers(&ast)?;
        lower_program(&ast, &gate_table, &qreg_table, &creg_table)
    }

    fn supported_extensions(&self) -> &[&str] {
        &["qasm", "qasm2"]
    }
}
