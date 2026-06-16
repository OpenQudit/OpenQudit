// ============================================================================
// 3. Recursive Descent Parser
// ============================================================================

use crate::Result;

use super::ast::{
    Argument, BinaryOperator, Expr, GateOp, QASMGateDecl, QASMParsedStatement, QASMProgram,
    QASMStatement, Qop, UnaryOperator, Uop,
};
use super::lexer::{Lexer, Token};

struct Parser<'a> {
    lexer: Lexer<'a>,
    /// One-token lookahead buffer.
    peeked: Option<Token>,
}

impl<'a> Parser<'a> {
    fn new(source: &'a str) -> Self {
        Parser {
            lexer: Lexer::new(source),
            peeked: None,
        }
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
        crate::Error::LanguageError {
            message: message.into(),
            lineno: self.lexer.line,
        }
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
            Token::Int(n) => n as f64,
            t => return Err(self.parse_error(format!("expected version number, got {:?}", t))),
        };
        self.expect_punct(';')?;
        let statements = self.parse_body()?;
        Ok(QASMProgram {
            version,
            statements,
        })
    }

    /// `<program> ::= <statement>*`
    ///
    /// Parses a sequence of statements without a leading `OPENQASM` header.
    /// Used when parsing included files, which are not required to carry their
    /// own version declaration.
    pub(super) fn parse_body(&mut self) -> Result<Vec<QASMParsedStatement>> {
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
            Token::Gate => self.parse_gate_decl_stmt()?,
            Token::Opaque => self.parse_opaque()?,
            Token::If => self.parse_if()?,
            Token::Include => self.parse_include()?,
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
        let name = self.expect_ident()?;
        self.expect_punct('[')?;
        let size = self.expect_int()?;
        self.expect_punct(']')?;
        self.expect_punct(';')?;
        if is_qreg {
            Ok(QASMStatement::QReg(name, size))
        } else {
            Ok(QASMStatement::CReg(name, size))
        }
    }

    /// `<gatedecl> <goplist> } | <gatedecl> }`
    fn parse_gate_decl_stmt(&mut self) -> Result<QASMStatement> {
        self.advance()?; // consume 'gate'
        let name = self.expect_ident()?;
        let params = self.parse_optional_param_idlist()?;
        let qargs = self.parse_idlist()?;
        self.expect_punct('{')?;
        let body = if self.peek()? == Token::Punct('}') {
            vec![]
        } else {
            self.parse_goplist()?
        };
        self.expect_punct('}')?;
        Ok(QASMStatement::GateDecl(QASMGateDecl {
            name,
            params,
            qargs,
            body,
        }))
    }

    /// `opaque <id> <idlist> ; | opaque <id> () <idlist> ; | opaque <id> (<idlist>) <idlist> ;`
    fn parse_opaque(&mut self) -> Result<QASMStatement> {
        self.advance()?; // consume 'opaque'
        let name = self.expect_ident()?;
        let params = self.parse_optional_param_idlist()?;
        let qargs = self.parse_idlist()?;
        self.expect_punct(';')?;
        Ok(QASMStatement::OpaqueDecl {
            name,
            params,
            qargs,
        })
    }

    /// Parses an optional `( <idlist> )` or `()` parameter list, returning the identifiers.
    /// If there is no opening `(`, returns an empty `Vec`.
    fn parse_optional_param_idlist(&mut self) -> Result<Vec<String>> {
        if self.peek()? != Token::Punct('(') {
            return Ok(vec![]);
        }
        self.advance()?; // consume '('
        let params = if self.peek()? == Token::Punct(')') {
            vec![]
        } else {
            self.parse_idlist()?
        };
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
            t => Err(self.parse_error(format!(
                "expected string literal after 'include', got {:?}",
                t
            ))),
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
                    let args = self
                        .parse_idlist()?
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
                    return Err(self.parse_error(format!(
                        "U gate requires exactly 3 parameters, got {}",
                        params.len()
                    )));
                }
                self.expect_punct(')')?;
                let lambda = params.pop().unwrap();
                let phi = params.pop().unwrap();
                let theta = params.pop().unwrap();
                let target = self.parse_argument()?;
                self.expect_punct(';')?;
                Ok(Uop::U {
                    theta,
                    phi,
                    lambda,
                    target,
                })
            }
            Token::CX => {
                self.advance()?;
                let control = self.parse_argument()?;
                self.expect_punct(',')?;
                let target = self.parse_argument()?;
                self.expect_punct(';')?;
                Ok(Uop::CX { control, target })
            }
            Token::Ident(_) => {
                let name = self.expect_ident()?;
                let params = self.parse_optional_param_explist()?;
                let args = self.parse_anylist()?;
                self.expect_punct(';')?;
                Ok(Uop::Custom { name, params, args })
            }
            t => Err(self.parse_error(format!(
                "expected gate operation (U, CX, or identifier), got {:?}",
                t
            ))),
        }
    }

    /// Parses an optional `( <explist> )` or `()` expression parameter list.
    /// If there is no opening `(`, returns an empty `Vec`.
    fn parse_optional_param_explist(&mut self) -> Result<Vec<Expr>> {
        if self.peek()? != Token::Punct('(') {
            return Ok(vec![]);
        }
        self.advance()?; // consume '('
        let params = if self.peek()? == Token::Punct(')') {
            vec![]
        } else {
            self.parse_explist()?
        };
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
                Token::Plus => BinaryOperator::Plus,
                Token::Minus => BinaryOperator::Minus,
                _ => break,
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
                Token::Star => BinaryOperator::Multiply,
                Token::Slash => BinaryOperator::Divide,
                _ => break,
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
            Ok(Expr::BinaryOp(
                Box::new(base),
                BinaryOperator::Power,
                Box::new(self.parse_power()?),
            ))
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
            Token::Sin | Token::Cos | Token::Tan | Token::Exp | Token::Ln | Token::Sqrt => {
                let op = match self.advance()? {
                    Token::Sin => UnaryOperator::Sin,
                    Token::Cos => UnaryOperator::Cos,
                    Token::Tan => UnaryOperator::Tan,
                    Token::Exp => UnaryOperator::Exp,
                    Token::Ln => UnaryOperator::Ln,
                    Token::Sqrt => UnaryOperator::Sqrt,
                    _ => unreachable!(),
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

pub(super) fn parse_qasm_program(source: &str) -> Result<QASMProgram> {
    Parser::new(source).parse_program()
}

pub(super) fn parse_qasm_body(source: &str) -> Result<Vec<QASMParsedStatement>> {
    Parser::new(source).parse_body()
}
