use std::iter::Peekable;
use std::str::Chars;

use crate::Result;

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
    Plus,
    Minus,
    Star,
    Slash,
    Caret,

    // Unary Math Functions
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,

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
    pub chars: Peekable<Chars<'a>>,
    pub line: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Lexer {
            chars: source.chars().peekable(),
            line: 1,
        }
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
        crate::Error::LanguageError {
            message: message.into(),
            lineno: self.line,
        }
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
                            Some(nc) => s.push(nc),
                            None => return Err(self.lex_error("unterminated string literal")),
                        }
                    }
                    return Ok(Token::StringLit(s));
                }
                _ if c.is_alphabetic() || c == '_' => {
                    let mut s = String::new();
                    s.push(c);
                    while self
                        .chars
                        .peek()
                        .is_some_and(|&nc| nc.is_alphanumeric() || nc == '_')
                    {
                        // peek() confirmed Some, so advance() is infallible here
                        s.push(self.advance().expect("peek guaranteed Some"));
                    }
                    return Ok(match s.as_str() {
                        "OPENQASM" => Token::OpenQasm,
                        "include" => Token::Include,
                        "qreg" => Token::Qreg,
                        "creg" => Token::Creg,
                        "gate" => Token::Gate,
                        "opaque" => Token::Opaque,
                        "if" => Token::If,
                        "barrier" => Token::Barrier,
                        "measure" => Token::Measure,
                        "reset" => Token::Reset,
                        "U" => Token::U,
                        "CX" => Token::CX,
                        "pi" => Token::Pi,
                        "sin" => Token::Sin,
                        "cos" => Token::Cos,
                        "tan" => Token::Tan,
                        "exp" => Token::Exp,
                        "ln" => Token::Ln,
                        "sqrt" => Token::Sqrt,
                        _ => Token::Ident(s),
                    });
                }
                _ if c.is_ascii_digit() || c == '.' => {
                    let mut s = String::new();
                    s.push(c);
                    let mut is_real = c == '.';

                    while let Some(&nc) = self.chars.peek() {
                        if nc.is_ascii_digit()
                            || nc == '.'
                            || nc == 'e'
                            || nc == 'E'
                            || ((nc == '+' || nc == '-') && (s.ends_with('e') || s.ends_with('E')))
                        {
                            if nc == '.' || nc == 'e' || nc == 'E' {
                                is_real = true;
                            }
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
