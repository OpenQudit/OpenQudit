use super::expr::{Expression, ParsedDefinition, UnitaryDefinition};
use super::lexer::{Lexer, Token};

#[derive(Debug, Clone, PartialEq)]
pub struct ParserError {
    pub error: String,
    pub index: usize,
}

impl ParserError {
    pub fn new(error: &str) -> Self {
        ParserError {
            error: String::from(error),
            index: 0,
        }
    }

    pub fn with_index(error: &str, index: usize) -> Self {
        ParserError {
            error: String::from(error),
            index,
        }
    }
}

pub type ParserResult<T> = Result<T, ParserError>;

/// Recursive Descent Parser
pub struct Parser {
    scalar: bool,
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(input: &str, scalar: bool) -> Self {
        let mut lexer = Lexer::new(input);
        let tokens: Vec<Token> = lexer.by_ref().collect();

        // for token in &tokens {
        //     println!("{:?}", token);
        // }

        let mut pos = 0;
        while tokens[pos] == Token::Comment {
            pos += 1;
        }

        Parser { scalar, tokens, pos }
    }

    pub fn parse(&mut self) -> ParserResult<Vec<ParsedDefinition>> {
        if self.scalar {
            return Err(ParserError::new("Cannot parse an expression definition in scalar mode."));
        }

        let mut statements = Vec::new();

        while !self.at_end() {
            statements.push(self.parse_statement()?);
        }

        Ok(statements)
    }

    pub fn parse_scalar(&mut self) -> ParserResult<Expression> {
        if !self.scalar {
            return Err(ParserError::new("Cannot parse a scalar expression in array mode."));
        }

        let expr = self.parse_expr()?;

        if !self.at_end() {
            return Err(ParserError::with_index("Expected EOF.", self.pos));
        }

        Ok(expr)
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn lookahead(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.pos + offset)
    }

    fn advance(&mut self) {
        self.pos += 1;

        while self.peek() == Some(&Token::Comment) {
            self.pos += 1;
        }
    }

    fn conditional_advance(&mut self, token: Token) -> bool {
        if self.peek() == Some(&token) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn expect(&mut self, expected: Token) -> ParserResult<()> {
        if self.peek() == Some(&expected) {
            self.advance();
            Ok(())
        } else {
            Err(ParserError::with_index(
                &format!(
                    "Unexpected token: {:?}, expected: {:?}",
                    self.peek(),
                    expected
                ),
                self.pos,
            ))
        }
    }

    /// Returns the current token, optionally consuming it.
    ///
    /// Expects the input to not be empty.
    fn current(&mut self, consume: bool) -> ParserResult<Token> {
        let token = self.peek().ok_or(ParserError::new("unexpected EOF"))?;
        let token = token.clone();
        if consume {
            self.advance();
        }
        Ok(token)
    }

    fn parse_statement(&mut self) -> ParserResult<ParsedDefinition> {
        let token = self.current(false)?;

        match token {
            Token::Ident(_) => self.parse_qobj_def(),
            _ => Err(ParserError::with_index(
                &format!(
                    "Unexpected token: {:?} during statement parse.",
                    self.peek()
                ),
                self.pos,
            )),
        }
    }

    fn parse_qobj_def(&mut self) -> ParserResult<ParsedDefinition> {
        // self.advance();
        // self.expect(Token::Utry)?;

        let name = self.parse_identifier()?;

        let mut radices = None;

        if self.current(false)? == Token::LAngle {
            self.advance();
            radices = Some(self.parse_constlist(Token::RAngle)?);
            self.expect(Token::RAngle)?;
        }

        self.expect(Token::LParen)?;

        let variables = self.parse_varlist(Token::RParen)?;

        for var in &variables {
            if var == "pi"
                || var == "e"
                || var == "i"
                || var == "π"
                || var == "τ"
                || var == "sqrt"
                || var == "exp"
                || var == "ln"
                || var == "sin"
                || var == "cos"
                || var == "tan"
                || var == "sec"
                || var == "csc"
                || var == "cot"
                || var == "asin"
                || var == "acos"
                || var == "atan"
                || var == "asec"
                || var == "acsc"
                || var == "acot"
                || var == "sinh"
                || var == "cosh"
                || var == "tanh"
                || var == "sech"
                || var == "csch"
                || var == "coth"
                || var == "asinh"
                || var == "acosh"
                || var == "atanh"
                || var == "asech"
                || var == "acsch"
                || var == "acoth"
                || var == "abs"
                || var == "sign"
                || var == "floor"
                || var == "ceil"
                || var == "round"
                || var == "trunc"
                || var == "real"
                || var == "imag"
                || var == "conj"
                || var == "norm"
                || var == "arg"
            {
                return Err(ParserError::with_index(
                    &format!("Variable name {} is reserved.", var),
                    self.pos,
                ));
            }
        }

        self.expect(Token::RParen)?;

        self.expect(Token::LBrace)?;

        let body = self.parse_expr()?;

        self.expect(Token::RBrace)?;

        Ok(ParsedDefinition {
            name,
            radices,
            variables,
            body,
        })
    }

    fn parse_expr(&mut self) -> ParserResult<Expression> {
        let mut expr = self.parse_term()?;

        while self.peek() == Some(&Token::Op('+')) || self.peek() == Some(&Token::Op('-')) {
            let op = self.current(true)?;
            let rhs = self.parse_term()?;
            match op {
                Token::Op('+') => {
                    expr = Expression::Binary {
                        op: '+',
                        lhs: Box::new(expr),
                        rhs: Box::new(rhs),
                    }
                }
                Token::Op('-') => {
                    expr = Expression::Binary {
                        op: '-',
                        lhs: Box::new(expr),
                        rhs: Box::new(rhs),
                    }
                }
                _ => unreachable!(),
            }
        }

        Ok(expr)
    }

    fn parse_term(&mut self) -> ParserResult<Expression> {
        if self.peek() == Some(&Token::Negation) {
            self.advance();
            let expr = self.parse_term()?;
            return Ok(Expression::Unary {
                op: '~',
                expr: Box::new(expr),
            });
        }

        let mut expr = self.parse_factor()?;

        while self.peek() == Some(&Token::Op('*')) || self.peek() == Some(&Token::Op('/')) {
            let op = self.current(true)?;
            let rhs = self.parse_factor()?;
            match op {
                Token::Op('*') => {
                    expr = Expression::Binary {
                        op: '*',
                        lhs: Box::new(expr),
                        rhs: Box::new(rhs),
                    }
                }
                Token::Op('/') => {
                    expr = Expression::Binary {
                        op: '/',
                        lhs: Box::new(expr),
                        rhs: Box::new(rhs),
                    }
                }
                _ => unreachable!(),
            }
        }

        Ok(expr)
    }

    fn parse_factor(&mut self) -> ParserResult<Expression> {
        let mut expr = self.parse_primary()?;

        while self.peek() == Some(&Token::Op('^')) {
            self.advance();
            let rhs = self.parse_primary()?;
            expr = Expression::Binary {
                op: '^',
                lhs: Box::new(expr),
                rhs: Box::new(rhs),
            };
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> ParserResult<Expression> {
        let token = self.current(false)?;

        match token {
            Token::Number(num) => {
                self.advance();
                Ok(Expression::Number(num))
            }

            Token::LBracket => self.parse_array(),

            Token::Ident(ident) => {
                self.advance();
                if self.peek() == Some(&Token::LParen) {
                    self.advance();
                    let args = self.parse_exprlist(Token::RParen)?;
                    self.expect(Token::RParen)?;
                    Ok(Expression::Call {
                        fn_name: ident,
                        args,
                    })
                } else {
                    Ok(Expression::Variable(ident))
                }
            }
            Token::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            _ => Err(ParserError::with_index(
                &format!("Unexpected token: {:?} during primary parse.", self.peek()),
                self.pos,
            )),
        }
    }

    fn parse_array(&mut self) -> ParserResult<Expression> {
        if self.lookahead(1) != Some(&Token::LBracket) {
            return self.parse_vector();
        }
        if self.lookahead(2) != Some(&Token::LBracket) {
            return self.parse_matrix();
        }
        self.parse_tensor()
    }

    fn parse_tensor(&mut self) -> ParserResult<Expression> {
        self.expect(Token::LBracket)?;

        let mut mats = Vec::new();

        while self.current(false)? != Token::RBracket {
            if let Expression::Matrix(mat) = self.parse_matrix()? {
                mats.push(mat);
            } else {
                return Err(ParserError::with_index(
                    "Expected a matrix in tensor parse.",
                    self.pos,
                ));
            }
            self.conditional_advance(Token::Comma);
        }

        self.expect(Token::RBracket)?;

        Ok(Expression::Tensor(mats))
    }

    fn parse_matrix(&mut self) -> ParserResult<Expression> {
        self.expect(Token::LBracket)?;

        let mut rows = Vec::new();

        while self.current(false)? != Token::RBracket {
            if let Expression::Vector(row) = self.parse_vector()? {
                rows.push(row);
            } else {
                return Err(ParserError::with_index(
                    "Expected a vector in matrix parse.",
                    self.pos,
                ));
            }
            self.conditional_advance(Token::Comma);
        }

        self.expect(Token::RBracket)?;

        Ok(Expression::Matrix(rows))
    }

    fn parse_vector(&mut self) -> ParserResult<Expression> {
        self.expect(Token::LBracket)?;

        let row = self.parse_exprlist(Token::RBracket)?;

        self.expect(Token::RBracket)?;

        Ok(Expression::Vector(row))
    }

    fn parse_exprlist(&mut self, terminating_token: Token) -> ParserResult<Vec<Expression>> {
        let mut expressions = Vec::new();

        while self.current(false)? != terminating_token {
            expressions.push(self.parse_expr()?);

            self.conditional_advance(Token::Comma);
        }

        Ok(expressions)
    }

    fn parse_varlist(&mut self, terminating_token: Token) -> ParserResult<Vec<String>> {
        let mut variables = Vec::new();

        while self.current(false)? != terminating_token {
            variables.push(self.parse_identifier()?);

            self.conditional_advance(Token::Comma);
        }

        Ok(variables)
    }

    fn parse_constlist(&mut self, terminating_token: Token) -> ParserResult<Vec<usize>> {
        let mut constants = Vec::new();

        while self.current(false)? != terminating_token {
            match self.current(false)? {
                Token::Number(num) => {
                    self.advance();
                    match num.parse() {
                        Ok(num) => constants.push(num),
                        Err(_) => {
                            return Err(ParserError::with_index(
                                &format!("Failed to parse integer: {}", num),
                                self.pos,
                            ))
                        }
                    }
                }
                _ => {
                    return Err(ParserError::with_index(
                        &format!("Unexpected token: {:?} during constant parse.", self.peek()),
                        self.pos,
                    ))
                }
            };

            self.conditional_advance(Token::Comma);
        }

        Ok(constants)
    }

    fn parse_identifier(&mut self) -> ParserResult<String> {
        let token = self.current(true)?;

        match token {
            Token::Ident(ident) => Ok(ident),
            _ => Err(ParserError::with_index("expected identifier", self.pos)),
        }
    }
}
