use std::iter::Peekable;
use std::ops::DerefMut;
use std::str::Chars;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Comma,
    Comment,
    EOF,
    Ident(String),
    LAngle,
    LBrace,
    LBracket,
    LParen,
    Negation,
    Number(String),
    Op(char),
    RAngle,
    RBrace,
    RBracket,
    RParen,
    State,
    Utry,
    Qobj,
}

fn is_greek_letter(c: char) -> bool {
    // Greek and Coptic block in Unicode
    ('\u{0391}'..='\u{03A1}').contains(&c) || // Uppercase Greek letters (Α to Ρ)
    ('\u{03A3}'..='\u{03A9}').contains(&c) || // Uppercase Greek letters (Σ to Ω)
    ('\u{03B1}'..='\u{03C1}').contains(&c) || // Lowercase Greek letters (α to ρ)
    ('\u{03C3}'..='\u{03C9}').contains(&c) || // Lowercase Greek letters (σ to ω)
    c == '\u{03B2}' || // Special case for lowercase beta (β)
    c == '\u{03B8}' || // Special case for lowercase theta (θ)
    c == '\u{03B4}' // Special case for lowercase delta (δ)
}

#[allow(dead_code)]
pub struct LexError {
    pub error: &'static str,
    pub index: usize,
}

impl LexError {
    pub fn new(error: &'static str) -> Self {
        LexError { error, index: 0 }
    }
}

pub type LexResult = Result<Token, LexError>;

pub struct Lexer<'a> {
    input: &'a str,
    chars: Box<Peekable<Chars<'a>>>,
    pos: usize,
}

impl<'a> Lexer<'a> {
    /// Creates a new `Lexer` instance ready to tokenize the given input.
    ///
    /// # Arguments
    ///
    /// * `input` - The input string to tokenize.
    pub fn new(input: &'a str) -> Lexer<'a> {
        Lexer {
            input,
            chars: Box::new(input.chars().peekable()),
            pos: 0,
        }
    }

    /// Skips the next set of whitespaces in the input.
    pub fn skip_whitespaces(&mut self) {
        let chars = self.chars.deref_mut();

        loop {
            let ch = chars.peek();

            if ch.is_none() {
                break;
            }

            if !ch.unwrap().is_whitespace() {
                break;
            }

            chars.next();
            self.pos += 1;
        }
    }

    /// Lexes and returns the next `Token` in the input.
    pub fn lex(&mut self) -> LexResult {
        self.skip_whitespaces();

        let chars = self.chars.deref_mut();
        let src = self.input;
        let mut pos = self.pos;
        let start = pos;
        let next = chars.next();

        if next.is_none() {
            return Ok(Token::EOF);
        }

        pos += 1;

        let next = next.unwrap();

        if is_greek_letter(next) {
            pos += 1;
            loop {
                let ch = match chars.peek() {
                    Some(ch) => *ch,
                    None => return Ok(Token::EOF),
                };

                // A word-like identifier only contains underscores and alphanumeric characters.
                if ch != '_' && !ch.is_alphanumeric() {
                    break;
                }

                chars.next();
                pos += 1;
            }
            let ident = &src[start..pos];
            self.pos = pos;
            return Ok(Token::Ident(ident.to_string()));
        }

        let result = match next {
            '(' => Ok(Token::LParen),
            ')' => Ok(Token::RParen),
            '[' => Ok(Token::LBracket),
            ']' => Ok(Token::RBracket),
            '{' => Ok(Token::LBrace),
            '}' => Ok(Token::RBrace),
            '<' => Ok(Token::LAngle),
            '>' => Ok(Token::RAngle),
            ',' => Ok(Token::Comma),
            '~' => Ok(Token::Negation),
            '+' | '-' | '*' | '/' | '^' => Ok(Token::Op(next)),
            '#' => {
                loop {
                    let ch = chars.next();
                    pos += 1;

                    if ch.is_none() || ch.unwrap() == '\n' {
                        break;
                    }
                }

                Ok(Token::Comment)
            }
            '.' | '0'..='9' => {
                // Parse number literal
                while let Some(&ch) = chars.peek() {
                    // Parse float.
                    if ch != '.' && !ch.is_ascii_hexdigit() {
                        break;
                    }

                    chars.next();
                    pos += 1;
                }

                Ok(Token::Number(src[start..pos].parse().unwrap()))
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                // Parse identifier
                while let Some(&ch) = chars.peek() {
                    // A word-like identifier only contains underscores and alphanumeric characters.
                    if ch != '_' && !ch.is_alphanumeric() {
                        break;
                    }

                    chars.next();
                    pos += 1;
                }

                match &src[start..pos] {
                    "utry" => Ok(Token::Utry),
                    "qobj" => Ok(Token::Qobj),
                    "state" => Ok(Token::State),
                    ident => Ok(Token::Ident(ident.to_string())),
                }
            }
            _ => Err(LexError::new("unexpected character")),
        };

        self.pos = pos;
        result
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token;

    /// Lexes the next `Token` and returns it.
    /// On EOF or failure, `None` will be returned.
    fn next(&mut self) -> Option<Self::Item> {
        match self.lex() {
            Ok(Token::EOF) | Err(_) => None,
            Ok(token) => Some(token),
        }
    }
}
