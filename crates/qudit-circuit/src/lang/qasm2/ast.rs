#[derive(Debug, PartialEq)]
pub struct QASMProgram {
    pub version: f64,
    pub statements: Vec<QASMParsedStatement>,
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
    UnaryOp(UnaryOperator, Box<Expr>),              // sin, cos, etc.
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
