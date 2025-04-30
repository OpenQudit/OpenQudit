#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Binary {
        op: char,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
    },

    Unary {
        op: char,
        expr: Box<Expression>,
    },

    Call {
        fn_name: String,
        args: Vec<Expression>,
    },

    Number(String),

    Variable(String),

    Matrix(Vec<Vec<Expression>>),
}

impl Expression {
    pub fn dim(&self) -> usize {
        match self {
            Expression::Matrix(mat) => {
                let nrows = mat.len();
                let ncols = mat[0].len();

                assert!(nrows == ncols, "Parsed matrix is not square");

                for row in mat.iter() {
                    assert!(row.len() == ncols, "Matrix has inconsistent row lengths");
                }

                nrows
            }
            Expression::Binary { op: _op, lhs, rhs } => {
                let lhs_dim = lhs.dim();
                let rhs_dim = rhs.dim();

                match (lhs_dim, rhs_dim) {
                    (0, 0) => 0,
                    (0, _) => rhs_dim,
                    (_, 0) => lhs_dim,
                    _ => {
                        assert!(
                            lhs_dim == rhs_dim,
                            "Binary operands have different dimensions"
                        );
                        lhs_dim
                    }
                }
            }
            Expression::Unary { expr, .. } => expr.dim(),
            Expression::Call { fn_name: _fn_name, args, .. } => {
                for arg in args.iter() {
                    assert!(
                        arg.dim() == args[0].dim(),
                        "Function arguments have different dimensions"
                    );
                }
                args[0].dim()
            }
            Expression::Number(_) => 0,
            Expression::Variable(_) => 0,
        }
    }

    fn matrix_multiply(&self, other: &Expression) -> Self {
        match (self, other) {
            (Expression::Matrix(lhs), Expression::Matrix(rhs)) => {
                let lhs_nrows = lhs.len();
                let lhs_ncols = lhs[0].len();
                let rhs_nrows = rhs.len();
                let rhs_ncols = rhs[0].len();

                assert!(
                    lhs_ncols == rhs_nrows,
                    "Matrix dimensions are incompatible for multiplication"
                );

                let mut result =
                    vec![vec![Expression::Number("0".to_string()); rhs_ncols]; lhs_nrows];

                for i in 0..lhs_nrows {
                    for j in 0..rhs_ncols {
                        for k in 0..lhs_ncols {
                            result[i][j] = Expression::Binary {
                                op: '+',
                                lhs: Box::new(result[i][j].clone()),
                                rhs: Box::new(Expression::Binary {
                                    op: '*',
                                    lhs: Box::new(lhs[i][k].clone()),
                                    rhs: Box::new(rhs[k][j].clone()),
                                }),
                            };
                        }
                    }
                }

                Expression::Matrix(result)
            }
            _ => panic!("Matrix multiplication requires two matrices"),
        }
    }

    pub fn into_element_wise(self) -> Self {
        match self {
            Expression::Binary { op, lhs, rhs } => {
                let lhs = lhs.into_element_wise();
                let rhs = rhs.into_element_wise();
                let lhs_dim = lhs.dim();
                let rhs_dim = rhs.dim();

                // Two scalar operands: already element-wise
                if lhs_dim == 0 && rhs_dim == 0 {
                    return Expression::Binary {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }

                // Exponentiation of a matrix
                if op == '^' {
                    match rhs {
                        Expression::Number(num) => {
                            let num = match num.parse::<usize>() {
                                Ok(num) => num,
                                Err(_) => {
                                    panic!("Power operator on matrix requires an integer exponent")
                                }
                            };
                            if num <= 0 {
                                panic!("Matrix power must be a positive integer")
                            }
                            let mut acm = lhs.clone();
                            for _ in 1..num {
                                acm = acm.matrix_multiply(&lhs);
                            }
                            acm
                        }
                        _ => panic!("Unexpected non-integer in matrix power operation"),
                    }

                // Scalar op Matrix
                } else if lhs_dim == 0 {
                    if op == '/' {
                        panic!("Division by matrix not supported")
                    }
                    match rhs {
                        Expression::Matrix(mat) => {
                            let mat = mat
                                .into_iter()
                                .map(|row| {
                                    row.into_iter()
                                        .map(|elem| Expression::Binary {
                                            op,
                                            lhs: Box::new(lhs.clone()),
                                            rhs: Box::new(elem),
                                        })
                                        .collect()
                                })
                                .collect();
                            Expression::Matrix(mat)
                        }
                        _ => panic!("Unexpected non-matrix expression in element-wise addition"),
                    }

                // Matrix op Scalar
                } else if rhs_dim == 0 {
                    match lhs {
                        Expression::Matrix(mat) => {
                            let mat = mat
                                .into_iter()
                                .map(|row| {
                                    row.into_iter()
                                        .map(|elem| Expression::Binary {
                                            op,
                                            lhs: Box::new(elem),
                                            rhs: Box::new(rhs.clone()),
                                        })
                                        .collect()
                                })
                                .collect();
                            Expression::Matrix(mat)
                        }
                        _ => panic!("Unexpected non-matrix expression in element-wise addition"),
                    }

                // Matrix op Matrix
                } else {
                    if op == '/' {
                        panic!("Division of matrices not supported")
                    }

                    match op {
                        '+' | '-' => match (lhs, rhs) {
                            (Expression::Matrix(lhs), Expression::Matrix(rhs)) => {
                                let mat = lhs
                                    .into_iter()
                                    .zip(rhs.into_iter())
                                    .map(|(lhs_row, rhs_row)| {
                                        lhs_row
                                            .into_iter()
                                            .zip(rhs_row.into_iter())
                                            .map(|(lhs_elem, rhs_elem)| Expression::Binary {
                                                op,
                                                lhs: Box::new(lhs_elem),
                                                rhs: Box::new(rhs_elem),
                                            })
                                            .collect()
                                    })
                                    .collect();
                                Expression::Matrix(mat)
                            }
                            _ => {
                                panic!("Unexpected non-matrix expression in element-wise addition")
                            }
                        },

                        '*' => lhs.matrix_multiply(&rhs),
                        _ => panic!("Unsupported binary operator for matrix-matrix operation"),
                    }
                }
            }
            Expression::Unary { op, expr } => {
                let expr = expr.into_element_wise();
                match expr {
                    Expression::Matrix(mat) => {
                        let mat = mat
                            .into_iter()
                            .map(|row| {
                                row.into_iter()
                                    .map(|elem| Expression::Unary {
                                        op,
                                        expr: Box::new(elem),
                                    })
                                    .collect()
                            })
                            .collect();
                        Expression::Matrix(mat)
                    }
                    _ => {
                        panic!("Unexpected non-matrix expression in unary element-wise conversion")
                    }
                }
            }
            Expression::Call { fn_name, args } => {
                assert!(
                    args.iter().all(|arg| arg.dim() == 0),
                    "Function arguments must be scalars."
                );
                let args = args
                    .into_iter()
                    .map(|arg| arg.into_element_wise())
                    .collect();
                Expression::Call { fn_name, args }
            }
            Expression::Number(num) => Expression::Number(num),
            Expression::Variable(var) => Expression::Variable(var),
            Expression::Matrix(mat) => {
                assert!(
                    mat.iter().all(|row| row.iter().all(|elem| elem.dim() == 0)),
                    "Matrix elements must be scalars."
                );
                Expression::Matrix(mat)
            }
        }
    }
}

/// An untyped definition of a unitary in the QGL language.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitaryDefinition {
    pub name: String,
    pub radices: Option<Vec<usize>>,
    pub variables: Vec<String>,
    pub body: Expression,
}

impl UnitaryDefinition {
    pub fn get_radices(&self) -> Vec<usize> {
        match &self.radices {
            Some(radices) => radices.clone(),
            None => {
                let mut d = self.dim();
                let power_of_2 = d & (d - 1) == 0 && d != 0;
                if !power_of_2 {
                    panic!("Dimension must be a power of 2");
                }
                let mut radices = Vec::new();
                while d > 1 {
                    radices.push(2);
                    d >>= 1;
                }
                radices
            }
        }
    }

    pub fn dim(&self) -> usize {
        self.body.dim()
    }
}
