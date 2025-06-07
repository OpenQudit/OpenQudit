use qudit_core::TensorShape;

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

    Vector(Vec<Expression>),
    Matrix(Vec<Vec<Expression>>),
    Tensor(Vec<Vec<Vec<Expression>>>),
}

impl Expression {

    pub fn gen_shape(&self) -> TensorShape {
        match self {
            Expression::Vector(vec) => {
                TensorShape::Vector(vec.len())
            }
            Expression::Matrix(mat) => {
                let nrows = mat.len();
                let ncols = mat[0].len();

                for row in mat.iter() {
                    assert!(row.len() == ncols, "Matrix has inconsistent row lengths");
                }

                TensorShape::Matrix(nrows, ncols)
            }
            Expression::Tensor(tensor) => {
                let nmats = tensor.len();
                let nrows = tensor[0].len();
                let ncols = tensor[0][0].len();
                
                for mat in tensor.iter() {
                    assert!(mat.len() == nrows, "Tensor has inconsistent matrix lengths");
                    for row in mat.iter() {
                        assert!(row.len() == ncols, "Matrix has inconsistent row lengths");
                    }
                }

                TensorShape::Tensor3D(nmats, nrows, ncols)
            }
            Expression::Binary { op, lhs, rhs } => {
                let lhs_shape = lhs.gen_shape();
                let rhs_shape = rhs.gen_shape();

                if lhs_shape == TensorShape::Scalar {
                    rhs_shape
                } else if rhs_shape == TensorShape::Scalar {
                    lhs_shape
                } else {
                    match (lhs_shape, rhs_shape) {
                        (TensorShape::Vector(l), TensorShape::Vector(r)) => {
                            assert!(l == r, "Vectors must have same length");
                            TensorShape::Vector(l)
                        }
                        (TensorShape::Matrix(lrows, lcols), TensorShape::Matrix(rrows, rcols)) => {
                            if op.to_string() == "*" {
                                assert!(lrows == rcols, "Left matrix columns must match right matrix rows");
                                TensorShape::Matrix(lrows, rcols)
                            } else {
                                assert!(lrows == rrows && lcols == rcols, "Matrices must have same dimensions");
                                TensorShape::Matrix(lrows, lcols)
                            }
                        }
                        (TensorShape::Tensor3D(lnmats, lnrows, lncols), TensorShape::Tensor3D(rnmats, rnrows, rncols)) => {
                            assert!(lnmats == rnmats && lnrows == rnrows && lncols == rncols, "Tensors must have same dimensions");
                            TensorShape::Tensor3D(lnmats, lnrows, lncols)
                        }
                        _ => panic!("Incompatible shapes for binary operation"),
                    }
                }
            }
            Expression::Unary { op: _op, expr } => {
                expr.gen_shape()
            }
            Expression::Call { fn_name: _fn_name, args, .. } => {
                for arg in args.iter() {
                    assert!(
                        arg.gen_shape() == args[0].gen_shape(),
                        "Function arguments have different shapes"
                    );
                }
                args[0].gen_shape()
            }
            Expression::Number(_) => {
                TensorShape::Scalar
            }
            Expression::Variable(_) => {
                TensorShape::Scalar
            }
        }
    }

    // pub fn dim(&self) -> usize {
    //     match self {
    //         Expression::Matrix(mat) => {
    //             let nrows = mat.len();
    //             let ncols = mat[0].len();

    //             assert!(nrows == ncols, "Parsed matrix is not square");

    //             for row in mat.iter() {
    //                 assert!(row.len() == ncols, "Matrix has inconsistent row lengths");
    //             }

    //             nrows
    //         }
    //         Expression::Binary { op: _op, lhs, rhs } => {
    //             let lhs_dim = lhs.dim();
    //             let rhs_dim = rhs.dim();

    //             match (lhs_dim, rhs_dim) {
    //                 (0, 0) => 0,
    //                 (0, _) => rhs_dim,
    //                 (_, 0) => lhs_dim,
    //                 _ => {
    //                     assert!(
    //                         lhs_dim == rhs_dim,
    //                         "Binary operands have different dimensions"
    //                     );
    //                     lhs_dim
    //                 }
    //             }
    //         }
    //         Expression::Unary { expr, .. } => expr.dim(),
    //         Expression::Call { fn_name: _fn_name, args, .. } => {
    //             for arg in args.iter() {
    //                 assert!(
    //                     arg.dim() == args[0].dim(),
    //                     "Function arguments have different dimensions"
    //                 );
    //             }
    //             args[0].dim()
    //         }
    //         Expression::Number(_) => 0,
    //         Expression::Variable(_) => 0,
    //     }
    // }

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
                let lhs_shape = lhs.gen_shape();
                let rhs_shape = rhs.gen_shape();

                // Two scalar operands: already element-wise
                if lhs_shape == TensorShape::Scalar && rhs_shape == TensorShape::Scalar {
                    return Expression::Binary {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }

                // supported ops (+,-,*,/,^)
                // obj + - obj; obj have same dimension
                // obj + - scalar;
                // scalar + - obj;
                // obj / scalar;
                // scalar / obj; (broadcast)
                // matrix ^ scalar; scalar has to be an integer
                // obj * scalar;
                // scalar * obj;
                // matrix * matrix;
                // matrix * vector;
                // vector * matrix;

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
                            // if num <= 0 {
                            //     panic!("Matrix power must be a positive integer")
                            // }
                            let mut acm = lhs.clone();
                            for _ in 1..num {
                                acm = acm.matrix_multiply(&lhs);
                            }
                            acm
                        }
                        _ => panic!("Unexpected non-integer in matrix power operation"),
                    }

                // Scalar op Matrix
                } else if lhs_shape == TensorShape::Scalar {
                    // if op == '/' {
                    //     panic!("Division by vector, matrix, or tensor not supported")
                    // }
                    match rhs {
                        Expression::Vector(vec) => {
                            let vec = vec
                                .into_iter()
                                .map(|elem| Expression::Binary {
                                    op,
                                    lhs: Box::new(lhs.clone()),
                                    rhs: Box::new(elem),
                                })
                                .collect();
                            Expression::Vector(vec)
                        }
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
                        Expression::Tensor(tensor) => {
                            let tensor = tensor
                                .into_iter()
                                .map(|mat| {
                                    mat.into_iter()
                                        .map(|row| {
                                            row.into_iter()
                                                .map(|elem| Expression::Binary {
                                                    op,
                                                    lhs: Box::new(lhs.clone()),
                                                    rhs: Box::new(elem),
                                                })
                                                .collect()
                                        })
                                        .collect()
                                })
                                .collect();
                            Expression::Tensor(tensor)
                        }
                        _ => panic!("Unexpected non-matrix expression in element-wise addition"),
                    }

                // Matrix op Scalar
                } else if rhs_shape == TensorShape::Scalar {
                    match lhs {
                        Expression::Vector(vec) => {
                            let vec = vec
                                .into_iter()
                                .map(|elem| Expression::Binary {
                                    op,
                                    lhs: Box::new(elem),
                                    rhs: Box::new(rhs.clone()),
                                })
                                .collect();
                            Expression::Vector(vec)
                        }
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
                        Expression::Tensor(tensor) => {
                            let tensor = tensor
                                .into_iter()
                                .map(|mat| {
                                    mat.into_iter()
                                        .map(|row| {
                                            row.into_iter()
                                                .map(|elem| Expression::Binary {
                                                    op,
                                                    lhs: Box::new(elem),
                                                    rhs: Box::new(rhs.clone()),
                                                })
                                                .collect()
                                        })
                                        .collect()
                                })
                                .collect();
                            Expression::Tensor(tensor)
                        }
                        _ => panic!("Unexpected non-matrix expression in element-wise addition"),
                    }

                // Matrix op Matrix
                } else {
                    if op == '/' {
                        panic!("Division of vectors, matrices, or tensors is not supported")
                    }

                    match op {
                        '+' | '-' => match (lhs, rhs) {
                            // TODO: assert that lhs and rhs have the same shape
                            (Expression::Vector(lhs), Expression::Vector(rhs)) => {
                                let vec = lhs
                                    .into_iter()
                                    .zip(rhs.into_iter())
                                    .map(|(lhs_elem, rhs_elem)| Expression::Binary {
                                        op,
                                        lhs: Box::new(lhs_elem),
                                        rhs: Box::new(rhs_elem),
                                    })
                                    .collect();
                                Expression::Vector(vec)
                            }
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
                            (Expression::Tensor(lhs), Expression::Tensor(rhs)) => {
                                let tensor = lhs
                                    .into_iter()
                                    .zip(rhs.into_iter())
                                    .map(|(lhs_mat, rhs_mat)| {
                                        lhs_mat
                                            .into_iter()
                                            .zip(rhs_mat.into_iter())
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
                                            .collect()
                                    })
                                    .collect();
                                Expression::Tensor(tensor)
                            }
                            _ => {
                                panic!("Unexpected non-matrix expression in element-wise addition")
                            }
                        },

                        '*' => {
                            todo!()
                        // match(lhs, rhs) {
                        }
                        // lhs.matrix_multiply(&rhs),
                        _ => panic!("Unsupported binary operator for matrix-matrix operation"),
                    }
                }
            }
            Expression::Unary { op, expr } => {
                let expr = expr.into_element_wise();
                match expr {
                    Expression::Vector(vec) => {
                        let vec = vec
                            .into_iter()
                            .map(|elem| Expression::Unary {
                                op,
                                expr: Box::new(elem),
                            })
                            .collect();
                        Expression::Vector(vec)
                    }
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
                    Expression::Tensor(tensor) => {
                        let tensor = tensor
                            .into_iter()
                            .map(|mat| {
                                mat.into_iter()
                                    .map(|row| {
                                        row.into_iter()
                                            .map(|elem| Expression::Unary {
                                                op,
                                                expr: Box::new(elem),
                                            })
                                            .collect()
                                    })
                                    .collect()
                            })
                            .collect();
                        Expression::Tensor(tensor)
                    }
                    _ => {
                        Expression::Unary {
                            op,
                            expr: Box::new(expr),
                        }
                    }
                }
            }
            Expression::Call { fn_name, args } => {
                assert!(
                    args.iter().all(|arg| arg.gen_shape() == TensorShape::Scalar),
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
            Expression::Vector(vec) => {
                let element_wise_list: Vec<_> = vec.into_iter().map(|elem| elem.into_element_wise()).collect();
                let element_shape = element_wise_list[0].gen_shape();
                assert!(
                    element_wise_list.iter().all(|elem| elem.gen_shape() == element_shape),
                    "Vector elements must have same shape."
                );
                match element_shape {
                    TensorShape::Scalar => {
                        Expression::Vector(element_wise_list)
                    }
                    TensorShape::Vector(_) => {
                        let exprs = element_wise_list.into_iter()
                            .map(|elem| {
                                if let Expression::Vector(vec) = elem {
                                    vec
                                } else {
                                    panic!("Unexpected shape for vector element-wise conversion")
                                }
                            })
                            .collect::<Vec<Vec<Expression>>>();
                        Expression::Matrix(exprs)
                    }
                    TensorShape::Matrix(_, _) => {
                        let exprs = element_wise_list.into_iter()
                            .map(|elem| {
                                if let Expression::Matrix(mat) = elem {
                                    mat
                                } else {
                                    panic!("Unexpected shape for vector element-wise conversion")
                                }
                            })
                            .collect::<Vec<Vec<Vec<Expression>>>>();
                        Expression::Tensor(exprs)
                    }
                    _ => panic!("Unexpected shape for vector element-wise conversion"),
                }
            }
            Expression::Matrix(mat) => {
                let element_wise_list: Vec<Vec<_>> = mat.into_iter()
                    .map(|row| {
                        row.into_iter()
                            .map(|elem| elem.into_element_wise())
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let element_shape = element_wise_list[0][0].gen_shape();
                assert!(
                    element_wise_list.iter().all(|row| {
                        row.iter().all(|elem| elem.gen_shape() == element_shape)
                    }),
                    "Matrix elements must have same shape."
                );
                match element_shape {
                    TensorShape::Scalar => {
                        Expression::Matrix(element_wise_list)
                    }
                    TensorShape::Vector(_) => {
                        let exprs = element_wise_list.into_iter()
                            .map(|row| {
                                row.into_iter()
                                    .map(|elem| {
                                        if let Expression::Vector(vec) = elem {
                                            vec
                                        } else {
                                            panic!("Unexpected shape for matrix element-wise conversion")
                                        }
                                    })
                                    .collect::<Vec<Vec<Expression>>>()
                            })
                            .collect::<Vec<Vec<Vec<Expression>>>>();
                        Expression::Tensor(exprs)
                    }
                    _ => panic!("Unexpected shape for matrix element-wise conversion"),
                }
            }
            Expression::Tensor(tensor) => {
                let element_wise_list: Vec<Vec<Vec<_>>> = tensor.into_iter()
                    .map(|mat| {
                        mat.into_iter()
                            .map(|row| {
                                row.into_iter()
                                    .map(|elem| elem.into_element_wise())
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<Vec<_>>>()
                    })
                    .collect();
                assert!(
                    element_wise_list.iter().all(|mat| {
                        mat.iter().all(|row| {
                            row.iter().all(|elem| elem.gen_shape() == TensorShape::Scalar)
                        })
                    }),
                    "Tensor elements must all be scalar."
                );
                Expression::Tensor(element_wise_list)
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

// impl UnitaryDefinition {
    // pub fn get_radices(&self) -> Vec<usize> {
    //     match &self.radices {
    //         Some(radices) => radices.clone(),
    //         None => {
    //             let mut d = self.dim();
    //             let power_of_2 = d & (d - 1) == 0 && d != 0;
    //             if !power_of_2 {
    //                 panic!("Dimension must be a power of 2");
    //             }
    //             let mut radices = Vec::new();
    //             while d > 1 {
    //                 radices.push(2);
    //                 d >>= 1;
    //             }
    //             radices
    //         }
    //     }
    // }

    // pub fn dim(&self) -> usize {
    //     self.body.dim()
    // }
// }

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedDefinition {
    pub name: String,
    pub radices: Option<Vec<usize>>,
    pub variables: Vec<String>,
    pub body: Expression,
}

fn get_power_of_two(d: usize) -> Option<usize> {
    if d == 0 {
        return None;
    }
    if d & (d - 1) != 0 {
        return None;
    }
    let mut power = 1;
    let mut exponent = 0;
    while d != power {
        power <<= 1;
        exponent += 1;
    }
    Some(exponent)
}

impl ParsedDefinition {
    pub fn get_radices(&self) -> Vec<usize> {
        match &self.radices {
            Some(radices) => radices.clone(),
            None => {
                let shape = self.body.gen_shape();
                match shape {
                    TensorShape::Scalar => vec![],
                    TensorShape::Vector(d) => {
                        match get_power_of_two(d) {
                            Some(n) => vec![2; n],
                            None => vec![d],
                        }
                    },
                    TensorShape::Matrix(nrows, ncols) => {
                        if nrows == ncols {
                            match get_power_of_two(nrows) {
                                Some(n) => vec![2; n * 2],
                                None => vec![nrows, ncols],
                            }
                        } else {
                            match (get_power_of_two(nrows), get_power_of_two(ncols)) {
                                (Some(n), Some(m)) => vec![2; n * m],
                                (Some(n), None) => vec![2; n].into_iter().chain(vec![ncols].into_iter()).collect(),
                                (None, Some(m)) => vec![nrows].into_iter().chain(vec![2; m].into_iter()).collect(),
                                (None, None) => vec![nrows, ncols],
                            }
                        }
                    },
                    TensorShape::Tensor3D(nmats, nrows, ncols) => {
                        if nrows == ncols {
                            match get_power_of_two(nrows) {
                                Some(n) => vec![2; n * 2].into_iter().chain(vec![nmats].into_iter()).collect(),
                                None => vec![nrows, ncols, nmats],
                            }
                        } else {
                            match (get_power_of_two(nrows), get_power_of_two(ncols)) {
                                (Some(n), Some(m)) => vec![2; n * m].into_iter().chain(vec![nmats].into_iter()).collect(),
                                (Some(n), None) => vec![2; n].into_iter().chain(vec![ncols, nmats].into_iter()).collect(),
                                (None, Some(m)) => vec![nrows].into_iter().chain(vec![2; m].into_iter()).chain(vec![nmats].into_iter()).collect(),
                                (None, None) => vec![nrows, ncols, nmats],
                            }
                        }
                    },
                    _ => panic!("Dynamic tensor shape unsupported"),
                }
            }
        }
    }
}

enum ElementExpression {
    Number(String),
    Variable(String),
    Binary {
        op: char,
        lhs: Box<ElementExpression>,
        rhs: Box<ElementExpression>,
    },
    Unary {
        op: char,
        expr: Box<ElementExpression>,
    },
    Call {
        fn_name: String,
        args: Vec<ElementExpression>,
    },
}

pub struct ElementWiseDefinition {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub variables: Vec<String>,
    pub body: Vec<ElementExpression>,
    pub shape: TensorShape,
}

impl ElementWiseDefinition {
    // pub fn new(pdef: ParsedDefinition) -> Self {
    //     let mut body = Vec::new();
    // }
}
