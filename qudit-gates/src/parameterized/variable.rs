use std::collections::HashMap;
use std::ops::Mul;
use qudit_core::{HasParams, QuditRadices, QuditSystem, ToRadix, ComplexScalar, matrix::Mat, accel::matmul_unchecked};
use qudit_expr::{UnitaryExpression, UnitaryExpressionGenerator, ComplexExpression};
use num_traits::identities::Zero;

//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------

fn factorial(n: u64) -> f64 {
    let product: u64 = (1..=n).product();
    return product as f64;
}

fn expr_add(matrix_expr1: &UnitaryExpression, matrix_expr2: &UnitaryExpression) -> UnitaryExpression {
    let matrix_expr2 = matrix_expr2.as_ref();
    let lhs_nrows = matrix_expr1.body.len();
    let lhs_ncols = matrix_expr1.body[0].len();
    let rhs_nrows = matrix_expr2.body.len();
    let rhs_ncols = matrix_expr2.body[0].len();

    if lhs_ncols != rhs_ncols || lhs_nrows != rhs_nrows {
        panic!("Matrix dimensions do not match for addition");
    }

    let mut variables = Vec::new();
    let mut var_map_matrix_expr1 = HashMap::new();
    let mut i = 0;
    for var in matrix_expr1.variables.iter() {
        variables.push(format!("x{}", i));
        var_map_matrix_expr1.insert(var.clone(), format!("x{}", i));
        i += 1;
    }
    let mut var_map_matrix_expr2 = HashMap::new();
    for var in matrix_expr2.variables.iter() {
        variables.push(format!("x{}", i));
        var_map_matrix_expr2.insert(var.clone(), format!("x{}", i));
        i += 1;
    }

    let mut out_body = vec![vec![]; lhs_nrows];

    for i in 0..lhs_nrows {
        for j in 0..rhs_ncols {
            let left_expr = matrix_expr1.body[i][j].map_var_names(&var_map_matrix_expr1);
            let right_expr = matrix_expr2.body[i][j].map_var_names(&var_map_matrix_expr2);
            out_body[i].push(left_expr + right_expr);
        }
    }

    UnitaryExpression {
        name: format!("{} + {}", matrix_expr1.name, matrix_expr2.name),
        radices: matrix_expr1.radices.clone(),
        variables,
        body: out_body,
    }
}

fn expr_sub(matrix_expr1: &UnitaryExpression, matrix_expr2: &UnitaryExpression) -> UnitaryExpression {
    let minus_matrix_expr2 = expr_scalar_mul(&matrix_expr2, -1.0);
    return expr_add(&matrix_expr1, &minus_matrix_expr2);
}

fn expr_scalar_mul(matrix_expr1: &UnitaryExpression, scalar: f64) -> UnitaryExpression {
    let lhs_nrows = matrix_expr1.body.len();
    let lhs_ncols = matrix_expr1.body[0].len();

    let mut variables = Vec::new();
    let mut var_map_matrix_expr1 = HashMap::new();
    let mut i = 0;
    for var in matrix_expr1.variables.iter() {
        variables.push(format!("x{}", i));
        var_map_matrix_expr1.insert(var.clone(), format!("x{}", i));
        i += 1;
    }

    let mut out_body = vec![vec![]; lhs_nrows];

    let formatted_scalar = ComplexExpression{real: scalar.into(), imag: 0.0.into()};
    for i in 0..lhs_nrows {
        for j in 0..lhs_ncols {
            let left_expr = matrix_expr1.body[i][j].map_var_names(&var_map_matrix_expr1);
            out_body[i].push(left_expr.mul(&formatted_scalar));
        }
    }

    UnitaryExpression {
        name: format!("{} * {}", matrix_expr1.name, scalar),
        radices: matrix_expr1.radices.clone(),
        variables,
        body: out_body,
    }
}

fn expr_pow(matrix_expr: &UnitaryExpression, pow: u64) -> UnitaryExpression {
    let mut matrix_pow_expr = UnitaryExpression::identity("Matrix_Pow", matrix_expr.radices());
    for _ in 0..pow {
        matrix_pow_expr = matrix_pow_expr.dot(matrix_expr);
    }
    return matrix_pow_expr;
}

fn expr_exp(matrix_expr: &UnitaryExpression) -> UnitaryExpression {
    let order = 100;
    let mut matrix_exp_expr = UnitaryExpression::identity("Matrix_Exp", matrix_expr.radices());
    for n in 1..order {
        matrix_exp_expr = expr_add(&matrix_exp_expr, &expr_scalar_mul( &expr_pow(matrix_expr, n), 1.0/factorial(n)));
    }
    return matrix_exp_expr;
}

//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct VariableUnitary {
    radices: Vec<usize>,
}

impl VariableUnitary {
    pub fn new(radices: &[usize]) -> Self {
        VariableUnitary {
            radices: radices.to_vec()
        }
    }

    // It's weird that the optimization function doesn't reference the parameters of `self`.
    // Double check that the original bqskitrs code is correct.
    // 
    // u, _, v = la.svd( ( 1 - slowdown_factor ) * env
    //                     + slowdown_factor * self.utry.conj().T )
    // self.utry = v.conj().T @ u.conj().T
    pub fn optimize<C: ComplexScalar>(&self, env: &Mat<C>, slowdown_factor: f64) -> Vec<C::R> {
        let env_svd = Mat::svd(env).unwrap();
        let u = env_svd.U();
        let v = env_svd.V();

        //let mat = Mat::zeros(v.nrows(), u.ncols()).as_mut();
        //matmul_unchecked(v, u.adjoint(), mat);
        let mat = v * u.adjoint();
        assert_eq!(self.num_params(), 2 * mat.nrows() * mat.ncols());

        let mut index_acc: usize;
        let half = self.num_params() / 2;
        let mut ret = vec![C::R::zero(); self.num_params()];
        for (r, row) in mat.row_iter().enumerate() {
            for (c, cmplx) in row.iter().enumerate() {
                index_acc = r * mat.ncols() + c;
                ret[index_acc] = cmplx.real();
                ret[index_acc + half] = cmplx.imag();
            }
        }
        ret
    }

    fn slice_index(&self, row_index: usize, col_index: usize, is_imag: bool) -> usize {
        let mut acc: usize = row_index * self.dimension() + col_index;
        if is_imag {
            acc += self.dimension() * self.dimension();
        }
        return acc;
    }

    fn base_matrix(&self) -> UnitaryExpression {
        let dim = self.dimension();
        let radices_str = self.radices.iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let mut variables = Vec::<String>::new();
        for i in 0..self.num_params() {
            variables.push(format!("θ{}", i));
        }
        let proto = format!("VariableUnitary<{}>(", radices_str) + &variables.join(", ") + ")";

        let mut body = String::from("[\n");
        for i in 0..dim {
            body.push_str("  [");
            for j in 0..dim {
                let real_index = self.slice_index(i, j, false);
                let imag_index = self.slice_index(i, j, true);
                let elem_string = format!("θ{} + i*θ{}", real_index, imag_index);
                body.push_str(&elem_string);
                if j < dim - 1 {
                    body.push_str(", ");
                }
            }
            body.push(']');
            if i < dim - 1 {
                body.push_str(",\n");
            }
        }
        body.push_str("\n]");
        
        UnitaryExpression::new(proto + "{" + &body + "}")
    }

    fn diag_matrix(&self) -> UnitaryExpression {
        let dim = self.dimension();
        let radices_str = self.radices.iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let mut variables = Vec::<String>::new();
        for i in 0..self.num_params() {
            variables.push(format!("θ{}", i));
        }
        let proto = format!("VariableUnitary<{}>(", radices_str) + &variables.join(", ") + ")";

        let mut body = String::from("[\n");
        let mut elem_string: String;
        for i in 0..dim {
            body.push_str("  [");
            for j in 0..dim {
                if i == j {
                    let real_index = self.slice_index(i, j, false);
                    let imag_index = self.slice_index(i, j, true);
                    elem_string = format!("θ{} + i*θ{}", real_index, imag_index);
                } else {
                    elem_string = "0".to_string();
                }
                body.push_str(&elem_string);
                if j < dim - 1 {
                    body.push_str(", ");
                }
            }
            body.push(']');
            if i < dim - 1 {
                body.push_str(",\n");
            }
        }
        body.push_str("\n]");
        
        UnitaryExpression::new(proto + "{" + &body + "}")
    }

    fn perturbation_matrix(&self) -> UnitaryExpression {
        let dim = self.dimension();
        let radices_str = self.radices.iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let mut variables = Vec::<String>::new();
        for i in 0..self.num_params() {
            variables.push(format!("θ{}", i));
        }
        let proto = format!("VariableUnitary<{}>(", radices_str) + &variables.join(", ") + ")";

        let mut body = String::from("[\n");
        let mut elem_string: String;
        for i in 0..dim {
            body.push_str("  [");
            for j in 0..dim {
                if i == j {
                    elem_string = "0".to_string();
                } else {
                    let real_index = self.slice_index(i, j, false);
                    let imag_index = self.slice_index(i, j, true);
                    elem_string = format!("θ{} + i*θ{}", real_index, imag_index);
                }
                body.push_str(&elem_string);
                if j < dim - 1 {
                    body.push_str(", ");
                }
            }
            body.push(']');
            if i < dim - 1 {
                body.push_str(",\n");
            }
        }
        body.push_str("\n]");
        
        UnitaryExpression::new(proto + "{" + &body + "}")
    }

    fn abs_diag_matrix_pow(&self, pow: i32) -> UnitaryExpression {
        let dim = self.dimension();
        let radices_str = self.radices.iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let mut variables = Vec::<String>::new();
        for i in 0..self.num_params() {
            variables.push(format!("θ{}", i));
        }
        let proto = format!("VariableUnitary<{}>(", radices_str) + &variables.join(", ") + ")";

        let mut body = String::from("[\n");
        let mut elem_string: String;
        for i in 0..dim {
            body.push_str("  [");
            for j in 0..dim {
                if i == j {
                    let real_index = self.slice_index(i, j, false);
                    let imag_index = self.slice_index(i, j, true);
                    if pow >= 0 {
                        elem_string = format!("sqrt(θ{}^2 + θ{}^2)^({})", real_index, imag_index, pow);
                    } else {
                        elem_string = format!("1 / sqrt(θ{}^2 + θ{}^2)^({})", real_index, imag_index, -1 * pow);
                    }
                    
                } else {
                    elem_string = "0".to_string();
                }
                body.push_str(&elem_string);
                if j < dim - 1 {
                    body.push_str(", ");
                }
            }
            body.push(']');
            if i < dim - 1 {
                body.push_str(",\n");
            }
        }
        body.push_str("\n]");
        
        UnitaryExpression::new(proto + "{" + &body + "}")
    }

}

impl HasParams for VariableUnitary {
    #[inline]
    fn num_params(&self) -> usize {
        2 * self.dimension() * self.dimension()
    }
}

impl QuditSystem for VariableUnitary {
    #[inline]
    fn radices(&self) -> QuditRadices {
        QuditRadices::new(&self.radices)
    }

    #[inline]
    fn num_qudits(&self) -> usize {
        self.radices.len()
    }

    #[inline]
    fn dimension(&self) -> usize {
        self.radices.iter().product()
    }
}

impl UnitaryExpressionGenerator for VariableUnitary {
    
    // WARNING: This is an approximate (2nd order) solution. (and even then, requires verification)
    // Symbolic polar decomposition is not generally possible.
    // Q = M(M^{\dagger}M)^{-1/2}
    #[inline]
    #[allow(non_snake_case)]
    fn gen_expr(&self) -> UnitaryExpression {
        let M = self.base_matrix();
        let MtM = M.conjugate().transpose().dot(&M);
        let S0 = self.abs_diag_matrix_pow(2);
        let E = expr_sub(&MtM, &S0);
        
        let Q_tilde = M.dot(
            expr_add(
                &expr_sub(
                    &self.abs_diag_matrix_pow(-1), 
                    &expr_scalar_mul(&self.abs_diag_matrix_pow(-3), 0.5).dot(&E)
                ),
                &expr_scalar_mul(
                    &self.abs_diag_matrix_pow(-5)
                    .dot(expr_pow(&E, 2)),3.0/8.0)
            )
        );

        let D = self.diag_matrix();
        let Phi = D.dot(&self.abs_diag_matrix_pow(-1));
        let I = UnitaryExpression::identity("I", self.radices());
        let X = expr_sub(&Phi.conjugate().transpose().dot(Q_tilde), &I);

        // Truncated log K ≈ X - 1/2 X^2
        let mut K = expr_sub(&X, &expr_scalar_mul(&expr_pow(&X, 2), 0.5));

        // Enforcing Hermicity
        K = expr_scalar_mul(&expr_sub(&K, &K.conjugate().transpose()), 0.5);

        return Phi.dot(expr_exp(&K));
    }
}