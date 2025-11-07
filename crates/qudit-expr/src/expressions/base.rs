use num::FromPrimitive;
use num::ToPrimitive;
use num::bigint::BigInt;
use num::rational::Ratio;
use qudit_core::RealScalar;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::analysis::simplify;

pub type Rational = Ratio<BigInt>;
pub type Constant = Rational;

#[derive(Clone)]
pub enum Expression {
    Pi,
    Variable(String),
    Constant(Constant),
    Neg(Box<Expression>),
    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Mul(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
    Pow(Box<Expression>, Box<Expression>),
    Sqrt(Box<Expression>),
    Sin(Box<Expression>),
    Cos(Box<Expression>),
}

impl Expression {
    pub fn zero() -> Self {
        Expression::Constant(Constant::new(BigInt::from(0), BigInt::from(1)))
    }

    pub fn one() -> Self {
        Expression::Constant(Constant::new(BigInt::from(1), BigInt::from(1)))
    }

    pub fn from_int(n: i64) -> Self {
        Expression::Constant(Constant::new(BigInt::from(n), BigInt::from(1)))
    }

    pub fn from_float(f: f64) -> Self {
        Self::from_float_64(f)
    }

    pub fn from_float_32(f: f32) -> Self {
        Expression::Constant(Constant::from_f32(f).unwrap())
    }

    pub fn from_float_64(f: f64) -> Self {
        Expression::Constant(Constant::from_f64(f).unwrap())
    }

    pub fn to_float(&self) -> f64 {
        match self {
            Expression::Constant(c) => c.to_f64().unwrap(),
            Expression::Variable(_) => panic!("Cannot convert variable to float"),
            Expression::Pi => std::f64::consts::PI,
            Expression::Neg(expr) => -expr.to_float(),
            Expression::Add(lhs, rhs) => lhs.to_float() + rhs.to_float(),
            Expression::Sub(lhs, rhs) => lhs.to_float() - rhs.to_float(),
            Expression::Mul(lhs, rhs) => lhs.to_float() * rhs.to_float(),
            Expression::Div(lhs, rhs) => lhs.to_float() / rhs.to_float(),
            Expression::Pow(lhs, rhs) => lhs.to_float().powf(rhs.to_float()),
            Expression::Sqrt(expr) => expr.to_float().sqrt(),
            Expression::Sin(expr) => expr.to_float().sin(),
            Expression::Cos(expr) => expr.to_float().cos(),
        }
    }

    pub fn to_constant(&self) -> Constant {
        // TODO: Figure out how to maintain precision by doing math over Constants
        Constant::from_float(self.to_float()).unwrap()
    }

    pub fn gather_context(&self) -> HashSet<String> {
        let mut context = HashSet::new();
        context.insert(self.to_string());
        match self {
            Expression::Pi => {
                context.insert("pi".to_string());
            }
            Expression::Variable(var) => {
                context.insert(var.clone());
            }
            Expression::Constant(_) => {
                context.insert(self.to_string());
                context.insert(self.to_float().to_string());
            }
            Expression::Neg(expr) => {
                context.extend(expr.gather_context());
            }
            Expression::Add(lhs, rhs) => {
                context.extend(lhs.gather_context());
                context.extend(rhs.gather_context());
            }
            Expression::Sub(lhs, rhs) => {
                context.extend(lhs.gather_context());
                context.extend(rhs.gather_context());
            }
            Expression::Mul(lhs, rhs) => {
                context.extend(lhs.gather_context());
                context.extend(rhs.gather_context());
            }
            Expression::Div(lhs, rhs) => {
                context.extend(lhs.gather_context());
                context.extend(rhs.gather_context());
            }
            Expression::Pow(lhs, rhs) => {
                context.extend(lhs.gather_context());
                context.extend(rhs.gather_context());
            }
            Expression::Sqrt(expr) => {
                context.extend(expr.gather_context());
            }
            Expression::Sin(expr) => {
                context.extend(expr.gather_context());
            }
            Expression::Cos(expr) => {
                context.extend(expr.gather_context());
            }
        }
        context
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Expression::Constant(c) => *c.numer() == BigInt::from(0),
            Expression::Neg(expr) => expr.is_zero(),
            Expression::Add(lhs, rhs) => lhs.is_zero() && rhs.is_zero(),
            Expression::Sub(lhs, rhs) => (lhs.is_zero() && rhs.is_zero()) || lhs == rhs,
            Expression::Mul(lhs, rhs) => lhs.is_zero() || rhs.is_zero(),
            Expression::Div(lhs, _) => lhs.is_zero(),
            Expression::Pow(lhs, rhs) => lhs.is_zero() && !rhs.is_zero(),
            Expression::Sqrt(expr) => expr.is_zero(),
            Expression::Sin(expr) => expr.is_zero(),
            Expression::Cos(expr) => {
                !expr.is_parameterized()
                    && (expr.eval::<f64>(&HashMap::new()) - std::f64::consts::PI / 2.0) < 1e-6
            }
            Expression::Pi => false,
            Expression::Variable(_) => false,
        }
    }

    /// Conservative check for zero. This is faster than the exact check.
    pub fn is_zero_fast(&self) -> bool {
        match self {
            Expression::Constant(c) => *c.numer() == BigInt::from(0),
            Expression::Neg(expr) => expr.is_zero_fast(),
            Expression::Add(lhs, rhs) => lhs.is_zero_fast() && rhs.is_zero_fast(),
            Expression::Sub(lhs, rhs) => lhs.is_zero_fast() && rhs.is_zero_fast(),
            Expression::Mul(lhs, rhs) => lhs.is_zero_fast() || rhs.is_zero_fast(),
            Expression::Div(lhs, _) => lhs.is_zero_fast(),
            Expression::Pow(lhs, rhs) => lhs.is_zero_fast() && !rhs.is_zero_fast(),
            Expression::Sqrt(expr) => expr.is_zero_fast(),
            Expression::Sin(expr) => expr.is_zero_fast(),
            Expression::Cos(_expr) => false,
            Expression::Pi => false,
            Expression::Variable(_) => false,
        }
    }

    pub fn is_one(&self) -> bool {
        match self {
            Expression::Constant(c) => *c.numer() == *c.denom(),
            Expression::Neg(expr) => {
                !expr.is_parameterized() && expr.eval::<f64>(&HashMap::new()) == -1.0
            }
            Expression::Add(lhs, rhs) => {
                lhs.is_one() && rhs.is_zero() || lhs.is_zero() && rhs.is_one()
            }
            Expression::Sub(lhs, rhs) => lhs.is_one() && rhs.is_zero(),
            Expression::Mul(lhs, rhs) => lhs.is_one() && rhs.is_one(),
            Expression::Div(lhs, rhs) => lhs == rhs && !rhs.is_zero(),
            Expression::Pow(lhs, _rhs) => lhs.is_one(),
            Expression::Sqrt(expr) => expr.is_one(),
            Expression::Sin(expr) => {
                !expr.is_parameterized()
                    && (expr.eval::<f64>(&HashMap::new()) - std::f64::consts::PI / 2.0) < 1e-6
            }
            Expression::Cos(expr) => expr.is_zero(),
            Expression::Pi => false,
            Expression::Variable(_) => false,
        }
    }

    pub fn is_one_fast(&self) -> bool {
        match self {
            Expression::Constant(c) => *c.numer() == *c.denom(),
            Expression::Neg(_expr) => false,
            Expression::Add(lhs, rhs) => {
                lhs.is_one_fast() && rhs.is_zero_fast() || lhs.is_zero_fast() && rhs.is_one_fast()
            }
            Expression::Sub(lhs, rhs) => lhs.is_one_fast() && rhs.is_zero_fast(),
            Expression::Mul(lhs, rhs) => lhs.is_one_fast() && rhs.is_one_fast(),
            Expression::Div(lhs, rhs) => lhs.is_one_fast() && rhs.is_one_fast(),
            Expression::Pow(lhs, _rhs) => lhs.is_one_fast(),
            Expression::Sqrt(expr) => expr.is_one_fast(),
            Expression::Sin(_expr) => false,
            Expression::Cos(expr) => expr.is_zero_fast(),
            Expression::Pi => false,
            Expression::Variable(_) => false,
        }
    }

    pub fn contains_variable<T: AsRef<str>>(&self, var: T) -> bool {
        let var = var.as_ref();
        match self {
            Expression::Pi => false,
            Expression::Variable(v) => v == var,
            Expression::Constant(_) => false,
            Expression::Neg(expr) => expr.contains_variable(var),
            Expression::Add(lhs, rhs) => lhs.contains_variable(var) || rhs.contains_variable(var),
            Expression::Sub(lhs, rhs) => lhs.contains_variable(var) || rhs.contains_variable(var),
            Expression::Mul(lhs, rhs) => lhs.contains_variable(var) || rhs.contains_variable(var),
            Expression::Div(lhs, rhs) => lhs.contains_variable(var) || rhs.contains_variable(var),
            Expression::Pow(lhs, rhs) => lhs.contains_variable(var) || rhs.contains_variable(var),
            Expression::Sqrt(expr) => expr.contains_variable(var),
            Expression::Sin(expr) => expr.contains_variable(var),
            Expression::Cos(expr) => expr.contains_variable(var),
        }
    }

    pub fn is_parameterized(&self) -> bool {
        match self {
            Expression::Pi => false,
            Expression::Variable(_) => true,
            Expression::Constant(_) => false,
            Expression::Neg(expr) => expr.is_parameterized(),
            Expression::Add(lhs, rhs) => lhs.is_parameterized() || rhs.is_parameterized(),
            Expression::Sub(lhs, rhs) => lhs.is_parameterized() || rhs.is_parameterized(),
            Expression::Mul(lhs, rhs) => lhs.is_parameterized() || rhs.is_parameterized(),
            Expression::Div(lhs, rhs) => lhs.is_parameterized() || rhs.is_parameterized(),
            Expression::Pow(lhs, rhs) => lhs.is_parameterized() || rhs.is_parameterized(),
            Expression::Sqrt(expr) => expr.is_parameterized(),
            Expression::Sin(expr) => expr.is_parameterized(),
            Expression::Cos(expr) => expr.is_parameterized(),
        }
    }

    pub fn eval<R: RealScalar>(&self, args: &HashMap<&str, R>) -> R {
        match self {
            Expression::Pi => R::PI(),
            Expression::Variable(var) => {
                if let Some(val) = args.get(var.as_str()) {
                    *val
                } else {
                    panic!("Variable {} not found in arguments", var)
                }
            }
            Expression::Constant(c) => R::from_ratio(c.clone()).unwrap(),
            Expression::Neg(expr) => -expr.eval(args),
            Expression::Add(lhs, rhs) => lhs.eval(args) + rhs.eval(args),
            Expression::Sub(lhs, rhs) => lhs.eval(args) - rhs.eval(args),
            Expression::Mul(lhs, rhs) => lhs.eval(args) * rhs.eval(args),
            Expression::Div(lhs, rhs) => lhs.eval(args) / rhs.eval(args),
            Expression::Pow(lhs, rhs) => lhs.eval(args).powf(rhs.eval(args)),
            Expression::Sqrt(expr) => expr.eval(args).sqrt(),
            Expression::Sin(expr) => expr.eval(args).sin(),
            Expression::Cos(expr) => expr.eval(args).cos(),
        }
    }

    /// Uses a magic value to evaluate the expression. This is useful for hashing expressions.
    pub fn hash_eval(&self) -> f64 {
        let val = match self {
            Expression::Pi => self.to_float(),
            Expression::Variable(_) => 1.7,
            Expression::Constant(_) => self.to_float(),
            Expression::Neg(expr) => -expr.hash_eval(),
            Expression::Add(lhs, rhs) => lhs.hash_eval() + rhs.hash_eval(),
            Expression::Sub(lhs, rhs) => lhs.hash_eval() - rhs.hash_eval(),
            Expression::Mul(lhs, rhs) => lhs.hash_eval() * rhs.hash_eval(),
            Expression::Div(lhs, rhs) => lhs.hash_eval() / rhs.hash_eval(),
            Expression::Pow(lhs, rhs) => lhs.hash_eval().powf(rhs.hash_eval()),
            Expression::Sqrt(expr) => expr.hash_eval().sqrt(),
            Expression::Sin(expr) => expr.hash_eval().sin(),
            Expression::Cos(expr) => expr.hash_eval().cos(),
        };

        if val.is_nan() || val.is_subnormal() {
            0.0
        } else {
            val
        }
    }

    pub fn map_var_names(&self, var_map: &HashMap<String, String>) -> Self {
        match self {
            Expression::Pi => Expression::Pi,
            Expression::Variable(var) => {
                if let Some(new_var) = var_map.get(var.as_str()) {
                    Expression::Variable(new_var.to_string())
                } else {
                    Expression::Variable(var.clone())
                }
            }
            Expression::Constant(c) => Expression::Constant(c.clone()),
            Expression::Neg(expr) => Expression::Neg(Box::new(expr.map_var_names(var_map))),
            Expression::Add(lhs, rhs) => Expression::Add(
                Box::new(lhs.map_var_names(var_map)),
                Box::new(rhs.map_var_names(var_map)),
            ),
            Expression::Sub(lhs, rhs) => Expression::Sub(
                Box::new(lhs.map_var_names(var_map)),
                Box::new(rhs.map_var_names(var_map)),
            ),
            Expression::Mul(lhs, rhs) => Expression::Mul(
                Box::new(lhs.map_var_names(var_map)),
                Box::new(rhs.map_var_names(var_map)),
            ),
            Expression::Div(lhs, rhs) => Expression::Div(
                Box::new(lhs.map_var_names(var_map)),
                Box::new(rhs.map_var_names(var_map)),
            ),
            Expression::Pow(lhs, rhs) => Expression::Pow(
                Box::new(lhs.map_var_names(var_map)),
                Box::new(rhs.map_var_names(var_map)),
            ),
            Expression::Sqrt(expr) => Expression::Sqrt(Box::new(expr.map_var_names(var_map))),
            Expression::Sin(expr) => Expression::Sin(Box::new(expr.map_var_names(var_map))),
            Expression::Cos(expr) => Expression::Cos(Box::new(expr.map_var_names(var_map))),
        }
    }

    pub fn rename_variable<S: AsRef<str>, T: AsRef<str>>(&self, original: S, new: T) -> Self {
        let original = original.as_ref();
        let new = new.as_ref();
        match self {
            Expression::Pi => Expression::Pi,
            Expression::Variable(var) => {
                if var == original {
                    Expression::Variable(new.to_string())
                } else {
                    Expression::Variable(var.clone())
                }
            }
            Expression::Constant(c) => Expression::Constant(c.clone()),
            Expression::Neg(expr) => Expression::Neg(Box::new(expr.rename_variable(original, new))),
            Expression::Add(lhs, rhs) => Expression::Add(
                Box::new(lhs.rename_variable(original, new)),
                Box::new(rhs.rename_variable(original, new)),
            ),
            Expression::Sub(lhs, rhs) => Expression::Sub(
                Box::new(lhs.rename_variable(original, new)),
                Box::new(rhs.rename_variable(original, new)),
            ),
            Expression::Mul(lhs, rhs) => Expression::Mul(
                Box::new(lhs.rename_variable(original, new)),
                Box::new(rhs.rename_variable(original, new)),
            ),
            Expression::Div(lhs, rhs) => Expression::Div(
                Box::new(lhs.rename_variable(original, new)),
                Box::new(rhs.rename_variable(original, new)),
            ),
            Expression::Pow(lhs, rhs) => Expression::Pow(
                Box::new(lhs.rename_variable(original, new)),
                Box::new(rhs.rename_variable(original, new)),
            ),
            Expression::Sqrt(expr) => {
                Expression::Sqrt(Box::new(expr.rename_variable(original, new)))
            }
            Expression::Sin(expr) => Expression::Sin(Box::new(expr.rename_variable(original, new))),
            Expression::Cos(expr) => Expression::Cos(Box::new(expr.rename_variable(original, new))),
        }
    }

    pub fn differentiate<S: AsRef<str>>(&self, wrt: S) -> Self {
        let wrt = wrt.as_ref();
        match self {
            Expression::Pi => Expression::zero(),
            Expression::Variable(var) => {
                if var == wrt {
                    Expression::one()
                } else {
                    Expression::zero()
                }
            }
            Expression::Constant(_) => Expression::zero(),
            Expression::Neg(expr) => Expression::Neg(Box::new(expr.differentiate(wrt))),
            Expression::Add(lhs, rhs) => Expression::Add(
                Box::new(lhs.differentiate(wrt)),
                Box::new(rhs.differentiate(wrt)),
            ),
            Expression::Sub(lhs, rhs) => Expression::Sub(
                Box::new(lhs.differentiate(wrt)),
                Box::new(rhs.differentiate(wrt)),
            ),
            Expression::Mul(lhs, rhs) => {
                lhs.differentiate(wrt) * *rhs.clone() + *lhs.clone() * rhs.differentiate(wrt)
            }
            Expression::Div(lhs, rhs) => {
                (lhs.differentiate(wrt) * *rhs.clone() - *lhs.clone() * rhs.differentiate(wrt))
                    / (*rhs.clone() * *rhs.clone())
            }
            Expression::Pow(lhs, rhs) => {
                let base_fn_x = lhs.contains_variable(wrt);
                let exponent_fn_x = rhs.contains_variable(wrt);

                if !base_fn_x && !exponent_fn_x {
                    Expression::zero()
                } else if !base_fn_x && exponent_fn_x {
                    if lhs.is_parameterized() {
                        todo!(
                            "Cannot differentiate with respect to a parameterized power base until ln is implemented"
                        )
                    } else {
                        self.clone()
                            * rhs.differentiate(wrt)
                            * Expression::from_float(lhs.eval::<f64>(&HashMap::new()).ln())
                    }
                } else if base_fn_x && !exponent_fn_x {
                    *rhs.clone()
                        * Expression::Pow(
                            Box::new(*lhs.clone()),
                            Box::new(*rhs.clone() - Expression::one()),
                        )
                        * lhs.differentiate(wrt)
                } else {
                    todo!(
                        "Cannot differentiate with respect to a parameterized base and exponent until ln is implemented"
                    )
                }
            }
            Expression::Sqrt(expr) => {
                let two = Expression::from_int(2);
                (Expression::one() / (two * self.clone())) * expr.differentiate(wrt)
            }
            Expression::Sin(expr) => {
                Expression::Cos(Box::new(*expr.clone())) * expr.differentiate(wrt)
            }
            Expression::Cos(expr) => {
                Expression::Neg(Box::new(Expression::Sin(Box::new(*expr.clone()))))
                    * expr.differentiate(wrt)
            }
        }
    }

    pub fn get_ancestors<S: AsRef<str>>(&self, variable: S) -> Vec<Expression> {
        let variable = variable.as_ref();
        let mut ancestors = Vec::new();
        match self {
            Expression::Pi => {}
            Expression::Variable(var) => {
                if var == variable {
                    ancestors.push(self.clone());
                }
            }
            Expression::Constant(_) => {}
            Expression::Neg(expr) => {
                let node_ancsestors = expr.get_ancestors(variable);
                let is_empty = node_ancsestors.is_empty();
                ancestors.extend(node_ancsestors);
                if !is_empty {
                    ancestors.push(self.clone());
                }
            }
            Expression::Add(lhs, rhs) => {
                let lhs_ancestors = lhs.get_ancestors(variable);
                let rhs_ancestors = rhs.get_ancestors(variable);
                let is_empty = lhs_ancestors.is_empty() && rhs_ancestors.is_empty();
                ancestors.extend(lhs_ancestors);
                ancestors.extend(rhs_ancestors);
                if !is_empty {
                    ancestors.push(self.clone());
                }
            }
            Expression::Sub(lhs, rhs) => {
                let lhs_ancestors = lhs.get_ancestors(variable);
                let rhs_ancestors = rhs.get_ancestors(variable);
                let is_empty = lhs_ancestors.is_empty() && rhs_ancestors.is_empty();
                ancestors.extend(lhs_ancestors);
                ancestors.extend(rhs_ancestors);
                if !is_empty {
                    ancestors.push(self.clone());
                }
            }
            Expression::Mul(lhs, rhs) => {
                let lhs_ancestors = lhs.get_ancestors(variable);
                let rhs_ancestors = rhs.get_ancestors(variable);
                let is_empty = lhs_ancestors.is_empty() && rhs_ancestors.is_empty();
                ancestors.extend(lhs_ancestors);
                ancestors.extend(rhs_ancestors);
                if !is_empty {
                    ancestors.push(self.clone());
                }
            }
            Expression::Div(lhs, rhs) => {
                let lhs_ancestors = lhs.get_ancestors(variable);
                let rhs_ancestors = rhs.get_ancestors(variable);
                let is_empty = lhs_ancestors.is_empty() && rhs_ancestors.is_empty();
                ancestors.extend(lhs_ancestors);
                ancestors.extend(rhs_ancestors);
                if !is_empty {
                    ancestors.push(self.clone());
                }
            }
            Expression::Pow(lhs, rhs) => {
                let lhs_ancestors = lhs.get_ancestors(variable);
                let rhs_ancestors = rhs.get_ancestors(variable);
                let is_empty = lhs_ancestors.is_empty() && rhs_ancestors.is_empty();
                ancestors.extend(lhs_ancestors);
                ancestors.extend(rhs_ancestors);
                if !is_empty {
                    ancestors.push(self.clone());
                }
            }
            Expression::Sqrt(expr) => {
                let node_ancsestors = expr.get_ancestors(variable);
                let is_empty = node_ancsestors.is_empty();
                ancestors.extend(node_ancsestors);
                if !is_empty {
                    ancestors.push(self.clone());
                }
            }
            Expression::Sin(expr) => {
                let node_ancsestors = expr.get_ancestors(variable);
                let is_empty = node_ancsestors.is_empty();
                ancestors.extend(node_ancsestors);
                if !is_empty {
                    ancestors.push(self.clone());
                }
            }
            Expression::Cos(expr) => {
                let node_ancsestors = expr.get_ancestors(variable);
                let is_empty = node_ancsestors.is_empty();
                ancestors.extend(node_ancsestors);
                if !is_empty {
                    ancestors.push(self.clone());
                }
            }
        }
        ancestors
    }

    pub fn fast_eq(&self, other: &Expression) -> bool {
        match (self, other) {
            (Expression::Pi, Expression::Pi) => true,
            (Expression::Variable(var1), Expression::Variable(var2)) => var1 == var2,
            (Expression::Constant(c1), Expression::Constant(c2)) => c1 == c2,
            (Expression::Neg(expr1), Expression::Neg(expr2)) => expr1.fast_eq(expr2),
            (Expression::Add(lhs1, rhs1), Expression::Add(lhs2, rhs2)) => {
                (lhs1.fast_eq(lhs2) && rhs1.fast_eq(rhs2))
                    || (lhs1.fast_eq(rhs2) && rhs1.fast_eq(lhs2))
            }
            (Expression::Sub(lhs1, rhs1), Expression::Sub(lhs2, rhs2)) => {
                (lhs1.fast_eq(lhs2) && rhs1.fast_eq(rhs2))
                    || (lhs1.fast_eq(rhs2) && rhs1.fast_eq(lhs2))
            }
            (Expression::Mul(lhs1, rhs1), Expression::Mul(lhs2, rhs2)) => {
                (lhs1.fast_eq(lhs2) && rhs1.fast_eq(rhs2))
                    || (lhs1.fast_eq(rhs2) && rhs1.fast_eq(lhs2))
            }
            (Expression::Div(lhs1, rhs1), Expression::Div(lhs2, rhs2)) => {
                (lhs1.fast_eq(lhs2) && rhs1.fast_eq(rhs2))
                    || (lhs1.fast_eq(rhs2) && rhs1.fast_eq(lhs2))
            }
            (Expression::Pow(lhs1, rhs1), Expression::Pow(lhs2, rhs2)) => {
                (lhs1.fast_eq(lhs2) && rhs1.fast_eq(rhs2))
                    || (lhs1.fast_eq(rhs2) && rhs1.fast_eq(lhs2))
            }
            (Expression::Sqrt(expr1), Expression::Sqrt(expr2)) => expr1.fast_eq(expr2),
            (Expression::Sin(expr1), Expression::Sin(expr2)) => expr1.fast_eq(expr2),
            (Expression::Cos(expr1), Expression::Cos(expr2)) => expr1.fast_eq(expr2),
            _ => false,
        }
    }

    pub fn substitute<S: AsRef<Expression>, T: AsRef<Expression>>(
        &self,
        original: S,
        substitution: T,
    ) -> Self {
        let original = original.as_ref();
        let substitution = substitution.as_ref();
        if self.fast_eq(original) {
            return substitution.clone();
        }
        match self {
            Expression::Pi => self.clone(),
            Expression::Variable(_) => self.clone(),
            Expression::Constant(_) => self.clone(),
            Expression::Neg(expr) => {
                Expression::Neg(Box::new(expr.substitute(original, substitution)))
            }
            Expression::Add(lhs, rhs) => Expression::Add(
                Box::new(lhs.substitute(original, substitution)),
                Box::new(rhs.substitute(original, substitution)),
            ),
            Expression::Sub(lhs, rhs) => Expression::Sub(
                Box::new(lhs.substitute(original, substitution)),
                Box::new(rhs.substitute(original, substitution)),
            ),
            Expression::Mul(lhs, rhs) => Expression::Mul(
                Box::new(lhs.substitute(original, substitution)),
                Box::new(rhs.substitute(original, substitution)),
            ),
            Expression::Div(lhs, rhs) => Expression::Div(
                Box::new(lhs.substitute(original, substitution)),
                Box::new(rhs.substitute(original, substitution)),
            ),
            Expression::Pow(lhs, rhs) => Expression::Pow(
                Box::new(lhs.substitute(original, substitution)),
                Box::new(rhs.substitute(original, substitution)),
            ),
            Expression::Sqrt(expr) => {
                Expression::Sqrt(Box::new(expr.substitute(original, substitution)))
            }
            Expression::Sin(expr) => {
                Expression::Sin(Box::new(expr.substitute(original, substitution)))
            }
            Expression::Cos(expr) => {
                Expression::Cos(Box::new(expr.substitute(original, substitution)))
            }
        }
    }

    pub fn simplify(&self) -> Self {
        simplify(self)
    }

    pub fn get_unique_variables(&self) -> Vec<String> {
        match self {
            Expression::Pi => {
                vec![]
            }
            Expression::Variable(s) => {
                vec![s.clone()]
            }
            Expression::Constant(_) => {
                vec![]
            }
            Expression::Neg(expr) => expr.get_unique_variables(),
            Expression::Add(lhs, rhs) => {
                let mut l = lhs.get_unique_variables();
                for r in rhs.get_unique_variables().into_iter() {
                    if !l.contains(&r) {
                        l.push(r)
                    }
                }
                l
            }
            Expression::Sub(lhs, rhs) => {
                let mut l = lhs.get_unique_variables();
                for r in rhs.get_unique_variables().into_iter() {
                    if !l.contains(&r) {
                        l.push(r)
                    }
                }
                l
            }
            Expression::Mul(lhs, rhs) => {
                let mut l = lhs.get_unique_variables();
                for r in rhs.get_unique_variables().into_iter() {
                    if !l.contains(&r) {
                        l.push(r)
                    }
                }
                l
            }
            Expression::Div(lhs, rhs) => {
                let mut l = lhs.get_unique_variables();
                for r in rhs.get_unique_variables().into_iter() {
                    if !l.contains(&r) {
                        l.push(r)
                    }
                }
                l
            }
            Expression::Pow(lhs, rhs) => {
                let mut l = lhs.get_unique_variables();
                for r in rhs.get_unique_variables().into_iter() {
                    if !l.contains(&r) {
                        l.push(r)
                    }
                }
                l
            }
            Expression::Sqrt(expr) => expr.get_unique_variables(),
            Expression::Sin(expr) => expr.get_unique_variables(),
            Expression::Cos(expr) => expr.get_unique_variables(),
        }
    }
}

impl std::ops::Add<Expression> for Expression {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        &self + &other
    }
}

impl std::ops::Add<&Expression> for Expression {
    type Output = Expression;

    fn add(self, other: &Expression) -> Expression {
        &self + other
    }
}

impl std::ops::Add<Expression> for &Expression {
    type Output = Expression;

    fn add(self, other: Expression) -> Expression {
        self + &other
    }
}

impl std::ops::Add<&Expression> for &Expression {
    type Output = Expression;

    fn add(self, other: &Expression) -> Expression {
        if let Expression::Constant(c1) = self
            && let Expression::Constant(c2) = other
        {
            return Expression::Constant(c1 + c2);
        }
        if other.is_zero_fast() {
            self.clone()
        } else if self.is_zero_fast() {
            other.clone()
        } else {
            Expression::Add(Box::new(self.clone()), Box::new(other.clone()))
        }
    }
}

impl std::ops::Sub<Expression> for Expression {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        &self - &other
    }
}

impl std::ops::Sub<&Expression> for Expression {
    type Output = Expression;

    fn sub(self, other: &Expression) -> Expression {
        &self - other
    }
}

impl std::ops::Sub<Expression> for &Expression {
    type Output = Expression;

    fn sub(self, other: Expression) -> Expression {
        self - &other
    }
}

impl std::ops::Sub<&Expression> for &Expression {
    type Output = Expression;

    fn sub(self, other: &Expression) -> Expression {
        if let Expression::Constant(c1) = self
            && let Expression::Constant(c2) = other
        {
            return Expression::Constant(c1 - c2);
        }
        if other.is_zero_fast() {
            self.clone()
        } else if self.is_zero_fast() {
            -other.clone()
        } else {
            Expression::Sub(Box::new(self.clone()), Box::new(other.clone()))
        }
    }
}

impl std::ops::Mul<Expression> for Expression {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        &self * &other
    }
}

impl std::ops::Mul<&Expression> for Expression {
    type Output = Expression;

    fn mul(self, other: &Expression) -> Expression {
        &self * other
    }
}

impl std::ops::Mul<Expression> for &Expression {
    type Output = Expression;

    fn mul(self, other: Expression) -> Expression {
        self * &other
    }
}

impl std::ops::Mul<&Expression> for &Expression {
    type Output = Expression;

    fn mul(self, other: &Expression) -> Expression {
        if let Expression::Constant(c1) = self
            && let Expression::Constant(c2) = other
        {
            return Expression::Constant(c1 * c2);
        }
        if other.is_zero_fast() || self.is_zero_fast() {
            Expression::zero()
        } else if other.is_one_fast() {
            self.clone()
        } else if self.is_one_fast() {
            other.clone()
        } else {
            Expression::Mul(Box::new(self.clone()), Box::new(other.clone()))
        }
    }
}

impl std::ops::Div<Expression> for Expression {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        &self / &other
    }
}

impl std::ops::Div<&Expression> for Expression {
    type Output = Expression;

    fn div(self, other: &Expression) -> Expression {
        &self / other
    }
}

impl std::ops::Div<Expression> for &Expression {
    type Output = Expression;

    fn div(self, other: Expression) -> Expression {
        self / &other
    }
}

impl std::ops::Div<&Expression> for &Expression {
    type Output = Expression;

    fn div(self, other: &Expression) -> Expression {
        if other.is_zero_fast() {
            panic!("Cannot divide by zero")
        } else if let (Expression::Constant(c1), Expression::Constant(c2)) = (self, other) {
            Expression::Constant(c1 / c2)
        } else if self.is_zero_fast() {
            Expression::zero()
        } else {
            Expression::Div(Box::new(self.clone()), Box::new(other.clone()))
        }
    }
}

impl std::ops::Neg for Expression {
    type Output = Self;

    fn neg(self) -> Self {
        -&self
    }
}

impl std::ops::Neg for &Expression {
    type Output = Expression;

    fn neg(self) -> Expression {
        if self.is_zero_fast() {
            self.clone()
        } else {
            Expression::Neg(Box::new(self.clone()))
        }
    }
}

impl std::fmt::Debug for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl PartialEq for Expression {
    fn eq(&self, other: &Self) -> bool {
        self.fast_eq(other)
        // if self.fast_eq(other) {
        //     return true;
        // }
        // check_equality(self, other)
    }
}

impl Eq for Expression {}

impl std::hash::Hash for Expression {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let val = self.hash_eval();
        (val * 1e5_f64).round().to_bits().hash(state);
    }
}

impl AsRef<Expression> for Expression {
    fn as_ref(&self) -> &Expression {
        self
    }
}

impl std::fmt::Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let inner = match self {
            Expression::Pi => "pi".to_string(),
            Expression::Variable(var) => var.clone(),
            Expression::Constant(_c) => self.to_float().to_string(),
            Expression::Neg(expr) => format!("~ {}", expr),
            Expression::Add(lhs, rhs) => format!("+ {} {}", lhs, rhs),
            Expression::Sub(lhs, rhs) => format!("- {} {}", lhs, rhs),
            Expression::Mul(lhs, rhs) => format!("* {} {}", lhs, rhs),
            Expression::Div(lhs, rhs) => format!("/ {} {}", lhs, rhs),
            Expression::Pow(lhs, rhs) => format!("pow {} {}", lhs, rhs),
            Expression::Sqrt(expr) => format!("sqrt {}", expr),
            Expression::Sin(expr) => format!("sin {}", expr),
            Expression::Cos(expr) => format!("cos {}", expr),
        };
        write!(f, "({})", inner)
    }
}

impl<R: RealScalar> From<R> for Expression {
    fn from(value: R) -> Self {
        Expression::from_float(value.to64())
    }
}
