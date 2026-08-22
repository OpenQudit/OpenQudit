use std::collections::HashMap;

use qudit_core::ComplexScalar;
use serde::Deserialize;
use serde::Serialize;

use crate::expressions::Constant;
use crate::expressions::Expression;
use crate::qgl::Expression as CiscExpression;
use crate::qgl::parse_scalar;

#[derive(Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct ComplexExpression {
    pub real: Expression,
    pub imag: Expression,
}

impl ComplexExpression {
    pub fn from_string(input: impl AsRef<str>) -> Self {
        ComplexExpression::new(
            parse_scalar(input.as_ref()).unwrap_or_else(|e| panic!("Invalid input string: {e}")),
        )
    }

    pub fn from_real_64(input: f64) -> Self {
        ComplexExpression {
            real: Expression::from_float_64(input),
            imag: Expression::zero(),
        }
    }

    pub fn from_real_32(input: f32) -> Self {
        ComplexExpression {
            real: Expression::from_float_32(input),
            imag: Expression::zero(),
        }
    }

    pub fn new(cisc_expr: CiscExpression) -> Self {
        match cisc_expr {
            CiscExpression::Number(num) => ComplexExpression {
                real: Expression::Constant(
                    Constant::from_float(num.parse::<f64>().unwrap()).unwrap(),
                ),
                imag: Expression::zero(),
            },
            CiscExpression::Variable(var) => {
                if var == "i" {
                    ComplexExpression {
                        real: Expression::zero(),
                        imag: Expression::one(),
                    }
                } else if var == "π" || var == "pi" {
                    ComplexExpression {
                        real: Expression::Pi,
                        imag: Expression::zero(),
                    }
                } else {
                    ComplexExpression {
                        real: Expression::Variable(var),
                        imag: Expression::zero(),
                    }
                }
            }
            CiscExpression::Unary { op, expr } => {
                let risc_expr = ComplexExpression::new(*expr);
                match op {
                    '~' => ComplexExpression {
                        real: Expression::Neg(Box::new(risc_expr.real)),
                        imag: Expression::Neg(Box::new(risc_expr.imag)),
                    },
                    _ => panic!("Invalid unary operator: {}", op),
                }
            }
            CiscExpression::Binary { op, lhs, rhs } => {
                let risc_lhs = ComplexExpression::new(*lhs);
                let risc_rhs = ComplexExpression::new(*rhs);
                match op {
                    '+' => risc_lhs + risc_rhs,
                    '-' => risc_lhs - risc_rhs,
                    '*' => risc_lhs * risc_rhs,
                    '/' => risc_lhs / risc_rhs,
                    '^' => {
                        if risc_lhs.is_e() {
                            // checking `is_imag()` requires a non-zero imaginary part
                            // and therefore rejects an exponent that simplifies to zero
                            // (e.g. 'i*0')
                            assert!(
                                risc_rhs.real.is_zero(),
                                "Exponential power must be imaginary",
                            );
                            ComplexExpression {
                                real: Expression::Cos(Box::new(risc_rhs.imag.clone())),
                                imag: Expression::Sin(Box::new(risc_rhs.imag)),
                            }
                        } else {
                            assert!(risc_lhs.is_real(), "Power base must be real");
                            assert!(risc_rhs.is_real(), "Power exponent must be real");
                            ComplexExpression {
                                real: Expression::Pow(
                                    Box::new(risc_lhs.real),
                                    Box::new(risc_rhs.real),
                                ),
                                imag: Expression::zero(),
                            }
                        }
                    }
                    _ => panic!("Invalid binary operator: {}", op),
                }
            }
            CiscExpression::Call { fn_name, args } => match fn_name.as_str() {
                "sqrt" => {
                    let risc_arg = ComplexExpression::new(args[0].clone());
                    assert!(args.len() == 1, "sqrt function takes exactly one argument");
                    assert!(
                        risc_arg.is_real(),
                        "sqrt function is only supported for real numbers"
                    );
                    ComplexExpression {
                        real: Expression::Sqrt(Box::new(risc_arg.real)),
                        imag: Expression::zero(),
                    }
                }
                "sin" => {
                    let risc_arg = ComplexExpression::new(args[0].clone());
                    assert!(args.len() == 1, "sin function takes exactly one argument");
                    assert!(
                        risc_arg.is_real(),
                        "sin function is only supported for real numbers"
                    );
                    ComplexExpression {
                        real: Expression::Sin(Box::new(risc_arg.real)),
                        imag: Expression::zero(),
                    }
                }
                "cos" => {
                    let risc_arg = ComplexExpression::new(args[0].clone());
                    assert!(args.len() == 1, "cos function takes exactly one argument");
                    assert!(
                        risc_arg.is_real(),
                        "cos function is only supported for real numbers"
                    );
                    ComplexExpression {
                        real: Expression::Cos(Box::new(risc_arg.real)),
                        imag: Expression::zero(),
                    }
                }
                "tan" => {
                    let risc_arg = ComplexExpression::new(args[0].clone());
                    assert!(args.len() == 1, "tan function takes exactly one argument");
                    assert!(
                        risc_arg.is_real(),
                        "tan function is only supported for real numbers"
                    );
                    ComplexExpression {
                        real: Expression::Div(
                            Box::new(Expression::Sin(Box::new(risc_arg.real.clone()))),
                            Box::new(Expression::Cos(Box::new(risc_arg.real))),
                        ),
                        imag: Expression::zero(),
                    }
                }
                "csc" => {
                    let risc_arg = ComplexExpression::new(args[0].clone());
                    assert!(args.len() == 1, "csc function takes exactly one argument");
                    assert!(
                        risc_arg.is_real(),
                        "csc function is only supported for real numbers"
                    );
                    ComplexExpression {
                        real: Expression::Div(
                            Box::new(Expression::one()),
                            Box::new(Expression::Sin(Box::new(risc_arg.real))),
                        ),
                        imag: Expression::zero(),
                    }
                }
                "sec" => {
                    let risc_arg = ComplexExpression::new(args[0].clone());
                    assert!(args.len() == 1, "sec function takes exactly one argument");
                    assert!(
                        risc_arg.is_real(),
                        "sec function is only supported for real numbers"
                    );
                    ComplexExpression {
                        real: Expression::Div(
                            Box::new(Expression::one()),
                            Box::new(Expression::Cos(Box::new(risc_arg.real))),
                        ),
                        imag: Expression::zero(),
                    }
                }
                "cot" => {
                    let risc_arg = ComplexExpression::new(args[0].clone());
                    assert!(args.len() == 1, "cot function takes exactly one argument");
                    assert!(
                        risc_arg.is_real(),
                        "cot function is only supported for real numbers"
                    );
                    ComplexExpression {
                        real: Expression::Div(
                            Box::new(Expression::Cos(Box::new(risc_arg.real.clone()))),
                            Box::new(Expression::Sin(Box::new(risc_arg.real))),
                        ),
                        imag: Expression::zero(),
                    }
                }
                _ => panic!("Invalid function name: {}", fn_name),
            },
            _ => panic!("Unexpected expression during complex conversion"),
        }
    }

    pub fn one() -> Self {
        ComplexExpression {
            real: Expression::one(),
            imag: Expression::zero(),
        }
    }

    pub fn zero() -> Self {
        ComplexExpression {
            real: Expression::zero(),
            imag: Expression::zero(),
        }
    }

    pub fn i() -> Self {
        ComplexExpression {
            real: Expression::zero(),
            imag: Expression::one(),
        }
    }

    pub fn pi() -> Self {
        ComplexExpression {
            real: Expression::Pi,
            imag: Expression::zero(),
        }
    }

    pub fn is_real(&self) -> bool {
        self.imag.is_zero()
    }

    pub fn is_imag(&self) -> bool {
        self.real.is_zero() && !self.imag.is_zero()
    }

    pub fn is_cplx(&self) -> bool {
        !self.real.is_zero() && !self.imag.is_zero()
    }

    pub fn is_real_fast(&self) -> bool {
        self.imag.is_zero_fast()
    }

    pub fn is_imag_fast(&self) -> bool {
        self.real.is_zero_fast() && !self.imag.is_zero_fast()
    }

    pub fn is_cplx_fast(&self) -> bool {
        !self.real.is_zero_fast() && !self.imag.is_zero_fast()
    }

    pub fn is_e(&self) -> bool {
        match &self.real {
            Expression::Variable(var) => var == "e",
            _ => false,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.real.is_zero() && self.imag.is_zero()
    }

    pub fn is_one(&self) -> bool {
        self.real.is_one() && self.imag.is_zero()
    }

    pub fn is_zero_fast(&self) -> bool {
        self.real.is_zero_fast() && self.imag.is_zero_fast()
    }

    pub fn is_one_fast(&self) -> bool {
        self.real.is_one_fast() && self.imag.is_zero_fast()
    }

    pub fn eval<C: ComplexScalar>(&self, args: &HashMap<&str, C::R>) -> C {
        C::new(self.real.eval(args), self.imag.eval(args))
    }

    pub fn map_var_names(&self, var_map: &HashMap<String, String>) -> Self {
        ComplexExpression {
            real: self.real.map_var_names(var_map),
            imag: self.imag.map_var_names(var_map),
        }
    }

    pub fn conjugate(&self) -> Self {
        ComplexExpression {
            real: self.real.clone(),
            imag: Expression::Neg(Box::new(self.imag.clone())),
        }
    }

    pub fn conjugate_in_place(&mut self) {
        self.imag = Expression::Neg(Box::new(self.imag.clone()));
    }

    pub fn differentiate(&self, wrt: &str) -> Self {
        ComplexExpression {
            real: self.real.differentiate(wrt),
            imag: self.imag.differentiate(wrt),
        }
    }

    pub fn simplify(&self) -> Self {
        ComplexExpression {
            real: self.real.simplify(),
            imag: self.imag.simplify(),
        }
    }

    pub fn get_ancestors(&self, variable: &str) -> Vec<Expression> {
        let mut ancestors = self.real.get_ancestors(variable);
        let im_ancestors = self.imag.get_ancestors(variable);
        if ancestors.is_empty() {
            return im_ancestors;
        }
        if !im_ancestors.is_empty() {
            ancestors.retain(|x| im_ancestors.contains(x));
        }
        ancestors
    }

    pub fn substitute<S: AsRef<Expression>, T: AsRef<Expression>>(
        &self,
        original: S,
        substitution: T,
    ) -> Self {
        // TODO: Allow ComplexExpression substitution
        ComplexExpression {
            real: self
                .real
                .substitute(original.as_ref(), substitution.as_ref()),
            imag: self
                .imag
                .substitute(original.as_ref(), substitution.as_ref()),
        }
    }

    pub fn rename_variable<S: AsRef<str>, T: AsRef<str>>(&self, original: S, new: T) -> Self {
        ComplexExpression {
            real: self.real.rename_variable(original.as_ref(), new.as_ref()),
            imag: self.imag.rename_variable(original.as_ref(), new.as_ref()),
        }
    }

    pub fn get_unique_variables(&self) -> Vec<String> {
        let mut real = self.real.get_unique_variables();
        for i in self.imag.get_unique_variables().into_iter() {
            if !real.contains(&i) {
                real.push(i)
            }
        }
        real
    }

    pub fn is_parameterized(&self) -> bool {
        self.real.is_parameterized() || self.imag.is_parameterized()
    }
}

impl std::ops::Mul<ComplexExpression> for ComplexExpression {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        &self * &rhs
    }
}

impl std::ops::Mul<&ComplexExpression> for ComplexExpression {
    type Output = ComplexExpression;

    fn mul(self, rhs: &ComplexExpression) -> ComplexExpression {
        &self * rhs
    }
}

impl std::ops::Mul<ComplexExpression> for &ComplexExpression {
    type Output = ComplexExpression;

    fn mul(self, rhs: ComplexExpression) -> ComplexExpression {
        self * &rhs
    }
}

impl std::ops::Mul<&ComplexExpression> for &ComplexExpression {
    type Output = ComplexExpression;

    fn mul(self, rhs: &ComplexExpression) -> ComplexExpression {
        ComplexExpression {
            real: &self.real * &rhs.real - &self.imag * &rhs.imag,
            imag: &self.real * &rhs.imag + &self.imag * &rhs.real,
        }
    }
}

impl std::ops::Add<ComplexExpression> for ComplexExpression {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        &self + &rhs
    }
}

impl std::ops::Add<&ComplexExpression> for ComplexExpression {
    type Output = ComplexExpression;

    fn add(self, rhs: &ComplexExpression) -> ComplexExpression {
        &self + rhs
    }
}

impl std::ops::Add<ComplexExpression> for &ComplexExpression {
    type Output = ComplexExpression;

    fn add(self, rhs: ComplexExpression) -> ComplexExpression {
        self + &rhs
    }
}

impl std::ops::Add<&ComplexExpression> for &ComplexExpression {
    type Output = ComplexExpression;

    fn add(self, rhs: &ComplexExpression) -> ComplexExpression {
        ComplexExpression {
            real: &self.real + &rhs.real,
            imag: &self.imag + &rhs.imag,
        }
    }
}

impl std::ops::Sub<ComplexExpression> for ComplexExpression {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        &self - &rhs
    }
}

impl std::ops::Sub<&ComplexExpression> for ComplexExpression {
    type Output = ComplexExpression;

    fn sub(self, rhs: &ComplexExpression) -> ComplexExpression {
        &self - rhs
    }
}

impl std::ops::Sub<ComplexExpression> for &ComplexExpression {
    type Output = ComplexExpression;

    fn sub(self, rhs: ComplexExpression) -> ComplexExpression {
        self - &rhs
    }
}

impl std::ops::Sub<&ComplexExpression> for &ComplexExpression {
    type Output = ComplexExpression;

    fn sub(self, rhs: &ComplexExpression) -> ComplexExpression {
        ComplexExpression {
            real: &self.real - &rhs.real,
            imag: &self.imag - &rhs.imag,
        }
    }
}

impl std::ops::Neg for ComplexExpression {
    type Output = Self;

    fn neg(self) -> Self {
        -&self
    }
}

impl std::ops::Neg for &ComplexExpression {
    type Output = ComplexExpression;

    fn neg(self) -> ComplexExpression {
        ComplexExpression {
            real: -&self.real,
            imag: -&self.imag,
        }
    }
}

impl std::ops::Div<ComplexExpression> for ComplexExpression {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        &self / &rhs
    }
}

impl std::ops::Div<&ComplexExpression> for ComplexExpression {
    type Output = ComplexExpression;

    fn div(self, rhs: &ComplexExpression) -> ComplexExpression {
        &self / rhs
    }
}

impl std::ops::Div<ComplexExpression> for &ComplexExpression {
    type Output = ComplexExpression;

    fn div(self, rhs: ComplexExpression) -> ComplexExpression {
        self / &rhs
    }
}

impl std::ops::Div<&ComplexExpression> for &ComplexExpression {
    type Output = ComplexExpression;

    fn div(self, rhs: &ComplexExpression) -> ComplexExpression {
        let dem = &rhs.real * &rhs.real + &rhs.imag * &rhs.imag;
        ComplexExpression {
            real: (&self.real * &rhs.real + &self.imag * &rhs.imag) / &dem,
            imag: (&self.imag * &rhs.real - &self.real * &rhs.imag) / &dem,
        }
    }
}

impl std::ops::AddAssign for ComplexExpression {
    fn add_assign(&mut self, rhs: Self) {
        *self = &*self + &rhs;
    }
}

impl std::ops::AddAssign<&ComplexExpression> for ComplexExpression {
    fn add_assign(&mut self, rhs: &Self) {
        *self = &*self + rhs;
    }
}

impl std::ops::SubAssign for ComplexExpression {
    fn sub_assign(&mut self, rhs: Self) {
        *self = &*self - &rhs;
    }
}

impl std::ops::SubAssign<&ComplexExpression> for ComplexExpression {
    fn sub_assign(&mut self, rhs: &Self) {
        *self = &*self - rhs;
    }
}

impl std::ops::MulAssign for ComplexExpression {
    fn mul_assign(&mut self, rhs: Self) {
        *self = &*self * &rhs;
    }
}

impl std::ops::MulAssign<&ComplexExpression> for ComplexExpression {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = &*self * rhs;
    }
}

impl std::ops::DivAssign for ComplexExpression {
    fn div_assign(&mut self, rhs: Self) {
        *self = &*self / &rhs;
    }
}

impl std::ops::DivAssign<&ComplexExpression> for ComplexExpression {
    fn div_assign(&mut self, rhs: &Self) {
        *self = &*self / rhs;
    }
}

impl std::fmt::Debug for ComplexExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("ComplexExpression")
            .field("real", &self.real)
            .field("imag", &self.imag)
            .finish()
    }
}

impl<C: ComplexScalar> From<C> for ComplexExpression {
    fn from(value: C) -> Self {
        ComplexExpression {
            real: value.re().into(),
            imag: value.im().into(),
        }
    }
}

#[cfg(feature = "python")]
mod python {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash as _;
    use std::hash::Hasher as _;

    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::sync::PyOnceLock;
    use pyo3::types::PyDict;
    use pyo3_stub_gen::derive::*;
    use pyo3_stub_gen::impl_stub_type;
    use qudit_core::c64;

    use super::*;
    use crate::expressions::base::python::PyExpression;
    use crate::expressions::base::python::to_python;
    use crate::python::PyExpressionRegistrar;

    /// A symbolic complex scalar, stored as a pair of real expression trees.
    ///
    /// This is the element type of every expression body: a `UnitaryExpression`
    /// over `d` qudit levels is `d * d` of these.
    #[gen_stub_pyclass]
    #[pyclass(name = "ComplexExpression", module = "openqudit.expressions", frozen)]
    pub struct PyComplexExpression {
        expr: ComplexExpression,

        /// Converting a tree into Python allocates one object per node, so the
        /// two halves are materialized at most once per wrapper.
        real: PyOnceLock<Py<PyExpression>>,
        imag: PyOnceLock<Py<PyExpression>>,
    }

    impl PyComplexExpression {
        /// Returns one half of the pair, materializing it on first access.
        fn part<'py>(
            py: Python<'py>,
            cache: &PyOnceLock<Py<PyExpression>>,
            expr: &Expression,
        ) -> PyResult<Py<PyExpression>> {
            cache
                .get_or_try_init(py, || Ok(to_python(py, expr)?.unbind()))
                .map(|node| node.clone_ref(py))
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyComplexExpression {
        /// The real part of this expression.
        #[getter]
        fn real(&self, py: Python<'_>) -> PyResult<Py<PyExpression>> {
            Self::part(py, &self.real, &self.expr.real)
        }

        /// The imaginary part of this expression.
        #[getter]
        fn imag(&self, py: Python<'_>) -> PyResult<Py<PyExpression>> {
            Self::part(py, &self.imag, &self.expr.imag)
        }

        /// Returns the names of the free parameters in this expression, in the
        /// order they are first encountered.
        fn variables(&self) -> Vec<String> {
            self.expr.get_unique_variables()
        }

        /// Returns whether this expression references any free parameter.
        fn is_parameterized(&self) -> bool {
            self.expr.is_parameterized()
        }

        /// Returns whether the imaginary part is structurally zero.
        fn is_real(&self) -> bool {
            self.expr.is_real()
        }

        /// Returns whether the real part is structurally zero and the imaginary
        /// part is not.
        fn is_imag(&self) -> bool {
            self.expr.is_imag()
        }

        /// Returns whether both parts are structurally non-zero.
        fn is_cplx(&self) -> bool {
            self.expr.is_cplx()
        }

        /// Returns whether both parts are structurally equivalent to zero.
        fn is_zero(&self) -> bool {
            self.expr.is_zero()
        }

        /// Returns whether this expression is structurally equivalent to one.
        fn is_one(&self) -> bool {
            self.expr.is_one()
        }

        /// Returns the complex conjugate of this expression.
        fn conjugate(&self) -> Self {
            self.expr.conjugate().into()
        }

        /// Returns an algebraically simplified version of this expression.
        fn simplify(&self) -> Self {
            self.expr.simplify().into()
        }

        /// Returns the partial derivative of this expression with respect to a
        /// parameter.
        ///
        /// # Arguments
        ///
        /// * `wrt` - The name of the parameter to differentiate with respect to.
        fn differentiate(&self, wrt: &str) -> Self {
            self.expr.differentiate(wrt).into()
        }

        /// Returns this expression with one parameter renamed.
        ///
        /// # Arguments
        ///
        /// * `original` - The current parameter name.
        /// * `new` - The replacement name.
        fn rename_variable(&self, original: &str, new: &str) -> Self {
            self.expr.rename_variable(original, new).into()
        }

        /// Returns this expression with every occurrence of one subtree
        /// replaced by another.
        ///
        /// # Arguments
        ///
        /// * `original` - The subtree to search for.
        /// * `substitution` - The subtree to put in its place.
        fn substitute(&self, original: Expression, substitution: Expression) -> Self {
            self.expr.substitute(&original, &substitution).into()
        }

        /// Evaluates this expression numerically.
        ///
        /// # Arguments
        ///
        /// * `values` - A value for each free parameter, passed by name.
        #[pyo3(signature = (**values))]
        fn evaluate(&self, values: Option<&Bound<'_, PyDict>>) -> PyResult<c64> {
            let mut bindings: HashMap<String, f64> = match values {
                Some(values) => values.extract()?,
                None => HashMap::new(),
            };

            let variables = self.expr.get_unique_variables();
            for variable in &variables {
                if !bindings.contains_key(variable) {
                    return Err(PyValueError::new_err(format!(
                        "no value given for variable '{variable}'"
                    )));
                }
            }
            bindings.retain(|name, _| variables.contains(name));

            let args: HashMap<&str, f64> = bindings.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            Ok(self.expr.eval(&args))
        }

        fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
            match other.cast::<PyComplexExpression>() {
                Ok(other) => self.expr == other.get().expr,
                Err(_) => false,
            }
        }

        fn __hash__(&self) -> u64 {
            let mut hasher = DefaultHasher::new();
            self.expr.hash(&mut hasher);
            hasher.finish()
        }

        fn __repr__(&self) -> String {
            format!(
                "ComplexExpression(real={}, imag={})",
                self.expr.real, self.expr.imag
            )
        }
    }

    impl From<ComplexExpression> for PyComplexExpression {
        fn from(value: ComplexExpression) -> Self {
            PyComplexExpression {
                expr: value,
                real: PyOnceLock::new(),
                imag: PyOnceLock::new(),
            }
        }
    }

    impl From<PyComplexExpression> for ComplexExpression {
        fn from(value: PyComplexExpression) -> Self {
            value.expr
        }
    }

    impl_stub_type!(ComplexExpression = PyComplexExpression);

    impl<'py> IntoPyObject<'py> for ComplexExpression {
        type Target = PyComplexExpression;
        type Output = Bound<'py, Self::Target>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            Bound::new(py, PyComplexExpression::from(self))
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for ComplexExpression {
        type Error = PyErr;

        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let wrapper = ob.cast::<PyComplexExpression>()?;
            Ok(wrapper.get().expr.clone())
        }
    }

    /// Registers the ComplexExpression class with the Python module.
    fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
        parent_module.add_class::<PyComplexExpression>()?;
        Ok(())
    }
    inventory::submit!(PyExpressionRegistrar { func: register });
}
