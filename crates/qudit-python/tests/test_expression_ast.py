import math
from decimal import Decimal
from fractions import Fraction

import pytest

from openqudit.expressions import Expression, UnitaryExpression


class TestConstant:
    """`Expression.Constant` is backed by an exact `fractions.Fraction`."""

    def test_fraction_roundtrips_exactly(self):
        node = Expression.Constant(Fraction(1, 3))
        assert isinstance(node.value, Fraction)
        assert node.value == Fraction(1, 3)
        assert node.value * 3 == 1

    def test_accepts_int(self):
        assert Expression.Constant(7).value == Fraction(7, 1)

    def test_accepts_str(self):
        assert Expression.Constant("1/3").value == Fraction(1, 3)
        assert Expression.Constant("0.25").value == Fraction(1, 4)

    def test_normalizes(self):
        assert Expression.Constant(Fraction(-2, 4)).value == Fraction(-1, 2)

    def test_big_values_survive(self):
        big = Fraction(10**40 + 1, 3**30)
        assert Expression.Constant(big).value == big

    def test_rejects_float(self):
        with pytest.raises(TypeError, match="not an exact constant"):
            Expression.Constant(0.1)

    @pytest.mark.parametrize("value", [None, [1], Decimal("1.5"), object()])
    def test_rejects_non_numbers(self, value):
        with pytest.raises(TypeError, match="expected an int"):
            Expression.Constant(value)

    def test_rejects_unparseable_str(self):
        with pytest.raises(ValueError):
            Expression.Constant("nope")

    def test_rejects_zero_denominator(self):
        class Weird:
            numerator = 1
            denominator = 0

        with pytest.raises(ZeroDivisionError):
            Expression.Constant(Weird())



class TestTree:
    """The AST is inspectable, immutable, and structurally comparable."""

    def test_variants_subclass_expression(self):
        node = Expression.Variable("theta")
        assert isinstance(node, Expression)
        assert isinstance(node, Expression.Variable)
        assert node.name == "theta"

    def test_children_are_expressions(self):
        node = Expression.Add(Expression.Pi(), Expression.Constant(1))
        assert isinstance(node.lhs, Expression.Pi)
        assert isinstance(node.rhs, Expression.Constant)

    def test_structural_pattern_matching(self):
        theta = Expression.Variable("theta")
        node = Expression.Sin(theta * Expression.Constant("1/2"))

        match node:
            case Expression.Sin(Expression.Mul(Expression.Variable(name), _)):
                assert name == "theta"
            case _:
                pytest.fail("expression did not match")

    def test_deep_tree_roundtrips(self):
        node = Expression.Constant(0)
        for _ in range(200):
            node = Expression.Add(node, Expression.Constant(1))
        assert node.to_float() == pytest.approx(200.0)

    def test_equality_and_hashing(self):
        def build():
            return Expression.Variable("x") + Expression.Constant(1)

        assert build() == build()
        assert hash(build()) == hash(build())
        assert len({build(), build()}) == 1

    def test_equality_against_other_types(self):
        assert Expression.Pi() != "pi"

    def test_repr_names_the_constructor(self):
        node = Expression.Neg(Expression.Constant(Fraction(1, 2)))
        assert repr(node) == (
            "Expression.Neg(operand=Expression.Constant(value=Fraction(1, 2)))"
        )


class TestArithmetic:
    def test_operators_build_nodes(self):
        x = Expression.Variable("x")
        assert isinstance(x + x, Expression.Add)
        assert isinstance(x - x, Expression.Sub)
        assert isinstance(-x, Expression.Neg)
        assert isinstance(x**2, Expression.Pow)

    def test_reflected_operators_coerce_constants(self):
        x = Expression.Variable("x")
        assert (2 * x).evaluate(x=3.0) == pytest.approx(6.0)
        assert (2 - x).evaluate(x=3.0) == pytest.approx(-1.0)
        assert (Fraction(1, 2) + x).evaluate(x=3.0) == pytest.approx(3.5)

    def test_float_operand_is_rejected(self):
        with pytest.raises(TypeError, match="not an exact constant"):
            Expression.Variable("x") + 0.5

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            Expression.Variable("x") / 0  # type: ignore

    def test_modular_pow_is_rejected(self):
        with pytest.raises(ValueError, match="modular"):
            pow(Expression.Variable("x"), 2, 3)


class TestEvaluation:
    def test_evaluate(self):
        x = Expression.Variable("x")
        node = Expression.Sin(x) + Expression.Pi()
        assert node.evaluate(x=math.pi / 2) == pytest.approx(1 + math.pi)

    def test_evaluate_reports_missing_variables(self):
        with pytest.raises(ValueError, match="no value given for variable 'x'"):
            Expression.Variable("x").evaluate()

    def test_to_float_requires_a_closed_expression(self):
        assert Expression.Pi().to_float() == pytest.approx(math.pi)
        with pytest.raises(ValueError, match="parameterized"):
            Expression.Variable("x").to_float()

    def test_variables_and_membership(self):
        node = Expression.Variable("a") * Expression.Variable("b")
        assert node.variables() == ["a", "b"]
        assert node.contains_variable("a")
        assert not node.contains_variable("c")
        assert node.is_parameterized()


class TestRewriting:
    def test_differentiate(self):
        x = Expression.Variable("x")
        assert (x * x).differentiate("x").evaluate(x=4.0) == pytest.approx(8.0)

    def test_substitute(self):
        x = Expression.Variable("x")
        assert x.substitute(x, Expression.Constant(3)) == Expression.Constant(3)

    def test_rename_variable(self):
        x = Expression.Variable("x")
        assert (x + x).rename_variable("x", "y").variables() == ["y"]

    def test_simplify_preserves_value(self):
        x = Expression.Variable("x")
        node = (x + Expression.Constant(0)) * Expression.Constant(1)
        assert node.simplify().evaluate(x=2.0) == pytest.approx(2.0)


class TestComplexExpression:
    """`ComplexExpression`s are read off the body of a top-level expression."""

    @staticmethod
    def body(source):
        return UnitaryExpression(source).elements()

    @pytest.fixture
    def rz(self):
        # diag(cos(t) - i*sin(t), cos(t) + i*sin(t))
        return self.body("RZish<2>(t){[[e^(~i*t),0,],[0,e^(i*t),],]}")

    def test_parts_are_expressions(self, rz):
        assert isinstance(rz[0].real, Expression)
        assert isinstance(rz[0].imag, Expression)

    def test_parts_are_cached(self, rz):
        assert rz[0].real is rz[0].real

    def test_classification(self, rz):
        assert rz[1].is_zero()
        assert rz[1].is_real()
        assert rz[0].is_cplx()
        assert not rz[0].is_real()

    def test_variables(self, rz):
        assert rz[0].variables() == ["t"]
        assert rz[0].is_parameterized()
        assert not rz[1].is_parameterized()

    def test_evaluate(self, rz):
        assert rz[0].evaluate(t=0.0) == pytest.approx(complex(1, 0))
        assert rz[3].evaluate(t=math.pi / 2) == pytest.approx(complex(0, 1))

    def test_evaluate_reports_missing_variables(self, rz):
        with pytest.raises(ValueError, match="no value given for variable 't'"):
            rz[0].evaluate()

    def test_conjugate(self, rz):
        assert rz[3].conjugate().evaluate(t=math.pi / 2) == pytest.approx(complex(0, -1))

    def test_differentiate(self, rz):
        derivative = rz[3].differentiate("t").evaluate(t=0.0)
        assert derivative == pytest.approx(complex(0, 1))

    def test_rename_variable(self, rz):
        assert rz[0].rename_variable("t", "u").variables() == ["u"]

    def test_simplify_preserves_value(self, rz):
        assert rz[0].simplify().evaluate(t=0.0) == pytest.approx(rz[0].evaluate(t=0.0))

    def test_equality_and_hashing(self, rz):
        again = self.body("RZish<2>(t){[[e^(~i*t),0,],[0,e^(i*t),],]}")
        assert rz[0] == again[0]
        assert hash(rz[0]) == hash(again[0])
        assert rz[0] != rz[3]

    def test_equality_against_other_types(self, rz):
        assert rz[0] != Expression.Constant(1)
