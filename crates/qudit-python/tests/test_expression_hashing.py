import pytest

from openqudit.expressions import (
    BraExpression,
    BraSystemExpression,
    KetExpression,
    KetSystemExpression,
    KrausOperatorsExpression,
    UnitaryExpression,
    UnitarySystemExpression,
)

# (class, source, a source with a different body, the same body under another name)
CASES = [
    (
        BraExpression,
        "B<2>(){[1,0,]}",
        "B<2>(){[0,1,]}",
        "Renamed<2>(){[1,0,]}",
    ),
    (
        BraSystemExpression,
        "BS<2>(){[[[1,0,]],[[0,1,]],]}",
        "BS<2>(){[[[0,1,]],[[1,0,]],]}",
        "Renamed<2>(){[[[1,0,]],[[0,1,]],]}",
    ),
    (
        KetExpression,
        "K<2>(){[[1,],[0,],]}",
        "K<2>(){[[0,],[1,],]}",
        "Renamed<2>(){[[1,],[0,],]}",
    ),
    (
        KetSystemExpression,
        "KS<2>(){[[[1,],[0,],],[[0,],[1,],],]}",
        "KS<2>(){[[[0,],[1,],],[[1,],[0,],],]}",
        "Renamed<2>(){[[[1,],[0,],],[[0,],[1,],],]}",
    ),
    (
        KrausOperatorsExpression,
        "KR<2>(){[[[1,0,],[0,0,],],[[0,0,],[0,1,],],]}",
        "KR<2>(){[[[0,0,],[0,1,],],[[1,0,],[0,0,],],]}",
        "Renamed<2>(){[[[1,0,],[0,0,],],[[0,0,],[0,1,],],]}",
    ),
    (
        UnitaryExpression,
        "U<2>(){[[1,0,],[0,1,],]}",
        "U<2>(){[[0,1,],[1,0,],]}",
        "Renamed<2>(){[[1,0,],[0,1,],]}",
    ),
    (
        UnitarySystemExpression,
        "US<2>(){[[[1,0,],[0,1,],],[[0,1,],[1,0,],],]}",
        "US<2>(){[[[0,1,],[1,0,],],[[1,0,],[0,1,],],]}",
        "Renamed<2>(){[[[1,0,],[0,1,],],[[0,1,],[1,0,],],]}",
    ),
]

IDS = [case[0].__name__ for case in CASES]


@pytest.mark.parametrize("cls, source, other, renamed", CASES, ids=IDS)
class TestHash:
    def test_is_hashable(self, cls, source, other, renamed):
        assert isinstance(hash(cls(source)), int)

    def test_is_stable(self, cls, source, other, renamed):
        expr = cls(source)
        assert hash(expr) == hash(expr)

    def test_equal_bodies_hash_alike(self, cls, source, other, renamed):
        assert hash(cls(source)) == hash(cls(source))

    def test_different_bodies_hash_apart(self, cls, source, other, renamed):
        assert hash(cls(source)) != hash(cls(other))

    def test_name_is_not_hashed(self, cls, source, other, renamed):
        """The hash mirrors Rust equality, which ignores the name."""
        assert cls(source).name() != cls(renamed).name()
        assert hash(cls(source)) == hash(cls(renamed))

    def test_usable_in_sets_and_dicts(self, cls, source, other, renamed):
        expr = cls(source)
        assert {expr: "value"}[expr] == "value"
        assert len({expr, expr}) == 1


def test_parameterized_expressions_hash_by_body():
    rz = "RZish<2>(t){[[e^(~i*t),0,],[0,e^(i*t),],]}"
    assert hash(UnitaryExpression(rz)) == hash(UnitaryExpression(rz))
    assert hash(UnitaryExpression(rz)) != hash(
        UnitaryExpression("RZish<2>(t){[[e^(i*t),0,],[0,e^(~i*t),],]}")
    )


def test_hash_distinguishes_radices():
    two = UnitaryExpression("I<2>(){[[1,0,],[0,1,],]}")
    three = UnitaryExpression("I<3>(){[[1,0,0,],[0,1,0,],[0,0,1,],]}")
    assert hash(two) != hash(three)
