import abc
from typing import Self

from openqudit import _openqudit
from openqudit.circuit import QuditCircuit  # TODO: Barrier
from openqudit.expressions import (
    BraSystemExpression,
    KetExpression,
    KrausOperatorsExpression,
    UnitaryExpression,
    UnitarySystemExpression,
)

DirectiveOperation = _openqudit.circuit.DirectiveOperation


class Operation(abc.ABC): ...


class ExpressionOperation(Operation):
    def num_params(self) -> int: ...
    def name(self) -> str: ...
    def num_qudits(self) -> int: ...
    def dimension(self) -> int: ...
    def __eq__(self, other: Self) -> bool: ...
    def __hash__(self) -> int: ...


class CircuitOperation(Operation): ...


for cls in (
    UnitaryExpression,
    KrausOperatorsExpression,
    BraSystemExpression,
    UnitarySystemExpression,
    KetExpression,
):
    ExpressionOperation.register(cls)

Operation.register(_openqudit.circuit.DirectiveOperation)

CircuitOperation.register(QuditCircuit)
