from openqudit.circuit import QuditCircuit
from openqudit.expressions import U3Gate
from openqudit.instantiation import DefaultInstantiater

U3 = U3Gate()
circuit = QuditCircuit(1)
circuit.append(U3, 0)
result = DefaultInstantiater().instantiate(circuit, U3(1.7, 1.7, 1.7))
assert result.fun < 1e-4
