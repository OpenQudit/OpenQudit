use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

mod common;
use common::FlamegraphProfiler;
use pprof::criterion::{PProfProfiler, Output};
use pprof::flamegraph::Options;

use qudit_inst::*;
use qudit_core::c32;
use qudit_core::c64;
use qudit_core::unitary::UnitaryMatrix;
use qudit_core::QuditRadices;
use qudit_circuit::QuditCircuit;
use qudit_circuit::CircuitLocation;
use qudit_expr::UnitaryExpression;
use qudit_gates::Gate;
use qudit_tensor::TNVM;
use qudit_expr::FUNCTION;
use qudit_expr::GRADIENT;
use qudit_inst::numerical::functions::HSProblem;
use qudit_inst::numerical::initializers::GreedyFurthestPoint;
use qudit_inst::numerical::runners::MultiStartRunner;
use qudit_inst::numerical::minimizers::LM;
use qudit_inst::numerical::initializers::Zeros;
use qudit_inst::numerical::initializers::Uniform;
use qudit_inst::numerical::MinimizingInstantiater;
use qudit_core::QuditSystem;
use qudit_expr::ExpressionGenerator;
use qudit_circuit::Operation;
use qudit_circuit::ControlState;

pub fn build_qsearch_thin_step_circuit(n: usize) -> QuditCircuit {
    let block_expr = Gate::U3().generate_expression().otimes(&Gate::U3().generate_expression()).dot(&Gate::CX().generate_expression());
    let mut circ = QuditCircuit::pure(vec![2; n]);
    for i in 0..n {
        circ.append_parameterized(Gate::U3(), [i]);
    }
    for _ in 0..2 {
        for i in 0..(n - 1) {
            circ.append_parameterized(Gate::Expression(block_expr.clone()), [i, i+1]);
        }
    }
    circ
} 


pub fn unitary_inst_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("QSearch-Thin-Instantiation");

    for num_qudits in [2, 3, 4, 5].iter() {

        let circ = build_qsearch_thin_step_circuit(*num_qudits);

        // sample target
        let network = circ.to_tensor_network();
        let code = qudit_tensor::compile_network(network);
        let mut tnvm = qudit_tensor::TNVM::<c64, GRADIENT>::new(&code);
        let result = tnvm.evaluate::<FUNCTION>(&vec![1.7; circ.num_params()]).get_fn_result().unpack_matrix();
        let target_utry = UnitaryMatrix::new(circ.radices(), result.to_owned());
        let target = InstantiationTarget::UnitaryMatrix(target_utry);

        // build instantiater
        let minimizer = LM::default();
        let initializer = Uniform::default();
        let runner = MultiStartRunner { minimizer, guess_generator: initializer, num_starts: 1 };
        let instantiater = MinimizingInstantiater::<_, HSProblem<f64>>::new(runner);
        let data = std::collections::HashMap::new();
 
        group.bench_function(
            BenchmarkId::from_parameter(num_qudits),
            |b| {
                b.iter(|| instantiater.instantiate(&circ, &target, &data)) 
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = unitary;
    config = Criterion::default().with_profiler(FlamegraphProfiler::new(100));
    targets = unitary_inst_benchmarks 
}
criterion_main!(unitary);
