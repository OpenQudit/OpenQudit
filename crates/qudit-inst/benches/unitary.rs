use std::sync::Arc;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;

mod common;
use common::FlamegraphProfiler;

use qudit_circuit::QuditCircuit;
use qudit_core::QuditSystem;
use qudit_core::UnitaryMatrix;
use qudit_core::c64;
use qudit_expr::FUNCTION;
use qudit_expr::GRADIENT;
use qudit_expr::library::Controlled;
use qudit_expr::library::U3Gate;
use qudit_expr::library::XGate;
use qudit_inst::numerical::MinimizingInstantiater;
use qudit_inst::numerical::functions::HSProblem;
use qudit_inst::numerical::initializers::Uniform;
use qudit_inst::numerical::minimizers::LM;
use qudit_inst::numerical::runners::MultiStartRunner;
use qudit_inst::*;

pub fn build_qsearch_thin_step_circuit(n: usize) -> QuditCircuit {
    let block_expr = U3Gate()
        .otimes(U3Gate())
        .dot(Controlled(XGate(2), [2].into(), None));
    let mut circ = QuditCircuit::pure(vec![2; n]);
    for i in 0..n {
        circ.append(U3Gate(), [i], None);
    }
    for _ in 0..2 {
        for i in 0..(n - 1) {
            circ.append(block_expr.clone(), [i, i + 1], None);
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
        let mut tnvm = qudit_tensor::TNVM::<c64, GRADIENT>::new(&code, None);
        let result = tnvm
            .evaluate::<FUNCTION>(&vec![1.7; circ.num_params()])
            .get_fn_result()
            .unpack_matrix();
        let target_utry = UnitaryMatrix::new(circ.radices(), result.to_owned());
        let target = InstantiationTarget::UnitaryMatrix(target_utry);

        // build instantiater
        let minimizer = LM::default();
        let initializer = Uniform::default();
        let runner = MultiStartRunner {
            minimizer,
            guess_generator: initializer,
            num_starts: 1,
        };
        let instantiater = MinimizingInstantiater::<_, HSProblem<f64>>::new(runner);
        let data = std::collections::HashMap::new();

        let circ = Arc::new(circ);
        let target = Arc::new(target);
        let data = Arc::new(data);

        group.bench_function(BenchmarkId::from_parameter(num_qudits), |b| {
            b.iter(|| instantiater.instantiate(circ.clone(), target.clone(), data.clone()))
        });
    }
    group.finish();
}

criterion_group! {
    name = unitary;
    config = Criterion::default().with_profiler(FlamegraphProfiler::new(100));
    targets = unitary_inst_benchmarks
}
criterion_main!(unitary);
