use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;

mod common;
use common::FlamegraphProfiler;
use common::build_qft_circuit;

use crate::common::build_dtc_circuit;

pub fn circuit_struct_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_qft");

    for num_qudits in [4, 8, 16, 32, 64, 128, 256, 512, 1023].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_qudits),
            num_qudits,
            |b, &num_qudits| b.iter_with_large_drop(|| build_qft_circuit(num_qudits)),
        );
    }
    group.finish();

    let mut group = c.benchmark_group("build_dtc");

    for num_qudits in [4, 8, 16, 32, 64, 128, 256, 512].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_qudits),
            num_qudits,
            |b, &num_qudits| b.iter_with_large_drop(|| build_dtc_circuit(num_qudits)),
        );
    }
    group.finish();
}

criterion_group! {
    name = circuit_struct;
    config = Criterion::default().with_profiler(FlamegraphProfiler::new(100));
    targets = circuit_struct_benchmarks
}
criterion_main!(circuit_struct);

