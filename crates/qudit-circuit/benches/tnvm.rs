use common::build_qsearch_thin_step_circuit;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;

mod common;
use common::FlamegraphProfiler;
use qudit_core::BitWidthConvertible;
use qudit_core::ComplexScalar;
use qudit_core::c32;
use qudit_core::c64;
use qudit_expr::GRADIENT;
use qudit_tensor::TNVM;
use qudit_tensor::compile_network;

pub fn qsearch_thin_gradient_calculation<C: ComplexScalar>(c: &mut Criterion) {
    let data_type = core::any::type_name::<C>();
    for num_qudits in [2, 3, 4, 5].iter() {
        let circ = build_qsearch_thin_step_circuit(*num_qudits);
        let params = vec![C::R::from64(1.7); circ.num_params()];
        let code = compile_network(circ.to_tensor_network());
        let mut tnvm = TNVM::<C, GRADIENT>::new(&code, None);
        c.bench_function(
            &format!("qvm-{data_type}-qsearch-thin-grad-{num_qudits}"),
            |b| {
                b.iter(|| {
                    let _ = tnvm.evaluate::<GRADIENT>(&params);
                })
            },
        );
    }
}

criterion_group! {
    name = tnvm_qsearch_thin;
    config = Criterion::default().with_profiler(FlamegraphProfiler::new(100));
    targets =
        qsearch_thin_gradient_calculation::<c32>,
        qsearch_thin_gradient_calculation::<c64>,
}

criterion_main!(tnvm_qsearch_thin);
