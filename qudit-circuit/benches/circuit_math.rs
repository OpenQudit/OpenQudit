// use criterion::criterion_group;
// use criterion::criterion_main;
// use criterion::Criterion;

// mod common;
// use common::build_qsearch_thin_step_circuit;
// use common::FlamegraphProfiler;

// use qudit_core::c64;
// use qudit_expr::DifferentiationLevel;
// use qudit_tree::compile;
// use qudit_tree::QVM;

// pub fn circuit_thin_qsearch_bench(c: &mut Criterion) {
//     let circ = build_qsearch_thin_step_circuit(3);
//     let tree = circ.to_tree();
//     let code = compile(&tree);

//     let mut qvm: QVM<c64> = QVM::new(code, DifferentiationLevel::None);
//     let params = vec![1.7; 3 * 2 * 2 * 3 + 9];
//     let mut utry = qudit_core::matrix::Mat::zeros(8, 8);

//     c.bench_function("qsearch-thin-3", |b| {
//         b.iter(|| {
//             qvm.write_unitary(&params, utry.as_mut());
//         })
//     });
// }

// criterion_group! {
//     name = circuit_math;
//     config = Criterion::default().with_profiler(FlamegraphProfiler::new(100));
//     targets = circuit_thin_qsearch_bench
// }
// criterion_main!(circuit_math);

// pub fn circuit_math_benchmarks(_c: &mut Criterion) {
//     // {
//     //     let mut group = c.benchmark_group("qft_utry_calc");
//     //     for num_qudits in [4, 5, 6, 7, 8, 9, 10].iter()
//     //     {
//     //         let circ = build_qft_circuit(*num_qudits);
//     //         group.bench_with_input(BenchmarkId::from_parameter(num_qudits),
//     // &circ, |b, circ| {             b.iter(|| {
//     //                 let _: UnitaryMatrix<f64> = circ.get_unitary(&vec![]);
//     //             })
//     //         });
//     //     }
//     // }

//     // {
//     //     let mut group = c.benchmark_group("qft_grad_calc");
//     //     for num_qudits in [4, 5, 6, 7, 8, 9].iter()
//     //     {
//     //         let circ = build_qft_circuit(*num_qudits);
//     //         group.bench_with_input(BenchmarkId::from_parameter(num_qudits),
//     // &circ, |b, circ| {             b.iter(|| {
//     //                 let _: (UnitaryMatrix<f64>, Array3<c64>) =
//     //                     circ.get_unitary_and_gradient(&vec![]);
//     //             })
//     //         });
//     //     }
//     // }

//     // {
//     //     let mut group = c.benchmark_group("qft_hess_calc");
//     //     for num_qudits in [4, 5, 6, 7, 8].iter()
//     //     {
//     //         let circ = build_qft_circuit(*num_qudits);
//     //         group.bench_with_input(BenchmarkId::from_parameter(num_qudits),
//     // &circ, |b, circ| {             b.iter(|| {
//     //                 let _: (UnitaryMatrix<f64>, Array3<c64>, Array4<c64>) =
//     //                     circ.get_unitary_gradient_and_hessian(&vec![]);
//     //             })
//     //         });
//     //     }
//     // }

//     // {
//     //     let mut group = c.benchmark_group("qft_hess_calc_op_tree");
//     //     for num_qudits in [4, 5, 6].iter() {
//     //         let circ = build_qft_circuit(*num_qudits).expression_tree();
//     //         group.bench_with_input(
//     //             BenchmarkId::from_parameter(num_qudits),
//     //             &circ,
//     //             |b, circ| {
//     //                 b.iter(|| {
//     //                     let _ =
//     //                         circ.get_unitary_gradient_and_hessian(
//     //                             &vec![0.0; circ.get_num_params()],
//     //                         );
//     //                 })
//     //             },
//     //         );
//     //     }
//     // }
// }

// criterion_group! {
//     name = circuit_math;
//     config = Criterion::default().with_profiler(FlamegraphProfiler::new());
//     targets = circuit_math_benchmarks
// }
// criterion_main!(circuit_math);
