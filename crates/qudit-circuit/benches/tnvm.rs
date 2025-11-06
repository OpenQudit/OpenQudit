use common::build_qsearch_thick_step_circuit;
use common::build_qsearch_thin_step_circuit;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;

mod common;
use common::FlamegraphProfiler;
use common::build_qft_circuit;
use qudit_core::BitWidthConvertible;
use qudit_core::ComplexScalar;
use qudit_core::c32;
use qudit_core::c64;
use qudit_expr::{FUNCTION, GRADIENT, HESSIAN};
use qudit_tensor::TNVM;
use qudit_tensor::compile_network;

// // TODO: qft compilation time

// pub fn qft_unitary_calculation<C: ComplexScalar>(c: &mut Criterion) {
//     let data_type = core::any::type_name::<C>();
//     for num_qudits in [4, 5, 6, 7, 8, 9, 10].iter() {
//         let circ = build_qft_circuit(*num_qudits);
//         let params = vec![C::real(1.7); circ.get_num_params()];
//         let code = compile(&circ.expression_tree());
//         let mut qvm: QVM<C> = QVM::new(code, QVMType::Unitary);
//         c.bench_function(
//             &format!("qvm-{data_type}-qft-utry-{num_qudits}"),
//             |b| {
//                 b.iter(|| {
//                     let _ = qvm.get_unitary(&params);
//                 })
//             },
//         );
//     }
// }

// pub fn qft_gradient_calculation<C: ComplexScalar>(c: &mut Criterion) {
//     let data_type = core::any::type_name::<C>();
//     for num_qudits in [4, 5, 6, 7, 8, 9].iter() {
//         let circ = build_qft_circuit(*num_qudits);
//         let params = vec![C::real(1.7); circ.get_num_params()];
//         let code = compile(&circ.expression_tree());
//         let mut qvm: QVM<C> = QVM::new(code, QVMType::UnitaryAndGradient);
//         c.bench_function(
//             &format!("qvm-{data_type}-qft-grad-{num_qudits}"),
//             |b| {
//                 b.iter(|| {
//                     let _ = qvm.get_unitary_and_gradient(&params);
//                 })
//             },
//         );
//     }
// }

// pub fn qsearch_thin_unitary_calculation<C: ComplexScalar>(c: &mut Criterion) {
//     let data_type = core::any::type_name::<C>();
//     for num_qudits in [2, 3, 4, 5].iter() {
//         let circ = build_qsearch_thin_step_circuit(*num_qudits);
//         let params = vec![C::real(1.7); circ.get_num_params()];
//         let code = compile(&circ.expression_tree());
//         let mut qvm: QVM<C> = QVM::new(code, QVMType::Unitary);
//         c.bench_function(
//             &format!("qvm-{data_type}-qsearch-thin-utry-{num_qudits}"),
//             |b| {
//                 b.iter(|| {
//                     let _ = qvm.get_unitary(&params);
//                 })
//             },
//         );
//     }
// }

pub fn qsearch_thin_gradient_calculation<C: ComplexScalar>(c: &mut Criterion) {
    let data_type = core::any::type_name::<C>();
    for num_qudits in [2, 3, 4, 5].iter() {
        let circ = build_qsearch_thin_step_circuit(*num_qudits);
        let params = vec![C::R::from64(1.7); circ.num_params()];
        let code = compile_network(circ.to_tensor_network());
        let mut tnvm = TNVM::<C, GRADIENT>::new(&code);
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

// pub fn qsearch_thick_unitary_calculation<C: ComplexScalar>(c: &mut Criterion) {
//     let data_type = core::any::type_name::<C>();
//     for num_qudits in [2, 3, 4, 5].iter() {
//         let circ = build_qsearch_thick_step_circuit(*num_qudits);
//         let params = vec![C::real(1.7); circ.get_num_params()];
//         let code = compile(&circ.expression_tree());
//         let mut qvm: QVM<C> = QVM::new(code, QVMType::Unitary);
//         c.bench_function(
//             &format!("qvm-{data_type}-qsearch-thick-utry-{num_qudits}"),
//             |b| {
//                 b.iter(|| {
//                     let _ = qvm.get_unitary(&params);
//                 })
//             },
//         );
//     }
// }

// pub fn qsearch_thick_gradient_calculation<C: ComplexScalar>(c: &mut Criterion) {
//     let data_type = core::any::type_name::<C>();
//     for num_qudits in [2, 3, 4, 5].iter() {
//         let circ = build_qsearch_thick_step_circuit(*num_qudits);
//         let params = vec![C::real(1.7); circ.get_num_params()];
//         let code = compile(&circ.expression_tree());
//         let mut qvm: QVM<C> = QVM::new(code, QVMType::UnitaryAndGradient);
//         c.bench_function(
//             &format!("qvm-{data_type}-qsearch-thick-grad-{num_qudits}"),
//             |b| {
//                 b.iter(|| {
//                     let _ = qvm.get_unitary_and_gradient(&params);
//                 })
//             },
//         );
//     }
// }

// criterion_group! {
//     name = qvm_qft;
//     config = Criterion::default().with_profiler(FlamegraphProfiler::new());
//     targets =
//         qft_unitary_calculation::<c32>,
//         qft_unitary_calculation::<c64>,
//         // qft_gradient_calculation::<c32>,
//         // qft_gradient_calculation::<c64>,
// }
criterion_group! {
    name = tnvm_qsearch_thin;
    config = Criterion::default().with_profiler(FlamegraphProfiler::new(100));
    targets =
        // qsearch_thin_unitary_calculation::<c32>,
        // qsearch_thin_unitary_calculation::<c64>,
        qsearch_thin_gradient_calculation::<c32>,
        qsearch_thin_gradient_calculation::<c64>,
}
// criterion_group! {
//     name = qvm_qsearch_thick;
//     config = Criterion::default().with_profiler(FlamegraphProfiler::new());
//     targets =
//         qsearch_thick_unitary_calculation::<c32>,
//         qsearch_thick_unitary_calculation::<c64>,
//         qsearch_thick_gradient_calculation::<c32>,
//         qsearch_thick_gradient_calculation::<c64>
// }
// criterion_main!(qvm_qft, qvm_qsearch_thin, qvm_qsearch_thick);
criterion_main!(tnvm_qsearch_thin);
