use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;

use qudit_expr::CompilableUnit;
use qudit_expr::Module;
use qudit_expr::ModuleBuilder;
use qudit_expr::UnitaryExpression;
use qudit_expr::library::{U3Gate, XGate, Controlled};
use qudit_expr::simplify_expressions_iter;

mod common;
use common::FlamegraphProfiler;

fn build_module(expr: UnitaryExpression) -> Module<f64> {
    let func = simplify_expressions_iter(expr.iter().flat_map(|c| [&c.real, &c.imag]));

    let mut grad_exprs = vec![];
    for variable in expr.variables() {
        for expr_elem in expr.elements() {
            let grad_expr = expr_elem.differentiate(variable);
            grad_exprs.push(grad_expr);
        }
    }

    let grad = simplify_expressions_iter(expr.elements()
        .iter()
        .chain(grad_exprs.iter())
        .flat_map(|c| [&c.real, &c.imag]));

    let unit = CompilableUnit::new(
        "func",
        &func,
        expr.variables().to_vec(),
        func.len() * 2,
    );

    let grad_unit = CompilableUnit::new(
        "grad",
        &grad,
        expr.variables().to_vec(),
        func.len() * 2,
    );

    ModuleBuilder::new("test").add_unit(unit).add_unit(grad_unit).build()
}

pub fn cnotu3u3_benchmarks(c: &mut Criterion) {
    c.bench_function("qsearch-step-module-gen", |b| {
        b.iter(|| {
            let u3 = U3Gate();
            let cx = Controlled(XGate(2), [2].into(), None);
            let step = u3.otimes(&u3).dot(cx);
            let _ = build_module(step);
        })
    });

    c.bench_function("cnot-module-gen", |b| {
        b.iter(|| {
            let cx = Controlled(XGate(2), [2].into(), None);
            let _ = build_module(cx);
        })
    });

    c.bench_function("u3-module-gen", |b| {
        b.iter(|| {
            let u3 = U3Gate();
            let _ = build_module(u3);
        })
    });
}

criterion_group! {
    name = cnotu3u3_module;
    config = Criterion::default().with_profiler(FlamegraphProfiler::new(100));
    targets = cnotu3u3_benchmarks
}

criterion_main!(cnotu3u3_module);
