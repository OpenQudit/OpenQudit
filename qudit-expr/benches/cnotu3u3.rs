use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

use qudit_core::c64;
use qudit_core::matrix::Mat;
use qudit_core::matrix::MatVec;
use qudit_expr::DifferentiationLevel;
use qudit_expr::Module;
use qudit_expr::ModuleBuilder;
use qudit_expr::UnitaryExpression;
use qudit_expr::UnitaryExpressionGenerator;

mod common;
use common::FlamegraphProfiler;

struct U3Gate;

impl UnitaryExpressionGenerator for U3Gate {
    fn gen_expr(&self) -> UnitaryExpression {
        UnitaryExpression::new(
            String::from("
                utry U3(f1, f2, f3) {
                    [
                        [ cos(f1/2), ~e^(i*f3)*sin(f1/2) ],
                        [ e^(i*f2)*sin(f1/2), e^(i*(f2+f3))*cos(f1/2) ]
                    ]
                }
            "),
        )
    }
}

struct CNOTGate;

impl UnitaryExpressionGenerator for CNOTGate {
    fn gen_expr(&self) -> UnitaryExpression {
        UnitaryExpression::new(
            String::from("
                utry CNOT() {
                    [
                        [ 1, 0, 0, 0 ],
                        [ 0, 1, 0, 0 ],
                        [ 0, 0, 0, 1 ],
                        [ 0, 0, 1, 0 ]
                    ]
                }
            "),
        )
    }
}

pub fn cnotu3u3_benchmarks(c: &mut Criterion) {
    c.bench_function("u3-module-gen", |b| {
        b.iter(|| {
            let u3_gate = U3Gate;
            let expr = u3_gate.gen_expr();

            let _module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::Gradient)
                .add_expression_with_stride(expr, 2)
                .build();
        })
    });

    c.bench_function("cnot-module-gen", |b| {
        b.iter(|| {
            let cnot_gate = CNOTGate;
            let expr = cnot_gate.gen_expr();

            let _module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::Gradient)
                .add_expression_with_stride(expr, 4)
                .build();
        })
    });

    c.bench_function("cnotu3u3-module-gen", |b| {
        b.iter(|| {
            let u3_gate = U3Gate;
            let u3_expr = u3_gate.gen_expr();

            let cnot_gate = CNOTGate;
            let cnot_expr = cnot_gate.gen_expr();

            let cnotu3u3_expr = cnot_expr.dot(&u3_expr.otimes(&u3_expr));

            let _module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::Gradient)
                .add_expression_with_stride(cnotu3u3_expr, 4)
                .build();
        })
    });
}

pub fn cnotu3u3eval_benchmarks(c: &mut Criterion) {
    let u3_gate = U3Gate;
    let u3_expr = u3_gate.gen_expr();

    let cnot_gate = CNOTGate;
    let cnot_expr = cnot_gate.gen_expr();

    let cnotu3u3_expr = cnot_expr.dot(&u3_expr.otimes(&u3_expr));

    let u3_name = u3_expr.name();
    let cnot_name = cnot_expr.name();
    let cnotu3u3_name = cnotu3u3_expr.name();

    let mut u3_out_utry: Mat<c64> = Mat::zeros(2, 2);
    let mut cnot_out_utry: Mat<c64> = Mat::zeros(4, 4);
    let mut cnotu3u3_out_utry: Mat<c64> = Mat::zeros(4, 4);
    let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::None)
        .add_expression_with_stride(u3_expr, u3_out_utry.col_stride().try_into().unwrap())
        .add_expression_with_stride(cnot_expr, cnot_out_utry.col_stride().try_into().unwrap())
        .add_expression_with_stride(cnotu3u3_expr, cnotu3u3_out_utry.col_stride().try_into().unwrap())
        .build();

    let u3_func = module.get_function(&u3_name).unwrap();
    let cnot_func = module.get_function(&cnot_name).unwrap();
    let cnotu3u3_func = module.get_function(&cnotu3u3_name).unwrap();

    let u3_ptr = u3_out_utry.as_mut().as_ptr_mut() as *mut f64;
    let cnot_ptr = cnot_out_utry.as_mut().as_ptr_mut() as *mut f64;
    let cnotu3u3_ptr = cnotu3u3_out_utry.as_mut().as_ptr_mut() as *mut f64;

    let params = vec![1.7, 2.3, 3.1, 3.1, 2.3, 1.7f64];
    c.bench_function("u3-eval", |b| {
        b.iter(|| {
            unsafe {
                u3_func.call(params.as_ptr(), u3_ptr);
            }
        })
    });

    c.bench_function("cnot-eval", |b| {
        b.iter(|| {
            unsafe {
                cnot_func.call(params.as_ptr(), cnot_ptr);
            }
        })
    });

    c.bench_function("cnotu3u3-eval", |b| {
        b.iter(|| {
            unsafe {
                cnotu3u3_func.call(params.as_ptr(), cnotu3u3_ptr);
            }
        })
    });
}

pub fn cnotu3u3evalgrad_benchmarks(c: &mut Criterion) {
    let u3_gate = U3Gate;
    let u3_expr = u3_gate.gen_expr();

    let cnot_gate = CNOTGate;
    let cnot_expr = cnot_gate.gen_expr();

    let cnotu3u3_expr = cnot_expr.dot(&u3_expr.otimes(&u3_expr));

    let u3_name = u3_expr.name();
    let cnot_name = cnot_expr.name();
    let cnotu3u3_name = cnotu3u3_expr.name();

    let mut u3_out_utry: Mat<c64> = Mat::zeros(2, 2);
    let mut cnot_out_utry: Mat<c64> = Mat::zeros(4, 4);
    let mut cnotu3u3_out_utry: Mat<c64> = Mat::zeros(4, 4);
    let mut u3_out_grad: MatVec<c64> = MatVec::zeros(2, 2, 3);
    let mut cnot_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 0);
    let mut cnotu3u3_out_grad: MatVec<c64> = MatVec::zeros(4, 4, 6);
    let module: Module<c64> = ModuleBuilder::new("test", DifferentiationLevel::Gradient)
        .add_expression_with_stride(u3_expr, u3_out_utry.col_stride().try_into().unwrap())
        .add_expression_with_stride(cnot_expr, cnot_out_utry.col_stride().try_into().unwrap())
        .add_expression_with_stride(cnotu3u3_expr, cnotu3u3_out_utry.col_stride().try_into().unwrap())
        .build();

    let u3_func = module.get_function_and_gradient(&u3_name).unwrap();
    let cnot_func = module.get_function_and_gradient(&cnot_name).unwrap();
    let cnotu3u3_func = module.get_function_and_gradient(&cnotu3u3_name).unwrap();

    let u3_ptr = u3_out_utry.as_mut().as_ptr_mut() as *mut f64;
    let cnot_ptr = cnot_out_utry.as_mut().as_ptr_mut() as *mut f64;
    let cnotu3u3_ptr = cnotu3u3_out_utry.as_mut().as_ptr_mut() as *mut f64;
    let u3_grad_ptr = u3_out_grad.as_mut().as_mut_ptr().as_ptr() as *mut f64;
    let cnot_grad_ptr = cnot_out_grad.as_mut().as_mut_ptr().as_ptr() as *mut f64;
    let cnotu3u3_grad_ptr = cnotu3u3_out_grad.as_mut().as_mut_ptr().as_ptr() as *mut f64;

    let params = vec![1.7, 2.3, 3.1, 3.1, 2.3, 1.7f64];
    c.bench_function("u3-eval-grad", |b| {
        b.iter(|| {
            unsafe {
                u3_func.call(params.as_ptr(), u3_ptr, u3_grad_ptr);
            }
        })
    });

    c.bench_function("cnot-eval-grad", |b| {
        b.iter(|| {
            unsafe {
                cnot_func.call(params.as_ptr(), cnot_ptr, cnot_grad_ptr);
            }
        })
    });

    c.bench_function("cnotu3u3-eval-grad", |b| {
        b.iter(|| {
            unsafe {
                cnotu3u3_func.call(params.as_ptr(), cnotu3u3_ptr, cnotu3u3_grad_ptr);
            }
        })
    });
}

criterion_group! {
    name = cnotu3u3_module;
    config = Criterion::default().sample_size(10).with_profiler(FlamegraphProfiler::new(100));
    targets = cnotu3u3_benchmarks
}

criterion_group! {
    name = cnotu3u3_eval;
    config = Criterion::default().with_profiler(FlamegraphProfiler::new(100));
    targets = cnotu3u3eval_benchmarks, cnotu3u3evalgrad_benchmarks
}

criterion_main!(cnotu3u3_module, cnotu3u3_eval);
