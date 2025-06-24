use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

mod common;
use common::FlamegraphProfiler;
use pprof::criterion::{PProfProfiler, Output};
use pprof::flamegraph::Options;

use qudit_core::accel::fused_reshape_permute_reshape_into_impl;
use qudit_core::accel::fused_reshape_permute_reshape_into_prepare;
use qudit_core::c64;
use qudit_core::matrix::{Mat, MatMut, MatRef};
use qudit_core::memory::{alloc_zeroed_memory, calc_col_stride};

pub fn frpr_benchmarks(c: &mut Criterion) {
    let test_cases = vec![
        // (in_nrows, in_ncols, shape, perm, out_nrows, out_ncols)
        (4, 4, vec![2, 2, 2, 2], vec![0, 2, 1, 3], 4, 4),
        (4, 4, vec![2, 2, 2, 2], vec![0, 1, 2, 3], 8, 2),
        (4, 4, vec![2, 2, 2, 2], vec![0, 1, 2, 3], 2, 8),
        (8, 8, vec![2, 2, 2, 2, 2, 2], vec![0, 3, 1, 4, 2, 5], 16, 4),
        (9, 9, vec![3, 3, 3, 3], vec![0, 3, 1, 2], 9, 9),
        (16, 16, vec![4, 2, 2, 2, 2, 4], vec![0, 3, 1, 4, 2, 5], 16, 16),
        (16, 16, vec![4, 2, 2, 2, 2, 4], vec![0, 3, 1, 4, 2, 5], 32, 8),
        (16, 16, vec![4, 2, 2, 2, 2, 4], vec![0, 3, 1, 4, 2, 5], 64, 4),
    ];

    let mut group = c.benchmark_group("fused_reshape_permute_reshape");

    for (in_nrows, in_ncols, shape, perm, out_nrows, out_ncols) in test_cases {
        let col_stride_in = calc_col_stride::<c64>(in_nrows, in_ncols);
        let mut memory_in = alloc_zeroed_memory::<c64>(in_ncols * col_stride_in);
        let inp = unsafe {
            faer::mat::MatRef::from_raw_parts(
                memory_in.as_ptr(),
                in_nrows,
                in_ncols,
                1,
                col_stride_in as isize,
            )
        };

        let col_stride_out = calc_col_stride::<c64>(out_nrows, out_ncols);
        let mut memory_out = alloc_zeroed_memory::<c64>(out_ncols * col_stride_out);
        let mut out: MatMut<c64> = unsafe {
            faer::mat::MatMut::from_raw_parts_mut(
                memory_out.as_mut_ptr(),
                out_nrows,
                out_ncols,
                1,
                col_stride_out as isize,
            )
        };

        let (is, os, dims) = fused_reshape_permute_reshape_into_prepare(
            inp.nrows(),
            inp.ncols(),
            inp.col_stride(),
            out.nrows(),
            out.ncols(),
            out.col_stride(),
            &shape,
            &perm,
        );

        group.bench_function(
            BenchmarkId::new("frpr_impl", format!("{}x{} as {:?} -> {:?} -> {}x{}", in_nrows, in_ncols, shape, perm, out_nrows, out_ncols)),
            |b| {
                b.iter(|| unsafe {
                    fused_reshape_permute_reshape_into_impl(inp.as_ref(), out.as_mut(), &is, &os, &dims);
                });
            },
        );
    }
}

criterion_group! {
    name = frpr_bench;
    config = Criterion::default().with_profiler(FlamegraphProfiler::new(100));
    targets = frpr_benchmarks
}
criterion_main!(frpr_bench);
