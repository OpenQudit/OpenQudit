use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;

mod common;
use common::FlamegraphProfiler;

use qudit_core::accel::fused_reshape_permute_reshape_into_impl;
use qudit_core::accel::tensor_fused_reshape_permute_reshape_into_prepare;
use qudit_core::c64;
use qudit_core::memory::{alloc_zeroed_memory, calc_col_stride};

pub fn frpr_benchmarks(c: &mut Criterion) {
    let test_cases_4_elements = vec![
        // (in_nrows, in_ncols, shape, perm, out_nrows, out_ncols)
        (4, 4, [2, 2, 2, 2], [0, 2, 1, 3], 4, 4),
        (4, 4, [2, 2, 2, 2], [0, 1, 2, 3], 8, 2),
        (4, 4, [2, 2, 2, 2], [0, 1, 2, 3], 2, 8),
        (9, 9, [3, 3, 3, 3], [0, 3, 1, 2], 9, 9),
    ];

    let mut group = c.benchmark_group("fused_reshape_permute_reshape_4_elements");

    for (in_nrows, in_ncols, shape, perm, out_nrows, out_ncols) in test_cases_4_elements {
        let col_stride_in = calc_col_stride::<c64>(in_nrows, in_ncols);
        let memory_in = alloc_zeroed_memory::<c64>(in_ncols * col_stride_in);

        let col_stride_out = calc_col_stride::<c64>(out_nrows, out_ncols);
        let mut memory_out = alloc_zeroed_memory::<c64>(out_ncols * col_stride_out);

        let (is, os, dims) = tensor_fused_reshape_permute_reshape_into_prepare(
            &[in_nrows, in_ncols],
            &[1, col_stride_in as isize],
            &[out_nrows, out_ncols],
            &[1, col_stride_out as isize],
            &shape,
            &perm,
        );
        println!("{:?}, {:?}, {:?}", is, os, dims);

        group.bench_function(
            BenchmarkId::new(
                "frpr_impl",
                format!(
                    "{}x{} as {:?} -> {:?} -> {}x{}",
                    in_nrows, in_ncols, shape, perm, out_nrows, out_ncols
                ),
            ),
            |b| {
                b.iter(|| unsafe {
                    fused_reshape_permute_reshape_into_impl(
                        memory_in.as_ptr(),
                        memory_out.as_mut_ptr(),
                        &is,
                        &os,
                        &dims,
                    );
                });
            },
        );
    }

    let test_cases_6_elements = vec![
        (8, 8, [2, 2, 2, 2, 2, 2], [0, 3, 1, 4, 2, 5], 16, 4),
        (16, 16, [4, 2, 2, 2, 2, 4], [0, 3, 1, 4, 2, 5], 16, 16),
        (16, 16, [4, 2, 2, 2, 2, 4], [0, 3, 1, 4, 2, 5], 32, 8),
        (16, 16, [4, 2, 2, 2, 2, 4], [0, 3, 1, 4, 2, 5], 64, 4),
    ];

    for (in_nrows, in_ncols, shape, perm, out_nrows, out_ncols) in test_cases_6_elements {
        let col_stride_in = calc_col_stride::<c64>(in_nrows, in_ncols);
        let memory_in = alloc_zeroed_memory::<c64>(in_ncols * col_stride_in);

        let col_stride_out = calc_col_stride::<c64>(out_nrows, out_ncols);
        let mut memory_out = alloc_zeroed_memory::<c64>(out_ncols * col_stride_out);

        let (is, os, dims) = tensor_fused_reshape_permute_reshape_into_prepare(
            &[in_nrows, in_ncols],
            &[1, col_stride_in as isize],
            &[out_nrows, out_ncols],
            &[1, col_stride_out as isize],
            &shape,
            &perm,
        );
        println!("{:?}, {:?}, {:?}", is, os, dims);

        group.bench_function(
            BenchmarkId::new(
                "frpr_impl",
                format!(
                    "{}x{} as {:?} -> {:?} -> {}x{}",
                    in_nrows, in_ncols, shape, perm, out_nrows, out_ncols
                ),
            ),
            |b| {
                b.iter(|| unsafe {
                    fused_reshape_permute_reshape_into_impl(
                        memory_in.as_ptr(),
                        memory_out.as_mut_ptr(),
                        &is,
                        &os,
                        &dims,
                    );
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
