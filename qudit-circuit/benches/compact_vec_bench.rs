use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::{hint::black_box as hint_black_box, time::Duration};

// Assuming the module structure based on the file path
use qudit_circuit::utils::CompactVec;

fn bench_core_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_operations");
    
    // Fast benchmark settings
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(500));
    group.sample_size(50);
    
    // Creation benchmarks - just the key types
    group.bench_function("new_u8_infallible", |b| {
        b.iter(|| black_box(CompactVec::<u8>::new()))
    });
    
    group.bench_function("new_u32_fallible", |b| {
        b.iter(|| black_box(CompactVec::<u32>::new()))
    });
    
    // Push benchmarks - key scenarios only
    group.bench_function("push_u8_inline", |b| {
        b.iter(|| {
            let mut vec = CompactVec::<u8>::new();
            for i in 0u32..7 {
                vec.push(black_box(i as u8));
            }
            black_box(vec)
        })
    });
    
    group.bench_function("push_u8_to_heap", |b| {
        b.iter(|| {
            let mut vec = CompactVec::<u8>::new();
            for i in 0u32..10 {
                vec.push(black_box(i as u8));
            }
            black_box(vec)
        })
    });
    
    group.bench_function("push_u32_small", |b| {
        b.iter(|| {
            let mut vec = CompactVec::<u32>::new();
            for i in 0u32..7 {
                vec.push(black_box(i));
            }
            black_box(vec)
        })
    });
    
    group.bench_function("push_u32_large_values", |b| {
        b.iter(|| {
            let mut vec = CompactVec::<u32>::new();
            for i in 300u32..307 {
                vec.push(black_box(i)); // Forces heap immediately
            }
            black_box(vec)
        })
    });
    
    // Vec comparison
    group.bench_function("vec_push_u8", |b| {
        b.iter(|| {
            let mut vec = Vec::<u8>::new();
            for i in 0u32..7 {
                vec.push(black_box(i as u8));
            }
            black_box(vec)
        })
    });
    
    group.finish();
}

fn bench_access_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("access_operations");
    
    // Fast benchmark settings
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(500));
    group.sample_size(50);
    
    // Setup test data - simplified
    let mut small_u32 = CompactVec::<u32>::new();
    let mut large_u32 = CompactVec::<u32>::new();
    
    for i in 0u32..7 { small_u32.push(i); }
    for i in 0u32..20 { large_u32.push(i); }
    
    let large_vec_u32: Vec<u32> = (0u32..20).collect();
    
    group.bench_function("get_u32_inline", |b| {
        b.iter(|| {
            for i in [0, 3, 6, 2, 5, 1, 4] {
                hint_black_box(small_u32.get(black_box(i)));
            }
        })
    });
    
    group.bench_function("get_u32_heap", |b| {
        b.iter(|| {
            for i in [0, 10, 15, 5, 18, 2, 12] {
                hint_black_box(large_u32.get(black_box(i)));
            }
        })
    });
    
    group.bench_function("vec_get_u32", |b| {
        b.iter(|| {
            for i in [0, 10, 15, 5, 18, 2, 12] {
                hint_black_box(large_vec_u32.get(black_box(i)));
            }
        })
    });
    
    group.bench_function("iter_u32_inline", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for value in small_u32.iter() {
                sum += black_box(value as u64);
            }
            black_box(sum)
        })
    });
    
    group.bench_function("iter_u32_heap", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for value in large_u32.iter() {
                sum += black_box(value as u64);
            }
            black_box(sum)
        })
    });
    
    group.finish();
}


criterion_group!(
    benches,
    bench_core_operations,
    bench_access_operations
);
criterion_main!(benches);
