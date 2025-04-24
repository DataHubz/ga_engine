// benches/bench.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::prelude::*;

const BATCH_SIZE: usize = 1_000;

/// Benchmark n×n matrix multiplication.
fn bench_matrix_mult(c: &mut Criterion) {
    let n = black_box(8);
    let a: Vec<f64> = (0..n * n).map(|i| (i % 10) as f64).collect();
    let b = a.clone();

    c.bench_function("matrix 8×8 × 1000 batch", |bencher| {
        bencher.iter(|| {
            let mut res = Vec::with_capacity(n * n);
            for _ in 0..BATCH_SIZE {
                res = multiply_matrices(black_box(&a), black_box(&b), n);
            }
            black_box(res)
        })
    });
}

/// Benchmark full 8-component multivector geometric product.
fn bench_geometric_product_full(c: &mut Criterion) {
    let a: [f64; 8] = black_box([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]);
    let b = a;

    c.bench_function("GA full product 8D × 1000 batch", |bencher| {
        bencher.iter(|| {
            let mut out = [0.0; 8];
            for _ in 0..BATCH_SIZE {
                geometric_product_full(black_box(&a), black_box(&b), &mut out);
            }
            black_box(out)
        })
    });
}

/// Benchmark rotating a 3D point about Z: classical vs GA (sandwich, fast, SIMD).
fn bench_rotate_point(c: &mut Criterion) {
    let v0 = Vec3::new(1.0, 0.0, 0.0);
    let m: [f64; 9] = [
        0.0, -1.0, 0.0,
        1.0,  0.0, 0.0,
        0.0,  0.0, 1.0,
    ];
    let rotor = Rotor3::from_axis_angle(
        Vec3::new(0.0, 0.0, 1.0),
        std::f64::consts::FRAC_PI_2,
    );

    c.bench_function("rotate 3D point classical", |bencher| {
        bencher.iter(|| {
            let mut res = v0;
            for _ in 0..BATCH_SIZE {
                res = apply_matrix3(black_box(&m), black_box(res));
            }
            black_box(res)
        })
    });

    c.bench_function("rotate 3D point GA (sandwich)", |bencher| {
        bencher.iter(|| {
            let mut res = v0;
            for _ in 0..BATCH_SIZE {
                res = rotor.rotate(black_box(res));
            }
            black_box(res)
        })
    });

    c.bench_function("rotate 3D point GA (fast)", |bencher| {
        bencher.iter(|| {
            let mut res = v0;
            for _ in 0..BATCH_SIZE {
                res = rotor.rotate_fast(black_box(res));
            }
            black_box(res)
        })
    });

    c.bench_function("rotate 3D point GA (SIMD 4×)", |bencher| {
        bencher.iter(|| {
            let mut vs = [v0, v0, v0, v0];
            for _ in 0..BATCH_SIZE {
                vs = rotor.rotate_simd(black_box(vs));
            }
            black_box(vs)
        })
    });

    c.bench_function("rotate 3D point GA (SIMD 8×)", |bencher| {
        bencher.iter(|| {
            let mut vs = [v0, v0, v0, v0, v0, v0, v0, v0];
            for _ in 0..BATCH_SIZE {
                vs = rotor.rotate_simd8(black_box(vs));
            }
            black_box(vs)
        })
    });
}

criterion_group!(
    benches,
    bench_matrix_mult,
    bench_geometric_product_full,
    bench_rotate_point,
);
criterion_main!(benches);