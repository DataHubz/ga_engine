// benches/classical.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::multiply_matrices;
use matrixmultiply::dgemm;
use nalgebra::DMatrix;

const BATCH_SIZE: usize = 1_000;
const N: usize = 8;

/// 1) Naïve triple‐loop
fn bench_naive(c: &mut Criterion) {
    let a: Vec<f64> = (0..N * N).map(|i| (i % 10) as f64).collect();
    let b = a.clone();
    c.bench_function("naive 8×8 × 1000", |bencher| {
        bencher.iter(|| {
            let mut res = Vec::with_capacity(N * N);
            for _ in 0..BATCH_SIZE {
                res = multiply_matrices(black_box(&a), black_box(&b), N);
            }
            black_box(res)
        })
    });
}

/// 2) `matrixmultiply::dgemm` (BLAS‐style)
fn bench_matrixmultiply(c: &mut Criterion) {
    let a: Vec<f64> = (0..N * N).map(|i| (i % 10) as f64).collect();
    let b = a.clone();
    let mut cmat = vec![0.0; N * N];

    c.bench_function("matrixmultiply dgemm 8×8 × 1000", |bencher| {
        bencher.iter(|| {
            for _ in 0..BATCH_SIZE {
                unsafe {
                    dgemm(
                        // m, k, n
                        N,
                        N,
                        N,
                        // α
                        1.0,
                        // A pointer, row stride, col stride
                        a.as_ptr(),
                        N as isize,
                        1,
                        // B pointer, row stride, col stride
                        b.as_ptr(),
                        N as isize,
                        1,
                        // β
                        0.0,
                        // C pointer, row stride, col stride
                        cmat.as_mut_ptr(),
                        N as isize,
                        1,
                    );
                }
            }
            // Make sure we don't accidentally return &cmat here
            black_box(&cmat);
        })
    });
}

/// 3) `nalgebra` dynamic matrix
fn bench_nalgebra(c: &mut Criterion) {
    // Build DMatrix from a flat iterator
    let a = DMatrix::from_iterator(N, N, (0..N * N).map(|i| (i % 10) as f64));
    let b = a.clone();
    c.bench_function("nalgebra DMatrix 8×8 × 1000", |bencher| {
        bencher.iter(|| {
            let mut res = a.clone();
            for _ in 0..BATCH_SIZE {
                res = &a * &b;
            }
            black_box(res)
        })
    });
}

criterion_group!(
    classical_benches,
    bench_naive,
    bench_matrixmultiply,
    bench_nalgebra,
);
criterion_main!(classical_benches);
