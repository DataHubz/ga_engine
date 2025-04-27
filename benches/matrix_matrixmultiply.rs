// benches/matrix_matrixmultiply.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use matrixmultiply::dgemm;

const BATCH_SIZE: usize = 1_000;

fn bench_matrixmultiply(c: &mut Criterion) {
    let n = black_box(128);
    let a: Vec<f64> = (0..n * n).map(|i| (i % 10) as f64).collect();
    let mut out = vec![0.0; n * n];

    c.bench_function("matrixmultiply dgemm 128×128 × 1000", |bencher| {
        bencher.iter(|| {
            for _ in 0..BATCH_SIZE {
                unsafe {
                    dgemm(
                        // m, k, n
                        n,
                        n,
                        n,
                        // α
                        1.0,
                        // A pointer and strides
                        a.as_ptr(),
                        n as isize,
                        1,
                        // B pointer and strides
                        a.as_ptr(),
                        n as isize,
                        1,
                        // β
                        0.0,
                        // C pointer and strides
                        out.as_mut_ptr(),
                        n as isize,
                        1,
                    );
                }
            }
            black_box(&out);
        })
    });
}

criterion_group!(matrixmultiply_benches, bench_matrixmultiply);
criterion_main!(matrixmultiply_benches);
