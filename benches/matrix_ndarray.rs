// benches/matrix_ndarray.rs
#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {}

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, ShapeBuilder};

const BATCH_SIZE: usize = 1_000;

fn bench_ndarray(c: &mut Criterion) {
    let n = black_box(128);
    let a = Array2::from_shape_fn((n, n).f(), |(i, j)| ((i * n + j) % 10) as f64);
    let mut cmat = Array2::<f64>::zeros((n, n).f());

    c.bench_function("ndarray + blas 128×128 × 1000", |bencher| {
        bencher.iter(|| {
            for _ in 0..BATCH_SIZE {
                general_mat_mul(1.0, &a, &a, 0.0, &mut cmat);
            }
            black_box(&cmat);
        })
    });
}

criterion_group!(ndarray_benches, bench_ndarray);
criterion_main!(ndarray_benches);
