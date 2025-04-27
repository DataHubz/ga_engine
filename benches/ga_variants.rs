// benches/ga_variants.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::{apply_matrix3, geometric_product_full, Rotor3, Vec3};

const BATCH: usize = 1_000;

fn bench_ga_full(c: &mut Criterion) {
    let a: [f64; 8] = black_box([1., 2., 3., 4., 5., 6., 7., 8.]);
    let b = a;
    let mut out = [0.0; 8];

    c.bench_function("GA full product 8D × 1000", |bencher| {
        bencher.iter(|| {
            for _ in 0..BATCH {
                geometric_product_full(black_box(&a), black_box(&b), &mut out);
            }
            black_box(out)
        })
    });
}

fn bench_ga_rotors(c: &mut Criterion) {
    // 3D rotation about Z by 90°
    let pts = Vec3::new(1.0, 0.0, 0.0);
    let rotor = Rotor3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);
    let matrix = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    // Classical for reference
    c.bench_function("rotate 3D classical × 1000", |bencher| {
        bencher.iter(|| {
            let mut p = pts;
            for _ in 0..BATCH {
                p = apply_matrix3(black_box(&matrix), black_box(p));
            }
            black_box(p)
        })
    });

    // sandwich-product (full GA)
    c.bench_function("rotate GA sandwich × 1000", |bencher| {
        bencher.iter(|| {
            let mut p = pts;
            for _ in 0..BATCH {
                p = rotor.rotate(black_box(p));
            }
            black_box(p)
        })
    });

    // fast quaternion‐style
    c.bench_function("rotate GA fast × 1000", |bencher| {
        bencher.iter(|| {
            let mut p = pts;
            for _ in 0..BATCH {
                p = rotor.rotate_fast(black_box(p));
            }
            black_box(p)
        })
    });

    // SIMD 4×
    c.bench_function("rotate GA simd4 × 1000", |bencher| {
        bencher.iter(|| {
            let mut vs = [pts; 4];
            for _ in 0..BATCH {
                vs = rotor.rotate_simd(black_box(vs));
            }
            black_box(vs)
        })
    });

    // SIMD 8×
    c.bench_function("rotate GA simd8 × 1000", |bencher| {
        bencher.iter(|| {
            let mut vs = [pts; 8];
            for _ in 0..BATCH {
                vs = rotor.rotate_simd8(black_box(vs));
            }
            black_box(vs)
        })
    });
}

criterion_group!(ga_variants_benches, bench_ga_full, bench_ga_rotors);
criterion_main!(ga_variants_benches);
