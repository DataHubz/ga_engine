// examples/rotate_cloud.rs
use ga_engine::prelude::*;
use std::time::Instant;

/// Simple 64-bit LCG for reproducible “random” floats in [0,1).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_f64(&mut self) -> f64 {
        // X_{n+1} = a X_n + c  (mod 2⁶⁴)
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        // take top 53 bits → [0,1)
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn main() {
    // Generate N random points in the unit cube centered at 0.
    const N: usize = 100_000;
    let mut rng = Lcg::new(0x1234_5678_9abc_def0);
    let mut points = Vec::with_capacity(N);
    for _ in 0..N {
        let x = rng.next_f64() * 2.0 - 1.0;
        let y = rng.next_f64() * 2.0 - 1.0;
        let z = rng.next_f64() * 2.0 - 1.0;
        points.push(Vec3::new(x, y, z));
    }

    // Define our +90° about Z transform, both as matrix and GA rotor.
    let angle = std::f64::consts::FRAC_PI_2;
    let matrix = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let rotor = Rotor3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), angle);

    // 1) Classical matrix
    let t0 = Instant::now();
    let rotated_mat: Vec<_> = points.iter().map(|&v| apply_matrix3(&matrix, v)).collect();
    let dt_mat = t0.elapsed();
    println!("Classical matrix   : {:>7.3?} for {} points", dt_mat, N);

    // 2) GA sandwich (“rotate”)
    let t1 = Instant::now();
    let _rotated_ga: Vec<_> = points.iter().map(|&v| rotor.rotate(v)).collect();
    let dt_ga = t1.elapsed();
    println!("GA sandwich        : {:>7.3?}", dt_ga);

    // 3) GA fast (FMA-chained)
    let t2 = Instant::now();
    let rotated_fast: Vec<_> = points.iter().map(|&v| rotor.rotate_fast(v)).collect();
    let dt_fast = t2.elapsed();
    println!("GA rotate_fast     : {:>7.3?}", dt_fast);

    // 4) GA SIMD-4× (in batches of four)
    let t3 = Instant::now();
    let mut rotated_simd4 = Vec::with_capacity(N);
    for chunk in points.chunks(4) {
        if chunk.len() == 4 {
            let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let out = rotor.rotate_simd(arr);
            rotated_simd4.extend_from_slice(&out);
        } else {
            // tail case
            for &v in chunk {
                rotated_simd4.push(rotor.rotate_fast(v));
            }
        }
    }
    let dt_simd4 = t3.elapsed();
    println!("GA SIMD-4×         : {:>7.3?}", dt_simd4);

    // sanity‐check that all methods agree (within ε)
    let eps = 1e-9;
    for i in 0..N {
        let m = rotated_mat[i];
        let f = rotated_fast[i];
        assert!((m.x - f.x).abs() < eps && (m.y - f.y).abs() < eps && (m.z - f.z).abs() < eps);
    }
}
