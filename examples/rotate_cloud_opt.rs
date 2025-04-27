// examples/rotate_cloud_opt.rs
use ga_engine::prelude::*;
use std::time::Instant;

const N_POINTS: usize = 100_000;
const BATCH_SIZE: usize = 4;

fn main() {
    // build a simple point cloud (unit circle in XY)
    let mut points = Vec::with_capacity(N_POINTS);
    for i in 0..N_POINTS {
        let theta = (i as f64) * std::f64::consts::TAU / (N_POINTS as f64);
        points.push(Vec3::new(theta.cos(), theta.sin(), 0.0));
    }

    // classical rotation matrix +90° about Z
    let m = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    // GA rotor for +90° about Z
    let rotor = Rotor3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);

    // 1) Classical matrix
    let mut out_mat = vec![Vec3::new(0.0, 0.0, 0.0); N_POINTS];
    let t0 = Instant::now();
    for i in 0..N_POINTS {
        out_mat[i] = apply_matrix3(&m, points[i]);
    }
    let dt0 = t0.elapsed();
    let sum0: f64 = out_mat.iter().map(|v| v.x + v.y + v.z).sum();
    println!("Classical matrix   : {:>8?}, checksum = {}", dt0, sum0);

    // 2) GA rotate_fast
    let mut out_fast = vec![Vec3::new(0.0, 0.0, 0.0); N_POINTS];
    let t1 = Instant::now();
    for i in 0..N_POINTS {
        out_fast[i] = rotor.rotate_fast(points[i]);
    }
    let dt1 = t1.elapsed();
    let sum1: f64 = out_fast.iter().map(|v| v.x + v.y + v.z).sum();
    println!("GA rotate_fast     : {:>8?}, checksum = {}", dt1, sum1);

    // 3) GA SIMD‐4× in tight chunks
    let mut out_simd4 = vec![Vec3::new(0.0, 0.0, 0.0); N_POINTS];
    let t2 = Instant::now();
    let chunks = N_POINTS / BATCH_SIZE;
    for chunk in 0..chunks {
        let base = chunk * BATCH_SIZE;
        let in4 = [
            points[base],
            points[base + 1],
            points[base + 2],
            points[base + 3],
        ];
        let out4 = rotor.rotate_simd(in4);
        out_simd4[base] = out4[0];
        out_simd4[base + 1] = out4[1];
        out_simd4[base + 2] = out4[2];
        out_simd4[base + 3] = out4[3];
    }
    // tail
    for i in (chunks * BATCH_SIZE)..N_POINTS {
        out_simd4[i] = rotor.rotate_fast(points[i]);
    }
    let dt2 = t2.elapsed();
    let sum2: f64 = out_simd4.iter().map(|v| v.x + v.y + v.z).sum();
    println!("GA SIMD-4×         : {:>8?}, checksum = {}", dt2, sum2);
}
