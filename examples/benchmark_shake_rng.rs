//! Benchmark SHAKE128 RNG vs rand::thread_rng()

use ga_engine::shake_rng::{sample_ternary_shake128, sample_small_error_shake128, generate_seed};
use ga_engine::clifford_ring_int::CliffordRingElementInt;
use std::time::Instant;
use rand::Rng;

const N: usize = 32;  // Polynomial degree
const NUM_OPS: usize = 1000;

fn discrete_poly_thread_rng() -> Vec<[i64; 8]> {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(N);
    for _ in 0..N {
        let mut mv = [0i64; 8];
        for j in 0..8 {
            mv[j] = rng.gen_range(-1..=1);
        }
        coeffs.push(mv);
    }
    coeffs
}

fn error_poly_thread_rng() -> Vec<[i64; 8]> {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(N);
    for _ in 0..N {
        let mut mv = [0i64; 8];
        for j in 0..8 {
            mv[j] = rng.gen_range(-2..=2);
        }
        coeffs.push(mv);
    }
    coeffs
}

fn discrete_poly_shake() -> Vec<[i64; 8]> {
    let mut coeffs = Vec::with_capacity(N);
    for _ in 0..N {
        let mut mv = [0i64; 8];
        for j in 0..8 {
            let seed = generate_seed();
            let values = sample_ternary_shake128(&seed, 1);
            mv[j] = values[0];
        }
        coeffs.push(mv);
    }
    coeffs
}

fn discrete_poly_shake_optimized() -> Vec<[i64; 8]> {
    let mut coeffs = Vec::with_capacity(N);

    // Generate all 8 components at once
    let mut component_data: Vec<Vec<i64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let seed = generate_seed();
        component_data.push(sample_ternary_shake128(&seed, N));
    }

    // Transpose to get coefficients
    for i in 0..N {
        let mut mv = [0i64; 8];
        for j in 0..8 {
            mv[j] = component_data[j][i];
        }
        coeffs.push(mv);
    }
    coeffs
}

fn error_poly_shake_optimized() -> Vec<[i64; 8]> {
    let mut coeffs = Vec::with_capacity(N);

    // Generate all 8 components at once
    let mut component_data: Vec<Vec<i64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let seed = generate_seed();
        component_data.push(sample_small_error_shake128(&seed, N, 2));
    }

    // Transpose to get coefficients
    for i in 0..N {
        let mut mv = [0i64; 8];
        for j in 0..8 {
            mv[j] = component_data[j][i];
        }
        coeffs.push(mv);
    }
    coeffs
}

fn main() {
    println!("=== RNG Performance Benchmark ===\n");
    println!("Polynomial degree: N = {}", N);
    println!("Components per coefficient: 8");
    println!("Total random values per polynomial: N × 8 = {}", N * 8);
    println!("Operations: {}\n", NUM_OPS);

    // Benchmark discrete polynomial (thread_rng)
    println!("--- Discrete Polynomial: thread_rng ({} ops) ---", NUM_OPS);
    let start = Instant::now();
    for _ in 0..NUM_OPS {
        let _ = discrete_poly_thread_rng();
    }
    let thread_rng_discrete = start.elapsed();
    let thread_rng_discrete_us = thread_rng_discrete.as_micros() as f64 / NUM_OPS as f64;
    println!("Average time: {:.2} µs\n", thread_rng_discrete_us);

    // Benchmark error polynomial (thread_rng)
    println!("--- Error Polynomial: thread_rng ({} ops) ---", NUM_OPS);
    let start = Instant::now();
    for _ in 0..NUM_OPS {
        let _ = error_poly_thread_rng();
    }
    let thread_rng_error = start.elapsed();
    let thread_rng_error_us = thread_rng_error.as_micros() as f64 / NUM_OPS as f64;
    println!("Average time: {:.2} µs\n", thread_rng_error_us);

    // Benchmark discrete polynomial (SHAKE - optimized)
    println!("--- Discrete Polynomial: SHAKE128 optimized ({} ops) ---", NUM_OPS);
    let start = Instant::now();
    for _ in 0..NUM_OPS {
        let _ = discrete_poly_shake_optimized();
    }
    let shake_discrete = start.elapsed();
    let shake_discrete_us = shake_discrete.as_micros() as f64 / NUM_OPS as f64;
    println!("Average time: {:.2} µs\n", shake_discrete_us);

    // Benchmark error polynomial (SHAKE - optimized)
    println!("--- Error Polynomial: SHAKE128 optimized ({} ops) ---", NUM_OPS);
    let start = Instant::now();
    for _ in 0..NUM_OPS {
        let _ = error_poly_shake_optimized();
    }
    let shake_error = start.elapsed();
    let shake_error_us = shake_error.as_micros() as f64 / NUM_OPS as f64;
    println!("Average time: {:.2} µs\n", shake_error_us);

    // Results
    println!("=== Results ===\n");
    println!("| Polynomial Type | thread_rng (µs) | SHAKE128 (µs) | Speedup |");
    println!("|-----------------|-----------------|---------------|---------|");
    println!("| Discrete | {:.2} | {:.2} | {:.2}× |",
             thread_rng_discrete_us, shake_discrete_us, thread_rng_discrete_us / shake_discrete_us);
    println!("| Error | {:.2} | {:.2} | {:.2}× |",
             thread_rng_error_us, shake_error_us, thread_rng_error_us / shake_error_us);
    println!();

    let discrete_speedup = thread_rng_discrete_us / shake_discrete_us;
    let error_speedup = thread_rng_error_us / shake_error_us;

    if discrete_speedup > 1.0 {
        println!("🎉 SHAKE128 discrete is {:.1}% faster!", (discrete_speedup - 1.0) * 100.0);
    } else {
        println!("⚠️  SHAKE128 discrete is {:.1}% slower", (1.0 - discrete_speedup) * 100.0);
    }

    if error_speedup > 1.0 {
        println!("🎉 SHAKE128 error is {:.1}% faster!", (error_speedup - 1.0) * 100.0);
    } else {
        println!("⚠️  SHAKE128 error is {:.1}% slower", (1.0 - error_speedup) * 100.0);
    }

    // Estimate impact on full encryption
    println!("\n=== Impact on Clifford-LWE Encryption ===");
    println!("Current encryption time: ~26 µs");
    println!("RNG time (2× discrete + 2× error): {:.2} µs ({:.1}%)",
             2.0 * thread_rng_discrete_us + 2.0 * thread_rng_error_us,
             100.0 * (2.0 * thread_rng_discrete_us + 2.0 * thread_rng_error_us) / 26.0);

    let rng_savings = 2.0 * (thread_rng_discrete_us - shake_discrete_us) +
                      2.0 * (thread_rng_error_us - shake_error_us);

    println!("Estimated savings with SHAKE: {:.2} µs", rng_savings);
    println!("Estimated new encryption time: {:.2} µs", 26.0 - rng_savings);
}
