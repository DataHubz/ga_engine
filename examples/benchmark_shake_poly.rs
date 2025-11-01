//! Benchmark SHAKE128 polynomial generation vs rand::thread_rng()

use ga_engine::shake_poly::{discrete_poly_shake, error_poly_shake, generate_seed};
use ga_engine::clifford_ring_int::{CliffordPolynomialInt, CliffordRingElementInt};
use std::time::Instant;
use rand::Rng;

const N: usize = 32;
const NUM_OPS: usize = 1000;

fn discrete_poly_thread_rng() -> CliffordPolynomialInt {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(N);
    for _ in 0..N {
        let mut mv = [0i64; 8];
        for j in 0..8 {
            mv[j] = rng.gen_range(-1..=1);
        }
        coeffs.push(CliffordRingElementInt::from_multivector(mv));
    }
    CliffordPolynomialInt::new(coeffs)
}

fn error_poly_thread_rng() -> CliffordPolynomialInt {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(N);
    for _ in 0..N {
        let mut mv = [0i64; 8];
        for j in 0..8 {
            mv[j] = rng.gen_range(-2..=2);
        }
        coeffs.push(CliffordRingElementInt::from_multivector(mv));
    }
    CliffordPolynomialInt::new(coeffs)
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

    // Benchmark discrete polynomial (SHAKE)
    println!("--- Discrete Polynomial: SHAKE128 ({} ops) ---", NUM_OPS);
    let start = Instant::now();
    for _ in 0..NUM_OPS {
        let seed = generate_seed();
        let _ = discrete_poly_shake(&seed, N);
    }
    let shake_discrete = start.elapsed();
    let shake_discrete_us = shake_discrete.as_micros() as f64 / NUM_OPS as f64;
    println!("Average time: {:.2} µs\n", shake_discrete_us);

    // Benchmark error polynomial (SHAKE)
    println!("--- Error Polynomial: SHAKE128 ({} ops) ---", NUM_OPS);
    let start = Instant::now();
    for _ in 0..NUM_OPS {
        let seed = generate_seed();
        let _ = error_poly_shake(&seed, N, 2);
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
    let current_rng_time = 2.0 * thread_rng_discrete_us + 2.0 * thread_rng_error_us;
    println!("RNG time (2× discrete + 2× error): {:.2} µs ({:.1}%)",
             current_rng_time,
             100.0 * current_rng_time / 26.0);

    let rng_savings = 2.0 * (thread_rng_discrete_us - shake_discrete_us) +
                      2.0 * (thread_rng_error_us - shake_error_us);

    if rng_savings > 0.0 {
        println!("Estimated savings with SHAKE: {:.2} µs", rng_savings);
        println!("Estimated new encryption time: {:.2} µs ⭐", 26.0 - rng_savings);
    } else {
        println!("SHAKE would add: {:.2} µs overhead", -rng_savings);
        println!("Estimated new encryption time: {:.2} µs", 26.0 - rng_savings);
    }
}
