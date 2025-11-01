//! Profile where time is spent in Clifford-LWE encryption

use ga_engine::clifford_ring_int::{CliffordPolynomialInt, CliffordRingElementInt};
use ga_engine::lazy_reduction::LazyReductionContext;
use std::time::Instant;
use rand::Rng;

struct CLWEParams {
    n: usize,
    q: i64,
    error_bound: i64,
}

impl Default for CLWEParams {
    fn default() -> Self {
        Self {
            n: 32,
            q: 3329,
            error_bound: 2,
        }
    }
}

fn discrete_poly_fast(n: usize) -> CliffordPolynomialInt {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut mv = [0i64; 8];
        for j in 0..8 {
            mv[j] = rng.gen_range(-1..=1);
        }
        coeffs.push(CliffordRingElementInt::from_multivector(mv));
    }
    CliffordPolynomialInt::new(coeffs)
}

fn error_poly_fast(n: usize, bound: i64) -> CliffordPolynomialInt {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut mv = [0i64; 8];
        for j in 0..8 {
            mv[j] = rng.gen_range(-bound..=bound);
        }
        coeffs.push(CliffordRingElementInt::from_multivector(mv));
    }
    CliffordPolynomialInt::new(coeffs)
}

fn main() {
    let params = CLWEParams::default();
    let lazy = LazyReductionContext::new(params.q);

    println!("=== Profiling Clifford-LWE Encryption ===\n");

    const NUM_OPS: usize = 100;

    // Setup
    let mut a = discrete_poly_fast(params.n);
    a.reduce_modulo_xn_minus_1(params.n, params.q);

    let mut b = discrete_poly_fast(params.n);
    b.reduce_modulo_xn_minus_1(params.n, params.q);

    // Profile RNG
    let start = Instant::now();
    for _ in 0..NUM_OPS {
        let _ = discrete_poly_fast(params.n);
    }
    let rng_time = start.elapsed().as_micros() as f64 / NUM_OPS as f64;
    println!("RNG (discrete):          {:.2} µs", rng_time);

    let start = Instant::now();
    for _ in 0..NUM_OPS {
        let _ = error_poly_fast(params.n, params.error_bound);
    }
    let error_time = start.elapsed().as_micros() as f64 / NUM_OPS as f64;
    println!("RNG (error):             {:.2} µs", error_time);

    // Profile Karatsuba multiplication
    let start = Instant::now();
    for _ in 0..NUM_OPS {
        let _ = a.multiply_karatsuba_lazy(&b, &lazy);
    }
    let karatsuba_time = start.elapsed().as_micros() as f64 / NUM_OPS as f64;
    println!("Karatsuba multiply:      {:.2} µs", karatsuba_time);

    // Profile SIMD Karatsuba multiplication
    let start = Instant::now();
    for _ in 0..NUM_OPS {
        let _ = a.multiply_karatsuba_lazy_simd(&b, &lazy);
    }
    let simd_karatsuba_time = start.elapsed().as_micros() as f64 / NUM_OPS as f64;
    println!("Karatsuba SIMD multiply: {:.2} µs", simd_karatsuba_time);

    // Profile modular reduction
    let mut c = a.multiply_karatsuba_lazy(&b, &lazy);
    let start = Instant::now();
    for _ in 0..NUM_OPS {
        c.reduce_modulo_xn_minus_1_lazy(params.n, &lazy);
    }
    let reduce_time = start.elapsed().as_micros() as f64 / NUM_OPS as f64;
    println!("Modular reduction:       {:.2} µs", reduce_time);

    // Profile addition
    let start = Instant::now();
    for _ in 0..NUM_OPS {
        let _ = a.add_lazy_poly(&b);
    }
    let add_time = start.elapsed().as_micros() as f64 / NUM_OPS as f64;
    println!("Polynomial addition:     {:.2} µs", add_time);

    println!("\n=== Encryption Breakdown ===");
    println!("Per encryption (~45 µs total):");
    println!("  3× RNG discrete:       {:.2} µs ({:.1}%)", 3.0 * rng_time, 100.0 * 3.0 * rng_time / 45.0);
    println!("  3× RNG error:          {:.2} µs ({:.1}%)", 3.0 * error_time, 100.0 * 3.0 * error_time / 45.0);
    println!("  2× Karatsuba multiply: {:.2} µs ({:.1}%)", 2.0 * karatsuba_time, 100.0 * 2.0 * karatsuba_time / 45.0);
    println!("  Other ops:             {:.2} µs", 45.0 - (3.0 * rng_time + 3.0 * error_time + 2.0 * karatsuba_time));

    println!("\n=== SIMD Impact ===");
    let simd_improvement = karatsuba_time - simd_karatsuba_time;
    println!("Karatsuba speedup:       {:.2} µs ({:.1}×)", simd_improvement, karatsuba_time / simd_karatsuba_time);
    println!("Expected total speedup:  {:.2} µs (2× Karatsuba per encrypt)", 2.0 * simd_improvement);
    println!("Expected new total:      {:.2} µs", 45.0 - 2.0 * simd_improvement);
}
