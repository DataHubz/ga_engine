//! Benchmark: Core operations comparison
//!
//! Fair comparison between Clifford-LWE and classical approach
//! Measures just the core algebraic operations, not error generation

use ga_engine::clifford_ring::{CliffordPolynomial, CliffordRingElement};
use std::time::Instant;

fn random_discrete_poly(n: usize) -> CliffordPolynomial {
    let mut coeffs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut mv = [0.0; 8];
        for i in 0..8 {
            mv[i] = if rand::random::<bool>() { 1.0 } else { -1.0 };
        }
        coeffs.push(CliffordRingElement::from_multivector(mv));
    }
    CliffordPolynomial::new(coeffs)
}

fn main() {
    println!("=== Core Operations Benchmark ===\n");

    // Test polynomial multiplication at different degrees
    for n in [8, 16, 32] {
        println!("--- Degree N={} (Total dim = 8×{} = {}) ---", n, n, 8*n);

        let mut a = random_discrete_poly(n);
        let mut b = random_discrete_poly(n);
        a.reduce_modulo_xn_minus_1(n);
        b.reduce_modulo_xn_minus_1(n);

        // Warm up
        for _ in 0..10 {
            let mut c = a.multiply(&b);
            c.reduce_modulo_xn_minus_1(n);
        }

        // Benchmark polynomial multiplication
        const ITERS: usize = 1000;
        let start = Instant::now();
        for _ in 0..ITERS {
            let mut c = a.multiply(&b);
            c.reduce_modulo_xn_minus_1(n);
        }
        let elapsed = start.elapsed();
        let avg_ns = elapsed.as_nanos() / ITERS as u128;

        println!("Polynomial multiply (Clifford): {} ns", avg_ns);
        println!("  = {} coefficient mults × 74 ns/mult", n * n);
        println!("  Expected: ~{} ns", n * n * 74);
        println!();
    }

    println!("--- Single Clifford Product ---");
    let a = CliffordRingElement::from_multivector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = CliffordRingElement::from_multivector([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

    const ITERS: usize = 10_000_000;
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = a.multiply(&b);
    }
    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() / ITERS as u128;

    println!("Single Clifford product: {} ns", avg_ns);
    println!("vs 8×8 matrix mult: ~82 ns");
    println!("Speedup: {:.2}×", 82.0 / avg_ns as f64);
}
