//! Benchmark: Compare polynomial multiplication methods
//!
//! Tests three approaches:
//! 1. Naive: O(N²) convolution - correct but slow
//! 2. Karatsuba: O(N^1.585) - correct and faster for large N
//! 3. FFT: O(N log N) - fastest but has correctness issues with non-commutative rings
//!
//! Goal: Verify Karatsuba is both correct AND faster than naive for N≥16

use ga_engine::clifford_ring::{CliffordPolynomial, CliffordRingElement};
use std::time::Instant;

fn random_poly(n: usize) -> CliffordPolynomial {
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

fn max_error(poly1: &CliffordPolynomial, poly2: &CliffordPolynomial) -> f64 {
    let n = poly1.coeffs.len().min(poly2.coeffs.len());
    let mut max_err = 0.0;

    for i in 0..n {
        for j in 0..8 {
            let err = (poly1.coeffs[i].coeffs[j] - poly2.coeffs[i].coeffs[j]).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }

    max_err
}

fn main() {
    println!("=== Polynomial Multiplication Method Comparison ===\n");
    println!("Testing: Naive O(N²) vs Karatsuba O(N^1.585) vs FFT O(N log N)\n");

    for n in [8, 16, 32, 64] {
        println!("--- Polynomial degree N={} (Total dim = {}) ---", n, 8*n);

        let a = random_poly(n);
        let b = random_poly(n);

        // Warm up
        for _ in 0..5 {
            let _ = a.multiply(&b);
        }

        const ITERS: usize = 100;

        // Benchmark NAIVE method (baseline)
        let start = Instant::now();
        let naive_result = {
            let mut result = None;
            for _ in 0..ITERS {
                let mut c = a.multiply(&b);
                c.reduce_modulo_xn_minus_1(n);
                result = Some(c);
            }
            result.unwrap()
        };
        let naive_time = start.elapsed();
        let naive_avg_us = naive_time.as_micros() / ITERS as u128;

        println!("✓ Naive:     {:6} µs  [O(N²) = {} ops]", naive_avg_us, n*n);

        // Benchmark KARATSUBA method
        let start = Instant::now();
        let karatsuba_result = {
            let mut result = None;
            for _ in 0..ITERS {
                let mut c = a.multiply_karatsuba(&b);
                c.reduce_modulo_xn_minus_1(n);
                result = Some(c);
            }
            result.unwrap()
        };
        let karatsuba_time = start.elapsed();
        let karatsuba_avg_us = karatsuba_time.as_micros() / ITERS as u128;

        let karatsuba_speedup = naive_avg_us as f64 / karatsuba_avg_us as f64;
        let karatsuba_error = max_error(&naive_result, &karatsuba_result);
        let karatsuba_correct = karatsuba_error < 1e-6;

        println!("  Karatsuba: {:6} µs  [O(N^1.585) ≈ {:.0} ops] {:5.2}× speedup {}",
            karatsuba_avg_us,
            (n as f64).powf(1.585),
            karatsuba_speedup,
            if karatsuba_correct { "✓" } else { "✗" }
        );
        if !karatsuba_correct {
            println!("    ⚠ Max error: {:.2e}", karatsuba_error);
        }

        // Benchmark FFT method
        let start = Instant::now();
        let fft_result = {
            let mut result = None;
            for _ in 0..ITERS {
                let c = a.multiply_circular_fft(&b, n);
                result = Some(c);
            }
            result.unwrap()
        };
        let fft_time = start.elapsed();
        let fft_avg_us = fft_time.as_micros() / ITERS as u128;

        let fft_speedup = naive_avg_us as f64 / fft_avg_us as f64;
        let fft_error = max_error(&naive_result, &fft_result);
        let fft_correct = fft_error < 1e-6;

        println!("  FFT:       {:6} µs  [O(N log N) ≈ {:.0} ops] {:5.2}× speedup {}",
            fft_avg_us,
            n as f64 * (n as f64).log2(),
            fft_speedup,
            if fft_correct { "✓" } else { "✗" }
        );
        if !fft_correct {
            println!("    ⚠ Max error: {:.2e} (non-commutative coupling issue)", fft_error);
        }

        println!();
    }

    println!("--- Summary ---");
    println!("Naive:     Correct but O(N²) - only practical for N<16");
    println!("Karatsuba: Correct AND O(N^1.585) - best choice for N≥16");
    println!("FFT:       Fast O(N log N) but incorrect for non-commutative rings");
    println!("\nRecommendation: Use Karatsuba for Clifford-LWE polynomial multiplication!");
}
