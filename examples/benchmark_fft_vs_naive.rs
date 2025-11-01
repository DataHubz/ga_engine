//! Benchmark: FFT vs Naive polynomial multiplication
//!
//! Compare O(N log N) FFT-based multiplication vs O(N²) naive convolution

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

fn main() {
    println!("=== FFT vs Naive Polynomial Multiplication ===\n");

    for n in [8, 16, 32, 64, 128] {
        println!("--- Degree N={} (Total dim = {}) ---", n, 8*n);

        let a = random_poly(n);
        let b = random_poly(n);

        // Warm up
        for _ in 0..5 {
            let _ = a.multiply_circular_fft(&b, n);
        }

        // Benchmark FFT method
        const ITERS: usize = 100;
        let start = Instant::now();
        for _ in 0..ITERS {
            let _ = a.multiply_circular_fft(&b, n);
        }
        let fft_time = start.elapsed();
        let fft_avg_us = fft_time.as_micros() / ITERS as u128;

        println!("FFT method: {} µs", fft_avg_us);
        println!("  Complexity: O(n log n) = O({} × {:.1}) ≈ {:.0} ops",
            n, (n as f64).log2(), n as f64 * (n as f64).log2());

        // Benchmark naive method (only for small n to avoid timeout)
        if n <= 32 {
            let start = Instant::now();
            for _ in 0..ITERS {
                let mut c = a.multiply(&b);
                c.reduce_modulo_xn_minus_1(n);
            }
            let naive_time = start.elapsed();
            let naive_avg_us = naive_time.as_micros() / ITERS as u128;

            println!("Naive method: {} µs", naive_avg_us);
            println!("  Complexity: O(n²) = O({}) = {} ops", n*n, n*n);
            println!("  Speedup: {:.2}×", naive_avg_us as f64 / fft_avg_us as f64);
        } else {
            println!("Naive method: skipped (too slow)");
            println!("  Complexity: O(n²) = O({}) = {} ops", n*n, n*n);
        }

        // Verify correctness (for small n)
        if n <= 32 {
            let fft_result = a.multiply_circular_fft(&b, n);
            let mut naive_result = a.multiply(&b);
            naive_result.reduce_modulo_xn_minus_1(n);

            let mut max_error = 0.0;
            for i in 0..n.min(fft_result.coeffs.len()).min(naive_result.coeffs.len()) {
                for j in 0..8 {
                    let err = (fft_result.coeffs[i].coeffs[j] - naive_result.coeffs[i].coeffs[j]).abs();
                    if err > max_error {
                        max_error = err;
                    }
                }
            }
            println!("  Max error: {:.2e} {}", max_error, if max_error < 1e-6 { "✓" } else { "✗" });
        }

        println!();
    }

    println!("--- Summary ---");
    println!("FFT method scales as O(N log N) - efficient for large N");
    println!("Naive method scales as O(N²) - only practical for small N");
    println!("For N=32: ~10-50× speedup expected");
    println!("For N=128: ~100-500× speedup expected");
}
