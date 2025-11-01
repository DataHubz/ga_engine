//! Analyze the exact relationship between FFT and CKKS canonical embedding
//!
//! This will help us understand if we need to modify our encoding or if
//! the automorphism formula is just different.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

fn main() {
    println!("=================================================================");
    println!("Analyzing FFT Structure for CKKS");
    println!("=================================================================\n");

    let n = 32; // Ring dimension
    let m = 2 * n; // M = 64
    let num_slots = n / 2; // 16 slots

    println!("Ring dimension N = {}", n);
    println!("Cyclotomic index M = {}", m);
    println!("Number of slots = {}\n", num_slots);

    // Create a simple test vector: [1, 0, 0, ..., 0] in slot 0
    let mut slots = vec![Complex::new(0.0, 0.0); num_slots];
    slots[0] = Complex::new(1.0, 0.0);

    println!("-----------------------------------------------------------------");
    println!("Test: Encode [1, 0, 0, ..., 0] and see coefficient pattern");
    println!("-----------------------------------------------------------------\n");

    // Method 1: Standard FFT (what rustfft does)
    println!("Method 1: Standard inverse FFT");

    let mut extended = vec![Complex::new(0.0, 0.0); n];
    for i in 0..num_slots {
        extended[i] = slots[i];
    }
    for i in 1..num_slots {
        extended[n - i] = slots[i].conj();
    }

    let mut fft_result = extended.clone();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);
    fft.process(&mut fft_result);

    println!("Coefficients (normalized):");
    for i in 0..16 {
        let val = fft_result[i] / (n as f64);
        println!("  coeff[{:2}] = {:.6} + {:.6}i", i, val.re, val.im);
    }

    println!("\n-----------------------------------------------------------------");
    println!("Method 2: CKKS canonical embedding (manual)");
    println!("-----------------------------------------------------------------\n");

    // Compute what CKKS canonical embedding should give
    // Inverse: coeff[j] = (1/N) Σ_k slot[k] * ζ_M^{-(2k+1)j}

    let mut ckks_coeffs = vec![Complex::new(0.0, 0.0); n];
    for j in 0..n {
        let mut sum = Complex::new(0.0, 0.0);
        for k in 0..num_slots {
            // ζ_M^{-(2k+1)j} where ζ_M = e^{2πi/M}
            let exponent = -(2 * k as i64 + 1) * j as i64;
            let angle = 2.0 * PI * (exponent as f64) / (m as f64);
            let root = Complex::new(angle.cos(), angle.sin());
            sum += extended[k] * root;
        }
        ckks_coeffs[j] = sum / (n as f64);
    }

    println!("CKKS coefficients:");
    for i in 0..16 {
        println!("  coeff[{:2}] = {:.6} + {:.6}i", i, ckks_coeffs[i].re, ckks_coeffs[i].im);
    }

    println!("\n-----------------------------------------------------------------");
    println!("Comparison: Are they the same or different?");
    println!("-----------------------------------------------------------------\n");

    let mut max_diff = 0.0;
    for i in 0..n {
        let diff = (fft_result[i] / (n as f64) - ckks_coeffs[i]).norm();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("Max difference: {:.2e}", max_diff);

    if max_diff < 1e-10 {
        println!("✓ They are THE SAME! rustfft already uses CKKS encoding!");
        println!("  The issue must be in the automorphism formula.\n");
    } else {
        println!("✗ They are DIFFERENT! We need to use custom encoding.");
        println!("  rustfft uses standard FFT, not CKKS canonical embedding.\n");
    }

    println!("=================================================================");
}
