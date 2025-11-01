//! Direct comparison: Clifford-LWE-256 with Naive vs Karatsuba
//!
//! Measures the actual speedup achieved by switching from O(N²) to O(N^1.585)

use ga_engine::clifford_ring::{CliffordPolynomial, CliffordRingElement};
use rand::Rng;
use std::time::Instant;

struct CLWEParams {
    n: usize,
    q: f64,
    error_stddev: f64,
}

impl Default for CLWEParams {
    fn default() -> Self {
        Self {
            n: 32,
            q: 3329.0,
            error_stddev: 1.0,
        }
    }
}

struct PublicKey {
    a: CliffordPolynomial,
    b: CliffordPolynomial,
}

fn random_discrete_poly(n: usize) -> CliffordPolynomial {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut mv = [0.0; 8];
        for i in 0..8 {
            let val: f64 = rng.gen_range(0..3) as f64 - 1.0;
            mv[i] = val;
        }
        coeffs.push(CliffordRingElement::from_multivector(mv));
    }
    CliffordPolynomial::new(coeffs)
}

fn gaussian_error_poly(n: usize, stddev: f64) -> CliffordPolynomial {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut mv = [0.0; 8];
        for i in 0..8 {
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            mv[i] = z * stddev;
        }
        coeffs.push(CliffordRingElement::from_multivector(mv));
    }
    CliffordPolynomial::new(coeffs)
}

fn scale_poly(poly: &CliffordPolynomial, scalar: f64) -> CliffordPolynomial {
    let coeffs: Vec<_> = poly.coeffs.iter()
        .map(|c| c.scalar_mul(scalar))
        .collect();
    CliffordPolynomial::new(coeffs)
}

fn encrypt_naive(pk: &PublicKey, message: &CliffordPolynomial, params: &CLWEParams) -> (CliffordPolynomial, CliffordPolynomial) {
    let mut r = random_discrete_poly(params.n);
    r.reduce_modulo_xn_minus_1(params.n);

    let mut e1 = gaussian_error_poly(params.n, params.error_stddev);
    e1.reduce_modulo_xn_minus_1(params.n);

    let mut e2 = gaussian_error_poly(params.n, params.error_stddev);
    e2.reduce_modulo_xn_minus_1(params.n);

    let scaled_msg = scale_poly(message, params.q / 2.0);

    // NAIVE O(N²) multiplication
    let mut u = pk.a.multiply(&r);
    u.reduce_modulo_xn_minus_1(params.n);
    u = u.add(&e1);

    let mut v = pk.b.multiply(&r);
    v.reduce_modulo_xn_minus_1(params.n);
    v = v.add(&e2);
    v = v.add(&scaled_msg);

    (u, v)
}

fn encrypt_karatsuba(pk: &PublicKey, message: &CliffordPolynomial, params: &CLWEParams) -> (CliffordPolynomial, CliffordPolynomial) {
    let mut r = random_discrete_poly(params.n);
    r.reduce_modulo_xn_minus_1(params.n);

    let mut e1 = gaussian_error_poly(params.n, params.error_stddev);
    e1.reduce_modulo_xn_minus_1(params.n);

    let mut e2 = gaussian_error_poly(params.n, params.error_stddev);
    e2.reduce_modulo_xn_minus_1(params.n);

    let scaled_msg = scale_poly(message, params.q / 2.0);

    // KARATSUBA O(N^1.585) multiplication
    let mut u = pk.a.multiply_karatsuba(&r);
    u.reduce_modulo_xn_minus_1(params.n);
    u = u.add(&e1);

    let mut v = pk.b.multiply_karatsuba(&r);
    v.reduce_modulo_xn_minus_1(params.n);
    v = v.add(&e2);
    v = v.add(&scaled_msg);

    (u, v)
}

fn main() {
    println!("=== Clifford-LWE-256: Naive vs Karatsuba Comparison ===\n");

    let params = CLWEParams::default();

    // Generate public key
    let mut a = random_discrete_poly(params.n);
    a.reduce_modulo_xn_minus_1(params.n);
    let mut b = random_discrete_poly(params.n);
    b.reduce_modulo_xn_minus_1(params.n);
    let pk = PublicKey { a, b };

    // Generate message
    let mut msg_coeffs = Vec::with_capacity(params.n);
    for i in 0..params.n {
        let mut mv = [0.0; 8];
        mv[0] = if i % 3 == 0 { 1.0 } else { 0.0 };
        msg_coeffs.push(CliffordRingElement::from_multivector(mv));
    }
    let message = CliffordPolynomial::new(msg_coeffs);

    println!("Parameters: N={}, dimension=8×{}={}", params.n, params.n, params.n * 8);
    println!();

    // Warm up
    for _ in 0..10 {
        let _ = encrypt_karatsuba(&pk, &message, &params);
    }

    const ITERS: usize = 100;

    // Benchmark NAIVE
    println!("--- Naive O(N²) Multiplication ---");
    let naive_start = Instant::now();
    for _ in 0..ITERS {
        let _ = encrypt_naive(&pk, &message, &params);
    }
    let naive_time = naive_start.elapsed();
    let naive_avg_us = naive_time.as_micros() as f64 / ITERS as f64;

    println!("Total time: {:?}", naive_time);
    println!("Average per encryption: {:.2} µs", naive_avg_us);
    println!("Complexity: O(N²) = {} coefficient multiplications", params.n * params.n);
    println!();

    // Benchmark KARATSUBA
    println!("--- Karatsuba O(N^1.585) Multiplication ---");
    let karatsuba_start = Instant::now();
    for _ in 0..ITERS {
        let _ = encrypt_karatsuba(&pk, &message, &params);
    }
    let karatsuba_time = karatsuba_start.elapsed();
    let karatsuba_avg_us = karatsuba_time.as_micros() as f64 / ITERS as f64;

    println!("Total time: {:?}", karatsuba_time);
    println!("Average per encryption: {:.2} µs", karatsuba_avg_us);
    println!("Complexity: O(N^1.585) ≈ {:.0} coefficient multiplications", (params.n as f64).powf(1.585));
    println!();

    // Summary
    let speedup = naive_avg_us / karatsuba_avg_us;
    let time_saved = naive_avg_us - karatsuba_avg_us;

    println!("--- Performance Improvement ---");
    println!("Naive:     {:.2} µs", naive_avg_us);
    println!("Karatsuba: {:.2} µs", karatsuba_avg_us);
    println!("Speedup:   {:.2}×", speedup);
    println!("Time saved: {:.2} µs per encryption ({:.1}%)", time_saved, (time_saved / naive_avg_us) * 100.0);
    println!();

    println!("--- Comparison with Kyber-512 ---");
    println!("Kyber-512 encryption: ~10-20 µs (using NTT)");
    println!("Clifford-LWE-256 (Karatsuba): {:.2} µs", karatsuba_avg_us);
    println!("Overhead factor: {:.1}× to {:.1}×", karatsuba_avg_us / 20.0, karatsuba_avg_us / 10.0);
    println!();

    println!("--- Next Steps for Performance ---");
    if speedup > 1.2 {
        println!("✓ Karatsuba provides {:.2}× speedup over naive!", speedup);
    }
    println!("• Current bottleneck: Polynomial multiplication still slower than NTT");
    println!("• Core Clifford product: 48 ns (1.71× faster than 8×8 matrix)");
    println!("• To match Kyber: Need ~{}× further speedup", (karatsuba_avg_us / 15.0) as usize);
    println!("• Possible approaches:");
    println!("  - Optimize Karatsuba implementation (reduce allocations)");
    println!("  - SIMD acceleration for Clifford product");
    println!("  - Precomputation for fixed public key");
}
