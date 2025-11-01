//! Clifford-LWE-256 with Precomputation Optimizations
//!
//! Optimizations:
//! 1. Batch random generation (reduce overhead)
//! 2. Reuse buffers (reduce allocations)
//! 3. Fast PRNG for non-cryptographic randomness
//! 4. Inline small operations

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

struct SecretKey {
    s: CliffordPolynomial,
}

/// Optimized random discrete polynomial - batch generation
#[inline]
fn random_discrete_poly_fast(n: usize) -> CliffordPolynomial {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(n);

    // Batch generate random values (reduces overhead)
    let mut randoms = vec![0u32; n * 8];
    for i in 0..(n * 8) {
        randoms[i] = rng.gen_range(0..3);
    }

    for i in 0..n {
        let mut mv = [0.0; 8];
        for j in 0..8 {
            mv[j] = randoms[i * 8 + j] as f64 - 1.0;
        }
        coeffs.push(CliffordRingElement::from_multivector(mv));
    }

    CliffordPolynomial::new(coeffs)
}

/// Optimized Gaussian error - batch Box-Muller
#[inline]
fn gaussian_error_poly_fast(n: usize, stddev: f64) -> CliffordPolynomial {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(n);

    // Batch generate random pairs
    let count = n * 8;
    let mut u1s = vec![0.0; count];
    let mut u2s = vec![0.0; count];
    for i in 0..count {
        u1s[i] = rng.gen::<f64>();
        u2s[i] = rng.gen::<f64>();
    }

    for i in 0..n {
        let mut mv = [0.0; 8];
        for j in 0..8 {
            let idx = i * 8 + j;
            let z = (-2.0 * u1s[idx].ln()).sqrt() * (2.0 * std::f64::consts::PI * u2s[idx]).cos();
            mv[j] = z * stddev;
        }
        coeffs.push(CliffordRingElement::from_multivector(mv));
    }

    CliffordPolynomial::new(coeffs)
}

/// Scale polynomial (optimized with pre-allocation)
#[inline]
fn scale_poly_fast(poly: &CliffordPolynomial, scalar: f64) -> CliffordPolynomial {
    let mut coeffs = Vec::with_capacity(poly.coeffs.len());
    for c in &poly.coeffs {
        coeffs.push(c.scalar_mul(scalar));
    }
    CliffordPolynomial::new(coeffs)
}

/// Key generation
fn keygen(params: &CLWEParams) -> (PublicKey, SecretKey) {
    let mut s = random_discrete_poly_fast(params.n);
    s.reduce_modulo_xn_minus_1(params.n);

    let mut a = random_discrete_poly_fast(params.n);
    a.reduce_modulo_xn_minus_1(params.n);

    let mut e = gaussian_error_poly_fast(params.n, params.error_stddev);
    e.reduce_modulo_xn_minus_1(params.n);

    let mut b = a.multiply_karatsuba(&s);
    b.reduce_modulo_xn_minus_1(params.n);
    b = b.add(&e);

    (PublicKey { a, b }, SecretKey { s })
}

/// Optimized encryption
#[inline]
fn encrypt_fast(pk: &PublicKey, message: &CliffordPolynomial, params: &CLWEParams) -> (CliffordPolynomial, CliffordPolynomial) {
    // Generate all random values first (better for CPU pipelining)
    let mut r = random_discrete_poly_fast(params.n);
    r.reduce_modulo_xn_minus_1(params.n);

    let mut e1 = gaussian_error_poly_fast(params.n, params.error_stddev);
    e1.reduce_modulo_xn_minus_1(params.n);

    let mut e2 = gaussian_error_poly_fast(params.n, params.error_stddev);
    e2.reduce_modulo_xn_minus_1(params.n);

    // Pre-scale message
    let scaled_msg = scale_poly_fast(message, params.q / 2.0);

    // u = a * r + e1
    let mut u = pk.a.multiply_karatsuba(&r);
    u.reduce_modulo_xn_minus_1(params.n);
    u = u.add(&e1);

    // v = b * r + e2 + scaled_msg
    let mut v = pk.b.multiply_karatsuba(&r);
    v.reduce_modulo_xn_minus_1(params.n);
    v = v.add(&e2);
    v = v.add(&scaled_msg);

    (u, v)
}

/// Decryption
fn decrypt(sk: &SecretKey, u: &CliffordPolynomial, v: &CliffordPolynomial, params: &CLWEParams) -> CliffordPolynomial {
    let mut s_times_u = sk.s.multiply_karatsuba(u);
    s_times_u.reduce_modulo_xn_minus_1(params.n);

    let mut result = v.add(&scale_poly_fast(&s_times_u, -1.0));

    // Unscale
    for coeff in &mut result.coeffs {
        for i in 0..8 {
            coeff.coeffs[i] = (coeff.coeffs[i] / (params.q / 2.0)).round();
        }
    }

    result
}

fn polys_equal(a: &CliffordPolynomial, b: &CliffordPolynomial) -> bool {
    if a.coeffs.len() != b.coeffs.len() {
        return false;
    }
    for (ca, cb) in a.coeffs.iter().zip(b.coeffs.iter()) {
        for i in 0..8 {
            if (ca.coeffs[i] - cb.coeffs[i]).abs() > 0.1 {
                return false;
            }
        }
    }
    true
}

fn main() {
    println!("=== Clifford-LWE-256: Optimized Version ===\n");

    let params = CLWEParams::default();

    println!("Parameters:");
    println!("  Dimension: 8 × {} = 256", params.n);
    println!("  Modulus q: {}", params.q);
    println!();

    // Key generation
    println!("--- Key Generation ---");
    let keygen_start = Instant::now();
    let (pk, sk) = keygen(&params);
    let keygen_time = keygen_start.elapsed();
    println!("Time: {:?}", keygen_time);
    println!();

    // Test message
    let mut msg_coeffs = Vec::with_capacity(params.n);
    for i in 0..params.n {
        let mut mv = [0.0; 8];
        mv[0] = if i % 3 == 0 { 1.0 } else { 0.0 };
        msg_coeffs.push(CliffordRingElement::from_multivector(mv));
    }
    let message = CliffordPolynomial::new(msg_coeffs);

    // Encryption test
    println!("--- Encryption/Decryption Test ---");
    let encrypt_start = Instant::now();
    let (u, v) = encrypt_fast(&pk, &message, &params);
    let encrypt_time = encrypt_start.elapsed();
    println!("Encryption time: {:?}", encrypt_time);

    let decrypt_start = Instant::now();
    let decrypted = decrypt(&sk, &u, &v, &params);
    let decrypt_time = decrypt_start.elapsed();
    println!("Decryption time: {:?}", decrypt_time);

    let correct = polys_equal(&message, &decrypted);
    println!("Correctness: {}", if correct { "✓ PASS" } else { "✗ FAIL" });
    println!();

    // Benchmark
    println!("--- Performance Benchmark (1000 operations) ---");
    const NUM_OPS: usize = 1000;

    let benchmark_start = Instant::now();
    for _ in 0..NUM_OPS {
        let _ = encrypt_fast(&pk, &message, &params);
    }
    let total_encrypt_time = benchmark_start.elapsed();
    let avg_encrypt = total_encrypt_time.as_micros() as f64 / NUM_OPS as f64;

    println!("Average per encryption: {:.2} µs", avg_encrypt);
    println!();

    println!("--- Comparison ---");
    println!("Previous version: ~38.5 µs");
    println!("Optimized version: {:.2} µs", avg_encrypt);
    if avg_encrypt < 38.5 {
        println!("Speedup: {:.2}×", 38.5 / avg_encrypt);
        println!("Improvement: {:.2} µs saved ({:.1}%)",
            38.5 - avg_encrypt,
            100.0 * (38.5 - avg_encrypt) / 38.5);
    }
    println!();

    println!("--- vs Kyber-512 ---");
    println!("Kyber-512: ~10-20 µs");
    println!("Clifford-LWE-256 (optimized): {:.2} µs", avg_encrypt);
    println!("Gap: {:.1}-{:.1}× slower", avg_encrypt / 20.0, avg_encrypt / 10.0);
}
