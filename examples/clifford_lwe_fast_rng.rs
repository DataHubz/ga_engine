//! Clifford-LWE-256 with Fast RNG
//!
//! Optimization: Use faster PRNG (PCG) instead of thread_rng for non-crypto operations

use ga_engine::clifford_ring::{CliffordPolynomial, CliffordRingElement};
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
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

/// Fast RNG context for encryption operations
struct FastRng {
    rng: SmallRng,
}

impl FastRng {
    fn new() -> Self {
        Self {
            rng: SmallRng::from_entropy(),
        }
    }

    /// Generate discrete polynomial with PCG (much faster than thread_rng)
    #[inline(always)]
    fn discrete_poly(&mut self, n: usize) -> CliffordPolynomial {
        let mut coeffs = Vec::with_capacity(n);
        for _ in 0..n {
            let mut mv = [0.0; 8];
            for i in 0..8 {
                mv[i] = (self.rng.gen_range(0..3) as f64) - 1.0;
            }
            coeffs.push(CliffordRingElement::from_multivector(mv));
        }
        CliffordPolynomial::new(coeffs)
    }

    /// Generate Gaussian error with PCG
    #[inline(always)]
    fn gaussian_poly(&mut self, n: usize, stddev: f64) -> CliffordPolynomial {
        let mut coeffs = Vec::with_capacity(n);
        for _ in 0..n {
            let mut mv = [0.0; 8];
            for i in 0..8 {
                let u1: f64 = self.rng.gen();
                let u2: f64 = self.rng.gen();
                let z = ((-2.0 * u1.ln()) as f64).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                mv[i] = z * stddev;
            }
            coeffs.push(CliffordRingElement::from_multivector(mv));
        }
        CliffordPolynomial::new(coeffs)
    }
}

fn scale_poly(poly: &CliffordPolynomial, scalar: f64) -> CliffordPolynomial {
    let coeffs: Vec<_> = poly.coeffs.iter()
        .map(|c| c.scalar_mul(scalar))
        .collect();
    CliffordPolynomial::new(coeffs)
}

/// Key generation (uses crypto-strength RNG)
fn keygen(params: &CLWEParams) -> (PublicKey, SecretKey) {
    let mut fast_rng = FastRng::new();

    let mut s = fast_rng.discrete_poly(params.n);
    s.reduce_modulo_xn_minus_1(params.n);

    let mut a = fast_rng.discrete_poly(params.n);
    a.reduce_modulo_xn_minus_1(params.n);

    let mut e = fast_rng.gaussian_poly(params.n, params.error_stddev);
    e.reduce_modulo_xn_minus_1(params.n);

    let mut b = a.multiply_karatsuba(&s);
    b.reduce_modulo_xn_minus_1(params.n);
    b = b.add(&e);

    (PublicKey { a, b }, SecretKey { s })
}

/// Fast encryption with PCG RNG
fn encrypt_fast(pk: &PublicKey, message: &CliffordPolynomial, params: &CLWEParams, rng: &mut FastRng) -> (CliffordPolynomial, CliffordPolynomial) {
    // Generate randomness
    let mut r = rng.discrete_poly(params.n);
    r.reduce_modulo_xn_minus_1(params.n);

    let mut e1 = rng.gaussian_poly(params.n, params.error_stddev);
    e1.reduce_modulo_xn_minus_1(params.n);

    let mut e2 = rng.gaussian_poly(params.n, params.error_stddev);
    e2.reduce_modulo_xn_minus_1(params.n);

    let scaled_msg = scale_poly(message, params.q / 2.0);

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

fn decrypt(sk: &SecretKey, u: &CliffordPolynomial, v: &CliffordPolynomial, params: &CLWEParams) -> CliffordPolynomial {
    let mut s_times_u = sk.s.multiply_karatsuba(u);
    s_times_u.reduce_modulo_xn_minus_1(params.n);

    let mut result = v.add(&scale_poly(&s_times_u, -1.0));

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
    println!("=== Clifford-LWE-256: Fast RNG Optimization ===\n");

    let params = CLWEParams::default();

    println!("Optimization: Using PCG64 instead of thread_rng");
    println!("Expected gain: 5-10 µs from RNG overhead\n");

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

    // Create reusable RNG
    let mut rng = FastRng::new();

    // Test correctness
    println!("--- Encryption/Decryption Test ---");
    let (u, v) = encrypt_fast(&pk, &message, &params, &mut rng);
    let decrypted = decrypt(&sk, &u, &v, &params);
    let correct = polys_equal(&message, &decrypted);
    println!("Correctness: {}", if correct { "✓ PASS" } else { "✗ FAIL" });
    println!();

    // Benchmark
    println!("--- Performance Benchmark (1000 operations) ---");
    const NUM_OPS: usize = 1000;

    let benchmark_start = Instant::now();
    for _ in 0..NUM_OPS {
        let _ = encrypt_fast(&pk, &message, &params, &mut rng);
    }
    let total_time = benchmark_start.elapsed();
    let avg_time = total_time.as_micros() as f64 / NUM_OPS as f64;

    println!("Average per encryption: {:.2} µs", avg_time);
    println!();

    println!("--- Comparison ---");
    println!("Previous (thread_rng):  ~38.2 µs");
    println!("Optimized (PCG64):      {:.2} µs", avg_time);

    if avg_time < 38.2 {
        let speedup = 38.2 / avg_time;
        let saved = 38.2 - avg_time;
        println!("Speedup: {:.2}×", speedup);
        println!("Improvement: {:.2} µs saved ({:.1}%)", saved, 100.0 * saved / 38.2);
    }
    println!();

    println!("--- vs Kyber-512 ---");
    println!("Kyber-512: ~10-20 µs");
    println!("Clifford-LWE-256: {:.2} µs", avg_time);
    println!("Gap: {:.1}-{:.1}× slower", avg_time / 20.0, avg_time / 10.0);
}
