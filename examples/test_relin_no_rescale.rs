//! Test relinearization without rescaling
//!
//! Verifies that after relinearization:
//! c0 + c1·s ≈ m1·m2·Δ² (with small error from EVK noise)

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt};
use ga_engine::clifford_fhe::rns::{rns_add, rns_sub, rns_multiply as rns_poly_multiply, decompose_base_pow2};

fn polynomial_multiply_ntt(a: &[i64], b: &[i64], q: i64, n: usize) -> Vec<i64> {
    let mut result = vec![0i128; n];
    let q128 = q as i128;
    for i in 0..n {
        for j in 0..n {
            let idx = i + j;
            let prod = (a[i] as i128) * (b[j] as i128);
            if idx < n {
                result[idx] += prod;
            } else {
                let wrapped_idx = idx % n;
                result[wrapped_idx] -= prod;
            }
        }
    }
    result.iter().map(|&x| {
        let r = x % q128;
        if r < 0 { (r + q128) as i64 } else { r as i64 }
    }).collect()
}

fn main() {
    println!("=== Relinearization Test (No Rescale) ===\n");

    // Setup: N=64, 2 primes, small noise
    let mut params = CliffordFHEParams::new_rns_mult();
    params.scale = 2f64.powi(20);
    params.n = 64;
    params.error_std = 3.2;  // Small but non-zero noise
    params.moduli = vec![params.moduli[0], params.moduli[1]];

    let n = params.n;
    let primes = &params.moduli;
    let delta = params.scale;

    println!("Parameters:");
    println!("  N = {}", n);
    println!("  Δ = {}", delta);
    println!("  σ = {}", params.error_std);
    println!();

    // Generate keys
    let (pk, sk, evk) = rns_keygen(&params);

    // Encrypt [2] and [3]
    let mut m1_coeffs = vec![0i64; n];
    m1_coeffs[0] = (2.0 * delta).round() as i64;
    let pt1 = RnsPlaintext::from_coeffs(m1_coeffs, delta, primes, 0);

    let mut m2_coeffs = vec![0i64; n];
    m2_coeffs[0] = (3.0 * delta).round() as i64;
    let pt2 = RnsPlaintext::from_coeffs(m2_coeffs, delta, primes, 0);

    let ct1 = rns_encrypt(&pk, &pt1, &params);
    let ct2 = rns_encrypt(&pk, &pt2, &params);

    // Tensor product
    println!("Step 1: Tensor product");
    let c0d0 = rns_poly_multiply(&ct1.c0, &ct2.c0, primes, polynomial_multiply_ntt);
    let c0d1 = rns_poly_multiply(&ct1.c0, &ct2.c1, primes, polynomial_multiply_ntt);
    let c1d0 = rns_poly_multiply(&ct1.c1, &ct2.c0, primes, polynomial_multiply_ntt);
    let c1d1 = rns_poly_multiply(&ct1.c1, &ct2.c1, primes, polynomial_multiply_ntt);

    let d0 = c0d0;
    let d1 = rns_add(&c0d1, &c1d0, primes);
    let d2 = c1d1;

    println!("  d0[0] residues: {:?}", &d0.rns_coeffs[0]);
    println!("  d1[0] residues: {:?}", &d1.rns_coeffs[0]);
    println!("  d2[0] residues: {:?}", &d2.rns_coeffs[0]);

    // Relinearization using gadget decomposition
    println!("\nStep 2: Relinearization");
    println!("  EVK base_w = {}", evk.base_w);
    println!("  EVK num_digits = {}", evk.evk0.len());

    // Decompose d2
    let d2_digits = decompose_base_pow2(&d2, primes, evk.base_w);
    println!("  d2_digits.len() = {}", d2_digits.len());

    for (t, digit) in d2_digits.iter().enumerate() {
        println!("  digit[{}][0] residues: {:?}", t, &digit.rns_coeffs[0][..2]);
    }

    // Relinearize: c0 = d0 - Σ d_t·evk0[t], c1 = d1 + Σ d_t·evk1[t]
    let mut c0 = d0.clone();
    let mut c1 = d1.clone();

    for t in 0..d2_digits.len() {
        let u0 = rns_poly_multiply(&d2_digits[t], &evk.evk0[t], primes, polynomial_multiply_ntt);
        let u1 = rns_poly_multiply(&d2_digits[t], &evk.evk1[t], primes, polynomial_multiply_ntt);

        println!("  u0[{}][0] residues: {:?}", t, &u0.rns_coeffs[0][..2]);
        println!("  u1[{}][0] residues: {:?}", t, &u1.rns_coeffs[0][..2]);

        c0 = rns_sub(&c0, &u0, primes);  // SUBTRACT (per expert's formula)
        c1 = rns_add(&c1, &u1, primes);  // ADD
    }

    println!("\nStep 3: Check result");
    println!("  c0[0] residues: {:?}", &c0.rns_coeffs[0]);
    println!("  c1[0] residues: {:?}", &c1.rns_coeffs[0]);

    // Decrypt: m' = c0 + c1·s
    let c1s = rns_poly_multiply(&c1, &sk.coeffs, primes, polynomial_multiply_ntt);
    let result = rns_add(&c0, &c1s, primes);

    println!("  c1·s[0] residues: {:?}", &c1s.rns_coeffs[0]);
    println!("  result[0] residues: {:?}", &result.rns_coeffs[0]);

    // Expected: 6·Δ²
    let expected = (6.0 * delta * delta).round() as i64;
    println!("\n  Expected (6·Δ²): {}", expected);

    // Decode from first prime
    let got = result.rns_coeffs[0][0];
    let centered = if got > primes[0] / 2 { got - primes[0] } else { got };

    println!("  Got (centered): {}", centered);
    println!("  Error: {}", (centered - expected).abs());

    let rel_error = ((centered - expected).abs() as f64) / (expected as f64);
    println!("  Relative error: {:.2e}", rel_error);

    if rel_error < 0.01 {  // 1% error tolerance
        println!("\n✅ Relinearization works!");
    } else {
        println!("\n❌ Relinearization FAILED!");
    }
}
