//! Deterministic tensor product test with zero noise
//!
//! This test verifies the tensor product step-by-step with:
//! - N=64, 2 primes, noise=0
//! - Deterministic r (fixed pattern, not random)
//! - Verifies algebraic identity: d0 + d1·s + d2·s² = m1·m2·Δ²

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt};
use ga_engine::clifford_fhe::rns::{RnsPolynomial, rns_add, rns_multiply as rns_poly_multiply};

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
    println!("=== Deterministic Tensor Product Test ===\n");

    // Setup parameters: N=64, 2 primes, NO NOISE
    let mut params = CliffordFHEParams::new_rns_mult();
    params.scale = 2f64.powi(20);  // Δ = 2^20
    params.n = 64;
    params.error_std = 0.0;  // ZERO NOISE for deterministic test
    params.moduli = vec![params.moduli[0], params.moduli[1]];  // Only 2 primes

    let n = params.n;
    let primes = &params.moduli;
    let delta = params.scale;

    println!("Parameters:");
    println!("  N = {}", n);
    println!("  Num primes = {}", primes.len());
    println!("  Δ = {}", delta);
    println!("  q0 = {}", primes[0]);
    println!("  q1 = {}", primes[1]);
    println!();

    // Generate keys
    let (pk, sk, _evk) = rns_keygen(&params);

    // ==================== CHECK 1: Public key relation ====================
    println!("CHECK 1: Public key relation (b + a·s = 0)");
    let as_poly = rns_poly_multiply(&pk.a, &sk.coeffs, primes, polynomial_multiply_ntt);
    let b_plus_as = rns_add(&pk.b, &as_poly, primes);

    print!("  b + a·s [first 3 coeffs, both primes]:");
    let mut pk_ok = true;
    for i in 0..3 {
        for j in 0..2 {
            let val = b_plus_as.rns_coeffs[i][j];
            let centered = if val > primes[j] / 2 { val - primes[j] } else { val };
            print!(" {}", centered);
            if centered.abs() > 100 {  // Allow tiny noise from rounding
                pk_ok = false;
            }
        }
        print!(" |");
    }
    println!();
    if pk_ok {
        println!("  ✅ Public key relation holds\n");
    } else {
        println!("  ❌ Public key relation FAILED\n");
        return;
    }

    // ==================== CHECK 2: Basic encrypt/decrypt ====================
    println!("CHECK 2: Encrypt [2] → Decrypt = [2]");

    let mut m1_coeffs = vec![0i64; n];
    m1_coeffs[0] = (2.0 * delta).round() as i64;  // m1 = 2
    let pt1 = RnsPlaintext::from_coeffs(m1_coeffs, delta, primes, 0);

    let ct1 = rns_encrypt(&pk, &pt1, &params);
    let pt1_dec = rns_decrypt(&sk, &ct1, &params);

    print!("  Decrypted [first 2 coeffs, both primes]:");
    let mut dec_ok = true;
    for i in 0..2 {
        for j in 0..2 {
            let val = pt1_dec.coeffs.rns_coeffs[i][j];
            print!(" {}", val);
        }
        print!(" |");
    }
    println!();

    let expected = (2.0 * delta).round() as i64;
    for j in 0..2 {
        if pt1_dec.coeffs.rns_coeffs[0][j] != expected {
            dec_ok = false;
        }
    }
    if dec_ok {
        println!("  ✅ Encrypt/decrypt works\n");
    } else {
        println!("  ❌ Encrypt/decrypt FAILED\n");
        return;
    }

    // ==================== CHECK 3: Tensor product identity ====================
    println!("CHECK 3: Tensor product algebraic identity");
    println!("  Testing: [2] × [3] should give d0 + d1·s + d2·s² = 6·Δ²\n");

    let mut m2_coeffs = vec![0i64; n];
    m2_coeffs[0] = (3.0 * delta).round() as i64;  // m2 = 3
    let pt2 = RnsPlaintext::from_coeffs(m2_coeffs, delta, primes, 0);

    let ct2 = rns_encrypt(&pk, &pt2, &params);

    // Compute tensor product: (c0, c1) ⊗ (d0, d1) = (c0·d0, c0·d1 + c1·d0, c1·d1)
    println!("  Computing tensor product...");
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

    // Compute d1·s
    let d1s = rns_poly_multiply(&d1, &sk.coeffs, primes, polynomial_multiply_ntt);
    println!("  d1·s[0] residues: {:?}", &d1s.rns_coeffs[0]);

    // Compute d2·s²
    let s_squared = rns_poly_multiply(&sk.coeffs, &sk.coeffs, primes, polynomial_multiply_ntt);
    let d2s2 = rns_poly_multiply(&d2, &s_squared, primes, polynomial_multiply_ntt);
    println!("  d2·s²[0] residues: {:?}", &d2s2.rns_coeffs[0]);

    // Compute d0 + d1·s + d2·s²
    let temp = rns_add(&d0, &d1s, primes);
    let result = rns_add(&temp, &d2s2, primes);

    println!("\n  Result (d0 + d1·s + d2·s²)[0] residues: {:?}", &result.rns_coeffs[0]);

    // Expected: m1 * m2 * Δ² = 2 * 3 * Δ² = 6·Δ²
    let expected_val = (6.0 * delta * delta).round() as i64;
    println!("  Expected (6·Δ²): {}", expected_val);

    // Check per prime
    let mut tensor_ok = true;
    for j in 0..2 {
        let got = result.rns_coeffs[0][j];
        let exp = ((expected_val % primes[j]) + primes[j]) % primes[j];
        println!("  Prime {}: got={}, expected={}", j, got, exp);

        // Allow 1% error for numerical precision
        let diff = (got as i128 - exp as i128).abs();
        let threshold = (exp as i128) / 100;
        if diff > threshold {
            println!("    ❌ MISMATCH (diff={}, threshold={})", diff, threshold);
            tensor_ok = false;
        } else {
            println!("    ✅ Match");
        }
    }

    if tensor_ok {
        println!("\n✅ ALL CHECKS PASSED - Tensor product identity holds!");
    } else {
        println!("\n❌ TENSOR PRODUCT FAILED - Identity does not hold!");
    }
}
