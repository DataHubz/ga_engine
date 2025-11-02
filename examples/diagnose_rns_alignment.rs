//! Diagnostic to check for RNS per-prime alignment issues
//!
//! This implements the checks suggested by the expert to identify
//! where modulus/residue array indexing diverges.

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt};
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
        if r < 0 {
            (r + q128) as i64
        } else {
            r as i64
        }
    }).collect()
}

/// CHECK 1: Per-prime independence verification
/// Each RNS component should compute independently
fn rns_mul_selfcheck(
    a: &RnsPolynomial,
    b: &RnsPolynomial,
    primes: &[i64],
    label: &str,
) -> RnsPolynomial {
    println!("\n[RNS MUL SELFCHECK: {}]", label);

    // Verify input dimensions match
    assert_eq!(a.num_primes(), primes.len(),
        "{}: a has {} primes but primes slice has {}", label, a.num_primes(), primes.len());
    assert_eq!(b.num_primes(), primes.len(),
        "{}: b has {} primes but primes slice has {}", label, b.num_primes(), primes.len());

    // Compute with rns_poly_multiply
    let c = rns_poly_multiply(a, b, primes, polynomial_multiply_ntt);

    println!("  Input a.num_primes: {}", a.num_primes());
    println!("  Input b.num_primes: {}", b.num_primes());
    println!("  Primes slice length: {}", primes.len());
    println!("  Result c.num_primes: {}", c.num_primes());

    // Independently recompute per prime and compare
    let mut mismatches = 0;
    for j in 0..primes.len() {
        let q = primes[j];

        // Extract coefficients for this prime
        let ai: Vec<i64> = (0..a.n).map(|i| a.rns_coeffs[i][j]).collect();
        let bi: Vec<i64> = (0..b.n).map(|i| b.rns_coeffs[i][j]).collect();

        // Compute expected product for this prime
        let expect = polynomial_multiply_ntt(&ai, &bi, q, a.n);

        // Compare coefficient-by-coefficient
        for i in 0..a.n.min(3) {  // Check first 3 coefficients
            let got = ((c.rns_coeffs[i][j] % q) + q) % q;
            let exp = ((expect[i] % q) + q) % q;

            if got != exp {
                println!("  ❌ MISMATCH at coeff[{}], prime_idx[{}] (q={})", i, j, q);
                println!("     Expected: {}", exp);
                println!("     Got:      {}", got);
                mismatches += 1;
            }
        }
    }

    if mismatches == 0 {
        println!("  ✅ All per-prime checks passed");
    } else {
        println!("  ❌ {} mismatches found!", mismatches);
    }

    c
}

/// CHECK 2: CRT consistency verification
/// Residues across primes should represent the same underlying value
fn check_crt_consistency(poly: &RnsPolynomial, primes: &[i64], label: &str) {
    println!("\n[CRT CONSISTENCY CHECK: {}]", label);
    println!("  Polynomial has {} primes, checking against slice of {}",
        poly.num_primes(), primes.len());

    if primes.len() < 2 {
        println!("  ⚠️  Need at least 2 primes for CRT check");
        return;
    }

    // Check first coefficient with 2 and 3 primes
    for &k in &[2usize, 3usize] {
        if k > primes.len() {
            continue;
        }

        println!("  Checking with {} primes:", k);

        // Get residues for first coefficient
        let residues: Vec<i64> = (0..k).map(|j| poly.rns_coeffs[0][j]).collect();
        let prime_slice = &primes[..k];

        println!("    Residues: {:?}", residues);
        println!("    Primes:   {:?}", prime_slice);

        // Simple 2-prime CRT reconstruction
        if k == 2 {
            let q0 = prime_slice[0] as i128;
            let q1 = prime_slice[1] as i128;
            let r0 = residues[0] as i128;
            let r1 = residues[1] as i128;

            // Find x such that x ≡ r0 (mod q0) and x ≡ r1 (mod q1)
            // x = r0 + q0 * ((r1 - r0) * q0^-1 mod q1)

            // Compute q0^-1 mod q1
            let q0_inv = mod_inverse(q0, q1);
            let diff = ((r1 - r0) % q1 + q1) % q1;
            let k_val = (diff * q0_inv) % q1;
            let x = r0 + q0 * k_val;

            // Verify reconstruction
            let check0 = ((x % q0) + q0) % q0;
            let check1 = ((x % q1) + q1) % q1;

            if check0 == r0 && check1 == r1 {
                println!("    ✅ CRT reconstruction: x = {} (matches both residues)", x);
            } else {
                println!("    ❌ CRT FAILED!");
                println!("       Reconstructed x = {}", x);
                println!("       x mod q0 = {} (expected {})", check0, r0);
                println!("       x mod q1 = {} (expected {})", check1, r1);
            }
        }
    }
}

/// Compute modular inverse using extended Euclidean algorithm
fn mod_inverse(a: i128, m: i128) -> i128 {
    let (mut old_r, mut r) = (a, m);
    let (mut old_s, mut s) = (1i128, 0i128);

    while r != 0 {
        let quotient = old_r / r;
        let temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        let temp_s = s;
        s = old_s - quotient * s;
        old_s = temp_s;
    }

    ((old_s % m) + m) % m
}

/// CHECK 3: Public key relation verification
fn check_pk_relation(params: &CliffordFHEParams) {
    println!("\n=== CHECK 3: PUBLIC KEY RELATION ===\n");

    let (pk, sk, _evk) = rns_keygen(&params);
    let primes = &params.moduli;

    // Compute b + a·s (should be small - just key generation noise)
    let as_poly = rns_poly_multiply(&pk.a, &sk.coeffs, primes, polynomial_multiply_ntt);
    let b_plus_as = rns_add(&pk.b, &as_poly, primes);

    // Check first few coefficients across all primes
    println!("Public key relation: b + a·s (should be small noise)");
    println!("Checking first 3 coefficients across all {} primes:\n", primes.len());

    let mut max_abs = 0i64;
    for i in 0..3 {
        print!("  coeff[{}]: ", i);
        for j in 0..primes.len() {
            let q = primes[j];
            let val = b_plus_as.rns_coeffs[i][j];

            // Center-lift
            let centered = if val > q / 2 { val - q } else { val };
            print!("{:4} ", centered);

            max_abs = max_abs.max(centered.abs());
        }
        println!();
    }

    println!("\nMax absolute value: {}", max_abs);

    if max_abs < 100 {
        println!("✅ Public key relation holds (small noise)");
    } else {
        println!("❌ Public key relation FAILED (noise too large: {})", max_abs);
    }
}

/// CHECK 4: Minimal reproducible test (N=8, 2 primes, no noise)
fn minimal_tensor_test() {
    println!("\n=== CHECK 4: MINIMAL TENSOR PRODUCT TEST ===");
    println!("Parameters: N=8, 2 primes, Δ=2^10, NO NOISE\n");

    // Create minimal parameters
    let mut params = CliffordFHEParams::new_rns_mult();
    params.n = 8;
    params.scale = 2f64.powi(10);  // Small scale
    params.error_std = 0.0;  // NO NOISE

    // Use only first 2 primes
    let two_primes = vec![params.moduli[0], params.moduli[1]];
    params.moduli = two_primes;

    println!("Primes: {:?}", params.moduli);
    println!("Scale:  {}", params.scale);
    println!("N:      {}", params.n);

    // Generate keys
    let (pk, sk, _evk) = rns_keygen(&params);
    let primes = &params.moduli;

    // First verify public key
    println!("\nVerifying public key relation...");
    let as_poly = rns_poly_multiply(&pk.a, &sk.coeffs, primes, polynomial_multiply_ntt);
    let b_plus_as = rns_add(&pk.b, &as_poly, primes);

    for i in 0..3 {
        print!("  b+a·s coeff[{}]: ", i);
        for j in 0..primes.len() {
            let val = b_plus_as.rns_coeffs[i][j];
            let centered = if val > primes[j] / 2 { val - primes[j] } else { val };
            print!("{:4} ", centered);
        }
        println!();
    }

    // Create plaintexts [2] and [3]
    let mut coeffs1 = vec![0i64; params.n];
    coeffs1[0] = (2.0 * params.scale).round() as i64;
    let pt1 = RnsPlaintext::from_coeffs(coeffs1, params.scale, primes, 0);

    let mut coeffs2 = vec![0i64; params.n];
    coeffs2[0] = (3.0 * params.scale).round() as i64;
    let pt2 = RnsPlaintext::from_coeffs(coeffs2, params.scale, primes, 0);

    println!("\nPlaintext 1 (value=2): coeff[0] = {}", pt1.coeffs.rns_coeffs[0][0]);
    println!("Plaintext 2 (value=3): coeff[0] = {}", pt2.coeffs.rns_coeffs[0][0]);

    // Encrypt
    let ct1 = rns_encrypt(&pk, &pt1, &params);
    let ct2 = rns_encrypt(&pk, &pt2, &params);

    println!("\nCiphertext 1:");
    println!("  c0[0] residues: {:?}", &ct1.c0.rns_coeffs[0]);
    println!("  c1[0] residues: {:?}", &ct1.c1.rns_coeffs[0]);

    println!("\nCiphertext 2:");
    println!("  c0[0] residues: {:?}", &ct2.c0.rns_coeffs[0]);
    println!("  c1[0] residues: {:?}", &ct2.c1.rns_coeffs[0]);

    // Tensor product with diagnostic checks
    println!("\n--- TENSOR PRODUCT WITH DIAGNOSTICS ---");

    let d0 = rns_mul_selfcheck(&ct1.c0, &ct2.c0, primes, "d0 = c0 × c0'");
    check_crt_consistency(&d0, primes, "d0");

    let c0d1 = rns_mul_selfcheck(&ct1.c0, &ct2.c1, primes, "c0 × c1'");
    let c1d0 = rns_mul_selfcheck(&ct1.c1, &ct2.c0, primes, "c1 × c0'");

    let d1 = rns_add(&c0d1, &c1d0, primes);
    check_crt_consistency(&d1, primes, "d1");

    let d2 = rns_mul_selfcheck(&ct1.c1, &ct2.c1, primes, "d2 = c1 × c1'");
    check_crt_consistency(&d2, primes, "d2");

    // Compute algebraic identity: d0 + d1·s + d2·s²
    println!("\n--- COMPUTING ALGEBRAIC IDENTITY ---");

    let s_squared = rns_mul_selfcheck(&sk.coeffs, &sk.coeffs, primes, "s²");
    let d1_s = rns_mul_selfcheck(&d1, &sk.coeffs, primes, "d1·s");
    let d2_s2 = rns_mul_selfcheck(&d2, &s_squared, primes, "d2·s²");

    let mut result = rns_add(&d0, &d1_s, primes);
    result = rns_add(&result, &d2_s2, primes);

    check_crt_consistency(&result, primes, "FINAL RESULT");

    // Extract and decode
    println!("\n--- FINAL RESULT ---");
    let result_val = result.rns_coeffs[0][0];
    let expected = (6.0 * params.scale * params.scale) as i64;

    println!("Result coeff[0] mod q0: {}", result_val);
    println!("Expected (6Δ²):         {}", expected);
    println!("Expected (6Δ²) mod q0:  {}", expected % primes[0]);

    let decoded = (result_val as f64) / (params.scale * params.scale);
    println!("\nDecoded value: {}", decoded);
    println!("Expected:      6.0");

    if (decoded - 6.0).abs() < 0.1 {
        println!("\n✅ ✅ ✅ MINIMAL TEST PASSED! ✅ ✅ ✅");
    } else {
        println!("\n❌ ❌ ❌ MINIMAL TEST FAILED! ❌ ❌ ❌");
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  RNS ALIGNMENT DIAGNOSTIC SUITE                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // CHECK 3: Public key relation (full parameters)
    let params = CliffordFHEParams::new_rns_mult();
    check_pk_relation(&params);

    // CHECK 4: Minimal reproducible test
    minimal_tensor_test();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  DIAGNOSTIC COMPLETE                                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
