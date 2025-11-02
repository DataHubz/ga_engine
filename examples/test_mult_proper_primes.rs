//! Test CKKS multiplication with PROPER prime chain
//!
//! Uses: Δ = 2^40, primes = [q60, q40_special]
//! This ensures rescaling works correctly: (6·Δ²) / Δ ≈ 6·Δ

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt, rns_multiply_ciphertexts};

fn main() {
    println!("=== CKKS Multiplication with Proper Prime Chain ===\n");

    // CORRECT SETUP: Δ = 2^40, primes chosen for rescaling
    let mut params = CliffordFHEParams::new_rns_mult();
    params.n = 64;
    params.scale = 2f64.powi(40);  // Δ = 2^40 = 1099511627776
    params.error_std = 3.2;  // TESTING: Try standard CKKS noise level

    // Prime chain: [q60, q40_special]
    // q60 for base ciphertext, q40 ≈ Δ for rescaling
    params.moduli = vec![
        1152921504606851201,  // 60-bit, ≡ 1 (mod 128)
        1099511628161,        // 40-bit, ≡ 1 (mod 128), ≈ Δ
    ];

    let delta = params.scale;
    let primes = &params.moduli;

    println!("Parameters:");
    println!("  N = {}", params.n);
    println!("  Δ = {} = 2^40", delta);
    println!("  q0 (base) = {} ≈ 2^{:.1}", primes[0], (primes[0] as f64).log2());
    println!("  q1 (special) = {} ≈ 2^{:.1}", primes[1], (primes[1] as f64).log2());
    println!("  q1 / Δ = {:.2}", (primes[1] as f64) / delta);
    println!();

    // Generate keys
    let (pk, sk, evk) = rns_keygen(&params);

    // Test: [2] × [3] = [6]
    println!("Test: [2] × [3] = [6]\n");

    // Encrypt [2]
    let mut m1_coeffs = vec![0i64; params.n];
    m1_coeffs[0] = (2.0 * delta).round() as i64;
    let pt1 = RnsPlaintext::from_coeffs(m1_coeffs, delta, primes, 0);

    let ct1 = rns_encrypt(&pk, &pt1, &params);
    println!("ct1 (encrypts 2):");
    println!("  level = {}, scale = {:.2e}", ct1.level, ct1.scale);

    // Decrypt to verify
    let pt1_dec = rns_decrypt(&sk, &ct1, &params);
    let val1 = (pt1_dec.coeffs.rns_coeffs[0][0] as f64) / delta;
    println!("  Decrypt: {:.6}", val1);

    // Encrypt [3]
    let mut m2_coeffs = vec![0i64; params.n];
    m2_coeffs[0] = (3.0 * delta).round() as i64;
    let pt2 = RnsPlaintext::from_coeffs(m2_coeffs, delta, primes, 0);

    let ct2 = rns_encrypt(&pk, &pt2, &params);
    println!("\nct2 (encrypts 3):");
    println!("  level = {}, scale = {:.2e}", ct2.level, ct2.scale);

    let pt2_dec = rns_decrypt(&sk, &ct2, &params);
    let val2 = (pt2_dec.coeffs.rns_coeffs[0][0] as f64) / delta;
    println!("  Decrypt: {:.6}", val2);

    // Multiply
    println!("\n--- Performing Multiplication ---");
    let ct_mult = rns_multiply_ciphertexts(&ct1, &ct2, &evk, &params);

    println!("\nct_mult (encrypts 2×3=6):");
    println!("  level = {}, scale = {:.2e}", ct_mult.level, ct_mult.scale);
    println!("  Expected scale: Δ²/q1 = (2^40)² / 2^40 ≈ 2^40 = {:.2e}", delta);

    // Decrypt
    let pt_mult = rns_decrypt(&sk, &ct_mult, &params);

    println!("\nAfter decryption:");
    println!("  pt_mult.level = {}", pt_mult.coeffs.level);
    println!("  Number of residues: {}", pt_mult.coeffs.rns_coeffs[0].len());
    println!("  Raw value (first residue): {}", pt_mult.coeffs.rns_coeffs[0][0]);
    if pt_mult.coeffs.rns_coeffs[0].len() > 1 {
        println!("  Raw value (second residue): {}", pt_mult.coeffs.rns_coeffs[0][1]);
    }
    println!("  ct_mult.scale = {:.2e}", ct_mult.scale);
    println!("  pt_mult.scale = {:.2e}", pt_mult.scale);

    // Decode
    let decoded = (pt_mult.coeffs.rns_coeffs[0][0] as f64) / ct_mult.scale;

    println!("\nResult:");
    println!("  Raw value: {}", pt_mult.coeffs.rns_coeffs[0][0]);
    println!("  Decoded (value / scale): {:.6}", decoded);
    println!("  Expected: 6.0");
    println!("  Error: {:.6}", (decoded - 6.0).abs());
    println!("  Relative error: {:.2e}", (decoded - 6.0).abs() / 6.0);

    if (decoded - 6.0).abs() < 0.1 {
        println!("\n✅ MULTIPLICATION WORKS!");
        println!("   With proper prime chain, rescaling is correct!");
    } else {
        println!("\n❌ MULTIPLICATION FAILED");
        println!("   Error too large: {:.6}", (decoded - 6.0).abs());
    }
}
