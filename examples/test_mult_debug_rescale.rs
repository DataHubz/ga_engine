//! Debug rescaling specifically

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt, rns_multiply_ciphertexts};

fn main() {
    println!("=== DEBUG: CKKS Multiplication Rescaling ===\n");

    let mut params = CliffordFHEParams::new_rns_mult();
    params.n = 64;
    params.scale = 2f64.powi(40);
    params.error_std = 3.2;  // WITH noise to trigger the bug

    params.moduli = vec![
        1152921504606851201,  // q0 ≈ 2^60
        1099511628161,        // q1 ≈ 2^40 ≈ Δ
    ];

    let delta = params.scale;
    println!("Δ = {}", delta);
    println!("q0 = {}", params.moduli[0]);
    println!("q1 = {}", params.moduli[1]);
    println!();

    let (pk, sk, evk) = rns_keygen(&params);

    // Encrypt [2] and [3]
    let mut m1_coeffs = vec![0i64; params.n];
    m1_coeffs[0] = (2.0 * delta).round() as i64;
    let pt1 = RnsPlaintext::from_coeffs(m1_coeffs, delta, &params.moduli, 0);

    let mut m2_coeffs = vec![0i64; params.n];
    m2_coeffs[0] = (3.0 * delta).round() as i64;
    let pt2 = RnsPlaintext::from_coeffs(m2_coeffs, delta, &params.moduli, 0);

    let ct1 = rns_encrypt(&pk, &pt1, &params);
    let ct2 = rns_encrypt(&pk, &pt2, &params);

    // Multiply (includes rescaling)
    let ct_mult = rns_multiply_ciphertexts(&ct1, &ct2, &evk, &params);

    println!("After multiplication:");
    println!("  ct_mult.level = {}", ct_mult.level);
    println!("  ct_mult.scale = {:.2e}", ct_mult.scale);
    println!();

    // Decrypt
    let pt_mult = rns_decrypt(&sk, &ct_mult, &params);

    println!("Decrypted:");
    println!("  Number of residues: {}", pt_mult.coeffs.rns_coeffs[0].len());
    println!("  Raw value: {}", pt_mult.coeffs.rns_coeffs[0][0]);
    println!();

    // Try to decode
    let decoded = (pt_mult.coeffs.rns_coeffs[0][0] as f64) / ct_mult.scale;

    println!("Result:");
    println!("  Decoded: {:.6}", decoded);
    println!("  Expected: 6.0");
    println!("  Error: {:.6}", (decoded - 6.0).abs());
    println!();

    // Also try center-lifting
    let q0 = params.moduli[0];
    let raw = pt_mult.coeffs.rns_coeffs[0][0];
    let centered = if raw > q0 / 2 { raw - q0 } else { raw };
    let decoded_centered = (centered as f64) / ct_mult.scale;

    println!("With center-lifting:");
    println!("  Centered value: {}", centered);
    println!("  Decoded: {:.6}", decoded_centered);
    println!("  Error: {:.6}", (decoded_centered - 6.0).abs());
}
