//! Multiplication test with ZERO noise for exact debugging

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt, rns_multiply_ciphertexts};

fn main() {
    println!("=== CKKS Multiplication - ZERO NOISE TEST ===\n");

    let mut params = CliffordFHEParams::new_rns_mult();
    params.n = 64;
    params.scale = 2f64.powi(40);
    params.error_std = 0.0;  // ZERO NOISE

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

    // Encrypt m1=2, m2=3
    let m1_val = (2.0 * delta).round() as i64;
    let m2_val = (3.0 * delta).round() as i64;

    println!("m1 = 2·Δ = {}", m1_val);
    println!("m2 = 3·Δ = {}", m2_val);
    println!();

    let mut m1_coeffs = vec![0i64; params.n];
    m1_coeffs[0] = m1_val;
    let pt1 = RnsPlaintext::from_coeffs(m1_coeffs, delta, &params.moduli, 0);

    let mut m2_coeffs = vec![0i64; params.n];
    m2_coeffs[0] = m2_val;
    let pt2 = RnsPlaintext::from_coeffs(m2_coeffs, delta, &params.moduli, 0);

    let ct1 = rns_encrypt(&pk, &pt1, &params);
    let ct2 = rns_encrypt(&pk, &pt2, &params);

    // Multiply
    let ct_mult = rns_multiply_ciphertexts(&ct1, &ct2, &evk, &params);

    println!("After multiplication:");
    println!("  ct_mult.level = {}", ct_mult.level);
    println!("  ct_mult.scale = {}", ct_mult.scale);
    println!("  Expected scale = Δ = {}", delta);
    println!();

    // Decrypt
    let pt_mult = rns_decrypt(&sk, &ct_mult, &params);

    let raw = pt_mult.coeffs.rns_coeffs[0][0];
    let decoded = (raw as f64) / ct_mult.scale;

    println!("Decrypted:");
    println!("  Raw value: {}", raw);
    println!("  Decoded: {:.6}", decoded);
    println!("  Expected: 6.0");
    println!("  Error: {:.10}", (decoded - 6.0).abs());

    // Also check if raw value makes sense
    let expected_raw = (6.0 * delta).round() as i64;
    println!("\n  Expected raw (6·Δ): {}", expected_raw);
    println!("  Raw error: {}", (raw - expected_raw).abs());

    if (decoded - 6.0).abs() < 0.01 {
        println!("\n✅ PASS");
    } else {
        println!("\n❌ FAIL");
    }
}
