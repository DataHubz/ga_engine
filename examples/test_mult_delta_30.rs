//! Test with Δ = 2^30 to see if noise scaling is the issue

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt, rns_multiply_ciphertexts};

fn main() {
    println!("=== Test with Δ = 2^30 ===\n");

    let mut params = CliffordFHEParams::new_rns_mult();
    params.n = 64;
    params.scale = 2f64.powi(30);  // Try 2^30 instead of 2^40
    params.error_std = 3.2;

    // Find primes ≈ 2^30 for rescaling
    // Using smaller primes for now
    params.moduli = vec![
        1152921504606851201,  // q0 ≈ 2^60 (keep this large for base)
        1073741909,           // q1 ≈ 2^30 (close to Δ)
    ];

    let delta = params.scale;
    println!("Δ = {} = 2^30", delta);
    println!("q1 / Δ = {:.2}", (params.moduli[1] as f64) / delta);
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

    // Verify encryption works
    let pt1_dec = rns_decrypt(&sk, &ct1, &params);
    let val1 = (pt1_dec.coeffs.rns_coeffs[0][0] as f64) / delta;
    println!("ct1 decrypts to: {:.6}", val1);

    let pt2_dec = rns_decrypt(&sk, &ct2, &params);
    let val2 = (pt2_dec.coeffs.rns_coeffs[0][0] as f64) / delta;
    println!("ct2 decrypts to: {:.6}", val2);
    println!();

    // Multiply
    let ct_mult = rns_multiply_ciphertexts(&ct1, &ct2, &evk, &params);

    println!("After multiplication:");
    println!("  level = {}, scale = {:.2e}", ct_mult.level, ct_mult.scale);
    println!();

    // Decrypt
    let pt_mult = rns_decrypt(&sk, &ct_mult, &params);
    let raw = pt_mult.coeffs.rns_coeffs[0][0];
    let decoded = (raw as f64) / ct_mult.scale;

    println!("Result:");
    println!("  Raw value: {}", raw);
    println!("  Decoded: {:.6}", decoded);
    println!("  Expected: 6.0");
    println!("  Error: {:.6}", (decoded - 6.0).abs());

    if (decoded - 6.0).abs() < 0.1 {
        println!("\n✅ WORKS with Δ = 2^30!");
    } else {
        println!("\n❌ Still fails with Δ = 2^30");
    }
}
