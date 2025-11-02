//! Test basic encryption/decryption WITH noise

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt};

fn main() {
    println!("=== Basic Encryption/Decryption WITH NOISE ===\n");

    let mut params = CliffordFHEParams::new_rns_mult();
    params.n = 64;
    params.scale = 2f64.powi(40);
    params.error_std = 3.2;  // Small noise

    params.moduli = vec![
        1152921504606851201,  // q0 ≈ 2^60
        1099511628161,        // q1 ≈ 2^40 ≈ Δ
    ];

    let delta = params.scale;
    let primes = &params.moduli;

    println!("Parameters:");
    println!("  Δ = {}", delta);
    println!("  error_std = {}", params.error_std);
    println!();

    let (pk, sk, _evk) = rns_keygen(&params);

    // Encrypt [2]
    let mut m_coeffs = vec![0i64; params.n];
    m_coeffs[0] = (2.0 * delta).round() as i64;
    let pt = RnsPlaintext::from_coeffs(m_coeffs, delta, primes, 0);

    println!("Plaintext:");
    println!("  m[0] = 2·Δ = {}", (2.0 * delta).round() as i64);
    println!();

    let ct = rns_encrypt(&pk, &pt, &params);
    println!("Ciphertext:");
    println!("  level = {}, scale = {:.2e}", ct.level, ct.scale);
    println!();

    // Decrypt
    let pt_dec = rns_decrypt(&sk, &ct, &params);

    let raw = pt_dec.coeffs.rns_coeffs[0][0];
    let decoded = (raw as f64) / ct.scale;

    println!("Decrypted:");
    println!("  Raw value: {}", raw);
    println!("  Decoded: {:.6}", decoded);
    println!("  Expected: 2.0");
    println!("  Error: {:.10}", (decoded - 2.0).abs());

    if (decoded - 2.0).abs() < 0.1 {
        println!("\n✅ PASS - Noise is reasonable");
    } else {
        println!("\n❌ FAIL - Noise is too large: {:.6}", (decoded - 2.0).abs());
    }
}
