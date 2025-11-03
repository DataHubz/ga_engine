//! Integration tests for Clifford FHE
//!
//! Run with: cargo test --test clifford_fhe_integration_tests

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{
    rns_encrypt, rns_decrypt, rns_add_ciphertexts, rns_multiply_ciphertexts,
    RnsPlaintext,
};

/// Helper to decode a value from RNS plaintext
fn decode_value(pt: &RnsPlaintext, scale: f64, all_primes: &[i64], level: usize) -> f64 {
    let active_primes = &all_primes[..all_primes.len() - level];

    // For now, ALWAYS use single-prime decoding (first prime)
    // This avoids CRT issues and works if message + noise << prime
    let val = pt.coeffs.rns_coeffs[0][0];
    let q = active_primes[0];
    let centered = if val > q / 2 { val - q } else { val };
    (centered as f64) / scale
}

#[test]
fn test_ntt_60bit_prime_basic() {
    // Test that 60-bit primes work with NTT
    let params = CliffordFHEParams {
        n: 1024,
        moduli: vec![1141392289560813569], // Single 60-bit prime
        scale: 2f64.powi(40),
        error_std: 3.2,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    };

    let q = params.moduli[0];
    let n = params.n;

    // Verify (q-1) divisible by 2n (required for NTT)
    assert_eq!((q - 1) % (2 * n as i64), 0, "q-1 must be divisible by 2n for NTT");
}

#[test]
fn test_single_prime_encryption_decryption() {
    let params = CliffordFHEParams {
        n: 1024,
        moduli: vec![1141392289560813569],
        scale: 2f64.powi(40),
        error_std: 3.2,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    };

    let (pk, sk, _evk) = rns_keygen(&params);

    // Test encrypting zero
    let coeffs_zero = vec![0i64; params.n];
    let pt_zero = RnsPlaintext::from_coeffs(coeffs_zero, params.scale, &params.moduli, 0);
    let ct_zero = rns_encrypt(&pk, &pt_zero, &params);
    let pt_dec_zero = rns_decrypt(&sk, &ct_zero, &params);

    // Check noise is reasonable (should be ~100, allow up to 1000)
    let noise = pt_dec_zero.coeffs.rns_coeffs[0][0];
    let q = params.moduli[0];
    let centered = if noise > q / 2 { noise - q } else { noise };
    let noise_magnitude = centered.abs();

    assert!(noise_magnitude < 1000, "Noise should be small (~100), got {}", noise_magnitude);
}

#[test]
fn test_two_prime_encryption_decryption() {
    // Use validated primes (old q1 = 1141173990025715713 was COMPOSITE!)
    let params = CliffordFHEParams {
        n: 1024,
        moduli: vec![
            1141392289560813569,  // q0 - 60-bit prime ✅
            1099511678977,        // q1 - 41-bit prime ✅
        ],
        scale: 2f64.powi(40),
        error_std: 3.2,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    };

    let (pk, sk, _evk) = rns_keygen(&params);

    let value = 1.5;
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = (value * params.scale).round() as i64;

    let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);
    let ct = rns_encrypt(&pk, &pt, &params);
    let pt_dec = rns_decrypt(&sk, &ct, &params);

    let decoded = decode_value(&pt_dec, params.scale, &params.moduli, 0);
    let error = (decoded - value).abs();

    assert!(error < 1e-6, "Decryption error too large: {}", error);
}

#[test]
fn test_homomorphic_addition() {
    // Use validated primes (old q1 = 1141173990025715713 was COMPOSITE!)
    let params = CliffordFHEParams {
        n: 1024,
        moduli: vec![
            1141392289560813569,  // q0 - 60-bit prime ✅
            1099511678977,        // q1 - 41-bit prime ✅
        ],
        scale: 2f64.powi(40),
        error_std: 3.2,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    };

    let (pk, sk, _evk) = rns_keygen(&params);

    let a = 1.5;
    let b = 2.7;
    let expected = a + b;

    let mut coeffs_a = vec![0i64; params.n];
    let mut coeffs_b = vec![0i64; params.n];
    coeffs_a[0] = (a * params.scale).round() as i64;
    coeffs_b[0] = (b * params.scale).round() as i64;

    let pt_a = RnsPlaintext::from_coeffs(coeffs_a, params.scale, &params.moduli, 0);
    let pt_b = RnsPlaintext::from_coeffs(coeffs_b, params.scale, &params.moduli, 0);

    let ct_a = rns_encrypt(&pk, &pt_a, &params);
    let ct_b = rns_encrypt(&pk, &pt_b, &params);

    let ct_sum = rns_add_ciphertexts(&ct_a, &ct_b, &params);
    let pt_sum = rns_decrypt(&sk, &ct_sum, &params);

    let result = decode_value(&pt_sum, params.scale, &params.moduli, ct_sum.level);
    let error = (result - expected).abs();

    assert!(error < 1e-6, "Addition error too large: {} (expected {}, got {})", error, expected, result);
}

#[test]
// FIXED: Now uses proper scaling primes (40-bit) for correct rescaling
fn test_homomorphic_multiplication() {
    // Use the corrected RNS-CKKS parameters with proper scaling primes
    // This has: 1 large prime (60-bit) + 2 scaling primes (40-bit ≈ Δ)
    let params = CliffordFHEParams::new_rns_mult();

    let (pk, sk, evk) = rns_keygen(&params);

    let a = 1.5;
    let b = 2.0;
    let expected = a * b;

    let mut coeffs_a = vec![0i64; params.n];
    let mut coeffs_b = vec![0i64; params.n];
    coeffs_a[0] = (a * params.scale).round() as i64;
    coeffs_b[0] = (b * params.scale).round() as i64;

    let pt_a = RnsPlaintext::from_coeffs(coeffs_a, params.scale, &params.moduli, 0);
    let pt_b = RnsPlaintext::from_coeffs(coeffs_b, params.scale, &params.moduli, 0);

    let ct_a = rns_encrypt(&pk, &pt_a, &params);
    let ct_b = rns_encrypt(&pk, &pt_b, &params);

    let ct_prod = rns_multiply_ciphertexts(&ct_a, &ct_b, &evk, &params);
    let pt_prod = rns_decrypt(&sk, &ct_prod, &params);

    let result = decode_value(&pt_prod, ct_prod.scale, &params.moduli, ct_prod.level);
    let error = (result - expected).abs();

    // After multiplication and rescaling, allow larger error
    assert!(error < 1e-3, "Multiplication error too large: {} (expected {}, got {})", error, expected, result);
}

#[test]
fn test_noise_growth() {
    // Use validated primes (old q1 = 1141173990025715713 was COMPOSITE!)
    let params = CliffordFHEParams {
        n: 1024,
        moduli: vec![
            1141392289560813569,  // q0 - 60-bit prime ✅
            1099511678977,        // q1 - 41-bit prime ✅
        ],
        scale: 2f64.powi(40),
        error_std: 3.2,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    };

    let (pk, sk, _evk) = rns_keygen(&params);

    let value = 1.0;
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = (value * params.scale).round() as i64;

    let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);
    let ct = rns_encrypt(&pk, &pt, &params);
    let pt_dec = rns_decrypt(&sk, &ct, &params);

    let dec1 = decode_value(&pt_dec, params.scale, &params.moduli, 0);
    let noise1 = ((dec1 - value) * params.scale).abs() as i128;

    // After one addition
    let ct_sum = rns_add_ciphertexts(&ct, &ct, &params);
    let pt_sum = rns_decrypt(&sk, &ct_sum, &params);
    let dec2 = decode_value(&pt_sum, params.scale, &params.moduli, ct_sum.level);
    let noise2 = ((dec2 - 2.0 * value) * params.scale).abs() as i128;

    // Noise should be reasonable (< 10000)
    assert!(noise1 < 10000, "Initial noise too large: {}", noise1);
    assert!(noise2 < 20000, "Noise after addition too large: {}", noise2);
}
