//! Test RNS-CKKS homomorphic multiplication
//!
//! Encrypt [2] and [3], multiply homomorphically, decrypt to get [6]

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt, rns_multiply_ciphertexts};

fn main() {
    println!("========================================================================");
    println!("Test: RNS-CKKS Homomorphic Multiplication");
    println!("========================================================================\n");

    let params = CliffordFHEParams::new_rns_mult();

    println!("Parameters:");
    println!("  N = {}", params.n);
    println!("  Moduli: {:?}", params.moduli);
    println!("  Scale: {:.2e}\n", params.scale);

    // Generate keys
    println!("Generating RNS keys...");
    let (pk, sk, evk) = rns_keygen(&params);
    println!("  Done\n");

    // Test: [2] × [3] = [6]
    println!("Test: Encrypt [2] and [3], multiply homomorphically");
    println!("Expected result: [6]\n");

    // Create plaintexts
    let value_a = 2.0;
    let value_b = 3.0;

    let scaled_a = (value_a * params.scale).round() as i64;
    let scaled_b = (value_b * params.scale).round() as i64;

    let mut coeffs_a = vec![0i64; params.n];
    coeffs_a[0] = scaled_a;

    let mut coeffs_b = vec![0i64; params.n];
    coeffs_b[0] = scaled_b;

    let pt_a = RnsPlaintext::from_coeffs(coeffs_a, params.scale, &params.moduli, 0);
    let pt_b = RnsPlaintext::from_coeffs(coeffs_b, params.scale, &params.moduli, 0);

    println!("Plaintext A: value = {}, scaled = {}", value_a, scaled_a);
    println!("Plaintext B: value = {}, scaled = {}\n", value_b, scaled_b);

    // Encrypt
    println!("Encrypting...");
    let ct_a = rns_encrypt(&pk, &pt_a, &params);
    let ct_b = rns_encrypt(&pk, &pt_b, &params);
    println!("  ct_a scale: {:.2e}", ct_a.scale);
    println!("  ct_b scale: {:.2e}\n", ct_b.scale);

    // Homomorphic multiplication
    println!("Multiplying ciphertexts...");
    let ct_result = rns_multiply_ciphertexts(&ct_a, &ct_b, &evk, &params);
    println!("  Result scale: {:.2e}", ct_result.scale);
    println!("  Result level: {}\n", ct_result.level);

    // Decrypt
    println!("Decrypting result...");
    let pt_result = rns_decrypt(&sk, &ct_result, &params);
    println!("  Plaintext scale: {:.2e}", pt_result.scale);
    println!("  Plaintext level: {}", pt_result.coeffs.level);
    println!("  Plaintext num_primes: {}", pt_result.coeffs.num_primes());

    // CKKS decoding after rescale: use single-prime extraction + center-lift
    // (After rescale, value is back at scale Δ, which is << q0)
    let num_active_primes = params.moduli.len() - ct_result.level;
    let active_primes = &params.moduli[..num_active_primes];
    let q0 = active_primes[0];

    // DEBUG: Show RNS coefficients
    println!("  First coefficient in RNS:");
    for (i, &prime) in active_primes.iter().enumerate() {
        println!("    mod q_{}: {} (prime = {})", i, pt_result.coeffs.rns_coeffs[0][i], prime);
    }
    println!();

    println!("Active primes at level {}: {} primes", ct_result.level, num_active_primes);
    println!("Using single-prime extraction (q0 = {})\n", q0);

    // RNS decoding via CRT (keep as i128 to avoid overflow!)
    use ga_engine::clifford_fhe::rns::mod_inverse;

    let c0 = pt_result.coeffs.rns_coeffs[0][0] as i128;
    let c1 = pt_result.coeffs.rns_coeffs[0][1] as i128;
    let q0 = active_primes[0] as i128;
    let q1 = active_primes[1] as i128;
    let Q = q0 * q1;

    // CRT reconstruction
    let Q0 = Q / q0;  // = q1
    let Q1 = Q / q1;  // = q0
    let Q0_inv = mod_inverse(Q0, q0);
    let Q1_inv = mod_inverse(Q1, q1);

    let mut c = ((c0 * Q0 % Q) * Q0_inv % Q + (c1 * Q1 % Q) * Q1_inv % Q) % Q;

    println!("CRT reconstruction (before center-lift): {}", c);
    println!("Q = {} (q0 * q1)", Q);

    // Center-lift (keep as i128!)
    if c > Q / 2 {
        c = c - Q;
    }

    println!("After center-lift: {}", c);
    println!("Expected value (6 * scale): {:.0}", 6.0 * pt_result.scale);
    println!("Scale: {:.2e}", pt_result.scale);

    // Decode: convert to f64 and divide by scale
    let recovered_value = (c as f64) / pt_result.scale;

    println!("Recovered value: {:.6}", recovered_value);
    println!("Expected value: {:.6}", value_a * value_b);

    let error = (recovered_value - (value_a * value_b)).abs();
    println!("\nError: {:.6}", error);

    if error < 0.01 {
        println!("✓ PASS: RNS-CKKS homomorphic multiplication works!");
    } else {
        println!("✗ FAIL: Error too large");
    }

    println!("\n========================================================================");
}
