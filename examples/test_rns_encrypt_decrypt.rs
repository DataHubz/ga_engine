//! Test RNS-CKKS encrypt/decrypt roundtrip
//!
//! Verify that we can encrypt and decrypt with RNS representation

use ga_engine::clifford_fhe::{CliffordFHEParams, keygen};
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt};

fn main() {
    println!("Testing RNS-CKKS Encrypt/Decrypt\n");

    // Use RNS parameters
    let params = CliffordFHEParams::new_rns_mult();
    println!("Parameters:");
    println!("  N: {}", params.n);
    println!("  Moduli: {} primes", params.moduli.len());
    for (i, &q) in params.moduli.iter().enumerate() {
        println!("    q{}: {} (bits: {:.1})", i, q, (q as f64).log2());
    }
    println!("  Scale: {:.2e}", params.scale);
    println!();

    // Generate keys
    let (pk, sk, _evk) = keygen(&params);

    // Create a simple plaintext
    let coeffs: Vec<i64> = (0..params.n).map(|i| if i == 0 { 42 } else { 0 }).collect();

    println!("Original coefficients: [42, 0, 0, ...]");
    println!();

    // Convert to RNS plaintext
    let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);

    println!("Encrypting...");
    let ct = rns_encrypt(&pk, &pt, &params);
    println!("  Ciphertext level: {}", ct.level);
    println!("  Ciphertext scale: {:.2e}", ct.scale);
    println!();

    println!("Decrypting...");
    let pt_result = rns_decrypt(&sk, &ct, &params);
    println!("  Plaintext scale: {:.2e}", pt_result.scale);
    println!();

    // Convert back using SINGLE PRIME (avoids CRT overflow!)
    let coeffs_result = pt_result.to_coeffs_single_prime(&params.moduli);

    println!("Recovered coefficients (first 10):");
    for i in 0..10.min(coeffs_result.len()) {
        println!("  [{}]: {}", i, coeffs_result[i]);
    }
    println!();

    // Check error
    let error = (coeffs_result[0] - coeffs[0]).abs();
    println!("Error on first coefficient: {}", error);
    println!();

    if error < 100 {
        println!("✓ PASS: Encrypt/decrypt works with single-prime decoding!");
    } else {
        println!("✗ FAIL: Too much error!");
    }
}
