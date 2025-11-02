//! Simplest possible RNS-CKKS test
//!
//! Test encrypt/decrypt with a scaled message

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt};

fn main() {
    println!("Testing RNS-CKKS with Scaled Message\n");

    let params = CliffordFHEParams::new_rns_mult();
    println!("Parameters:");
    println!("  Moduli: {:?}", params.moduli);
    println!("  modulus_at_level(0): {}", params.modulus_at_level(0));
    println!("  Scale: {:.2e}\n", params.scale);

    // Generate RNS keys
    println!("Generating RNS keys...");
    let (pk, sk, _evk) = rns_keygen(&params);
    println!("  Generated RNS keys (pk, sk, evk)\n");

    // Create a simple message: value 5.0
    let message_value = 5.0;

    // In CKKS, we encode as: coefficient = message * scale
    let scaled_coeff = (message_value * params.scale).round() as i64;

    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = scaled_coeff;  // Only first coefficient has the message

    println!("Message value: {}", message_value);
    println!("Scaled coefficient: {}", scaled_coeff);
    println!("Scale: {:.2e}\n", params.scale);

    // Create plaintext
    let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);

    // Encrypt
    println!("Encrypting...");
    let ct = rns_encrypt(&pk, &pt, &params);

    // Decrypt
    println!("Decrypting...");
    let pt_result = rns_decrypt(&sk, &ct, &params);

    // Extract coefficients using full CRT reconstruction
    // (single-prime won't work for large scaled values!)
    let coeffs_result = pt_result.to_coeffs(&params.moduli);

    // Decode: divide by scale to get message back
    let recovered_message = coeffs_result[0] as f64 / params.scale;

    println!("\nRecovered coefficient[0]: {}", coeffs_result[0]);
    println!("Recovered message value: {:.6}", recovered_message);
    println!("Original message value:  {:.6}", message_value);

    let error = (recovered_message - message_value).abs();
    println!("\nError: {:.6}", error);

    if error < 0.01 {
        println!("✓ PASS: RNS-CKKS encrypt/decrypt works!");
    } else {
        println!("✗ FAIL: Error too large!");
    }
}
