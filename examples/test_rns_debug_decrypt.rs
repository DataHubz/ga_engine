//! Debug RNS decrypt to understand the values
//!
//! Add logging to see what's happening during decryption

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt};

fn main() {
    println!("========================================================================");
    println!("Debug: RNS Decrypt with Logging");
    println!("========================================================================\n");

    let params = CliffordFHEParams::new_rns_mult();
    let primes = &params.moduli;

    println!("Parameters:");
    println!("  Moduli: {:?}", primes);
    println!("  Scale: {:.2e}\n", params.scale);

    // Generate RNS keys
    println!("Generating RNS keys...");
    let (pk, sk, _evk) = rns_keygen(&params);
    println!("  Done\n");

    // Create a simple message: value 5.0
    let message_value = 5.0;
    let scaled_coeff = (message_value * params.scale).round() as i64;

    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = scaled_coeff;

    println!("Message: {}", message_value);
    println!("Scaled coefficient: {}\n", scaled_coeff);

    // Create plaintext
    let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);

    println!("Plaintext RNS form (coefficient[0]):");
    for (i, &q) in primes.iter().enumerate() {
        let residue = pt.coeffs.rns_coeffs[0][i];
        println!("  Prime {}: {} mod {} = {}", i, scaled_coeff, q, residue);
    }
    println!();

    // Encrypt
    println!("Encrypting...");
    let ct = rns_encrypt(&pk, &pt, &params);
    println!("  Done\n");

    println!("Ciphertext c0 RNS form (coefficient[0]):");
    for (i, &q) in primes.iter().enumerate() {
        let residue = ct.c0.rns_coeffs[0][i];
        println!("  Prime {}: {}", i, residue);
    }
    println!();

    println!("Ciphertext c1 RNS form (coefficient[0]):");
    for (i, &q) in primes.iter().enumerate() {
        let residue = ct.c1.rns_coeffs[0][i];
        println!("  Prime {}: {}", i, residue);
    }
    println!();

    // Decrypt
    println!("Decrypting...");
    let pt_result = rns_decrypt(&sk, &ct, &params);
    println!("  Done\n");

    println!("Decrypted plaintext RNS form (coefficient[0]):");
    for (i, &q) in primes.iter().enumerate() {
        let residue = pt_result.coeffs.rns_coeffs[0][i];
        println!("  Prime {}: {} (should be close to {} mod {})", i, residue, scaled_coeff, q);
    }
    println!();

    // Try single-prime extraction
    println!("Extracting using single prime (prime 0)...");
    let coeffs_single = pt_result.to_coeffs_single_prime(&primes);
    let recovered_single = coeffs_single[0] as f64 / params.scale;

    println!("  Single prime coefficient: {}", coeffs_single[0]);
    println!("  Single prime message: {:.6}", recovered_single);
    println!("  Error: {:.6}\n", (recovered_single - message_value).abs());

    // Check if the decrypted value is small (< q/2)
    let q0 = primes[0];
    let residue_0 = pt_result.coeffs.rns_coeffs[0][0];
    println!("Analysis:");
    println!("  First prime q0 = {}", q0);
    println!("  Residue mod q0 = {}", residue_0);
    println!("  q0/2 = {}", q0 / 2);

    if residue_0 < q0 / 2 {
        println!("  ✓ Residue is in range [0, q0/2), so single-prime works!");
        println!("  Expected coefficient: {}", scaled_coeff);
        println!("  Got coefficient: {}", residue_0);
        if (residue_0 - scaled_coeff).abs() < 1000 {
            println!("  ✓ Close to expected (within noise)!");
        } else {
            println!("  ✗ Too far from expected!");
        }
    } else {
        println!("  ⚠️  Residue is in range [q0/2, q0), will be center-lifted to negative");
        let centered = residue_0 - q0;
        println!("  Center-lifted: {}", centered);
        println!("  Expected: {}", scaled_coeff);
    }

    println!("\n========================================================================");
}
