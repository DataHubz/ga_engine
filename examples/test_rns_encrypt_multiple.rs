//! Test RNS encrypt/decrypt multiple times to check noise levels

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt};

fn main() {
    println!("========================================================================");
    println!("Test: RNS Encrypt/Decrypt Noise Analysis");
    println!("========================================================================\n");

    let params = CliffordFHEParams::new_rns_mult();
    let primes = &params.moduli;

    println!("Parameters:");
    println!("  N = {}", params.n);
    println!("  Moduli: {:?}", primes);
    println!("  Scale: {:.2e}", params.scale);
    println!("  Error std: {}\n", params.error_std);

    // Generate keys once
    println!("Generating RNS keys...");
    let (pk, sk, _evk) = rns_keygen(&params);
    println!("  Done\n");

    // Test with a simple message value
    let message_value = 5.0;
    let scaled_coeff = (message_value * params.scale).round() as i64;

    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = scaled_coeff;

    let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);

    println!("Message value: {}", message_value);
    println!("Scaled coefficient: {}", scaled_coeff);
    println!("Plaintext RNS (coefficient[0]): {:?}\n", pt.coeffs.rns_coeffs[0]);

    // Run multiple encryptions to see noise distribution
    println!("Running 10 encrypt/decrypt cycles:\n");

    for i in 0..10 {
        let ct = rns_encrypt(&pk, &pt, &params);
        let pt_result = rns_decrypt(&sk, &ct, &params);

        let coeffs_result = pt_result.to_coeffs_single_prime(&primes);
        let recovered_value = coeffs_result[0] as f64 / params.scale;

        let error = recovered_value - message_value;
        let residue = pt_result.coeffs.rns_coeffs[0][0];
        let expected_residue = pt.coeffs.rns_coeffs[0][0];
        let residue_error = (residue as i64) - (expected_residue as i64);

        println!("  Run {}: residue = {} (expected {}), residue_error = {}, message_error = {:.6}",
                 i + 1, residue, expected_residue, residue_error, error);
    }

    println!("\n========================================================================");
}
