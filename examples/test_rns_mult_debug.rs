//! Debug RNS homomorphic multiplication

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt, rns_multiply_ciphertexts};

fn main() {
    let params = CliffordFHEParams::new_rns_mult();

    println!("Parameters:");
    println!("  Moduli: {:?}\n", params.moduli);

    let (pk, sk, evk) = rns_keygen(&params);

    // Encrypt [2] and [3]
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

    println!("Plaintext A RNS (coeff[0]): {:?}", pt_a.coeffs.rns_coeffs[0]);
    println!("Plaintext B RNS (coeff[0]): {:?}\n", pt_b.coeffs.rns_coeffs[0]);

    let ct_a = rns_encrypt(&pk, &pt_a, &params);
    let ct_b = rns_encrypt(&pk, &pt_b, &params);

    println!("Before multiplication:");
    println!("  ct_a.level = {}", ct_a.level);
    println!("  ct_b.level = {}", ct_b.level);
    println!("  ct_a.scale = {:.2e}", ct_a.scale);
    println!("  ct_b.scale = {:.2e}\n", ct_b.scale);

    let ct_result = rns_multiply_ciphertexts(&ct_a, &ct_b, &evk, &params);

    println!("After multiplication:");
    println!("  ct_result.level = {}", ct_result.level);
    println!("  ct_result.scale = {:.2e}", ct_result.scale);
    println!("  Active primes: {}\n", params.moduli.len() - ct_result.level);

    // Decrypt
    let pt_result = rns_decrypt(&sk, &ct_result, &params);

    println!("Decrypted plaintext:");
    println!("  pt_result.scale = {:.2e}", pt_result.scale);
    println!("  pt_result.level = {}", pt_result.coeffs.level);
    println!("  pt_result.num_primes = {}", pt_result.coeffs.num_primes());
    println!("  pt_result RNS (coeff[0]): {:?}\n", pt_result.coeffs.rns_coeffs[0]);

    // Get active primes
    let num_active = params.moduli.len() - ct_result.level;
    let active_primes = &params.moduli[..num_active];
    println!("Active primes: {:?}\n", active_primes);

    // Try to reconstruct
    println!("Attempting CRT reconstruction with {} active primes...", num_active);
    let coeffs_result = pt_result.to_coeffs(active_primes);

    println!("Recovered coefficient[0]: {}", coeffs_result[0]);

    let recovered_value = coeffs_result[0] as f64 / pt_result.scale;
    println!("Recovered value: {:.6}", recovered_value);
    println!("Expected value: {:.6}", value_a * value_b);
}
