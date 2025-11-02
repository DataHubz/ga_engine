//! Test RNS encrypt/decrypt at level 1 (with 2 primes instead of 3)

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt};
use ga_engine::clifford_fhe::rns::rns_rescale;

fn main() {
    let params = CliffordFHEParams::new_rns_mult();

    println!("Test: RNS encrypt/decrypt at level 1\n");

    let (pk, sk, _evk) = rns_keygen(&params);

    // Create plaintext at level 0 (all 3 primes)
    let value = 5.0;
    let scaled = (value * params.scale).round() as i64;

    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = scaled;

    let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);

    println!("Original plaintext (level 0):");
    println!("  Value: {}", value);
    println!("  Scaled: {}", scaled);
    println!("  RNS: {:?}\n", pt.coeffs.rns_coeffs[0]);

    // Encrypt
    let ct = rns_encrypt(&pk, &pt, &params);

    println!("Ciphertext (level 0):");
    println!("  Level: {}", ct.level);
    println!("  Scale: {:.2e}\n", ct.scale);

    // Manually rescale to level 1
    println!("Rescaling to level 1...");
    let ct_c0_rescaled = rns_rescale(&ct.c0, &params.moduli);
    let ct_c1_rescaled = rns_rescale(&ct.c1, &params.moduli);

    // Create ciphertext at level 1
    let q_last = params.moduli[params.moduli.len() - 1];
    let new_scale = ct.scale / (q_last as f64);

    use ga_engine::clifford_fhe::ckks_rns::RnsCiphertext;
    let ct_level1 = RnsCiphertext::new(ct_c0_rescaled, ct_c1_rescaled, 1, new_scale);

    println!("  New level: {}", ct_level1.level);
    println!("  New scale: {:.2e}", ct_level1.scale);
    println!("  Active primes: {}\n", params.moduli.len() - ct_level1.level);

    // Decrypt at level 1
    let pt_result = rns_decrypt(&sk, &ct_level1, &params);

    println!("Decrypted plaintext (level 1):");
    println!("  Num primes: {}", pt_result.coeffs.num_primes());
    println!("  RNS: {:?}", pt_result.coeffs.rns_coeffs[0]);
    println!("  Scale: {:.2e}\n", pt_result.scale);

    // Reconstruct with active primes
    let active_primes = &params.moduli[..2];
    println!("Active primes: {:?}\n", active_primes);

    let coeffs_result = pt_result.to_coeffs(active_primes);
    let recovered_value = coeffs_result[0] as f64 / pt_result.scale;

    println!("Recovered coefficient: {}", coeffs_result[0]);
    println!("Recovered value: {:.6}", recovered_value);
    println!("Expected value: {:.6}", value);
    println!("Error: {:.6}", (recovered_value - value).abs());
}
