//! Minimal encryption/decryption test to isolate the bug
// V1-specific test - only compile when v1 is enabled and v2 is NOT enabled
#![cfg(all(feature = "v1", not(feature = "v2")))]


use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt};

#[test]
fn test_minimal_enc_dec() {
    let mut params = CliffordFHEParams::new_rns_mult();
    params.scale = 1024.0; // Small scale for testing

    let (pk, sk, _evk) = rns_keygen(&params);

    // Encrypt simple value
    let value = 5.0;
    let scaled = (value * params.scale).round() as i64;
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = scaled;

    println!("\n=== INPUT ===");
    println!("Value: {}", value);
    println!("Scale: {}", params.scale);
    println!("Scaled coefficient: {}", scaled);

    let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);

    println!("\n=== PLAINTEXT RNS ===");
    println!("pt.coeffs.rns_coeffs[0]: {:?}", &pt.coeffs.rns_coeffs[0][..3]);

    let ct = rns_encrypt(&pk, &pt, &params);

    println!("\n=== CIPHERTEXT ===");
    println!("ct.c0.rns_coeffs[0]: {:?}", &ct.c0.rns_coeffs[0][..3]);
    println!("ct.c1.rns_coeffs[0]: {:?}", &ct.c1.rns_coeffs[0][..3]);

    // Manual decryption to see intermediate values
    use ga_engine::clifford_fhe::rns::rns_multiply as rns_poly_multiply;
    use ga_engine::clifford_fhe::rns::rns_sub;

    fn polynomial_multiply_ntt(a: &[i64], b: &[i64], q: i64, n: usize) -> Vec<i64> {
        let mut result = vec![0i128; n];
        let q128 = q as i128;
        for i in 0..n {
            for j in 0..n {
                let idx = i + j;
                let prod = (a[i] as i128) * (b[j] as i128) % q128;
                if idx < n {
                    result[idx] = (result[idx] + prod) % q128;
                } else {
                    let wrapped_idx = idx % n;
                    result[wrapped_idx] = (result[wrapped_idx] - prod) % q128;
                }
            }
        }
        result.iter().map(|&x| ((x % q128 + q128) % q128) as i64).collect()
    }

    let c1s = rns_poly_multiply(&ct.c1, &sk.coeffs, &params.moduli, polynomial_multiply_ntt);

    println!("\n=== DECRYPTION INTERMEDIATE ===");
    println!("c1·s residues[0]: {:?}", &c1s.rns_coeffs[0][..3]);

    let m_prime = rns_sub(&ct.c0, &c1s, &params.moduli);

    println!("c0 - c1·s residues[0]: {:?}", &m_prime.rns_coeffs[0][..3]);

    let recovered_coeffs = m_prime.to_coeffs_single_prime(0, params.moduli[0]);
    let recovered_value = (recovered_coeffs[0] as f64) / params.scale;

    println!("\n=== RESULT ===");
    println!("Recovered coefficient: {}", recovered_coeffs[0]);
    println!("Recovered value: {}", recovered_value);
    println!("Expected value: {}", value);
    println!("Error: {}", (recovered_value - value).abs());

    assert!((recovered_value - value).abs() < 1.0,
           "Encryption/decryption should recover value (got {}, expected {})", recovered_value, value);
}
