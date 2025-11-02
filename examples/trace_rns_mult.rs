//! Trace RNS values step-by-step through multiplication pipeline

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt};
use ga_engine::clifford_fhe::rns::mod_inverse;

fn main() {
    let params = CliffordFHEParams::new_rns_mult();
    let (pk, sk, evk) = rns_keygen(&params);

    println!("=== STEP 1: Encode [2] ===\n");
    let value = 2.0;
    let scaled = (value * params.scale).round() as i64;
    println!("Value: {}", value);
    println!("Scale Δ: {:.2e}", params.scale);
    println!("Scaled value (2*Δ): {}\n", scaled);

    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = scaled;
    let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);

    println!("RNS representation of plaintext[0]:");
    for (j, &q) in params.moduli.iter().enumerate() {
        println!("  mod q_{}: {} (prime = {})", j, pt.coeffs.rns_coeffs[0][j], q);
    }

    println!("\n=== STEP 2: Encrypt ===\n");
    let ct = rns_encrypt(&pk, &pt, &params);
    println!("Ciphertext level: {}", ct.level);
    println!("Ciphertext scale: {:.2e}\n", ct.scale);

    println!("=== STEP 3: Decrypt ===\n");
    let pt_dec = rns_decrypt(&sk, &ct, &params);
    println!("Decrypted level: {}", pt_dec.coeffs.level);
    println!("Decrypted scale: {:.2e}", pt_dec.scale);
    println!("Decrypted RNS coefficients[0]:");
    for (j, &q) in params.moduli.iter().enumerate() {
        println!("  mod q_{}: {}", j, pt_dec.coeffs.rns_coeffs[0][j]);
    }

    // In CKKS, message + noise is small relative to any single prime
    // So we can decode from a single prime's residue (standard CKKS approach)
    let c = pt_dec.coeffs.to_coeffs_single_prime(0, params.moduli[0])[0];

    println!("\nDecoding from single prime (standard CKKS):");
    println!("  Coefficient[0] mod q_0: {}", c);
    println!("  Expected (2*Δ): {}", scaled);
    println!("  Recovered value: {:.6}", (c as f64) / pt_dec.scale);
    println!("  Expected value: {:.6}", value);

    let enc_dec_error = ((c as f64) / pt_dec.scale - value).abs();
    if enc_dec_error < 0.01 {
        println!("  ✓ Encrypt/Decrypt works correctly!\n");
    } else {
        println!("  ✗ Encrypt/Decrypt FAILED!\n");
        return;
    }

    println!("=== STEP 3.5: Test direct polynomial multiply ===\n");
    // Before doing homomorphic multiply, let's test if direct polynomial multiplication works
    use ga_engine::clifford_fhe::rns::{RnsPolynomial, rns_multiply as rns_poly_multiply};

    // Create simple test polynomials: [2Δ, 0, 0, ...] and [3Δ, 0, 0, ...]
    let mut poly1_coeffs = vec![0i64; params.n];
    poly1_coeffs[0] = scaled;
    let mut poly2_coeffs = vec![0i64; params.n];
    poly2_coeffs[0] = (3.0 * params.scale).round() as i64;

    let poly1 = RnsPolynomial::from_coeffs(&poly1_coeffs, &params.moduli, params.n, 0);
    let poly2 = RnsPolynomial::from_coeffs(&poly2_coeffs, &params.moduli, params.n, 0);

    // Helper function for polynomial multiplication
    fn poly_mult_ntt(a: &[i64], b: &[i64], q: i64, n: usize) -> Vec<i64> {
        let mut result = vec![0i128; n];
        let q128 = q as i128;
        for i in 0..n {
            for j in 0..n {
                let idx = i + j;
                // Don't reduce modulo here - accumulate first
                let prod = (a[i] as i128) * (b[j] as i128);
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

    let product = rns_poly_multiply(&poly1, &poly2, &params.moduli, poly_mult_ntt);
    // Use single-prime decoding for direct multiply test too
    let product_val = product.to_coeffs_single_prime(0, params.moduli[0])[0];

    println!("Direct multiply: [2Δ] × [3Δ]");
    println!("  poly1[0] = {}", poly1_coeffs[0]);
    println!("  poly2[0] = {}", poly2_coeffs[0]);
    println!("  product[0] mod q_0 = {}", product_val);
    let expected = (6.0 * params.scale * params.scale);
    println!("  Expected (6Δ²) ≈ {:.3e}", expected);
    println!("  Ratio: {:.6}\n", product_val as f64 / expected);

    println!("=== STEP 4: Homomorphic Multiplication [2] × [3] ===\n");

    // Create second plaintext
    let value_b = 3.0;
    let scaled_b = (value_b * params.scale).round() as i64;
    let mut coeffs_b = vec![0i64; params.n];
    coeffs_b[0] = scaled_b;
    let pt_b = RnsPlaintext::from_coeffs(coeffs_b, params.scale, &params.moduli, 0);
    let ct_b = rns_encrypt(&pk, &pt_b, &params);

    // Multiply
    use ga_engine::clifford_fhe::ckks_rns::rns_multiply_ciphertexts;
    let ct_result = rns_multiply_ciphertexts(&ct, &ct_b, &evk, &params);

    println!("Result ciphertext level: {} (dropped 1 prime)", ct_result.level);
    println!("Result ciphertext scale: {:.2e}", ct_result.scale);
    println!("Expected scale: {:.2e}", (params.scale * params.scale) / (params.moduli[2] as f64));

    println!("\n=== STEP 5: Decrypt Result ===\n");
    let pt_result = rns_decrypt(&sk, &ct_result, &params);
    println!("Result level: {}", pt_result.coeffs.level);
    println!("Result num_primes: {}", pt_result.coeffs.num_primes());
    println!("Result scale: {:.2e}", pt_result.scale);
    println!("Result RNS coefficients[0]:");
    for j in 0..pt_result.coeffs.num_primes() {
        println!("  mod q_{}: {}", j, pt_result.coeffs.rns_coeffs[0][j]);
    }

    // Decode using single-prime (works for CKKS since message+noise << any prime)
    // For production with many primes, single-prime decoding is sufficient
    // because the decrypted message is always small relative to any single prime
    let active_primes = &params.moduli[..pt_result.coeffs.num_primes()];
    let c_r = pt_result.coeffs.to_coeffs_single_prime(0, active_primes[0])[0];

    println!("\nDecoding using single prime (standard CKKS):");
    println!("  Number of active primes: {}", active_primes.len());
    println!("  Using prime q_0: {}", active_primes[0]);
    println!("  Coefficient[0] (centered): {}", c_r);
    println!("  Expected (6*scale): {:.3e}", 6.0 * pt_result.scale);
    println!("  Recovered value: {:.6}", (c_r as f64) / pt_result.scale);
    println!("  Expected value: 6.000000");

    let final_error = ((c_r as f64) / pt_result.scale - 6.0).abs();
    if final_error < 0.1 {
        println!("\n✓✓✓ RNS-CKKS MULTIPLICATION WORKS! ✓✓✓");
    } else {
        println!("\n✗✗✗ RNS-CKKS MULTIPLICATION FAILED ✗✗✗");
        println!("Error magnitude: {:.2e}", final_error);
    }
}
