//! Test just d0 = c0 × c0' from tensor product
//!
//! This isolates whether the issue is in the polynomial multiplication
//! or in the subsequent steps with secret key multiplication.

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt};
use ga_engine::clifford_fhe::rns::rns_multiply as rns_poly_multiply;

// Include the polynomial multiply function
fn polynomial_multiply_ntt(a: &[i64], b: &[i64], q: i64, n: usize) -> Vec<i64> {
    let mut result = vec![0i128; n];
    let q128 = q as i128;

    for i in 0..n {
        for j in 0..n {
            let idx = i + j;
            let prod = (a[i] as i128) * (b[j] as i128);
            if idx < n {
                result[idx] += prod;
            } else {
                let wrapped_idx = idx % n;
                result[wrapped_idx] -= prod;
            }
        }
    }

    result.iter().map(|&x| {
        let r = x % q128;
        if r < 0 {
            (r + q128) as i64
        } else {
            r as i64
        }
    }).collect()
}

fn main() {
    let mut params = CliffordFHEParams::new_rns_mult();
    params.scale = 2f64.powi(20); // Use smaller scale for clearer debugging

    let (pk, _sk, _evk) = rns_keygen(&params);
    let primes = &params.moduli;

    // Create plaintexts [2Δ, 0, 0, ...] and [3Δ, 0, 0, ...]
    let mut coeffs_1 = vec![0i64; params.n];
    coeffs_1[0] = (2.0 * params.scale).round() as i64;
    let pt1 = RnsPlaintext::from_coeffs(coeffs_1, params.scale, primes, 0);

    let mut coeffs_2 = vec![0i64; params.n];
    coeffs_2[0] = (3.0 * params.scale).round() as i64;
    let pt2 = RnsPlaintext::from_coeffs(coeffs_2, params.scale, primes, 0);

    println!("=== PLAINTEXT MULTIPLY TEST ===\n");
    println!("Δ = 2^20 = {}", params.scale as i64);
    println!("pt1[0] = 2Δ = {}", pt1.coeffs.rns_coeffs[0][0]);
    println!("pt2[0] = 3Δ = {}", pt2.coeffs.rns_coeffs[0][0]);

    // Direct plaintext multiply
    let pt_product = rns_poly_multiply(&pt1.coeffs, &pt2.coeffs, primes, polynomial_multiply_ntt);
    let pt_prod_val = pt_product.to_coeffs_single_prime(0, primes[0])[0];

    let expected_pt = (6.0 * params.scale * params.scale) as i64;
    println!("\nDirect plaintext multiply:");
    println!("  pt1 × pt2 = {}", pt_prod_val);
    println!("  Expected 6Δ² = {}", expected_pt);
    println!("  Match: {}\n", (pt_prod_val - expected_pt).abs() < 100);

    // Now encrypt and compute tensor product d0
    println!("=== CIPHERTEXT TENSOR PRODUCT ===\n");

    let ct1 = rns_encrypt(&pk, &pt1, &params);
    let ct2 = rns_encrypt(&pk, &pt2, &params);

    println!("ct1.c0[0] residues: {:?}", &ct1.c0.rns_coeffs[0][..3]);
    println!("ct2.c0[0] residues: {:?}", &ct2.c0.rns_coeffs[0][..3]);

    // Compute d0 = ct1.c0 × ct2.c0
    let d0 = rns_poly_multiply(&ct1.c0, &ct2.c0, primes, polynomial_multiply_ntt);

    let d0_val = d0.to_coeffs_single_prime(0, primes[0])[0];

    println!("\nd0 = ct1.c0 × ct2.c0:");
    println!("  d0[0] (single prime) = {}", d0_val);
    println!("  Expected (approximately 6Δ² + noise) ≈ {}", expected_pt);

    let d0_value = (d0_val as f64) / (params.scale * params.scale);
    println!("  d0[0] / Δ² = {}", d0_value);
    println!("  Expected: ≈ 6.0");

    if (d0_value - 6.0).abs() < 10.0 {
        println!("\n✓ Tensor product d0 is approximately correct!");
    } else {
        println!("\n✗ Tensor product d0 is WRONG!");
        println!("  Error: {} times too large", d0_value / 6.0);
    }
}
