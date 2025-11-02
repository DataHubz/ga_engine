//! Detailed degree-2 test with all intermediate values

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt};
use ga_engine::clifford_fhe::rns::{rns_add, rns_sub, rns_multiply as rns_poly_multiply};

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

fn main() {
    let params = CliffordFHEParams::new_rns_mult();
    let primes = &params.moduli;
    let q0 = primes[0];

    let (_pk, sk, _evk) = rns_keygen(&params);

    // Use small messages for clarity
    let msg_a = 2.0;
    let msg_b = 3.0;

    println!("========================================================================");
    println!("Detailed Degree-2 Analysis");
    println!("========================================================================\n");
    println!("Messages: {} × {} = {}", msg_a, msg_b, msg_a * msg_b);
    println!("Scale Δ: {:.4e}", params.scale);
    println!("Scale² Δ²: {:.4e}", params.scale * params.scale);
    println!("Prime q0: {}\n", q0);

    // Show what we expect
    let expected_at_delta = msg_a * params.scale;
    let expected_at_delta2 = msg_a * msg_b * params.scale * params.scale;

    println!("Expected values:");
    println!("  msg_a × Δ = {:.4e}", expected_at_delta);
    println!("  (msg_a × msg_b) × Δ² = {:.4e}", expected_at_delta2);
    println!("  Δ² / q0 = {:.4e}", params.scale * params.scale / q0 as f64);
    println!();

    // Encrypt
    let scaled_a = (msg_a * params.scale).round() as i64;
    let scaled_b = (msg_b * params.scale).round() as i64;

    let mut coeffs_a = vec![0i64; params.n];
    coeffs_a[0] = scaled_a;
    let mut coeffs_b = vec![0i64; params.n];
    coeffs_b[0] = scaled_b;

    let pt_a = RnsPlaintext::from_coeffs(coeffs_a, params.scale, &params.moduli, 0);
    let pt_b = RnsPlaintext::from_coeffs(coeffs_b, params.scale, &params.moduli, 0);

    println!("Plaintexts (scaled coefficients):");
    println!("  pt_a coeff[0] = {}, RNS = {:?}", scaled_a, pt_a.coeffs.rns_coeffs[0]);
    println!("  pt_b coeff[0] = {}, RNS = {:?}\n", scaled_b, pt_b.coeffs.rns_coeffs[0]);

    let ct_a = rns_encrypt(&_pk, &pt_a, &params);
    let ct_b = rns_encrypt(&_pk, &pt_b, &params);

    // Tensor product
    let active_primes = &primes[..];

    let d0 = rns_poly_multiply(&ct_a.c0, &ct_b.c0, active_primes, polynomial_multiply_ntt);
    let c0d1 = rns_poly_multiply(&ct_a.c0, &ct_b.c1, active_primes, polynomial_multiply_ntt);
    let c1d0 = rns_poly_multiply(&ct_a.c1, &ct_b.c0, active_primes, polynomial_multiply_ntt);
    let c1d1 = rns_poly_multiply(&ct_a.c1, &ct_b.c1, active_primes, polynomial_multiply_ntt);

    let d1 = rns_add(&c0d1, &c1d0, active_primes);
    let d2 = c1d1;

    println!("Degree-2 ciphertext (d0, d1, d2) - residue mod q0:");
    println!("  d0[0] mod q0: {}", d0.rns_coeffs[0][0]);
    println!("  d1[0] mod q0: {}", d1.rns_coeffs[0][0]);
    println!("  d2[0] mod q0: {}\n", d2.rns_coeffs[0][0]);

    // Manual decrypt
    let s_squared = rns_poly_multiply(&sk.coeffs, &sk.coeffs, active_primes, polynomial_multiply_ntt);
    let d1_times_s = rns_poly_multiply(&d1, &sk.coeffs, active_primes, polynomial_multiply_ntt);
    let d2_times_s_sq = rns_poly_multiply(&d2, &s_squared, active_primes, polynomial_multiply_ntt);

    println!("Intermediate products - residue mod q0:");
    println!("  s²[0] mod q0: {}", s_squared.rns_coeffs[0][0]);
    println!("  (d1*s)[0] mod q0: {}", d1_times_s.rns_coeffs[0][0]);
    println!("  (d2*s²)[0] mod q0: {}\n", d2_times_s_sq.rns_coeffs[0][0]);

    // Step by step
    let temp = rns_sub(&d0, &d1_times_s, active_primes);
    println!("After d0 - d1*s:");
    println!("  result[0] mod q0: {}\n", temp.rns_coeffs[0][0]);

    let m_deg2 = rns_add(&temp, &d2_times_s_sq, active_primes);
    println!("After + d2*s²:");
    println!("  m[0] mod q0: {}\n", m_deg2.rns_coeffs[0][0]);

    // Extract and center-lift
    let coeffs_q0 = m_deg2.to_coeffs_single_prime(0, q0);
    let mut v = coeffs_q0[0];
    if v > q0 / 2 {
        v = v - q0;
    }

    println!("Final result:");
    println!("  Raw mod q0: {}", coeffs_q0[0]);
    println!("  Center-lifted: {}", v);
    println!("  Expected ≈ (msg × Δ²) mod q0 with noise\n");

    let scale_sq = params.scale * params.scale;
    let recovered = v as f64 / scale_sq;

    println!("Decoded:");
    println!("  v / Δ² = {:.6}", recovered);
    println!("  Expected: {:.6}", msg_a * msg_b);

    println!("\n========================================================================");
}
