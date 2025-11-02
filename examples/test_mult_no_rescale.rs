//! Test multiplication with relinearization but WITHOUT rescaling
//! This isolates whether the issue is in relinearization or rescaling

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, RnsCiphertext, rns_encrypt, rns_decrypt};
use ga_engine::clifford_fhe::rns::{rns_add, rns_multiply as rns_poly_multiply};

/// Simple polynomial multiplication
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

/// Relinearize degree-2 to degree-1 (same as in ckks_rns.rs)
fn rns_relin(
    d0: &ga_engine::clifford_fhe::rns::RnsPolynomial,
    d1: &ga_engine::clifford_fhe::rns::RnsPolynomial,
    d2: &ga_engine::clifford_fhe::rns::RnsPolynomial,
    evk: &ga_engine::clifford_fhe::keys_rns::RnsEvaluationKey,
    primes: &[i64],
) -> (ga_engine::clifford_fhe::rns::RnsPolynomial, ga_engine::clifford_fhe::rns::RnsPolynomial) {
    let (evk0, evk1) = &evk.relin_keys[0];

    let d2_evk0 = rns_poly_multiply(d2, evk0, primes, polynomial_multiply_ntt);
    let d2_evk1 = rns_poly_multiply(d2, evk1, primes, polynomial_multiply_ntt);

    let c0 = rns_add(d0, &d2_evk0, primes);
    let c1 = rns_add(d1, &d2_evk1, primes);

    (c0, c1)
}

fn main() {
    let params = CliffordFHEParams::new_rns_mult();
    let primes = &params.moduli;

    println!("========================================================================");
    println!("Test: Multiply with Relinearization (NO RESCALING)");
    println!("========================================================================\n");

    let (pk, sk, evk) = rns_keygen(&params);

    let msg_a = 2.0;
    let msg_b = 3.0;

    let scaled_a = (msg_a * params.scale).round() as i64;
    let scaled_b = (msg_b * params.scale).round() as i64;

    let mut coeffs_a = vec![0i64; params.n];
    coeffs_a[0] = scaled_a;

    let mut coeffs_b = vec![0i64; params.n];
    coeffs_b[0] = scaled_b;

    let pt_a = RnsPlaintext::from_coeffs(coeffs_a, params.scale, &params.moduli, 0);
    let pt_b = RnsPlaintext::from_coeffs(coeffs_b, params.scale, &params.moduli, 0);

    println!("Messages: {} × {} = {}", msg_a, msg_b, msg_a * msg_b);
    println!("Scale: {:.2e}\n", params.scale);

    let ct_a = rns_encrypt(&pk, &pt_a, &params);
    let ct_b = rns_encrypt(&pk, &pt_b, &params);

    // Multiply to degree-2
    let active_primes = &primes[..];

    let c0d0 = rns_poly_multiply(&ct_a.c0, &ct_b.c0, active_primes, polynomial_multiply_ntt);
    let c0d1 = rns_poly_multiply(&ct_a.c0, &ct_b.c1, active_primes, polynomial_multiply_ntt);
    let c1d0 = rns_poly_multiply(&ct_a.c1, &ct_b.c0, active_primes, polynomial_multiply_ntt);
    let c1d1 = rns_poly_multiply(&ct_a.c1, &ct_b.c1, active_primes, polynomial_multiply_ntt);

    let d1 = rns_add(&c0d1, &c1d0, active_primes);

    println!("After multiplication (degree-2):");
    println!("  Level: 0 (all 3 primes)");
    println!("  Scale: {:.2e} (scale²)\n", params.scale * params.scale);

    // Relinearize
    let (c0_relin, c1_relin) = rns_relin(&c0d0, &d1, &c1d1, &evk, active_primes);

    println!("After relinearization (degree-1):");
    println!("  c0[0]: {:?}", c0_relin.rns_coeffs[0]);
    println!("  c1[0]: {:?}\n", c1_relin.rns_coeffs[0]);

    // Decrypt (still at scale²)
    let ct_relin = RnsCiphertext::new(
        c0_relin,
        c1_relin,
        0,  // Still level 0
        params.scale * params.scale  // Still at scale²
    );

    let pt_result = rns_decrypt(&sk, &ct_relin, &params);

    println!("Decrypted (at scale²):");
    println!("  RNS[0]: {:?}\n", pt_result.coeffs.rns_coeffs[0]);

    // Reconstruct with CRT
    let coeffs_result = pt_result.to_coeffs(active_primes);
    println!("CRT reconstruction:");
    println!("  coeff[0] = {}\n", coeffs_result[0]);

    // Decode: divide by scale²
    let scale_sq = params.scale * params.scale;
    let recovered_msg = coeffs_result[0] as f64 / scale_sq;

    println!("Decoded value: {:.6}", recovered_msg);
    println!("Expected value: {:.6}", msg_a * msg_b);
    println!("Error: {:.6}", (recovered_msg - (msg_a * msg_b)).abs());

    if (recovered_msg - (msg_a * msg_b)).abs() < 0.1 {
        println!("\n✓ PASS: Relinearization works!");
    } else {
        println!("\n✗ FAIL: Relinearization has issues");
    }

    println!("\n========================================================================");
}
