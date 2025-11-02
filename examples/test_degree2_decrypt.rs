//! Probe A: Test degree-2 decryption without relinearization
//!
//! Multiply two ciphertexts, then manually decrypt as degree-2:
//! m = d0 - d1*s + d2*s²

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
    println!("========================================================================");
    println!("Probe A: Degree-2 Decryption (No Relinearization)");
    println!("========================================================================\n");

    let params = CliffordFHEParams::new_rns_mult();
    let primes = &params.moduli;

    let (_pk, sk, _evk) = rns_keygen(&params);

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
    println!("Scale: {:.2e}", params.scale);
    println!("Scale²: {:.2e}\n", params.scale * params.scale);

    let ct_a = rns_encrypt(&_pk, &pt_a, &params);
    let ct_b = rns_encrypt(&_pk, &pt_b, &params);

    // Multiply to get degree-2
    let active_primes = &primes[..];

    let c0d0 = rns_poly_multiply(&ct_a.c0, &ct_b.c0, active_primes, polynomial_multiply_ntt);
    let c0d1 = rns_poly_multiply(&ct_a.c0, &ct_b.c1, active_primes, polynomial_multiply_ntt);
    let c1d0 = rns_poly_multiply(&ct_a.c1, &ct_b.c0, active_primes, polynomial_multiply_ntt);
    let c1d1 = rns_poly_multiply(&ct_a.c1, &ct_b.c1, active_primes, polynomial_multiply_ntt);

    let d0 = c0d0;
    let d1 = rns_add(&c0d1, &c1d0, active_primes);
    let d2 = c1d1;

    println!("Degree-2 ciphertext:");
    println!("  d0[0]: {:?}", d0.rns_coeffs[0]);
    println!("  d1[0]: {:?}", d1.rns_coeffs[0]);
    println!("  d2[0]: {:?}\n", d2.rns_coeffs[0]);

    // Manual degree-2 decrypt: m = d0 - d1*s + d2*s²
    println!("Manually decrypting degree-2...");

    // Compute s²
    let s_squared = rns_poly_multiply(&sk.coeffs, &sk.coeffs, active_primes, polynomial_multiply_ntt);
    println!("  s²[0]: {:?}", s_squared.rns_coeffs[0]);

    // Compute d1*s
    let d1_times_s = rns_poly_multiply(&d1, &sk.coeffs, active_primes, polynomial_multiply_ntt);
    println!("  d1*s[0]: {:?}", d1_times_s.rns_coeffs[0]);

    // Compute d2*s²
    let d2_times_s_sq = rns_poly_multiply(&d2, &s_squared, active_primes, polynomial_multiply_ntt);
    println!("  d2*s²[0]: {:?}\n", d2_times_s_sq.rns_coeffs[0]);

    // m = d0 - d1*s + d2*s²
    let temp = rns_sub(&d0, &d1_times_s, active_primes);
    let m_deg2 = rns_add(&temp, &d2_times_s_sq, active_primes);

    println!("Result m = d0 - d1*s + d2*s²:");
    println!("  m[0]: {:?}\n", m_deg2.rns_coeffs[0]);

    // CKKS decoding: extract from single prime and center-lift
    // (NOT full CRT - that's for large values, not for decoding)
    let q0 = active_primes[0];
    let coeffs_q0 = m_deg2.to_coeffs_single_prime(0, q0);

    println!("Single-prime extraction (prime 0 = {}):", q0);
    println!("  Raw coefficient mod q0: {}", coeffs_q0[0]);

    // Center-lift to (-q0/2, q0/2]
    let mut v = coeffs_q0[0];
    if v > q0 / 2 {
        v = v - q0;
    }

    println!("  Center-lifted: {}\n", v);

    // Decode at scale²
    let scale_sq = params.scale * params.scale;
    let recovered_msg = v as f64 / scale_sq;

    println!("Decoded at scale²:");
    println!("  Recovered: {:.6}", recovered_msg);
    println!("  Expected:  {:.6}", msg_a * msg_b);
    println!("  Error:     {:.6}", (recovered_msg - (msg_a * msg_b)).abs());

    if (recovered_msg - (msg_a * msg_b)).abs() < 0.1 {
        println!("\n✓ PASS: Degree-2 decryption works!");
    } else {
        println!("\n✗ FAIL: Degree-2 decryption broken");
    }

    println!("\n========================================================================");
}
