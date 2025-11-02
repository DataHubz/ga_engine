//! Step-by-step debug of homomorphic multiplication

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt};
use ga_engine::clifford_fhe::rns::{rns_add, rns_multiply as rns_poly_multiply};

/// Simple polynomial multiplication (same as in ckks_rns.rs)
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

    println!("========================================================================");
    println!("Step-by-Step Homomorphic Multiplication Debug");
    println!("========================================================================\n");

    println!("Generating keys...");
    let (pk, sk, evk) = rns_keygen(&params);
    println!("Done\n");

    // Create simple messages
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

    println!("Plaintexts:");
    println!("  msg_a = {}, scaled = {}", msg_a, scaled_a);
    println!("  msg_b = {}, scaled = {}", msg_b, scaled_b);
    println!("  pt_a RNS[0]: {:?}", pt_a.coeffs.rns_coeffs[0]);
    println!("  pt_b RNS[0]: {:?}\n", pt_b.coeffs.rns_coeffs[0]);

    // Encrypt
    let ct_a = rns_encrypt(&pk, &pt_a, &params);
    let ct_b = rns_encrypt(&pk, &pt_b, &params);

    println!("Ciphertexts (level 0, scale = {:.2e}):", params.scale);
    println!("  ct_a.c0[0]: {:?}", ct_a.c0.rns_coeffs[0]);
    println!("  ct_a.c1[0]: {:?}", ct_a.c1.rns_coeffs[0]);
    println!("  ct_b.c0[0]: {:?}", ct_b.c0.rns_coeffs[0]);
    println!("  ct_b.c1[0]: {:?}\n", ct_b.c1.rns_coeffs[0]);

    // Step 1: Multiply to get degree-2 ciphertext (NO relinearization, NO rescaling)
    println!("==== STEP 1: Multiply (degree-2, scale²) ====\n");

    let active_primes = &primes[..];

    let c0d0 = rns_poly_multiply(&ct_a.c0, &ct_b.c0, active_primes, polynomial_multiply_ntt);
    let c0d1 = rns_poly_multiply(&ct_a.c0, &ct_b.c1, active_primes, polynomial_multiply_ntt);
    let c1d0 = rns_poly_multiply(&ct_a.c1, &ct_b.c0, active_primes, polynomial_multiply_ntt);
    let c1d1 = rns_poly_multiply(&ct_a.c1, &ct_b.c1, active_primes, polynomial_multiply_ntt);

    let d1 = rns_add(&c0d1, &c1d0, active_primes);

    println!("Degree-2 ciphertext (d0, d1, d2):");
    println!("  d0[0] = c0*c0: {:?}", c0d0.rns_coeffs[0]);
    println!("  d1[0] = c0*c1 + c1*c0: {:?}", d1.rns_coeffs[0]);
    println!("  d2[0] = c1*c1: {:?}\n", c1d1.rns_coeffs[0]);

    // Decrypt degree-2: m = d0 + d1*s + d2*s²
    // But we decrypt with c0 - c1*s, so we need to check the math
    println!("Attempting to decrypt degree-2 manually...");
    println!("  (This should fail or give wrong result - degree-2 needs relinearization)\n");

    // Create a pseudo-ciphertext treating d0 as c0, d1 as c1 (ignoring d2)
    use ga_engine::clifford_fhe::ckks_rns::RnsCiphertext;
    let ct_deg2_partial = RnsCiphertext::new(
        c0d0.clone(),
        d1.clone(),
        0,
        params.scale * params.scale  // scale²
    );

    let pt_deg2_partial = rns_decrypt(&sk, &ct_deg2_partial, &params);
    println!("Partial decrypt (ignoring d2*s² term):");
    println!("  RNS[0]: {:?}", pt_deg2_partial.coeffs.rns_coeffs[0]);
    println!("  (This is missing the d2*s² contribution)\n");

    println!("Expected: msg_a * msg_b * scale² = {} * {} * scale²", msg_a, msg_b);
    let expected_at_scale_sq = msg_a * msg_b * params.scale * params.scale;
    println!("             = {:.2e}\n", expected_at_scale_sq);

    println!("========================================================================");
}
