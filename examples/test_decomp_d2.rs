//! Debug: check d2 decomposition in actual multiplication

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt};
use ga_engine::clifford_fhe::rns::{rns_multiply as rns_poly_multiply, decompose_base_pow2};

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
        if r < 0 { (r + q128) as i64 } else { r as i64 }
    }).collect()
}

fn main() {
    println!("=== Debug d2 Decomposition ===\n");

    let mut params = CliffordFHEParams::new_rns_mult();
    params.n = 64;
    params.scale = 2f64.powi(40);
    params.error_std = 0.0;  // ZERO NOISE

    params.moduli = vec![
        1152921504606851201,  // q0
        1099511628161,        // q1
    ];

    let delta = params.scale;
    let primes = &params.moduli;

    let (pk, sk, _evk) = rns_keygen(&params);

    // Encrypt [2] and [3]
    let mut m1_coeffs = vec![0i64; params.n];
    m1_coeffs[0] = (2.0 * delta).round() as i64;
    let pt1 = RnsPlaintext::from_coeffs(m1_coeffs, delta, primes, 0);

    let mut m2_coeffs = vec![0i64; params.n];
    m2_coeffs[0] = (3.0 * delta).round() as i64;
    let pt2 = RnsPlaintext::from_coeffs(m2_coeffs, delta, primes, 0);

    let ct1 = rns_encrypt(&pk, &pt1, &params);
    let ct2 = rns_encrypt(&pk, &pt2, &params);

    // Compute tensor product to get d2 = c1·d1
    let d2 = rns_poly_multiply(&ct1.c1, &ct2.c1, primes, polynomial_multiply_ntt);

    println!("d2 (= c1·d1) [0] residues: {:?}", d2.rns_coeffs[0]);
    println!();

    // CRT reconstruct d2[0]
    let r0 = d2.rns_coeffs[0][0] as i128;
    let r1 = d2.rns_coeffs[0][1] as i128;
    let q0 = primes[0] as i128;
    let q1 = primes[1] as i128;

    // Simple 2-prime CRT
    let q_prod = q0 * q1;
    let inv0 = modinv(q1, q0);
    let inv1 = modinv(q0, q1);

    let value = (r0 * q1 * inv1 + r1 * q0 * inv0) % q_prod;

    // Center-lift
    let centered = if value > q_prod / 2 {
        value - q_prod
    } else {
        value
    };

    println!("CRT reconstruction of d2[0]:");
    println!("  Value (mod Q): {}", value);
    println!("  Centered:      {}", centered);
    println!();

    // What should d2[0] be?
    // d2 = c1·d1, and c1, d1 both have coefficient [0] ≈ 0 (they're random noise + small message)
    // Actually, ct encrypts m·Δ, so c0 ≈ m·Δ + noise, c1 ≈ noise
    // So d2 = c1 · c1 ≈ noise²

    println!("Expected d2[0] ≈ 0 (since c1 coeffs are small)");
    println!();

    // Decompose
    let digits = decompose_base_pow2(&d2, primes, 20);

    println!("Digits:");
    for (t, digit) in digits.iter().enumerate() {
        println!("  digit[{}][0] residues: {:?}", t, digit.rns_coeffs[0]);
    }
    println!();

    // Verify reconstruction PER PRIME (not globally)
    // For each prime, check that Σ dt·B^t ≡ d2[0] (mod qi)
    println!("Verifying decomposition per prime:");
    let b = 1i64 << 20;

    for j in 0..2 {
        let qi = primes[j];
        let mut reconstructed_mod_qi = 0i128;
        let mut b_pow = 1i128;

        for digit in &digits {
            let dt = digit.rns_coeffs[0][j] as i128;
            reconstructed_mod_qi = (reconstructed_mod_qi + dt * b_pow) % (qi as i128);
            b_pow = (b_pow * (b as i128)) % (qi as i128);
        }

        let expected_mod_qi = d2.rns_coeffs[0][j] as i128;

        println!("  Prime j={} (qi={}):", j, qi);
        println!("    Reconstructed: {}", reconstructed_mod_qi);
        println!("    Expected:      {}", expected_mod_qi);

        if reconstructed_mod_qi == expected_mod_qi {
            println!("    ✅ MATCH");
        } else {
            println!("    ❌ MISMATCH");
        }
    }
}

fn modinv(a: i128, m: i128) -> i128 {
    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (m, a % m);

    while new_r != 0 {
        let quotient = r / new_r;
        (t, new_t) = (new_t, t - quotient * new_t);
        (r, new_r) = (new_r, r - quotient * new_r);
    }

    if r > 1 {
        panic!("a is not invertible mod m");
    }
    if t < 0 {
        t += m;
    }
    t
}
