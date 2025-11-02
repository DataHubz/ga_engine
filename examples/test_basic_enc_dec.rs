//! Test basic encryption/decryption with fixed public key

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt};

fn main() {
    // Use minimal parameters
    let mut params = CliffordFHEParams::new_rns_mult();
    params.scale = 2f64.powi(20);  // Smaller scale for testing
    params.n = 64;  // Smaller N
    params.error_std = 0.0;  // NO NOISE for debugging

    // Use only 2 primes
    params.moduli = vec![params.moduli[0], params.moduli[1]];

    println!("Parameters:");
    println!("  N = {}", params.n);
    println!("  Num primes = {}", params.moduli.len());
    println!("  Δ = {}", params.scale);
    println!("  Primes: {:?}\n", params.moduli);

    // Generate keys
    let (pk, sk, _evk) = rns_keygen(&params);
    let primes = &params.moduli;

    // Verify public key relation: b + a·s ≈ 0
    use ga_engine::clifford_fhe::rns::{rns_add, rns_multiply as rns_poly_multiply};

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

    let as_poly = rns_poly_multiply(&pk.a, &sk.coeffs, primes, polynomial_multiply_ntt);
    let b_plus_as = rns_add(&pk.b, &as_poly, primes);

    println!("Public key verification (b + a·s, should be small):");
    for i in 0..3 {
        print!("  coeff[{}]: ", i);
        for j in 0..primes.len() {
            let val = b_plus_as.rns_coeffs[i][j];
            let centered = if val > primes[j] / 2 { val - primes[j] } else { val };
            print!("{:4} ", centered);
        }
        println!();
    }

    // Create plaintext [2]
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = (2.0 * params.scale).round() as i64;
    let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, primes, 0);

    println!("\nPlaintext:");
    println!("  Value: 2");
    println!("  coeff[0] = 2Δ = {}", pt.coeffs.rns_coeffs[0][0]);
    println!("  coeff[0] residues: {:?}", &pt.coeffs.rns_coeffs[0]);

    // Verify RNS consistency of plaintext
    let q0 = primes[0];
    let q1 = primes[1];
    let r0 = pt.coeffs.rns_coeffs[0][0];
    let r1 = pt.coeffs.rns_coeffs[0][1];
    let expected = (2.0 * params.scale).round() as i64;
    println!("  expected mod q0: {}", ((expected % q0) + q0) % q0);
    println!("  expected mod q1: {}", ((expected % q1) + q1) % q1);
    if r0 != ((expected % q0) + q0) % q0 || r1 != ((expected % q1) + q1) % q1 {
        println!("  ⚠️  WARNING: Plaintext RNS representation is inconsistent!");
    }

    // Encrypt
    let ct = rns_encrypt(&pk, &pt, &params);

    println!("\nCiphertext:");
    println!("  c0[0] residues: {:?}", &ct.c0.rns_coeffs[0]);
    println!("  c1[0] residues: {:?}", &ct.c1.rns_coeffs[0]);

    // Decrypt
    let pt_dec = rns_decrypt(&sk, &ct, &params);

    println!("\nDecrypted:");
    println!("  coeff[0] residues: {:?}", &pt_dec.coeffs.rns_coeffs[0]);

    // CHECK CRT consistency
    let r0 = pt_dec.coeffs.rns_coeffs[0][0];
    let r1 = pt_dec.coeffs.rns_coeffs[0][1];
    let expected_2delta = (2.0 * params.scale).round() as i64;

    // Check if residues are consistent with expected value
    let exp_r0 = ((expected_2delta % q0) + q0) % q0;
    let exp_r1 = ((expected_2delta % q1) + q1) % q1;
    println!("  Expected residues for 2Δ: [{}, {}]", exp_r0, exp_r1);

    if r0 != exp_r0 || r1 != exp_r1 {
        println!("  ⚠️  RESIDUES ARE INCONSISTENT - they represent different values!");

        // Try to reconstruct using CRT
        let p = q0 as i128;
        let q = q1 as i128;
        let p_inv_mod_q = {
            let mut t = 0i128;
            let mut new_t = 1i128;
            let mut r = q;
            let mut new_r = p % q;
            while new_r != 0 {
                let quotient = r / new_r;
                let temp = new_t;
                new_t = t - quotient * new_t;
                t = temp;
                let temp = new_r;
                new_r = r - quotient * new_r;
                r = temp;
            }
            if t < 0 { t += q; }
            t
        };

        let a = r0 as i128;
        let b = r1 as i128;
        let diff = ((b - a) % q + q) % q;
        let factor = (diff * p_inv_mod_q) % q;
        let x = a + p * factor;
        println!("  CRT reconstruction: {}", x);
        println!("  CRT reconstruction (mod q0): {}", x % p);
        println!("  CRT reconstruction (mod q1): {}", x % q);
    }

    // Extract value from first prime
    let dec_val = pt_dec.coeffs.rns_coeffs[0][0];
    let centered = if dec_val > primes[0] / 2 {
        dec_val - primes[0]
    } else {
        dec_val
    };

    println!("  coeff[0] centered (from prime 0): {}", centered);
    println!("  Expected (2Δ):     {}", expected_2delta);

    let recovered = (centered as f64) / params.scale;
    println!("\nRecovered value: {}", recovered);
    println!("Expected:        2.0");

    if (recovered - 2.0).abs() < 0.1 {
        println!("\n✅ Basic encryption/decryption WORKS!");
    } else {
        println!("\n❌ Basic encryption/decryption FAILED!");
        println!("   Error: {}", (recovered - 2.0).abs());
    }
}
