//! Probe B: Verify EVK encrypts s² correctly
//!
//! Decrypt the EVK to check: evk0 - evk1*s ≈ s²

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::rns::{rns_sub, rns_multiply as rns_poly_multiply};

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
    println!("Probe B: Verify EVK Correctness");
    println!("========================================================================\n");

    let params = CliffordFHEParams::new_rns_mult();
    let primes = &params.moduli;

    println!("Generating keys...");
    let (_pk, sk, evk) = rns_keygen(&params);
    println!("Done\n");

    // Compute s² manually
    println!("Computing s² manually...");
    let s_squared_manual = rns_poly_multiply(&sk.coeffs, &sk.coeffs, primes, polynomial_multiply_ntt);
    println!("  s²[0] (first few coeffs in RNS):");
    for i in 0..5.min(params.n) {
        println!("    coeff[{}]: {:?}", i, s_squared_manual.rns_coeffs[i]);
    }
    println!();

    // Decrypt EVK: evk0 - evk1*s should give s²
    println!("Decrypting EVK...");
    let (evk0, evk1) = &evk.relin_keys[0];

    // Compute evk1*s
    let evk1_times_s = rns_poly_multiply(evk1, &sk.coeffs, primes, polynomial_multiply_ntt);

    // Compute evk0 - evk1*s
    let decrypted_evk = rns_sub(evk0, &evk1_times_s, primes);

    println!("  Dec(EVK) = evk0 - evk1*s:");
    println!("  (first few coeffs in RNS)");
    for i in 0..5.min(params.n) {
        println!("    coeff[{}]: {:?}", i, decrypted_evk.rns_coeffs[i]);
    }
    println!();

    // Compare: should be close to s² (up to noise e_k)
    println!("Comparison (Dec(EVK) vs s²):");
    println!("  Checking first coefficient across all primes...\n");

    let mut all_close = true;
    for (prime_idx, &q) in primes.iter().enumerate() {
        let s_sq_val = s_squared_manual.rns_coeffs[0][prime_idx];
        let dec_evk_val = decrypted_evk.rns_coeffs[0][prime_idx];

        // Compute difference (handling wrap-around)
        let diff = if dec_evk_val >= s_sq_val {
            dec_evk_val - s_sq_val
        } else {
            q - (s_sq_val - dec_evk_val)
        };

        // Center the difference
        let centered_diff = if diff > q / 2 { diff - q } else { diff };

        println!("  Prime {}: s²={}, Dec(EVK)={}, diff={}",
                 prime_idx, s_sq_val, dec_evk_val, centered_diff);

        // Should be small (just noise e_k, std ≈ 3.2)
        if centered_diff.abs() > 100 {
            println!("    ⚠️  Difference too large!");
            all_close = false;
        }
    }

    println!();
    if all_close {
        println!("✓ PASS: EVK correctly encrypts s² (differences are just noise)");
    } else {
        println!("✗ FAIL: EVK does NOT encrypt s² correctly!");
        println!("   This indicates a bug in EVK generation.");
    }

    println!("\n========================================================================");
}
