//! Test EVK identity: evk0 - evk1·s = -B^t·s² + e (small)

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::rns::rns_multiply as rns_poly_multiply;

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
    println!("=== EVK Identity Verification ===\n");

    let mut params = CliffordFHEParams::new_rns_mult();
    params.n = 64;
    params.scale = 2f64.powi(40);
    params.error_std = 3.2;

    params.moduli = vec![
        1152921504606851201,  // q0
        1099511628161,        // q1
    ];

    println!("Generating keys with error_std = {}...", params.error_std);
    let (pk, sk, evk) = rns_keygen(&params);

    println!("EVK has {} digit keys, base_w = {}", evk.evk0.len(), evk.base_w);
    println!();

    let primes = &params.moduli;
    let b = 1i64 << evk.base_w;

    // Compute s²
    let s_squared = rns_poly_multiply(&sk.coeffs, &sk.coeffs, primes, polynomial_multiply_ntt);

    // Test EVK identity for first digit (t=0, B^0=1)
    let t = 0;
    println!("Testing digit t={} (B^t = {}):", t, b.pow(t as u32));

    // Compute evk0[t] - evk1[t]·s
    let evk1_s = rns_poly_multiply(&evk.evk1[t], &sk.coeffs, primes, polynomial_multiply_ntt);

    // Compute the LHS of identity: evk0[t] - evk1[t]·s
    let mut lhs_coeffs = vec![vec![0i64; 2]; params.n];
    for i in 0..params.n {
        for j in 0..2 {
            let qi = primes[j];
            let diff = ((evk.evk0[t].rns_coeffs[i][j] as i128
                        - evk1_s.rns_coeffs[i][j] as i128)
                       % (qi as i128) + (qi as i128)) % (qi as i128);
            lhs_coeffs[i][j] = diff as i64;
        }
    }

    // Compute the RHS: -B^t·s² (for t=0, B^t=1)
    let mut rhs_coeffs = vec![vec![0i64; 2]; params.n];
    for i in 0..params.n {
        for j in 0..2 {
            let qi = primes[j];
            // -s²[i][j]
            let neg = ((qi - s_squared.rns_coeffs[i][j] % qi) % qi);
            rhs_coeffs[i][j] = neg;
        }
    }

    // The difference (LHS - RHS) should be the small error e_t
    println!("Checking: (evk0 - evk1·s) - (-B^t·s²) = e_t (small)");
    println!();

    let mut max_error = 0i128;
    for i in 0..params.n.min(5) {
        for j in 0..2 {
            let qi = primes[j];
            let diff = ((lhs_coeffs[i][j] as i128 - rhs_coeffs[i][j] as i128)
                       % (qi as i128) + (qi as i128)) % (qi as i128);

            // Center-lift
            let centered = if diff > qi as i128 / 2 {
                diff - qi as i128
            } else {
                diff
            };

            if i < 3 {
                println!("  coeff[{}], prime[{}]: error = {}", i, j, centered);
            }

            max_error = max_error.max(centered.abs());
        }
    }

    println!();
    println!("Max centered error: {}", max_error);
    println!("Expected ~6σ = {:.1}", 6.0 * params.error_std);

    if (max_error as f64) < 10.0 * params.error_std {
        println!("\n✅ EVK identity holds! Errors are small.");
    } else {
        println!("\n❌ EVK identity BROKEN! Errors are too large.");
    }
}
