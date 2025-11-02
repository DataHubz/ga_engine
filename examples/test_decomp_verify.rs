//! Verify CRT-consistent decomposition

use ga_engine::clifford_fhe::rns::{RnsPolynomial, decompose_base_pow2};

fn main() {
    println!("=== Testing CRT-Consistent Decomposition ===\n");

    // Simple test with 2 primes
    let primes = vec![
        1152921504606851201i64,  // q0 ≈ 2^60
        1099511628161i64,         // q1 ≈ 2^40
    ];
    let n = 4;
    let w = 20;

    // Create a test polynomial with a known value
    // Let's use a simple value: 42
    let test_val = 42i64;

    let mut rns_coeffs = vec![vec![0i64; 2]; n];
    rns_coeffs[0][0] = test_val % primes[0];
    rns_coeffs[0][1] = test_val % primes[1];

    let poly = RnsPolynomial::new(rns_coeffs, n, 0);

    println!("Original polynomial:");
    println!("  coeff[0] residues: {:?}", poly.rns_coeffs[0]);
    println!("  Value: {}", test_val);
    println!();

    // Decompose
    let digits = decompose_base_pow2(&poly, &primes, w);

    println!("Decomposition (w={}, B=2^{}={}):", w, w, 1i64 << w);
    for (t, digit) in digits.iter().enumerate() {
        println!("  digit[{}][0] residues: {:?}", t, digit.rns_coeffs[0]);
    }
    println!();

    // Verify: reconstruct Σ d_t * B^t mod each prime
    let b = 1i64 << w;
    for j in 0..2 {
        let qi = primes[j];
        let mut reconstructed = 0i128;
        let mut b_pow = 1i128;

        for digit in &digits {
            let dt = digit.rns_coeffs[0][j] as i128;
            reconstructed = (reconstructed + dt * b_pow) % (qi as i128);
            b_pow = (b_pow * (b as i128)) % (qi as i128);
        }

        let expected = poly.rns_coeffs[0][j] as i128;

        println!("Prime j={} (qi={}):", j, qi);
        println!("  Reconstructed: {}", reconstructed);
        println!("  Expected:      {}", expected);

        if reconstructed == expected {
            println!("  ✅ MATCH");
        } else {
            println!("  ❌ MISMATCH!");
        }
        println!();
    }
}
