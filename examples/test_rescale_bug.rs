//! Minimal test to isolate the rescale bug
//!
//! Bug: rescaled polynomial values change between creation and use

use ga_engine::clifford_fhe::rns::{RnsPolynomial, rns_rescale_reference};

fn main() {
    println!("=== Testing rns_rescale_reference for value stability ===\n");

    let n = 1024;
    let primes = vec![
        1_099_511_627_689,  // q₀
        1_099_511_627_691,  // q₁
        1_099_511_627_693,  // q₂
    ];

    // Create a simple polynomial: [100, 0, 0, ..., 0] at level 0
    let coeffs = vec![100i64; n];
    let poly = RnsPolynomial::from_coeffs(&coeffs, &primes, n, 0);

    println!("Original polynomial:");
    println!("  level: {}", poly.level);
    println!("  num_primes: {}", poly.num_primes());
    println!("  coeffs[0] in RNS:");
    for j in 0..poly.num_primes() {
        println!("    mod q_{}: {}", j, poly.rns_coeffs[0][j]);
    }

    println!("\nCalling rns_rescale_reference...");
    let rescaled = rns_rescale_reference(&poly, &primes);

    println!("\nImmediately after rescale:");
    println!("  rescaled.level: {}", rescaled.level);
    println!("  rescaled.num_primes(): {}", rescaled.num_primes());
    println!("  rescaled.rns_coeffs[0][0]: {}", rescaled.rns_coeffs[0][0]);
    if rescaled.num_primes() > 1 {
        println!("  rescaled.rns_coeffs[0][1]: {}", rescaled.rns_coeffs[0][1]);
    }

    // Do nothing in between - just print again
    println!("\nA few lines later:");
    println!("  rescaled.rns_coeffs[0][0]: {}", rescaled.rns_coeffs[0][0]);
    if rescaled.num_primes() > 1 {
        println!("  rescaled.rns_coeffs[0][1]: {}", rescaled.rns_coeffs[0][1]);
    }

    println!("\nPrinting all residues:");
    for j in 0..rescaled.num_primes() {
        println!("  mod q_{}: {}", j, rescaled.rns_coeffs[0][j]);
    }

    // Check if values match
    let val1 = rescaled.rns_coeffs[0][0];
    let val2 = rescaled.rns_coeffs[0][0];

    if val1 == val2 {
        println!("\n✓ Values are stable!");
    } else {
        println!("\n✗ BUG: Values changed! {} != {}", val1, val2);
    }

    // Now test with TWO consecutive rescales to see if they interfere
    println!("\n=== Testing two consecutive rescales ===\n");

    let poly2 = RnsPolynomial::from_coeffs(&vec![200i64; n], &primes, n, 0);

    println!("Rescaling poly1...");
    let rescaled1 = rns_rescale_reference(&poly, &primes);
    println!("  rescaled1[0][0]: {}", rescaled1.rns_coeffs[0][0]);

    println!("Rescaling poly2...");
    let rescaled2 = rns_rescale_reference(&poly2, &primes);
    println!("  rescaled2[0][0]: {}", rescaled2.rns_coeffs[0][0]);

    println!("\nChecking rescaled1 after rescaled2 was created:");
    println!("  rescaled1[0][0]: {} (should still be the same)", rescaled1.rns_coeffs[0][0]);

    // Test with large values to match real usage
    println!("\n=== Testing with large values (matching real usage) ===\n");

    let large_coeffs = vec![2147470i64; n];  // Value from real rescale output
    let poly_large = RnsPolynomial::from_coeffs(&large_coeffs, &primes, n, 0);

    println!("Rescaling polynomial with coeff[0] = 2147470...");
    let rescaled_large = rns_rescale_reference(&poly_large, &primes);

    println!("  rescaled.rns_coeffs[0].len(): {}", rescaled_large.rns_coeffs[0].len());
    println!("  rescaled.num_primes(): {}", rescaled_large.num_primes());
    println!("  rescaled[0][0]: {}", rescaled_large.rns_coeffs[0][0]);
    if rescaled_large.rns_coeffs[0].len() > 1 {
        println!("  rescaled[0][1]: {}", rescaled_large.rns_coeffs[0][1]);
    }

    println!("\n=== Test complete ===");
}
