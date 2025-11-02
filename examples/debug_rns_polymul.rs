//! Debug RNS polynomial multiplication
//!
//! Test polynomial multiplication in isolation to verify the implementation

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::rns::RnsPolynomial;

/// Simple polynomial multiplication with negacyclic reduction
fn polynomial_multiply_naive(a: &[i64], b: &[i64], q: i64, n: usize) -> Vec<i64> {
    let mut result = vec![0i64; n];

    for i in 0..n {
        for j in 0..n {
            let idx = i + j;
            if idx < n {
                result[idx] = (result[idx] + a[i] * b[j]) % q;
            } else {
                // x^n = -1 reduction (negacyclic)
                let wrapped_idx = idx % n;
                result[wrapped_idx] = (result[wrapped_idx] - a[i] * b[j]) % q;
            }
        }
    }

    result.iter().map(|&x| ((x % q) + q) % q).collect()
}

fn main() {
    println!("========================================================================");
    println!("Debug: RNS Polynomial Multiplication");
    println!("========================================================================\n");

    let params = CliffordFHEParams::new_rns_mult();
    let primes = &params.moduli;
    let n = params.n;

    println!("Parameters:");
    println!("  N = {}", n);
    println!("  Primes: {:?}", primes);
    println!();

    // Test 1: Simple single-coefficient multiplication
    println!("Test 1: Simple multiplication [5, 0, ...] × [3, 0, ...]");
    println!("Expected: [15, 0, 0, ...]");
    println!("------------------------------------------------------------------------\n");

    let mut a_coeffs = vec![0i64; n];
    a_coeffs[0] = 5;

    let mut b_coeffs = vec![0i64; n];
    b_coeffs[0] = 3;

    // Convert to RNS
    let a_rns = RnsPolynomial::from_coeffs(&a_coeffs, primes, n, 0);
    let b_rns = RnsPolynomial::from_coeffs(&b_coeffs, primes, n, 0);

    println!("a[0] in RNS: {:?}", a_rns.rns_coeffs[0]);
    println!("b[0] in RNS: {:?}", b_rns.rns_coeffs[0]);
    println!();

    // Multiply for each prime separately
    for (prime_idx, &q) in primes.iter().enumerate() {
        println!("Prime {} (q = {}):", prime_idx, q);

        // Extract coefficients for this prime
        let a_mod_q: Vec<i64> = (0..n).map(|i| a_rns.rns_coeffs[i][prime_idx]).collect();
        let b_mod_q: Vec<i64> = (0..n).map(|i| b_rns.rns_coeffs[i][prime_idx]).collect();

        // Multiply
        let c_mod_q = polynomial_multiply_naive(&a_mod_q, &b_mod_q, q, n);

        println!("  a mod q: first 4 coeffs = {:?}", &a_mod_q[0..4]);
        println!("  b mod q: first 4 coeffs = {:?}", &b_mod_q[0..4]);
        println!("  c mod q: first 4 coeffs = {:?}", &c_mod_q[0..4]);
        println!("  Expected c[0] = 15");

        if c_mod_q[0] == 15 {
            println!("  ✓ PASS\n");
        } else {
            println!("  ✗ FAIL: got c[0] = {}\n", c_mod_q[0]);
        }
    }

    // Test 2: Verify negacyclic reduction works
    println!("Test 2: Negacyclic reduction [0, 0, ..., 1] × [0, 0, ..., 1]");
    println!("Expected: x^(N-1) × x^(N-1) = x^(2N-2) = x^(N-2) × x^N = -x^(N-2)");
    println!("So result should have coefficient[N-2] = -1 (or q-1 mod q)");
    println!("------------------------------------------------------------------------\n");

    let mut a_coeffs = vec![0i64; n];
    a_coeffs[n - 1] = 1;  // x^(N-1)

    let mut b_coeffs = vec![0i64; n];
    b_coeffs[n - 1] = 1;  // x^(N-1)

    let q = primes[0];
    let c_coeffs = polynomial_multiply_naive(&a_coeffs, &b_coeffs, q, n);

    println!("a: coefficient[{}] = 1 (rest zero)", n - 1);
    println!("b: coefficient[{}] = 1 (rest zero)", n - 1);
    println!("Result coefficients:");
    println!("  c[{}] = {} (expected {} for -1 mod {})", n - 2, c_coeffs[n - 2], q - 1, q);

    if c_coeffs[n - 2] == q - 1 {
        println!("  ✓ PASS: Negacyclic reduction works!\n");
    } else {
        println!("  ✗ FAIL: Negacyclic reduction broken!\n");
    }

    // Test 3: Check if RNS representation is consistent
    println!("Test 3: RNS consistency check");
    println!("Convert 5 to RNS, then back to integer, should get 5");
    println!("------------------------------------------------------------------------\n");

    let mut coeffs = vec![0i64; n];
    coeffs[0] = 5;

    let rns = RnsPolynomial::from_coeffs(&coeffs, primes, n, 0);

    println!("Original: coeffs[0] = 5");
    println!("RNS form: {:?}", rns.rns_coeffs[0]);

    // Extract using single prime
    let recovered = rns.to_coeffs_single_prime(0, primes[0]);
    println!("Recovered (single prime): {}", recovered[0]);

    if recovered[0] == 5 {
        println!("✓ PASS\n");
    } else {
        println!("✗ FAIL\n");
    }

    // Test 4: Scaled coefficient test (like in actual CKKS)
    println!("Test 4: Scaled coefficient (simulating CKKS encoding)");
    println!("------------------------------------------------------------------------\n");

    let scale = params.scale;
    let message = 5.0;
    let scaled = (message * scale).round() as i64;

    println!("Message: {}", message);
    println!("Scale: {:.2e}", scale);
    println!("Scaled coefficient: {}", scaled);

    let mut coeffs = vec![0i64; n];
    coeffs[0] = scaled;

    let rns = RnsPolynomial::from_coeffs(&coeffs, primes, n, 0);

    println!("RNS form of coefficient[0]: {:?}", rns.rns_coeffs[0]);

    // Check each prime
    for (idx, &q) in primes.iter().enumerate() {
        let expected = scaled % q;
        let actual = rns.rns_coeffs[0][idx];
        println!("  Prime {}: {} mod {} = {} (RNS has {})", idx, scaled, q, expected, actual);

        if expected != actual {
            println!("    ⚠️  MISMATCH!");
        }
    }

    // Recover using single prime
    let recovered_coeff = rns.to_coeffs_single_prime(0, primes[0]);
    let recovered_msg = recovered_coeff[0] as f64 / scale;

    println!("\nRecovered coefficient: {}", recovered_coeff[0]);
    println!("Recovered message: {:.6}", recovered_msg);
    println!("Original message: {:.6}", message);
    println!("Error: {:.6}", (recovered_msg - message).abs());

    if (recovered_msg - message).abs() < 0.001 {
        println!("✓ PASS\n");
    } else {
        println!("✗ FAIL\n");
    }

    println!("========================================================================");
}
