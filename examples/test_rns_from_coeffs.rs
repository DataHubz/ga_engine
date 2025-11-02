//! Test RnsPolynomial::from_coeffs for large values

use ga_engine::clifford_fhe::rns::RnsPolynomial;

fn main() {
    let primes = vec![1099511627689i64, 1099511627691, 1099511627693];
    let n = 4;

    // Test 1: Small positive value
    println!("Test 1: Small positive value (5)");
    let coeffs = vec![5, 0, 0, 0];
    let rns = RnsPolynomial::from_coeffs(&coeffs, &primes, n, 0);
    println!("  Input: {:?}", coeffs);
    println!("  RNS:   {:?}\n", rns.rns_coeffs[0]);

    // Test 2: Large positive value
    println!("Test 2: Large positive value (5497558138880)");
    let value = 5497558138880i64;
    let coeffs = vec![value, 0, 0, 0];
    let rns = RnsPolynomial::from_coeffs(&coeffs, &primes, n, 0);
    println!("  Input: {:?}", coeffs);
    println!("  RNS:   {:?}", rns.rns_coeffs[0]);

    // Manual calculation
    for (i, &q) in primes.iter().enumerate() {
        let expected = ((value % q) + q) % q;
        let actual = rns.rns_coeffs[0][i];
        println!("  Prime {}: {} mod {} = {} (expected {})", i, value, q, actual, expected);
        if actual != expected {
            println!("    ⚠️  MISMATCH!");
        }
    }
    println!();

    // Test 3: Negative value
    println!("Test 3: Negative value (-100)");
    let coeffs = vec![-100, 0, 0, 0];
    let rns = RnsPolynomial::from_coeffs(&coeffs, &primes, n, 0);
    println!("  Input: {:?}", coeffs);
    println!("  RNS:   {:?}", rns.rns_coeffs[0]);

    for (i, &q) in primes.iter().enumerate() {
        let expected = ((-100i64 % q) + q) % q;
        let actual = rns.rns_coeffs[0][i];
        println!("  Prime {}: -100 mod {} = {} (expected {})", i, q, actual, expected);
    }
    println!();

    // Test 4: Roundtrip with to_coeffs_single_prime
    println!("Test 4: Roundtrip with single prime extraction");
    let value = 5497558138880i64;
    let coeffs = vec![value, 0, 0, 0];
    let rns = RnsPolynomial::from_coeffs(&coeffs, &primes, n, 0);

    println!("  Original: {}", value);
    println!("  RNS: {:?}", rns.rns_coeffs[0]);

    let recovered = rns.to_coeffs_single_prime(0, primes[0]);
    println!("  Recovered (single prime): {}", recovered[0]);
    println!("  Expected: {} (residue mod prime 0)", value % primes[0]);

    if recovered[0] == value % primes[0] {
        println!("  ✓ PASS");
    } else {
        println!("  ✗ FAIL");
    }
}
