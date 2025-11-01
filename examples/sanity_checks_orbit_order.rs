//! Sanity checks for orbit-order CKKS implementation
//!
//! These are the 5 checks recommended by the expert to verify correctness

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::canonical_embedding::{encode_multivector_canonical, decode_multivector_canonical};
use ga_engine::clifford_fhe::automorphisms::apply_automorphism;

fn pow_mod(base: usize, exp: usize, modulus: usize) -> usize {
    if exp == 0 {
        return 1;
    }
    let mut result = 1usize;
    let mut base = base % modulus;
    let mut exp = exp;

    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }
    result
}

fn main() {
    println!("=================================================================");
    println!("CKKS Orbit Order - Sanity Checks");
    println!("=================================================================\n");

    let params = CliffordFHEParams::new_test();
    let n = params.n; // 32
    let m = 2 * n;    // 64
    let g = 5;        // Generator

    println!("Parameters:");
    println!("  N = {}", n);
    println!("  M = 2N = {}", m);
    println!("  g = {} (generator)", g);
    println!("  Number of slots = N/2 = {}\n", n/2);

    println!("-----------------------------------------------------------------");
    println!("Check 1: Order of generator g");
    println!("-----------------------------------------------------------------\n");

    let order_full = pow_mod(g, n/2, m);
    let order_half = pow_mod(g, n/4, m);

    println!("g^(N/2) mod M = {}^{} mod {} = {}", g, n/2, m, order_full);
    println!("g^(N/4) mod M = {}^{} mod {} = {}", g, n/4, m, order_half);

    if order_full == 1 && order_half != 1 {
        println!("✓ PASS: g has order N/2 = {}\n", n/2);
    } else {
        println!("✗ FAIL: g does not have correct order!\n");
    }

    println!("-----------------------------------------------------------------");
    println!("Check 2: Orbit length and properties");
    println!("-----------------------------------------------------------------\n");

    let mut orbit = vec![0usize; n/2];
    let mut cur = 1usize;
    for t in 0..(n/2) {
        orbit[t] = cur;
        cur = (cur * g) % m;
    }

    println!("Orbit: {:?}", &orbit);

    // Check all distinct
    let mut sorted = orbit.clone();
    sorted.sort();
    let all_distinct = sorted.windows(2).all(|w| w[0] != w[1]);

    // Check all odd
    let all_odd = orbit.iter().all(|&x| x % 2 == 1);

    println!("All elements distinct: {}", all_distinct);
    println!("All elements odd: {}", all_odd);

    if all_distinct && all_odd && orbit.len() == n/2 {
        println!("✓ PASS: Orbit has correct properties\n");
    } else {
        println!("✗ FAIL: Orbit is malformed!\n");
    }

    println!("-----------------------------------------------------------------");
    println!("Check 3: Rotation test (σ_g = rotate left by 1)");
    println!("-----------------------------------------------------------------\n");

    // Encode test vector [1, 2, 3, 4, 5, 6, 7, 8]
    let mut basis = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let coeffs = encode_multivector_canonical(&basis, params.scale, params.n);
    let coeffs_auto = apply_automorphism(&coeffs, g, n);
    let result = decode_multivector_canonical(&coeffs_auto, params.scale, params.n);

    println!("Input:  {:?}", &basis);
    println!("Output: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
             result[0], result[1], result[2], result[3],
             result[4], result[5], result[6], result[7]);
    println!("Expected: [2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 0.00]");

    let expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0];
    let max_error = result.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);

    if max_error < 0.01 {
        println!("✓ PASS: σ_g rotates left by 1 (error {:.2e})\n", max_error);
    } else {
        println!("✗ FAIL: σ_g doesn't rotate correctly (error {:.2e})!\n", max_error);
    }

    println!("-----------------------------------------------------------------");
    println!("Check 4: Inverse rotation (σ_{{g^{{-1}}}} = rotate right by 1)");
    println!("-----------------------------------------------------------------\n");

    // Compute g^(-1) mod M
    fn mod_inverse(a: usize, m: usize) -> usize {
        // Extended Euclidean algorithm
        let (mut old_r, mut r) = (a as i64, m as i64);
        let (mut old_s, mut s) = (1i64, 0i64);

        while r != 0 {
            let quotient = old_r / r;
            let temp_r = r;
            r = old_r - quotient * r;
            old_r = temp_r;

            let temp_s = s;
            s = old_s - quotient * s;
            old_s = temp_s;
        }

        ((old_s % (m as i64) + (m as i64)) % (m as i64)) as usize
    }

    let g_inv = mod_inverse(g, m);
    println!("g^(-1) mod M = {} (since {}·{} ≡ 1 mod {})", g_inv, g, g_inv, m);

    let coeffs_auto_inv = apply_automorphism(&coeffs, g_inv, n);
    let result_inv = decode_multivector_canonical(&coeffs_auto_inv, params.scale, params.n);

    println!("Input:  {:?}", &basis);
    println!("Output: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
             result_inv[0], result_inv[1], result_inv[2], result_inv[3],
             result_inv[4], result_inv[5], result_inv[6], result_inv[7]);
    println!("Expected: [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00]");

    let expected_inv = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let max_error_inv = result_inv.iter().zip(&expected_inv).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);

    if max_error_inv < 0.01 {
        println!("✓ PASS: σ_(g^(-1)) rotates right by 1 (error {:.2e})\n", max_error_inv);
    } else {
        println!("✗ FAIL: σ_(g^(-1)) doesn't rotate correctly (error {:.2e})!\n", max_error_inv);
    }

    println!("-----------------------------------------------------------------");
    println!("Check 5: Conjugate orbit");
    println!("-----------------------------------------------------------------\n");

    let conjugate_orbit: Vec<usize> = orbit.iter().map(|&e| (m - e) % m).collect();
    println!("Original orbit:   {:?}", &orbit);
    println!("Conjugate orbit:  {:?}", &conjugate_orbit);

    // Check they're disjoint
    let disjoint = orbit.iter().all(|e| !conjugate_orbit.contains(e));

    if disjoint {
        println!("✓ PASS: Conjugate orbit is disjoint from original\n");
    } else {
        println!("✗ FAIL: Orbits overlap!\n");
    }

    println!("=================================================================");
    println!("Summary");
    println!("=================================================================\n");

    let all_pass = order_full == 1 && order_half != 1 &&
                   all_distinct && all_odd &&
                   max_error < 0.01 &&
                   max_error_inv < 0.01 &&
                   disjoint;

    if all_pass {
        println!("🎉 ALL CHECKS PASS! Orbit-order CKKS is working correctly!");
        println!("   Rotations via automorphisms are now fully functional.\n");
    } else {
        println!("⚠️  Some checks failed - review implementation.\n");
    }
}
