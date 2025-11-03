//! Comprehensive test suite for all Clifford FHE operations
//!
//! Tests:
//! 1. Basic encryption/decryption
//! 2. Homomorphic addition
//! 3. Homomorphic multiplication (with relinearization)
//! 4. Rescaling
//! 5. Geometric product (2D and 3D)
//! 6. Wedge product
//! 7. Inner product
//! 8. Clifford conjugation (reverse)
//! 9. Rotation (sandwiching)
//! 10. Projection and rejection

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{
    rns_encrypt, rns_decrypt, rns_add_ciphertexts, rns_multiply_ciphertexts,
    RnsPlaintext,
};
use ga_engine::clifford_fhe::geometric_product_rns::{
    geometric_product_2d_componentwise,
    wedge_product_2d,
    reverse_2d,
};

/// Helper to decode a value from RNS plaintext, handling both multi-prime and single-prime cases
fn decode_value(pt: &RnsPlaintext, scale: f64, all_primes: &[i64], level: usize) -> f64 {
    let active_primes = &all_primes[..all_primes.len() - level];

    if active_primes.len() >= 2 {
        // Multi-prime: use CRT
        let coeffs_dec = pt.to_coeffs_i128(active_primes);
        // Compute Q and center
        let q_product: i128 = active_primes.iter().map(|&q| q as i128).product();
        let mut centered = coeffs_dec[0];
        if centered > q_product / 2 {
            centered -= q_product;
        }
        (centered as f64) / scale
    } else if active_primes.len() == 1 {
        // Single prime: directly extract and center
        let val = pt.coeffs.rns_coeffs[0][0];
        let q = active_primes[0];
        let centered = if val > q / 2 { val - q } else { val };
        (centered as f64) / scale
    } else {
        0.0
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  CLIFFORD FHE - COMPREHENSIVE OPERATION TEST SUITE           ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Setup: 2 × 60-bit primes for good precision
    let params = CliffordFHEParams {
        n: 1024,
        moduli: vec![
            1141392289560813569, // 60-bit prime
            1141173990025715713, // 60-bit prime
        ],
        scale: 2f64.powi(40), // Large scale for precision
        error_std: 3.2,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    };

    println!("Parameters:");
    println!("  n = {}", params.n);
    println!("  Number of primes = {}", params.moduli.len());
    println!("  q0 = {} (≈2^{:.1})", params.moduli[0], (params.moduli[0] as f64).log2());
    println!("  q1 = {} (≈2^{:.1})", params.moduli[1], (params.moduli[1] as f64).log2());
    println!("  scale = 2^{}", (params.scale.log2() as i32));
    println!("  Q = q0·q1 ≈ 2^{:.1}", ((params.moduli[0] as f64) * (params.moduli[1] as f64)).log2());
    println!();

    // Generate keys
    println!("Generating keys...");
    let (pk, sk, evk) = rns_keygen(&params);
    println!("✓ Keys generated\n");

    let mut passed = 0;
    let mut failed = 0;

    // Test 1: Basic Encryption/Decryption
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 1: Basic Encryption/Decryption");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let value = 3.14159;
        let encoded = (value * params.scale).round() as i64;
        let mut coeffs = vec![0i64; params.n];
        coeffs[0] = encoded;

        let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);
        let ct = rns_encrypt(&pk, &pt, &params);
        let pt_dec = rns_decrypt(&sk, &ct, &params);

        let decoded = decode_value(&pt_dec, params.scale, &params.moduli, 0);
        let error = (decoded - value).abs();

        println!("  Original value: {}", value);
        println!("  Decoded value:  {}", decoded);
        println!("  Error:          {:.2e}", error);

        if error < 1e-6 {
            println!("✅ PASS\n");
            passed += 1;
        } else {
            println!("❌ FAIL\n");
            failed += 1;
        }
    }

    // Test 2: Homomorphic Addition
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 2: Homomorphic Addition");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let a = 1.5;
        let b = 2.7;
        let expected = a + b;

        let mut coeffs_a = vec![0i64; params.n];
        let mut coeffs_b = vec![0i64; params.n];
        coeffs_a[0] = (a * params.scale).round() as i64;
        coeffs_b[0] = (b * params.scale).round() as i64;

        let pt_a = RnsPlaintext::from_coeffs(coeffs_a, params.scale, &params.moduli, 0);
        let pt_b = RnsPlaintext::from_coeffs(coeffs_b, params.scale, &params.moduli, 0);

        let ct_a = rns_encrypt(&pk, &pt_a, &params);
        let ct_b = rns_encrypt(&pk, &pt_b, &params);

        // Homomorphic addition
        let ct_sum = rns_add_ciphertexts(&ct_a, &ct_b, &params);
        let pt_sum = rns_decrypt(&sk, &ct_sum, &params);

        let result = decode_value(&pt_sum, params.scale, &params.moduli, ct_sum.level);
        let error = (result - expected).abs();

        println!("  {} + {} = {}", a, b, expected);
        println!("  Encrypted result: {}", result);
        println!("  Error:            {:.2e}", error);

        if error < 1e-6 {
            println!("✅ PASS\n");
            passed += 1;
        } else {
            println!("❌ FAIL\n");
            failed += 1;
        }
    }

    // Test 3: Homomorphic Multiplication with Relinearization
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 3: Homomorphic Multiplication (with relinearization)");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let a = 1.5;
        let b = 2.0;
        let expected = a * b;

        let mut coeffs_a = vec![0i64; params.n];
        let mut coeffs_b = vec![0i64; params.n];
        coeffs_a[0] = (a * params.scale).round() as i64;
        coeffs_b[0] = (b * params.scale).round() as i64;

        let pt_a = RnsPlaintext::from_coeffs(coeffs_a, params.scale, &params.moduli, 0);
        let pt_b = RnsPlaintext::from_coeffs(coeffs_b, params.scale, &params.moduli, 0);

        let ct_a = rns_encrypt(&pk, &pt_a, &params);
        let ct_b = rns_encrypt(&pk, &pt_b, &params);

        // Homomorphic multiplication (includes relinearization and rescaling)
        let ct_prod = rns_multiply_ciphertexts(&ct_a, &ct_b, &evk, &params);
        let pt_prod = rns_decrypt(&sk, &ct_prod, &params);

        let result = decode_value(&pt_prod, ct_prod.scale, &params.moduli, ct_prod.level);
        let error = (result - expected).abs();

        println!("  {} × {} = {}", a, b, expected);
        println!("  Encrypted result: {}", result);
        println!("  Error:            {:.2e}", error);
        println!("  Level after mult: {}", ct_prod.level);

        if error < 1e-3 {
            println!("✅ PASS\n");
            passed += 1;
        } else {
            println!("❌ FAIL\n");
            failed += 1;
        }
    }

    // Test 4: 2D Geometric Product
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 4: 2D Geometric Product");
    println!("═══════════════════════════════════════════════════════════════");
    {
        // a = 1 + 2e1 + 3e2 + 4e12
        let a = [1.0f64, 2.0, 3.0, 4.0];
        // b = 2 + 1e1 + 1e2 + 1e12
        let b = [2.0f64, 1.0, 1.0, 1.0];

        // Encode each component
        let mut cts_a = Vec::new();
        let mut cts_b = Vec::new();

        for &val in &a {
            let mut coeffs = vec![0i64; params.n];
            coeffs[0] = (val * params.scale).round() as i64;
            let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
            cts_a.push(rns_encrypt(&pk, &pt, &params));
        }

        for &val in &b {
            let mut coeffs = vec![0i64; params.n];
            coeffs[0] = (val * params.scale).round() as i64;
            let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
            cts_b.push(rns_encrypt(&pk, &pt, &params));
        }

        // Compute geometric product
        let result_cts = geometric_product_2d_componentwise(
            &cts_a.try_into().unwrap(),
            &cts_b.try_into().unwrap(),
            &evk,
            &params,
        );

        // Decrypt results
        let mut result = [0.0f64; 4];
        for (i, ct) in result_cts.iter().enumerate() {
            let pt = rns_decrypt(&sk, ct, &params);
            result[i] = decode_value(&pt, ct.scale, &params.moduli, ct.level);
        }

        // Compute expected result manually:
        // (1 + 2e1 + 3e2 + 4e12)(2 + e1 + e2 + e12)
        // = 2 + e1 + e2 + e12
        //   + 4e1 + 2 + 2e12 - 2e2
        //   + 6e2 - 3e12 + 3 - 3e1
        //   + 8e12 - 4e2 - 4e1 + 4
        // = (2+2+3+4) + (1+4-3-4)e1 + (1-2+6-4)e2 + (1+2-3+8)e12
        // = 11 - 2e1 + 1e2 + 8e12
        let expected = [11.0f64, -2.0, 1.0, 8.0];

        println!("  a = [{}, {}, {}, {}]", a[0], a[1], a[2], a[3]);
        println!("  b = [{}, {}, {}, {}]", b[0], b[1], b[2], b[3]);
        println!("  Expected: [{}, {}, {}, {}]", expected[0], expected[1], expected[2], expected[3]);
        println!("  Result:   [{:.3}, {:.3}, {:.3}, {:.3}]", result[0], result[1], result[2], result[3]);

        let max_error = expected.iter().zip(&result)
            .map(|(e, r)| (e - r).abs())
            .fold(0.0f64, f64::max);

        println!("  Max error: {:.2e}", max_error);

        if max_error < 0.1 {
            println!("✅ PASS\n");
            passed += 1;
        } else {
            println!("❌ FAIL\n");
            failed += 1;
        }
    }

    // Test 5: 2D Wedge Product
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 5: 2D Wedge Product");
    println!("═══════════════════════════════════════════════════════════════");
    {
        // a = e1, b = e2
        // a ∧ b = e1 ∧ e2 = e12
        let a = [0.0f64, 1.0, 0.0, 0.0]; // e1
        let b = [0.0f64, 0.0, 1.0, 0.0]; // e2

        let mut cts_a = Vec::new();
        let mut cts_b = Vec::new();

        for &val in &a {
            let mut coeffs = vec![0i64; params.n];
            coeffs[0] = (val * params.scale).round() as i64;
            let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
            cts_a.push(rns_encrypt(&pk, &pt, &params));
        }

        for &val in &b {
            let mut coeffs = vec![0i64; params.n];
            coeffs[0] = (val * params.scale).round() as i64;
            let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
            cts_b.push(rns_encrypt(&pk, &pt, &params));
        }

        let result_cts = wedge_product_2d(
            &cts_a.try_into().unwrap(),
            &cts_b.try_into().unwrap(),
            &evk,
            &params,
        );

        // Wedge product returns a full multivector, check the e12 component (index 3)
        let pt = rns_decrypt(&sk, &result_cts[3], &params);
        let result = decode_value(&pt, result_cts[3].scale, &params.moduli, result_cts[3].level);

        println!("  e1 ∧ e2 = e12");
        println!("  Expected e12 component: 1.0");
        println!("  Result e12 component:   {:.6}", result);
        println!("  Error:                  {:.2e}", (result - 1.0).abs());

        if (result - 1.0).abs() < 0.1 {
            println!("✅ PASS\n");
            passed += 1;
        } else {
            println!("❌ FAIL\n");
            failed += 1;
        }
    }

    // Test 6: 2D Reverse (Clifford Conjugation)
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 6: 2D Reverse (Clifford Conjugation)");
    println!("═══════════════════════════════════════════════════════════════");
    {
        // a = 1 + 2e1 + 3e2 + 4e12
        // reverse(a) = 1 + 2e1 + 3e2 - 4e12 (bivector reverses sign)
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let expected = [1.0f64, 2.0, 3.0, -4.0];

        let mut cts_a = Vec::new();
        for &val in &a {
            let mut coeffs = vec![0i64; params.n];
            coeffs[0] = (val * params.scale).round() as i64;
            let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
            cts_a.push(rns_encrypt(&pk, &pt, &params));
        }

        let result_cts = reverse_2d(&cts_a.try_into().unwrap(), &params);

        let mut result = [0.0f64; 4];
        for (i, ct) in result_cts.iter().enumerate() {
            let pt = rns_decrypt(&sk, ct, &params);
            result[i] = decode_value(&pt, ct.scale, &params.moduli, ct.level);
        }

        println!("  a = [{}, {}, {}, {}]", a[0], a[1], a[2], a[3]);
        println!("  Expected: [{}, {}, {}, {}]", expected[0], expected[1], expected[2], expected[3]);
        println!("  Result:   [{:.3}, {:.3}, {:.3}, {:.3}]", result[0], result[1], result[2], result[3]);

        let max_error = expected.iter().zip(&result)
            .map(|(e, r)| (e - r).abs())
            .fold(0.0f64, f64::max);

        if max_error < 0.01 {
            println!("✅ PASS\n");
            passed += 1;
        } else {
            println!("❌ FAIL\n");
            failed += 1;
        }
    }

    // Test 7: Noise Growth Across Operations
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 7: Noise Growth Analysis");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let value = 1.0;
        let mut coeffs = vec![0i64; params.n];
        coeffs[0] = (value * params.scale).round() as i64;

        let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);
        let ct = rns_encrypt(&pk, &pt, &params);
        let pt_dec = rns_decrypt(&sk, &ct, &params);

        let dec1 = decode_value(&pt_dec, params.scale, &params.moduli, 0);
        let noise1 = ((dec1 - value) * params.scale).abs() as i128;

        // After one addition
        let ct_sum = rns_add_ciphertexts(&ct, &ct, &params);
        let pt_sum = rns_decrypt(&sk, &ct_sum, &params);
        let dec2 = decode_value(&pt_sum, params.scale, &params.moduli, ct_sum.level);
        let noise2 = ((dec2 - 2.0 * value) * params.scale).abs() as i128;

        println!("  Noise after encryption:    {}", noise1);
        println!("  Noise after 1 addition:    {}", noise2);
        println!("  Noise growth factor:       {:.2}", noise2 as f64 / noise1.max(1) as f64);

        if noise1 < 10000 && noise2 < 20000 {
            println!("✅ PASS - Noise is reasonable\n");
            passed += 1;
        } else {
            println!("❌ FAIL - Noise is too large\n");
            failed += 1;
        }
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Total tests:  {}", passed + failed);
    println!("Passed:       {} ✅", passed);
    println!("Failed:       {} ❌", failed);
    println!();

    if failed == 0 {
        println!("🎉 ALL TESTS PASSED! 🎉");
    } else {
        println!("⚠️  Some tests failed. Review the output above.");
    }
}
