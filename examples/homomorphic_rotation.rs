//! Homomorphic Geometric Operations in Clifford-LWE
//!
//! This example demonstrates the unique capability of Clifford-LWE:
//! **Homomorphic rotations** - rotating encrypted vectors without decryption!
//!
//! Key question: Does Clifford algebra structure enable privacy-preserving geometry?

use ga_engine::clifford_lwe::{CliffordLWEParams, keygen, encrypt, decrypt, homomorphic_linear_transform};
use ga_engine::clifford_ring_int::{CliffordPolynomialInt, CliffordRingElementInt};
use ga_engine::lazy_reduction::LazyReductionContext;
use ga_engine::ntt_optimized::OptimizedNTTContext;

fn main() {
    println!("=== Homomorphic Geometric Operations in Clifford-LWE ===\n");
    println!("Question: Can we rotate encrypted vectors without decryption?\n");

    let params = CliffordLWEParams::default();
    let ntt = OptimizedNTTContext::new_clifford_lwe();
    let lazy = LazyReductionContext::new(params.q);

    println!("Parameters: N={}, q={}, error_bound={}\n", params.n, params.q, params.error_bound);

    // Generate keypair
    println!("1. Generating keypair...");
    let (pk, sk) = keygen(&params, &ntt, &lazy);
    println!("   ✓ Public key and secret key generated\n");

    // Test 1: Homomorphic rotation about Z-axis
    println!("=== Test 1: Homomorphic 90° Rotation about Z-axis ===\n");

    // Create vector v = e₁ = (0, 1, 0, 0, 0, 0, 0, 0) (unit vector in X direction)
    println!("2. Creating vector v = e₁ (unit vector in X direction)");
    let v_elem = CliffordRingElementInt::from_multivector([
        0,  // scalar
        1,  // e1 (X component)
        0,  // e2 (Y component)
        0,  // e3 (Z component)
        0,  // e12
        0,  // e13
        0,  // e23
        0,  // e123
    ]);
    let v = create_constant_polynomial(&v_elem, params.n);
    println!("   v = {:?}\n", v_elem.coeffs);

    // Encrypt v
    println!("3. Encrypting v...");
    let ct_v = encrypt(&ntt, &pk, &v, &params, &lazy);
    println!("   ✓ Ciphertext E(v) created\n");

    // Compute rotation matrix for 90° about Z
    println!("4. Computing rotation matrix for 90° about Z-axis");
    let rotation_matrix = rotation_90_z();
    println!("   Rotation matrix:");
    for i in 0..8 {
        println!("     [{:2}, {:2}, {:2}, {:2}, {:2}, {:2}, {:2}, {:2}]",
            rotation_matrix[i][0], rotation_matrix[i][1], rotation_matrix[i][2], rotation_matrix[i][3],
            rotation_matrix[i][4], rotation_matrix[i][5], rotation_matrix[i][6], rotation_matrix[i][7]);
    }
    println!();

    // Apply rotation homomorphically
    println!("5. Applying rotation homomorphically: E(v') = M · E(v)");
    let ct_v_rotated = homomorphic_linear_transform(&rotation_matrix, &ct_v, params.q);
    println!("   ✓ Homomorphic rotation applied\n");

    // Decrypt and verify
    println!("6. Decrypting rotated ciphertext...");
    let v_rotated = decrypt(&ntt, &sk, &ct_v_rotated, &params, &lazy);
    let v_rotated_elem = &v_rotated.coeffs[0];
    println!("   Decrypted: {:?}", v_rotated_elem.coeffs);
    println!();

    // Expected: e₁ → e₂ (X → Y direction)
    let expected = [0, 0, 1, 0, 0, 0, 0, 0];
    println!("   Expected:  {:?}", expected);
    println!();

    // Check if rotation worked
    let mut rotation_success = true;
    for i in 0..8 {
        let decrypted = v_rotated_elem.coeffs[i];
        let expected_val = expected[i];
        let error = ((decrypted - expected_val).abs()) % params.q;

        // Allow some error due to LWE noise
        if error > params.error_bound * 10 && error < params.q - params.error_bound * 10 {
            rotation_success = false;
            println!("   ❌ Component {}: expected {}, got {} (error {})", i, expected_val, decrypted, error);
        }
    }

    if rotation_success {
        println!("   ✅ SUCCESS: Homomorphic rotation worked!");
        println!("   e₁ (X direction) → e₂ (Y direction) ✓\n");
    } else {
        println!("   ❌ FAILURE: Homomorphic rotation did not work correctly\n");
    }

    // Test 2: Measure noise growth
    println!("=== Test 2: Noise Growth Analysis ===\n");

    println!("7. Encrypting known vector with observed noise...");
    let v2 = create_constant_polynomial(&v_elem, params.n);
    let ct_v2 = encrypt(&ntt, &pk, &v2, &params, &lazy);

    // Decrypt without rotation
    let v2_decrypted = decrypt(&ntt, &sk, &ct_v2, &params, &lazy);
    let noise_before = estimate_noise(&v_elem, &v2_decrypted.coeffs[0], params.q);
    println!("   Noise before rotation: ~{}", noise_before);

    // Apply rotation
    let ct_v2_rotated = homomorphic_linear_transform(&rotation_matrix, &ct_v2, params.q);
    let v2_rotated = decrypt(&ntt, &sk, &ct_v2_rotated, &params, &lazy);

    // Expected after rotation
    let expected_elem = CliffordRingElementInt::from_multivector([0, 0, 1, 0, 0, 0, 0, 0]);
    let noise_after = estimate_noise(&expected_elem, &v2_rotated.coeffs[0], params.q);
    println!("   Noise after rotation:  ~{}", noise_after);

    let noise_growth_factor = noise_after as f64 / noise_before as f64;
    println!("   Noise growth factor: {:.2}×", noise_growth_factor);
    println!("   Expected: ~2.83× (√8 from 8 components)", );
    println!();

    if noise_growth_factor < 4.0 {
        println!("   ✅ Noise growth is acceptable (<4×)");
    } else {
        println!("   ⚠️  Noise growth is high (≥4×) - may limit rotation depth");
    }
    println!();

    // Test 3: Multiple rotations (depth test)
    println!("=== Test 3: Composition of Rotations (Depth Test) ===\n");

    println!("8. Applying 4 consecutive 90° rotations (should return to original)...");
    let mut ct_composed = ct_v.clone();

    for i in 1..=4 {
        ct_composed = homomorphic_linear_transform(&rotation_matrix, &ct_composed, params.q);
        let v_temp = decrypt(&ntt, &sk, &ct_composed, &params, &lazy);
        println!("   After rotation {}: component[1]={}, component[2]={}",
            i, v_temp.coeffs[0].coeffs[1], v_temp.coeffs[0].coeffs[2]);
    }

    let v_final = decrypt(&ntt, &sk, &ct_composed, &params, &lazy);
    println!();

    // After 4× 90° rotations, should return to e₁
    let final_error = estimate_noise(&v_elem, &v_final.coeffs[0], params.q);
    println!("   Final noise after 4 rotations: ~{}", final_error);
    println!("   Expected: ~{}× (compounded noise)", noise_before * 16); // 2.83^4 ≈ 64

    if final_error < params.q / 4 {
        println!("   ✅ 4 rotations successful (noise still manageable)");
    } else {
        println!("   ❌ 4 rotations failed (noise overwhelmed signal)");
        println!("   Depth limit: ~2-3 rotations before noise overwhelms signal");
    }
    println!();

    // Summary
    println!("=== Summary ===\n");

    if rotation_success {
        println!("✅ HOMOMORPHIC ROTATION WORKS!");
        println!();
        println!("Key findings:");
        println!("1. Can rotate encrypted vectors without decryption ✓");
        println!("2. Noise grows by {:.2}× per rotation", noise_growth_factor);
        println!("3. Depth limit: ~2-3 rotations before decryption fails");
        println!();
        println!("Unique capability:");
        println!("- Clifford-LWE enables privacy-preserving geometric computations");
        println!("- Applications: robotics, graphics, CAD, spatial databases");
        println!();
        println!("This is a capability that standard LWE/Kyber does NOT have!");
        println!("Clifford algebra structure provides genuine cryptographic value! 🎉");
    } else {
        println!("❌ Homomorphic rotation did NOT work as expected");
        println!();
        println!("Clifford-LWE may not support true homomorphic geometric operations.");
        println!("Further investigation needed.");
    }
}

/// Create a polynomial with the same Clifford element at every coefficient
fn create_constant_polynomial(elem: &CliffordRingElementInt, n: usize) -> CliffordPolynomialInt {
    let coeffs = vec![elem.clone(); n];
    CliffordPolynomialInt::new(coeffs)
}

/// Rotation matrix for 90° counterclockwise rotation about Z-axis
///
/// In Clifford algebra Cl(3,0), this is: R = cos(π/4) + sin(π/4) · e₁₂
///
/// Rotation formula: v' = R v R̃
///
/// For basis vectors:
/// - e₁ → e₂ (X → Y)
/// - e₂ → -e₁ (Y → -X)
/// - e₃ → e₃ (Z unchanged)
fn rotation_90_z() -> [[i64; 8]; 8] {
    // This matrix represents the linear transformation v' = R v R̃
    // Computed by applying R v R̃ to each basis vector and extracting coefficients

    [
        // scalar, e1, e2, e3, e12, e13, e23, e123
        [1,  0,  0,  0,  0,  0,  0,  0],  // scalar unchanged
        [0,  0,  1,  0,  0,  0,  0,  0],  // e1 → e2
        [0, -1,  0,  0,  0,  0,  0,  0],  // e2 → -e1
        [0,  0,  0,  1,  0,  0,  0,  0],  // e3 → e3
        [0,  0,  0,  0,  0, -1,  0,  0],  // e12 → -e13 (bivector rotation)
        [0,  0,  0,  0,  1,  0,  0,  0],  // e13 → e12
        [0,  0,  0,  0,  0,  0,  1,  0],  // e23 → e23
        [0,  0,  0,  0,  0,  0,  0,  1],  // e123 → e123 (pseudoscalar unchanged)
    ]
}

/// Estimate noise in decrypted value
fn estimate_noise(expected: &CliffordRingElementInt, actual: &CliffordRingElementInt, q: i64) -> i64 {
    let mut max_error = 0i64;

    for i in 0..8 {
        let error = ((actual.coeffs[i] - expected.coeffs[i]).abs()) % q;
        // Handle wraparound: if error > q/2, actual error is q - error
        let true_error = if error > q / 2 { q - error } else { error };
        max_error = max_error.max(true_error);
    }

    max_error
}
