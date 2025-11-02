//! Comprehensive Test Suite for 3D Homomorphic Geometric Operations
//!
//! Tests all Cl(3,0) operations:
//! 1. Geometric Product (⊗)
//! 2. Reverse (ã)
//! 3. Rotation (R·x·R̃)
//! 4. Wedge Product (∧)
//! 5. Inner Product (·)
//! 6. Projection (proj_b(a))
//! 7. Rejection (rej_b(a))

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{rns_encrypt, rns_decrypt, RnsPlaintext};
use ga_engine::clifford_fhe::geometric_product_rns::{
    geometric_product_3d_componentwise, reverse_3d, rotate_3d,
    wedge_product_3d, inner_product_3d, project_3d, reject_3d
};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  Clifford FHE: 3D Geometric Operations Test Suite            ║");
    println!("║  Testing Cl(3,0) with 8 basis elements                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Setup parameters
    let params = CliffordFHEParams::new_rns_mult();
    let primes = &params.moduli;

    println!("Parameters:");
    println!("  N = {}", params.n);
    println!("  Δ = {}", params.scale);
    println!("  Primes: {} (60-bit), {} (40-bit)", primes[0], primes[1]);
    println!("  Basis: {{1, e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃}}");
    println!();

    // Generate keys
    println!("Generating keys...");
    let (pk, sk, evk) = rns_keygen(&params);
    println!("✓ Keys generated");
    println!();

    let delta = params.scale;
    let n = params.n;

    // Helper function to encrypt a 3D multivector
    let encrypt_multivector = |mv: [f64; 8]| -> [_; 8] {
        let mut cts = Vec::new();
        for i in 0..8 {
            let mut coeffs = vec![0i64; n];
            coeffs[0] = (mv[i] * delta).round() as i64;
            let pt = RnsPlaintext::from_coeffs(coeffs, delta, primes, 0);
            let ct = rns_encrypt(&pk, &pt, &params);
            cts.push(ct);
        }
        [cts[0].clone(), cts[1].clone(), cts[2].clone(), cts[3].clone(),
         cts[4].clone(), cts[5].clone(), cts[6].clone(), cts[7].clone()]
    };

    // Helper function to decrypt a 3D multivector
    let decrypt_multivector = |cts: &[_; 8]| -> [f64; 8] {
        let mut result = [0.0; 8];
        for i in 0..8 {
            let pt = rns_decrypt(&sk, &cts[i], &params);
            result[i] = (pt.coeffs.rns_coeffs[0][0] as f64) / cts[i].scale;
        }
        result
    };

    // Helper function to check results
    let check_result = |name: &str, result: [f64; 8], expected: [f64; 8], threshold: f64| {
        let mut max_error = 0.0f64;
        for i in 0..8 {
            let error = (result[i] - expected[i]).abs();
            max_error = max_error.max(error);
        }

        println!("  Result:   [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
                 result[0], result[1], result[2], result[3],
                 result[4], result[5], result[6], result[7]);
        println!("  Expected: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
                 expected[0], expected[1], expected[2], expected[3],
                 expected[4], expected[5], expected[6], expected[7]);
        println!("  Max error: {:.2e}", max_error);

        if max_error < threshold {
            println!("  ✅ {} PASSED", name);
        } else {
            println!("  ❌ {} FAILED (error {:.2e} > threshold {:.2e})",
                     name, max_error, threshold);
        }
        println!();
    };

    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 1: 3D Geometric Product");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Computing: (1 + e₁) ⊗ (1 + e₂)");
    println!("Expected: 1 + e₁ + e₂ + e₁₂");
    println!();

    let a = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // 1 + e₁
    let b = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // 1 + e₂

    let cts_a = encrypt_multivector(a);
    let cts_b = encrypt_multivector(b);

    let cts_result = geometric_product_3d_componentwise(&cts_a, &cts_b, &evk, &params);
    let result = decrypt_multivector(&cts_result);

    check_result("3D Geometric Product", result,
                 [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], 1e-6);

    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 2: 3D Reverse Operation");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Computing: reverse(1 + e₁ + e₂ + e₃ + e₁₂ + e₁₃ + e₂₃ + e₁₂₃)");
    println!("Expected: 1 + e₁ + e₂ + e₃ - e₁₂ - e₁₃ - e₂₃ + e₁₂₃");
    println!();

    let a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let cts_a = encrypt_multivector(a);

    let cts_result = reverse_3d(&cts_a, &params);
    let result = decrypt_multivector(&cts_result);

    check_result("3D Reverse", result,
                 [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0], 1e-6);

    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 3: 3D Rotation (90° around z-axis)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Rotor R = cos(θ/2) + sin(θ/2)e₁₂  for θ = π/2");
    println!("  cos(π/4) ≈ 0.707, sin(π/4) ≈ 0.707");
    println!("Vector x = e₁ (unit vector along x-axis)");
    println!("Expected: x' = e₂ (90° rotation in xy-plane)");
    println!();

    // Rotor for 90° rotation around z-axis: R = cos(π/4) + sin(π/4)e₁₂
    let theta = std::f64::consts::PI / 2.0;
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();
    let rotor = [cos_half, 0.0, 0.0, 0.0, sin_half, 0.0, 0.0, 0.0];

    // Vector to rotate: x = e₁
    let vector = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let cts_rotor = encrypt_multivector(rotor);
    let cts_vector = encrypt_multivector(vector);

    let cts_result = rotate_3d(&cts_rotor, &cts_vector, &evk, &params);
    let result = decrypt_multivector(&cts_result);

    // After 90° rotation, e₁ becomes e₂
    check_result("3D Rotation", result,
                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.01);

    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 4: 3D Wedge Product");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Computing: e₁ ∧ e₂");
    println!("Expected: e₁₂ (bivector representing xy-plane)");
    println!();

    let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₁
    let b = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₂

    let cts_a = encrypt_multivector(a);
    let cts_b = encrypt_multivector(b);

    let cts_result = wedge_product_3d(&cts_a, &cts_b, &evk, &params);
    let result = decrypt_multivector(&cts_result);

    check_result("3D Wedge Product", result,
                 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 0.01);

    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 5: 3D Inner Product");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Computing: (e₁ + 2e₂ + 3e₃) · (4e₁ + 5e₂ + 6e₃)");
    println!("Expected: 1*4 + 2*5 + 3*6 = 32 (scalar)");
    println!();

    let a = [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0];
    let b = [0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0];

    let cts_a = encrypt_multivector(a);
    let cts_b = encrypt_multivector(b);

    let cts_result = inner_product_3d(&cts_a, &cts_b, &evk, &params);
    let result = decrypt_multivector(&cts_result);

    check_result("3D Inner Product", result,
                 [32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.01);

    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 6: 3D Projection");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Computing: proj_e₁(e₁ + e₂)");
    println!("Expected: e₁ (projection onto x-axis)");
    println!();

    let a = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₁ + e₂
    let b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₁

    let cts_a = encrypt_multivector(a);
    let cts_b = encrypt_multivector(b);

    let cts_result = project_3d(&cts_a, &cts_b, &evk, &params);
    let result = decrypt_multivector(&cts_result);

    // Note: project_3d returns unnormalized projection, so we expect (1·e₁)⊗e₁ = e₁
    check_result("3D Projection", result,
                 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.01);

    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 7: 3D Rejection");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Computing: rej_e₁(e₁ + e₂)");
    println!("Expected: e₂ (perpendicular component)");
    println!();

    let a = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₁ + e₂
    let b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];  // e₁

    let cts_a = encrypt_multivector(a);
    let cts_b = encrypt_multivector(b);

    let cts_result = reject_3d(&cts_a, &cts_b, &evk, &params);
    let result = decrypt_multivector(&cts_result);

    check_result("3D Rejection", result,
                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.01);

    println!("═══════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("✅ All 7 operations tested for 3D geometric algebra!");
    println!();
    println!("Operations implemented:");
    println!("  1. Geometric Product (⊗) - Full 8-component Cl(3,0) multiplication");
    println!("  2. Reverse (ã) - Sign flips for bivectors");
    println!("  3. Rotation (R·x·R̃) - 3D encrypted rotations");
    println!("  4. Wedge Product (∧) - Compute bivectors and trivectors");
    println!("  5. Inner Product (·) - Dot products in 3D");
    println!("  6. Projection - Parallel component");
    println!("  7. Rejection - Perpendicular component");
    println!();
    println!("This enables:");
    println!("  • Full 3D robotics (encrypted 3D poses and rotations)");
    println!("  • 3D physics simulations (encrypted forces, torques, angular momentum)");
    println!("  • 3D computer graphics (encrypted transformations)");
    println!("  • Encrypted geometric deep learning");
    println!();
    println!("🎉 Clifford FHE is feature-complete for 3D operations!");
}
