//! Test homomorphic geometric product with rotation keys
//!
//! This example demonstrates the core innovation of Clifford-FHE:
//! performing geometric product operations on encrypted multivectors!
//!
//! Test case: (1 + 2e1) ⊗ (3 + 4e2)
//! Expected: 3 + 6e1 + 4e2 + 8e12

use ga_engine::clifford_fhe::{
    decrypt, encode_multivector, decode_multivector, encrypt,
    geometric_product_homomorphic, keygen_with_rotation,
    Plaintext, CliffordFHEParams,
};

fn main() {
    println!("=================================================================");
    println!("Clifford-FHE: Homomorphic Geometric Product (Rotation-Based)");
    println!("=================================================================\n");

    // Set up parameters (using test parameters for faster testing)
    let params = CliffordFHEParams::new_test();
    println!("Parameters (TEST - not secure!):");
    println!("  Ring dimension (N): {}", params.n);
    println!("  Scaling factor: 2^{}\n", params.scale.log2() as u32);

    // Generate keys (including rotation keys!)
    println!("Generating keys...");
    let (pk, sk, evk, rotk) = keygen_with_rotation(&params);
    println!("✓ Keys generated (including rotation keys for GP)\n");

    // Test case: (1 + 2e1) ⊗ (3 + 4e2)
    println!("Test: (1 + 2e₁) ⊗ (3 + 4e₂)");
    println!("-------------------------------------------------------------");

    let mv_a = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 1 + 2e1
    let mv_b = [3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 3 + 4e2

    println!("a = {:?}", &mv_a[..4]);
    println!("b = {:?}\n", &mv_b[..4]);

    // Expected result: (1 + 2e1) ⊗ (3 + 4e2)
    // = 1*3 + 1*4e2 + 2e1*3 + 2e1*4e2
    // = 3 + 4e2 + 6e1 + 8e12
    // = 3 + 6e1 + 4e2 + 8e12
    let expected = [3.0, 6.0, 4.0, 0.0, 8.0, 0.0, 0.0, 0.0];
    println!("Expected: a ⊗ b = {:?}", &expected[..5]);
    println!("  [scalar, e1, e2, e3, e12, ...]\n");

    // Encrypt multivectors
    println!("Encrypting multivectors...");
    let pt_a_coeffs = encode_multivector(&mv_a, params.scale, params.n);
    let pt_a = Plaintext::new(pt_a_coeffs, params.scale);
    let ct_a = encrypt(&pk, &pt_a, &params);

    let pt_b_coeffs = encode_multivector(&mv_b, params.scale, params.n);
    let pt_b = Plaintext::new(pt_b_coeffs, params.scale);
    let ct_b = encrypt(&pk, &pt_b, &params);
    println!("✓ Multivectors encrypted\n");

    // HOMOMORPHIC GEOMETRIC PRODUCT!
    println!("Computing homomorphic geometric product...");
    println!("(This uses rotation keys to compute structure constants)");
    let ct_result = geometric_product_homomorphic(&ct_a, &ct_b, &evk, &rotk, &params);
    println!("✓ Geometric product computed homomorphically!\n");

    // Decrypt result
    println!("Decrypting result...");
    let pt_result = decrypt(&sk, &ct_result, &params);
    let mv_result = decode_multivector(&pt_result.coeffs, params.scale);
    println!("✓ Result decrypted\n");

    // Display results
    println!("=================================================================");
    println!("Results");
    println!("=================================================================");
    println!("Result: {:?}", &mv_result[..5]);
    println!("Expected: {:?}\n", &expected[..5]);

    // Check each component
    println!("Component-wise comparison:");
    println!("-------------------------------------------------------------");
    let labels = ["scalar", "e1", "e2", "e3", "e12", "e13", "e23", "e123"];
    let mut max_error = 0.0f64;
    let mut all_correct = true;

    for i in 0..8 {
        let error = (mv_result[i] - expected[i]).abs();
        max_error = max_error.max(error);

        let status = if error < 0.5 {
            "✅"
        } else {
            all_correct = false;
            "❌"
        };

        println!(
            "  {} {:<6}: {:>8.2} (expected {:>6.2}, error: {:.2e})",
            status, labels[i], mv_result[i], expected[i], error
        );
    }

    println!("\n=================================================================");
    println!("Summary");
    println!("=================================================================");
    println!("Max error: {:.2e}", max_error);

    if all_correct {
        println!("✅ **SUCCESS!** Homomorphic geometric product works!");
        println!("   Computed (1 + 2e₁) ⊗ (3 + 4e₂) = 3 + 6e₁ + 4e₂ + 8e₁₂");
        println!("   entirely on encrypted data!");
    } else {
        println!("⚠️  Results have large errors. Further debugging needed.");
        println!("   This could be due to:");
        println!("   - Rotation implementation needs adjustment");
        println!("   - Coefficient positioning issues");
        println!("   - Noise accumulation");
    }

    println!("\n=================================================================");
    println!("This demonstrates THE KEY innovation of Clifford-FHE:");
    println!("Computing geometric algebra operations on ENCRYPTED multivectors!");
    println!("=================================================================");
}
