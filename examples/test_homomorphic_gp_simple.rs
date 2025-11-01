//! Test homomorphic geometric product with simple rotation
//!
//! This is the KEY test that validates the entire Clifford-FHE system!
//!
//! We test: Enc(a) ⊗ Enc(b) = Enc(a ⊗ b)

use ga_engine::clifford_fhe::{
    CliffordFHEParams, keygen,
    encode_multivector_slots, decode_multivector_slots,
    encrypt, decrypt, Plaintext,
};
use ga_engine::clifford_fhe::geometric_product::geometric_product_homomorphic_simple;

fn geometric_product_plaintext(a: &[f64; 8], b: &[f64; 8]) -> [f64; 8] {
    // Cl(3,0) geometric product
    // This is the reference implementation we're trying to match homomorphically!

    let mut result = [0.0; 8];

    // Component 0 (scalar)
    result[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
               - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];

    // Component 1 (e1)
    result[1] = a[0]*b[1] + a[1]*b[0] - a[2]*b[4] + a[4]*b[2]
               - a[3]*b[5] + a[5]*b[3] + a[6]*b[7] - a[7]*b[6];

    // Component 2 (e2)
    result[2] = a[0]*b[2] + a[2]*b[0] + a[1]*b[4] - a[4]*b[1]
               - a[3]*b[6] + a[6]*b[3] - a[5]*b[7] + a[7]*b[5];

    // Component 3 (e3)
    result[3] = a[0]*b[3] + a[3]*b[0] + a[1]*b[5] - a[5]*b[1]
               + a[2]*b[6] - a[6]*b[2] + a[4]*b[7] - a[7]*b[4];

    // Component 4 (e12)
    result[4] = a[0]*b[4] + a[4]*b[0] + a[1]*b[2] - a[2]*b[1]
               - a[3]*b[7] + a[7]*b[3] - a[5]*b[6] + a[6]*b[5];

    // Component 5 (e13)
    result[5] = a[0]*b[5] + a[5]*b[0] + a[1]*b[3] - a[3]*b[1]
               + a[2]*b[7] - a[7]*b[2] + a[4]*b[6] - a[6]*b[4];

    // Component 6 (e23)
    result[6] = a[0]*b[6] + a[6]*b[0] + a[2]*b[3] - a[3]*b[2]
               - a[1]*b[7] + a[7]*b[1] - a[4]*b[5] + a[5]*b[4];

    // Component 7 (e123)
    result[7] = a[0]*b[7] + a[7]*b[0] + a[1]*b[6] - a[6]*b[1]
               + a[2]*b[5] - a[5]*b[2] + a[3]*b[4] - a[4]*b[3];

    result
}

fn main() {
    println!("=================================================================");
    println!("Testing Homomorphic Geometric Product with Simple Rotation");
    println!("=================================================================\n");

    let params = CliffordFHEParams::new_test();
    let (pk, sk, evk) = keygen(&params);

    println!("-----------------------------------------------------------------");
    println!("Test 1: (1 + 2e1) ⊗ (3 + 4e2) = 3 + 6e1 + 4e2 + 8e12");
    println!("-----------------------------------------------------------------\n");

    let mv_a = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 1 + 2e1
    let mv_b = [3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 3 + 4e2

    println!("a = {:?}", mv_a);
    println!("b = {:?}", mv_b);

    // Compute expected result
    let expected = geometric_product_plaintext(&mv_a, &mv_b);
    println!("\nExpected a ⊗ b = {:?}", expected);
    println!("                 = 3 + 6e1 + 4e2 + 8e12\n");

    // Encrypt
    let coeffs_a = encode_multivector_slots(&mv_a, params.scale, params.n);
    let pt_a = Plaintext::new(coeffs_a, params.scale);
    let ct_a = encrypt(&pk, &pt_a, &params);

    let coeffs_b = encode_multivector_slots(&mv_b, params.scale, params.n);
    let pt_b = Plaintext::new(coeffs_b, params.scale);
    let ct_b = encrypt(&pk, &pt_b, &params);

    println!("✓ Encrypted a and b");

    // Homomorphic geometric product!
    println!("⊗ Computing homomorphic geometric product...");
    let ct_result = geometric_product_homomorphic_simple(&ct_a, &ct_b, &evk, &sk, &pk, &params);
    println!("✓ Computed Enc(a) ⊗ Enc(b)");

    // Decrypt
    let pt_result = decrypt(&sk, &ct_result, &params);
    let mv_result = decode_multivector_slots(&pt_result.coeffs, params.scale, params.n);

    println!("\nResult    = [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
             mv_result[0], mv_result[1], mv_result[2], mv_result[3],
             mv_result[4], mv_result[5], mv_result[6], mv_result[7]);
    println!("Expected  = [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
             expected[0], expected[1], expected[2], expected[3],
             expected[4], expected[5], expected[6], expected[7]);

    // Check correctness
    let mut max_error = 0.0;
    for i in 0..8 {
        let error = (mv_result[i] - expected[i]).abs();
        if error > max_error {
            max_error = error;
        }
    }

    println!("\nMax error: {:.2e}", max_error);

    if max_error < 1.0 {
        println!("✓ PASS: Homomorphic geometric product works!\n");
    } else {
        println!("✗ FAIL: Error too large\n");
    }

    println!("-----------------------------------------------------------------");
    println!("Test 2: (1+e1+e2+e3) ⊗ (1+e1+e2+e3) = Full multivector");
    println!("-----------------------------------------------------------------\n");

    let mv_c = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let mv_d = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];

    println!("c = {:?}", mv_c);
    println!("d = {:?}", mv_d);

    let expected2 = geometric_product_plaintext(&mv_c, &mv_d);
    println!("\nExpected c ⊗ d = {:?}", expected2);

    // Encrypt
    let coeffs_c = encode_multivector_slots(&mv_c, params.scale, params.n);
    let pt_c = Plaintext::new(coeffs_c, params.scale);
    let ct_c = encrypt(&pk, &pt_c, &params);

    let coeffs_d = encode_multivector_slots(&mv_d, params.scale, params.n);
    let pt_d = Plaintext::new(coeffs_d, params.scale);
    let ct_d = encrypt(&pk, &pt_d, &params);

    println!("✓ Encrypted c and d");

    // Homomorphic geometric product
    println!("⊗ Computing homomorphic geometric product...");
    let ct_result2 = geometric_product_homomorphic_simple(&ct_c, &ct_d, &evk, &sk, &pk, &params);
    println!("✓ Computed Enc(c) ⊗ Enc(d)");

    // Decrypt
    let pt_result2 = decrypt(&sk, &ct_result2, &params);
    let mv_result2 = decode_multivector_slots(&pt_result2.coeffs, params.scale, params.n);

    println!("\nResult    = [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
             mv_result2[0], mv_result2[1], mv_result2[2], mv_result2[3],
             mv_result2[4], mv_result2[5], mv_result2[6], mv_result2[7]);
    println!("Expected  = [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
             expected2[0], expected2[1], expected2[2], expected2[3],
             expected2[4], expected2[5], expected2[6], expected2[7]);

    // Check correctness
    let mut max_error2 = 0.0;
    for i in 0..8 {
        let error = (mv_result2[i] - expected2[i]).abs();
        if error > max_error2 {
            max_error2 = error;
        }
    }

    println!("\nMax error: {:.2e}", max_error2);

    if max_error2 < 1.0 {
        println!("✓ PASS: Complex geometric product works!\n");
    } else {
        println!("✗ FAIL: Error too large\n");
    }

    println!("=================================================================");
    println!("Summary");
    println!("=================================================================\n");

    if max_error < 1.0 && max_error2 < 1.0 {
        println!("🎉 SUCCESS! Homomorphic geometric product is working!");
        println!("   This is the KEY innovation of Clifford-FHE!");
        println!("   We can now compute Enc(a) ⊗ Enc(b) = Enc(a ⊗ b) ✓\n");
    } else {
        println!("❌ Some tests failed - debugging needed\n");
    }
}
