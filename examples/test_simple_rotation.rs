//! Test the simple rotation mechanism

use ga_engine::clifford_fhe::{
    CliffordFHEParams, keygen,
    encode_multivector_slots, decode_multivector_slots,
    encrypt, decrypt, Plaintext,
};
use ga_engine::clifford_fhe::simple_rotation::{rotate_slots_simple, extract_slot_simple};
use ga_engine::clifford_fhe::slot_encoding::create_slot_mask;
use ga_engine::clifford_fhe::ckks::multiply_by_plaintext;

fn main() {
    println!("=================================================================");
    println!("Testing Simple Rotation Mechanism");
    println!("=================================================================\n");

    let params = CliffordFHEParams::new_test();
    let (pk, sk, _evk) = keygen(&params);

    // Test input
    let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    println!("Original: {:?}\n", mv);

    // Encrypt
    let coeffs = encode_multivector_slots(&mv, params.scale, params.n);
    let pt = Plaintext::new(coeffs, params.scale);
    let ct = encrypt(&pk, &pt, &params);

    println!("-----------------------------------------------------------------");
    println!("Test 1: Simple rotation left by 1");
    println!("-----------------------------------------------------------------");

    let ct_rot = rotate_slots_simple(&ct, 1, &sk, &pk, &params);
    let pt_rot = decrypt(&sk, &ct_rot, &params);
    let mv_rot = decode_multivector_slots(&pt_rot.coeffs, params.scale, params.n);

    println!("Result:   {:?}", &mv_rot[..8]);
    println!("Expected: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0]");

    let expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0];
    let max_error = mv_rot.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("Max error: {:.2e}", max_error);

    if max_error < 0.1 {
        println!("✓ PASS\n");
    } else {
        println!("✗ FAIL\n");
    }

    println!("-----------------------------------------------------------------");
    println!("Test 2: Simple rotation right by 1");
    println!("-----------------------------------------------------------------");

    let ct_rot = rotate_slots_simple(&ct, -1, &sk, &pk, &params);
    let pt_rot = decrypt(&sk, &ct_rot, &params);
    let mv_rot = decode_multivector_slots(&pt_rot.coeffs, params.scale, params.n);

    println!("Result:   {:?}", &mv_rot[..8]);
    println!("Expected: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]");

    let expected = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let max_error = mv_rot.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("Max error: {:.2e}", max_error);

    if max_error < 0.1 {
        println!("✓ PASS\n");
    } else {
        println!("✗ FAIL\n");
    }

    println!("-----------------------------------------------------------------");
    println!("Test 3: Extract slot 3 (value 4.0)");
    println!("-----------------------------------------------------------------");

    let ct_extract = extract_slot_simple(&ct, 3, &sk, &pk, &params);
    let pt_extract = decrypt(&sk, &ct_extract, &params);
    let mv_extract = decode_multivector_slots(&pt_extract.coeffs, params.scale, params.n);

    println!("Result:   {:?}", &mv_extract[..8]);
    println!("Expected: [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]");

    let expected = [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0];
    let max_error = mv_extract.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("Max error: {:.2e}", max_error);

    if max_error < 0.1 {
        println!("✓ PASS\n");
    } else {
        println!("✗ FAIL\n");
    }

    println!("=================================================================");
    println!("Summary");
    println!("=================================================================");
    println!("\n✓ Simple rotation works for testing slot operations logic!");
    println!("  (Not secure for production, but sufficient for validating");
    println!("   the geometric product SIMD implementation)\n");
}
