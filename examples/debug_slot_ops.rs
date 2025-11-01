//! Debug slot operations to understand the issue

use ga_engine::clifford_fhe::{
    decode_multivector_slots, encode_multivector_slots, encrypt, decrypt,
    keygen_with_rotation, rotate_slots, multiply_by_plaintext,
    CliffordFHEParams, Plaintext,
};
use ga_engine::clifford_fhe::slot_encoding::create_slot_mask;

fn main() {
    println!("=================================================================");
    println!("Debugging SIMD Slot Operations");
    println!("=================================================================\n");

    let params = CliffordFHEParams::new_test();
    println!("Parameters: N = {}, scale = 2^{}\n", params.n, params.scale.log2() as u32);

    let (pk, sk, _evk, rotk) = keygen_with_rotation(&params);
    println!("✓ Keys generated\n");

    // Test 1: Basic encryption/decryption with SIMD slots
    println!("TEST 1: Basic SIMD encryption/decryption");
    println!("-------------------------------------------------------------");
    let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    println!("Original: {:?}", mv);

    let coeffs = encode_multivector_slots(&mv, params.scale, params.n);
    let pt = Plaintext::new(coeffs, params.scale);
    let ct = encrypt(&pk, &pt, &params);

    let pt_dec = decrypt(&sk, &ct, &params);
    let mv_dec = decode_multivector_slots(&pt_dec.coeffs, params.scale, params.n);
    println!("Decrypted: {:?}", &mv_dec[..8]);

    let max_error = mv.iter().zip(&mv_dec).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("Max error: {:.2e}\n", max_error);

    // Test 2: Rotation only
    println!("TEST 2: Rotation (left by 1)");
    println!("-------------------------------------------------------------");
    let ct_rotated = rotate_slots(&ct, 1, &rotk, &params);
    let pt_rot = decrypt(&sk, &ct_rotated, &params);
    let mv_rot = decode_multivector_slots(&pt_rot.coeffs, params.scale, params.n);
    println!("After rotate left 1: {:?}", &mv_rot[..8]);
    println!("Expected: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0]\n");

    // Test 3: Masking only
    println!("TEST 3: Masking (keep only slot 0)");
    println!("-------------------------------------------------------------");
    let mask_coeffs = create_slot_mask(0, params.scale, params.n);
    println!("Mask created with scale: {}", params.scale);
    let mask_pt = Plaintext::new(mask_coeffs.clone(), params.scale);

    let ct_masked = multiply_by_plaintext(&ct, &mask_pt, &params);
    let pt_mask = decrypt(&sk, &ct_masked, &params);
    let mv_mask = decode_multivector_slots(&pt_mask.coeffs, params.scale, params.n);
    println!("After masking: {:?}", &mv_mask[..8]);
    println!("Expected: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]");
    println!("Ciphertext scale before mask: {}", ct.scale);
    println!("Ciphertext scale after mask: {}\n", ct_masked.scale);

    // Test 4: Rotate then mask
    println!("TEST 4: Rotate to position 0, then mask");
    println!("-------------------------------------------------------------");
    // We want to extract slot 3 (value 4.0)
    let ct_to_zero = rotate_slots(&ct, -3, &rotk, &params);
    let pt_check = decrypt(&sk, &ct_to_zero, &params);
    let mv_check = decode_multivector_slots(&pt_check.coeffs, params.scale, params.n);
    println!("After rotating slot 3 to position 0: {:?}", &mv_check[..8]);
    println!("Expected: [4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 1.0]");

    let ct_masked2 = multiply_by_plaintext(&ct_to_zero, &mask_pt, &params);
    let pt_final = decrypt(&sk, &ct_masked2, &params);
    let mv_final = decode_multivector_slots(&pt_final.coeffs, params.scale, params.n);
    println!("After masking: {:?}", &mv_final[..8]);
    println!("Expected: [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n");

    println!("=================================================================");
    println!("Analysis Complete");
    println!("=================================================================");
}
