//! Analyze what each automorphism actually does to the slots

use ga_engine::clifford_fhe::{
    CliffordFHEParams,
    encode_multivector_slots, decode_multivector_slots,
};
use ga_engine::clifford_fhe::automorphisms::apply_automorphism;

fn main() {
    println!("=================================================================");
    println!("Analyzing Automorphism Effects on Slots");
    println!("=================================================================\n");

    let params = CliffordFHEParams::new_test();
    let n = params.n; // 32
    let m = 2 * n;    // 64

    // Test input: [1, 2, 3, 4, 5, 6, 7, 8]
    let mut mv = [0.0; 8];
    for i in 0..8 {
        mv[i] = (i + 1) as f64;
    }

    println!("Original slots: {:?}\n", mv);

    // Encode to coefficients
    let coeffs = encode_multivector_slots(&mv, params.scale, params.n);

    println!("-----------------------------------------------------------------");
    println!("Testing first few automorphism indices");
    println!("-----------------------------------------------------------------\n");

    // Test k = 1, 3, 5, 7, 9, 11, 13, 15
    for &k in &[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31] {
        let coeffs_auto = apply_automorphism(&coeffs, k, n);
        let mv_result = decode_multivector_slots(&coeffs_auto, params.scale, params.n);

        println!("k={:2}: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
                 k,
                 mv_result[0], mv_result[1], mv_result[2], mv_result[3],
                 mv_result[4], mv_result[5], mv_result[6], mv_result[7]);
    }

    println!("\n-----------------------------------------------------------------");
    println!("Analysis: Looking for patterns");
    println!("-----------------------------------------------------------------\n");

    // Check if k=1 is identity
    let coeffs_k1 = apply_automorphism(&coeffs, 1, n);
    let mv_k1 = decode_multivector_slots(&coeffs_k1, params.scale, params.n);
    let is_identity = mv.iter().zip(&mv_k1).all(|(a, b)| (a - b).abs() < 0.1);

    if is_identity {
        println!("✓ k=1 is identity (as expected)");
    } else {
        println!("✗ k=1 is NOT identity! Got: {:?}", &mv_k1[..8]);
    }

    // Check which k produces conjugation (reverse + conjugate for complex)
    // For real slots, conjugation should reverse the order
    let expected_reverse = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    for &k in &[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63] {
        let coeffs_auto = apply_automorphism(&coeffs, k, n);
        let mv_result = decode_multivector_slots(&coeffs_auto, params.scale, params.n);

        let matches_reverse = mv_result.iter().zip(&expected_reverse)
            .all(|(a, b)| (a - b).abs() < 0.1);

        if matches_reverse {
            println!("✓ k={} produces reversal (conjugation)", k);
        }
    }

    println!("\n=================================================================");
}
