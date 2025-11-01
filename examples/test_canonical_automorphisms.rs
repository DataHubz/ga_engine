//! Test which automorphism indices produce slot rotations with canonical embedding

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::canonical_embedding::{encode_multivector_canonical, decode_multivector_canonical};
use ga_engine::clifford_fhe::automorphisms::apply_automorphism;

fn main() {
    println!("=================================================================");
    println!("Testing Automorphisms with Canonical Embedding");
    println!("=================================================================\n");

    let params = CliffordFHEParams::new_test();
    let n = params.n; // 32

    // Test input
    let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    println!("Original slots: {:?}\n", mv);

    // Encode using canonical embedding
    let coeffs = encode_multivector_canonical(&mv, params.scale, params.n);

    println!("-----------------------------------------------------------------");
    println!("Testing automorphism indices k = 3, 5, 7, 9, ...");
    println!("-----------------------------------------------------------------\n");

    let expected_left_1 = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0];
    let expected_left_2 = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0];
    let expected_right_1 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    for &k in &[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31] {
        let coeffs_auto = apply_automorphism(&coeffs, k, n);
        let mv_result = decode_multivector_canonical(&coeffs_auto, params.scale, params.n);

        let matches_left_1 = mv_result.iter().zip(&expected_left_1)
            .all(|(a, b)| (a - b).abs() < 0.01);
        let matches_left_2 = mv_result.iter().zip(&expected_left_2)
            .all(|(a, b)| (a - b).abs() < 0.01);
        let matches_right_1 = mv_result.iter().zip(&expected_right_1)
            .all(|(a, b)| (a - b).abs() < 0.01);

        if matches_left_1 {
            println!("✓ k={:2} produces LEFT rotation by 1", k);
            println!("  Result: [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
                     mv_result[0], mv_result[1], mv_result[2], mv_result[3],
                     mv_result[4], mv_result[5], mv_result[6], mv_result[7]);
        } else if matches_left_2 {
            println!("✓ k={:2} produces LEFT rotation by 2", k);
            println!("  Result: [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
                     mv_result[0], mv_result[1], mv_result[2], mv_result[3],
                     mv_result[4], mv_result[5], mv_result[6], mv_result[7]);
        } else if matches_right_1 {
            println!("✓ k={:2} produces RIGHT rotation by 1", k);
            println!("  Result: [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
                     mv_result[0], mv_result[1], mv_result[2], mv_result[3],
                     mv_result[4], mv_result[5], mv_result[6], mv_result[7]);
        }
    }

    println!("\n-----------------------------------------------------------------");
    println!("Testing the standard formula: k = 5^r mod M");
    println!("-----------------------------------------------------------------\n");

    use ga_engine::clifford_fhe::automorphisms::rotation_to_automorphism;

    for r in -3..=3isize {
        let k = rotation_to_automorphism(r, n);
        let coeffs_auto = apply_automorphism(&coeffs, k, n);
        let mv_result = decode_multivector_canonical(&coeffs_auto, params.scale, params.n);

        println!("r={:2} → k={:2}: [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
                 r, k,
                 mv_result[0], mv_result[1], mv_result[2], mv_result[3],
                 mv_result[4], mv_result[5], mv_result[6], mv_result[7]);
    }

    println!("\n=================================================================");
}
