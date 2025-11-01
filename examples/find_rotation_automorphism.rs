//! Find which automorphism index produces left rotation by 1

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::canonical_embedding::{encode_multivector_canonical, decode_multivector_canonical};
use ga_engine::clifford_fhe::automorphisms::apply_automorphism;

fn main() {
    println!("=================================================================");
    println!("Finding Correct Rotation Automorphism");
    println!("=================================================================\n");

    let params = CliffordFHEParams::new_test();
    let n = params.n; // 32
    let m = 2 * n;    // 64

    // Test input
    let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    println!("Original: {:?}\n", mv);

    // Encode
    let coeffs = encode_multivector_canonical(&mv, params.scale, params.n);

    println!("-----------------------------------------------------------------");
    println!("Testing ALL valid automorphism indices (odd, < M)");
    println!("-----------------------------------------------------------------\n");

    let expected_left_1 = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0];

    let mut found_generator = None;

    for k in (1..m).filter(|&x| x % 2 == 1) {
        let coeffs_auto = apply_automorphism(&coeffs, k, n);
        let mv_result = decode_multivector_canonical(&coeffs_auto, params.scale, params.n);

        let matches = mv_result.iter().zip(&expected_left_1)
            .all(|(a, b)| (a - b).abs() < 0.01);

        if matches {
            println!("✓ FOUND! k={} produces left rotation by 1", k);
            println!("  Result: {:?}\n", &mv_result[..8]);
            found_generator = Some(k);
            break;
        }
    }

    if let Some(k1) = found_generator {
        println!("-----------------------------------------------------------------");
        println!("Testing powers of k={} to see if it generates all rotations", k1);
        println!("-----------------------------------------------------------------\n");

        for r in 0..=7 {
            let k_r = (k1.pow(r) % m as usize) | 1; // Ensure odd
            let k_r = if k_r >= m { k_r % m } else { k_r };

            let coeffs_auto = apply_automorphism(&coeffs, k_r, n);
            let mv_result = decode_multivector_canonical(&coeffs_auto, params.scale, params.n);

            println!("k^{} mod {} = {:2}: [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
                     r, m, k_r,
                     mv_result[0], mv_result[1], mv_result[2], mv_result[3],
                     mv_result[4], mv_result[5], mv_result[6], mv_result[7]);
        }
    } else {
        println!("❌ NO automorphism found that produces left rotation by 1!");
        println!("   This means the canonical embedding may still have issues.\n");
    }

    println!("\n=================================================================");
}
