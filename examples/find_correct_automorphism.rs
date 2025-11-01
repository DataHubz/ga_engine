//! Empirically find the correct automorphism indices for slot rotations
//!
//! This tests all possible automorphism indices k (odd, coprime to M=64)
//! to find which ones actually produce the desired slot rotations.

use ga_engine::clifford_fhe::{
    CliffordFHEParams,
    encode_multivector_slots, decode_multivector_slots,
};
use ga_engine::clifford_fhe::automorphisms::apply_automorphism;

fn main() {
    println!("=================================================================");
    println!("Finding Correct Automorphism Indices for Slot Rotations");
    println!("=================================================================\n");

    let params = CliffordFHEParams::new_test();
    let n = params.n; // 32
    let m = 2 * n;    // 64

    // Test input: [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, ...]
    let mut mv = [0.0; 8];
    for i in 0..8 {
        mv[i] = (i + 1) as f64;
    }

    println!("Original slots: {:?}", mv);

    // Encode to coefficients
    let coeffs = encode_multivector_slots(&mv, params.scale, params.n);

    println!("\n-----------------------------------------------------------------");
    println!("Testing all valid automorphism indices k (odd, coprime to M=64)");
    println!("-----------------------------------------------------------------\n");

    // Valid automorphism indices: k must be odd and coprime to M=64
    // For M=64=2^6, coprime means odd
    let valid_k: Vec<usize> = (1..m).filter(|&k| k % 2 == 1).collect();

    println!("Testing {} valid indices: {:?}\n", valid_k.len(), &valid_k[..std::cmp::min(10, valid_k.len())]);

    // Expected results for different rotations
    let expected_left_1 = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0];
    let expected_left_2 = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0];
    let expected_left_3 = [4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0];
    let expected_right_1 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    let mut found_left_1 = None;
    let mut found_left_2 = None;
    let mut found_left_3 = None;
    let mut found_right_1 = None;

    for &k in &valid_k {
        // Apply automorphism
        let coeffs_auto = apply_automorphism(&coeffs, k, n);

        // Decode back to slots
        let mv_result = decode_multivector_slots(&coeffs_auto, params.scale, params.n);

        // Check if this matches any expected rotation
        let matches_left_1 = mv_result.iter().zip(&expected_left_1)
            .all(|(a, b)| (a - b).abs() < 0.1);
        let matches_left_2 = mv_result.iter().zip(&expected_left_2)
            .all(|(a, b)| (a - b).abs() < 0.1);
        let matches_left_3 = mv_result.iter().zip(&expected_left_3)
            .all(|(a, b)| (a - b).abs() < 0.1);
        let matches_right_1 = mv_result.iter().zip(&expected_right_1)
            .all(|(a, b)| (a - b).abs() < 0.1);

        if matches_left_1 && found_left_1.is_none() {
            found_left_1 = Some(k);
            println!("✓ k={:2} produces LEFT rotation by 1:  {:?}", k, &mv_result[..8]);
        }
        if matches_left_2 && found_left_2.is_none() {
            found_left_2 = Some(k);
            println!("✓ k={:2} produces LEFT rotation by 2:  {:?}", k, &mv_result[..8]);
        }
        if matches_left_3 && found_left_3.is_none() {
            found_left_3 = Some(k);
            println!("✓ k={:2} produces LEFT rotation by 3:  {:?}", k, &mv_result[..8]);
        }
        if matches_right_1 && found_right_1.is_none() {
            found_right_1 = Some(k);
            println!("✓ k={:2} produces RIGHT rotation by 1: {:?}", k, &mv_result[..8]);
        }
    }

    println!("\n=================================================================");
    println!("Summary of Findings");
    println!("=================================================================\n");

    if let Some(k) = found_left_1 {
        println!("Left rotation by 1:  k = {}", k);
    } else {
        println!("Left rotation by 1:  NOT FOUND ❌");
    }

    if let Some(k) = found_left_2 {
        println!("Left rotation by 2:  k = {}", k);
    } else {
        println!("Left rotation by 2:  NOT FOUND ❌");
    }

    if let Some(k) = found_left_3 {
        println!("Left rotation by 3:  k = {}", k);
    } else {
        println!("Left rotation by 3:  NOT FOUND ❌");
    }

    if let Some(k) = found_right_1 {
        println!("Right rotation by 1: k = {}", k);
    } else {
        println!("Right rotation by 1: NOT FOUND ❌");
    }

    // Analyze pattern if we found multiple
    if let (Some(k1), Some(k2), Some(k3)) = (found_left_1, found_left_2, found_left_3) {
        println!("\n-----------------------------------------------------------------");
        println!("Pattern Analysis");
        println!("-----------------------------------------------------------------");
        println!("Left 1: k = {}", k1);
        println!("Left 2: k = {}", k2);
        println!("Left 3: k = {}", k3);

        // Check if it's a power relationship
        let k1_squared_mod = (k1 * k1) % m;
        let k1_cubed_mod = (k1 * k1 * k1) % m;

        println!("\nChecking if k² mod {} = k for next rotation:", m);
        println!("k₁² mod {} = {} (expected {} for left 2)", m, k1_squared_mod, k2);
        println!("k₁³ mod {} = {} (expected {} for left 3)", m, k1_cubed_mod, k3);

        if k1_squared_mod == k2 && k1_cubed_mod == k3 {
            println!("\n✓ Pattern confirmed! Rotation r uses k = k₁^r mod {}", m);
            println!("  Generator: k₁ = {}", k1);
        } else {
            println!("\n⚠ Pattern doesn't match simple power relationship");
        }
    }

    println!("\n=================================================================");
}
