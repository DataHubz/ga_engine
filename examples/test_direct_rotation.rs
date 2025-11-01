//! Test if there's a simple coefficient permutation that rotates slots
//!
//! Maybe we don't need automorphisms at all - maybe there's a direct
//! coefficient manipulation that rotates slots.

use ga_engine::clifford_fhe::{
    CliffordFHEParams,
    encode_multivector_slots, decode_multivector_slots,
};

fn main() {
    println!("=================================================================");
    println!("Testing Direct Coefficient Rotations");
    println!("=================================================================\n");

    let params = CliffordFHEParams::new_test();
    let n = params.n; // 32

    // Test input
    let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    println!("Original slots: {:?}\n", mv);

    // Encode
    let coeffs = encode_multivector_slots(&mv, params.scale, params.n);

    println!("-----------------------------------------------------------------");
    println!("Test 1: Cyclic rotation of coefficients");
    println!("-----------------------------------------------------------------\n");

    // Try rotating coefficients cyclically
    for shift in 1..=8 {
        let mut coeffs_rotated = coeffs.clone();
        coeffs_rotated.rotate_left(shift);

        let mv_result = decode_multivector_slots(&coeffs_rotated, params.scale, params.n);

        println!("Coeff rotate left {}: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
                 shift,
                 mv_result[0], mv_result[1], mv_result[2], mv_result[3],
                 mv_result[4], mv_result[5], mv_result[6], mv_result[7]);
    }

    println!("\n-----------------------------------------------------------------");
    println!("Test 2: Cyclic rotation of coefficients (right)");
    println!("-----------------------------------------------------------------\n");

    for shift in 1..=8 {
        let mut coeffs_rotated = coeffs.clone();
        coeffs_rotated.rotate_right(shift);

        let mv_result = decode_multivector_slots(&coeffs_rotated, params.scale, params.n);

        println!("Coeff rotate right {}: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
                 shift,
                 mv_result[0], mv_result[1], mv_result[2], mv_result[3],
                 mv_result[4], mv_result[5], mv_result[6], mv_result[7]);
    }

    println!("\n-----------------------------------------------------------------");
    println!("Test 3: Swapping even/odd coefficients");
    println!("-----------------------------------------------------------------\n");

    let mut coeffs_swapped = coeffs.clone();
    for i in 0..n/2 {
        coeffs_swapped.swap(2 * i, 2 * i + 1);
    }

    let mv_swapped = decode_multivector_slots(&coeffs_swapped, params.scale, params.n);
    println!("Even/odd swapped: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
             mv_swapped[0], mv_swapped[1], mv_swapped[2], mv_swapped[3],
             mv_swapped[4], mv_swapped[5], mv_swapped[6], mv_swapped[7]);

    println!("\n-----------------------------------------------------------------");
    println!("Test 4: Negating second half");
    println!("-----------------------------------------------------------------\n");

    let mut coeffs_negated = coeffs.clone();
    for i in n/2..n {
        coeffs_negated[i] = -coeffs_negated[i];
    }

    let mv_negated = decode_multivector_slots(&coeffs_negated, params.scale, params.n);
    println!("Second half negated: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
             mv_negated[0], mv_negated[1], mv_negated[2], mv_negated[3],
             mv_negated[4], mv_negated[5], mv_negated[6], mv_negated[7]);

    println!("\n=================================================================");
}
