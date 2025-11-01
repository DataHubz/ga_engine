//! Test Galois automorphisms on plaintexts (no encryption)

use ga_engine::clifford_fhe::{
    encode_multivector_slots, decode_multivector_slots,
    CliffordFHEParams,
};
use ga_engine::clifford_fhe::automorphisms::{apply_automorphism, rotation_to_automorphism};

fn main() {
    println!("Testing Galois Automorphisms on Plaintexts");
    println!("=================================================================\n");

    let params = CliffordFHEParams::new_test();
    let n = params.n;

    // Encode a multivector into polynomial via SIMD slots
    let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    println!("Original multivector: {:?}", mv);

    let coeffs = encode_multivector_slots(&mv, params.scale, n);
    println!("Encoded to {} coefficients\n", coeffs.len());

    // Decode to verify roundtrip
    let mv_decoded = decode_multivector_slots(&coeffs, params.scale, n);
    println!("Decoded multivector: {:?}", &mv_decoded);
    println!("Roundtrip error: {:.2e}\n",
        mv.iter().zip(&mv_decoded).map(|(a,b)| (a-b).abs()).fold(0.0, f64::max));

    // Apply automorphism for rotation left by 1
    println!("Applying rotation automorphism (left by 1):");
    let k = rotation_to_automorphism(1, n);
    println!("  Rotation 1 → automorphism index k = {}", k);

    let coeffs_rotated = apply_automorphism(&coeffs, k, n);

    // Decode the rotated coefficients
    let mv_rotated = decode_multivector_slots(&coeffs_rotated, params.scale, n);
    println!("After rotation: {:?}", &mv_rotated);
    println!("Expected:       [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0 or 1.0]");

    // The last value might be 1.0 due to wraparound in slots
    println!("\n=================================================================");
}
