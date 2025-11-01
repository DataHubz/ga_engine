//! SIMD Slot Operations for Clifford-FHE
//!
//! This module provides operations for manipulating individual SIMD slots in
//! CKKS ciphertexts. These are essential for implementing geometric product
//! over encrypted multivectors.
//!
//! # Core Operations
//!
//! - `extract_slot`: Isolates a value at a specific slot position
//! - `place_at_slot`: Places a value at a specific slot position
//!
//! # Strategy
//!
//! To extract slot i:
//! 1. Rotate slot i to position 0
//! 2. Multiply by mask [1, 0, 0, ...] to zero out other slots
//! 3. (Optional) Rotate back to original position
//!
//! To place value at slot i:
//! 1. Extract value at slot 0
//! 2. Rotate to position i

use crate::clifford_fhe::ckks::{multiply_by_plaintext, rotate_slots, Ciphertext, Plaintext};
use crate::clifford_fhe::keys::RotationKey;
use crate::clifford_fhe::params::CliffordFHEParams;
use crate::clifford_fhe::slot_encoding::create_slot_mask;

/// Extract value at specific SIMD slot
///
/// Returns a ciphertext where only the specified slot contains the original
/// value, and all other slots are zero.
///
/// # Strategy
///
/// 1. Rotate so target slot moves to position 0
/// 2. Multiply by mask [1, 0, 0, ...] to zero other slots
/// 3. Rotate back to original position
///
/// # Arguments
/// * `ct` - Input ciphertext
/// * `slot_index` - Index of slot to extract (0-7 for multivectors)
/// * `rotk` - Rotation keys
/// * `params` - FHE parameters
///
/// # Returns
/// Ciphertext with value only at the specified slot
///
/// # Example
/// ```rust,ignore
/// // ct encrypts [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
/// let ct_slot_2 = extract_slot(&ct, 2, rotk, params);
/// // ct_slot_2 encrypts [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
/// ```
pub fn extract_slot(
    ct: &Ciphertext,
    slot_index: usize,
    rotk: &RotationKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    assert!(slot_index < params.n / 2, "Slot index out of range");

    // Step 1: Rotate slot to position 0
    let ct_at_zero = if slot_index > 0 {
        rotate_slots(ct, -(slot_index as isize), rotk, params)
    } else {
        ct.clone()
    };

    // Step 2: Multiply by mask [1, 0, 0, ...]
    let mask_coeffs = create_slot_mask(0, params.scale, params.n);
    let mask_pt = Plaintext::new(mask_coeffs, params.scale);
    let ct_masked = multiply_by_plaintext(&ct_at_zero, &mask_pt, params);

    // Step 3: Rotate back to original position
    if slot_index > 0 {
        rotate_slots(&ct_masked, slot_index as isize, rotk, params)
    } else {
        ct_masked
    }
}

/// Extract value at slot 0 only (optimized version)
///
/// This is a common operation so we provide an optimized version
/// that doesn't require rotation.
pub fn extract_slot_zero(
    ct: &Ciphertext,
    params: &CliffordFHEParams,
) -> Ciphertext {
    let mask_coeffs = create_slot_mask(0, params.scale, params.n);
    let mask_pt = Plaintext::new(mask_coeffs, params.scale);
    multiply_by_plaintext(ct, &mask_pt, params)
}

/// Place value from slot 0 at specified slot position
///
/// Takes a ciphertext where the value is at slot 0, and moves it to
/// the target slot position. All other slots remain zero.
///
/// # Arguments
/// * `ct` - Input ciphertext (value should be at slot 0)
/// * `target_slot` - Index where to place the value
/// * `rotk` - Rotation keys
/// * `params` - FHE parameters
///
/// # Returns
/// Ciphertext with value at target slot
///
/// # Example
/// ```rust,ignore
/// // ct encrypts [5.0, 0.0, 0.0, ...]
/// let ct_at_3 = place_at_slot(&ct, 3, rotk, params);
/// // ct_at_3 encrypts [0.0, 0.0, 0.0, 5.0, 0.0, ...]
/// ```
pub fn place_at_slot(
    ct: &Ciphertext,
    target_slot: usize,
    rotk: &RotationKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    assert!(target_slot < params.n / 2, "Target slot out of range");

    if target_slot == 0 {
        ct.clone()
    } else {
        rotate_slots(ct, target_slot as isize, rotk, params)
    }
}

/// Extract and move slot to position 0
///
/// This is an optimized operation that extracts a slot and leaves the
/// result at position 0 (doesn't rotate back). Useful when you're going
/// to immediately use the value.
///
/// # Arguments
/// * `ct` - Input ciphertext
/// * `slot_index` - Index of slot to extract
/// * `rotk` - Rotation keys
/// * `params` - FHE parameters
///
/// # Returns
/// Ciphertext with extracted value at slot 0, other slots zero
pub fn extract_to_slot_zero(
    ct: &Ciphertext,
    slot_index: usize,
    rotk: &RotationKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    // Rotate to position 0
    let ct_at_zero = if slot_index > 0 {
        rotate_slots(ct, -(slot_index as isize), rotk, params)
    } else {
        ct.clone()
    };

    // Mask to keep only slot 0
    let mask_coeffs = create_slot_mask(0, params.scale, params.n);
    let mask_pt = Plaintext::new(mask_coeffs, params.scale);
    multiply_by_plaintext(&ct_at_zero, &mask_pt, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe::{
        decode_multivector_slots, encode_multivector_slots, encrypt, decrypt,
        keygen_with_rotation, CliffordFHEParams,
    };

    // Helper: Encrypt multivector using SIMD slot encoding
    fn encrypt_slots_mv(
        pk: &crate::clifford_fhe::keys::PublicKey,
        mv: &[f64; 8],
        params: &CliffordFHEParams,
    ) -> Ciphertext {
        let coeffs = encode_multivector_slots(mv, params.scale, params.n);
        let pt = Plaintext::new(coeffs, params.scale);
        encrypt(pk, &pt, params)
    }

    // Helper: Decrypt multivector from SIMD slots
    fn decrypt_slots_mv(
        sk: &crate::clifford_fhe::keys::SecretKey,
        ct: &Ciphertext,
        params: &CliffordFHEParams,
    ) -> [f64; 8] {
        let pt = decrypt(sk, ct, params);
        decode_multivector_slots(&pt.coeffs, params.scale, params.n)
    }

    #[test]
    fn test_extract_slot() {
        let params = CliffordFHEParams::new_test(); // Small N for fast testing
        let (pk, sk, _evk, rotk) = keygen_with_rotation(&params);

        // Encrypt multivector [1, 2, 3, 4, 5, 6, 7, 8] using SIMD
        let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ct = encrypt_slots_mv(&pk, &mv, &params);

        // Extract slot 3 (value should be 4.0)
        let ct_extracted = extract_slot(&ct, 3, &rotk, &params);

        // Decrypt using SIMD
        let mv_result = decrypt_slots_mv(&sk, &ct_extracted, &params);

        // Check: slot 3 should have 4.0, others should be ~0.0
        for i in 0..8 {
            let expected = if i == 3 { 4.0 } else { 0.0 };
            let error = (mv_result[i] - expected).abs();
            assert!(
                error < 0.1,
                "Slot {} error: {} (got {}, expected {})",
                i,
                error,
                mv_result[i],
                expected
            );
        }
    }

    #[test]
    fn test_extract_to_slot_zero() {
        let params = CliffordFHEParams::new_test();
        let (pk, sk, _evk, rotk) = keygen_with_rotation(&params);

        let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ct = encrypt_slots_mv(&pk, &mv, &params);

        // Extract slot 5 to position 0 (value 6.0)
        let ct_extracted = extract_to_slot_zero(&ct, 5, &rotk, &params);

        let mv_result = decrypt_slots_mv(&sk, &ct_extracted, &params);

        // Slot 0 should have 6.0, others ~0.0
        assert!((mv_result[0] - 6.0).abs() < 0.1);
        for i in 1..8 {
            assert!(mv_result[i].abs() < 0.1);
        }
    }

    #[test]
    fn test_place_at_slot() {
        let params = CliffordFHEParams::new_test();
        let (pk, sk, _evk, rotk) = keygen_with_rotation(&params);

        // Encrypt value at slot 0: [7.0, 0, 0, ...]
        let mv = [7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ct = encrypt_slots_mv(&pk, &mv, &params);

        // Move to slot 4
        let ct_placed = place_at_slot(&ct, 4, &rotk, &params);

        let mv_result = decrypt_slots_mv(&sk, &ct_placed, &params);

        // Slot 4 should have 7.0, others ~0.0
        for i in 0..8 {
            let expected = if i == 4 { 7.0 } else { 0.0 };
            let error = (mv_result[i] - expected).abs();
            assert!(
                error < 0.1,
                "Slot {} error: {} (got {}, expected {})",
                i,
                error,
                mv_result[i],
                expected
            );
        }
    }
}
