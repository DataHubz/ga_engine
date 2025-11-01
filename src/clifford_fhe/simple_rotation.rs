//! Simplified slot rotation for testing (bypasses complex Galois automorphisms)
//!
//! This module provides a simpler rotation mechanism that works correctly
//! for small test parameters by directly operating on the slot representation.
//!
//! WARNING: This is NOT the standard CKKS approach and should only be used
//! for testing/debugging. For production, proper Galois automorphisms are needed.

use crate::clifford_fhe::ckks::{Ciphertext, Plaintext, decrypt, encrypt, multiply_by_plaintext};
use crate::clifford_fhe::keys::{PublicKey, SecretKey, RotationKey};
use crate::clifford_fhe::params::CliffordFHEParams;
use crate::clifford_fhe::slot_encoding::{decode_multivector_slots, encode_multivector_slots};
use rustfft::num_complex::Complex;

/// Simplified slot rotation (for testing only)
///
/// This directly decrypts, rotates at the slot level, and re-encrypts.
/// Obviously not secure, but useful for testing the slot operations logic.
///
/// # Arguments
/// * `ct` - Ciphertext to rotate
/// * `rotation_amount` - Slots to rotate (positive = left, negative = right)
/// * `sk` - Secret key (for decrypt)
/// * `pk` - Public key (for re-encrypt)
/// * `params` - Parameters
pub fn rotate_slots_simple(
    ct: &Ciphertext,
    rotation_amount: isize,
    sk: &SecretKey,
    pk: &PublicKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    // Decrypt to slots
    let pt = decrypt(sk, ct, params);

    // Get all slots, not just the first 8
    use crate::clifford_fhe::slot_encoding::{coefficients_to_slots, slots_to_coefficients};
    use rustfft::num_complex::Complex;

    let num_slots = params.n / 2; // Total slots available
    let mut slots = coefficients_to_slots(&pt.coeffs, params.scale, params.n);

    // Rotate all slots (including zeros beyond position 8)
    let r = ((rotation_amount % num_slots as isize) + num_slots as isize) % num_slots as isize;

    let mut rotated = vec![Complex::new(0.0, 0.0); num_slots];
    for i in 0..num_slots {
        let src_idx = ((i as isize + r) % num_slots as isize) as usize;
        rotated[i] = slots[src_idx];
    }

    // Re-encrypt
    let coeffs = slots_to_coefficients(&rotated, params.scale, params.n);
    let pt_new = Plaintext::new(coeffs, params.scale);
    let mut ct_new = encrypt(pk, &pt_new, params);

    // Preserve the original ciphertext's level and scale
    ct_new.level = ct.level;
    ct_new.scale = ct.scale;
    ct_new
}

/// Extract slot using simple rotation
pub fn extract_slot_simple(
    ct: &Ciphertext,
    slot_index: usize,
    sk: &SecretKey,
    pk: &PublicKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    // Decrypt
    let pt = decrypt(sk, ct, params);
    let mv = decode_multivector_slots(&pt.coeffs, params.scale, params.n);

    // Create new MV with only the desired slot
    let mut extracted = [0.0; 8];
    extracted[slot_index] = mv[slot_index];

    // Re-encrypt
    let coeffs = encode_multivector_slots(&extracted, params.scale, params.n);
    let pt_new = Plaintext::new(coeffs, params.scale);
    encrypt(pk, &pt_new, params)
}
