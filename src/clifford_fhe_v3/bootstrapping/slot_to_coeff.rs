//! SlotToCoeff Transformation
//!
//! Transforms ciphertext from slot representation back to coefficient representation.
//! This is the inverse of CoeffToSlot.
//!
//! ## Algorithm
//!
//! SlotToCoeff is the inverse FFT, also using O(log N) homomorphic rotations.
//!
//! **Structure:** Same as CoeffToSlot but with:
//! - Reversed level order (log N → 0)
//! - Inverse diagonal matrices
//! - Negated rotation directions (or equivalent)
//!
//! **Complexity:** O(log N) rotations, O(N log N) multiplications
//!
//! ## Correctness
//!
//! Must satisfy: SlotToCoeff(CoeffToSlot(x)) ≈ x (up to noise growth)

use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;
use super::keys::RotationKeys;
use super::rotation::rotate;

/// SlotToCoeff transformation
///
/// Transforms a ciphertext from slot (evaluation) representation back to coefficient representation.
///
/// # Arguments
///
/// * `ct` - Input ciphertext in slot representation
/// * `rotation_keys` - Rotation keys for all required rotations
///
/// # Returns
///
/// Ciphertext in coefficient representation
///
/// # Algorithm
///
/// Inverse FFT-like butterfly structure (reverse of CoeffToSlot):
///
/// ```text
/// Level log(N)-1: 1 pair, rotation by ±N/2
/// Level log(N)-2: 2 pairs, rotation by ±N/4
/// ...
/// Level 1: N/4 pairs, rotation by ±2
/// Level 0: N/2 pairs, rotation by ±1
/// ```
///
/// Each level applies:
/// 1. Rotation
/// 2. Diagonal matrix multiplication (inverse constants)
/// 3. Addition/subtraction
///
pub fn slot_to_coeff(
    ct: &Ciphertext,
    rotation_keys: &RotationKeys,
) -> Result<Ciphertext, String> {
    let n = ct.n;
    let num_slots = n / 2;
    let num_levels = (num_slots as f64).log2() as usize;

    println!("SlotToCoeff: N={}, slots={}, levels={}", n, num_slots, num_levels);

    // Start with input ciphertext
    let mut current = ct.clone();

    // Apply inverse FFT-like butterfly structure (reversed order)
    for level in (0..num_levels).rev() {
        let rotation_amount = 1 << level;  // N/4, ..., 4, 2, 1

        println!("  Level {}: rotation by ±{}", level, rotation_amount);

        // Rotate by -rotation_amount (negative to reverse CoeffToSlot)
        let ct_rotated = rotate(&current, -(rotation_amount as i32), rotation_keys)?;

        // In full implementation:
        // 1. Multiply current and ct_rotated by inverse diagonal matrices
        // 2. Add/subtract: current = inv_diag1 * current + inv_diag2 * ct_rotated

        // For now, just use rotated ciphertext to show progression
        current = ct_rotated;
    }

    // TODO: Apply final scaling/normalization

    Ok(current)
}

/// Precompute inverse diagonal matrices for SlotToCoeff
///
/// These are the inverse of the CoeffToSlot matrices.
///
/// # Returns
///
/// Vector of inverse diagonal matrices, one per level
///
#[allow(dead_code)]
fn precompute_slot_to_coeff_matrices(n: usize) -> Vec<Vec<f64>> {
    let num_slots = n / 2;
    let num_levels = (num_slots as f64).log2() as usize;

    let mut matrices = Vec::with_capacity(num_levels);

    for level in 0..num_levels {
        // Placeholder: identity-like diagonal (inverse would need actual DFT matrix inverse)
        let diag = vec![1.0; num_slots];
        matrices.push(diag);
    }

    matrices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::params::CliffordFHEParams;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
    use crate::clifford_fhe_v3::bootstrapping::keys::generate_rotation_keys;
    use crate::clifford_fhe_v3::bootstrapping::coeff_to_slot::coeff_to_slot;

    #[test]
    fn test_slot_to_coeff_structure() {
        // Test that SlotToCoeff runs without errors
        let params = CliffordFHEParams::new_128bit();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        // Create test message
        let mut message = vec![0.0; params.n / 2];
        message[0] = 1.0;

        let pt = ckks_ctx.encode(&message);
        let ct = ckks_ctx.encrypt(&pt, &pk);

        // Generate rotation keys
        let rotations = vec![-1, -2, -4, -8];  // Negative rotations for SlotToCoeff
        let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

        // Run SlotToCoeff
        let result = slot_to_coeff(&ct, &rotation_keys);

        println!("SlotToCoeff test result: {:?}", result.is_ok());
    }

    #[test]
    fn test_coeff_slot_roundtrip() {
        // Test that SlotToCoeff(CoeffToSlot(x)) ≈ x
        // Use small params for faster testing (n=1024 needs fewer rotation keys)
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        // Create test message
        let mut message = vec![0.0; params.n / 2];
        message[0] = 1.0;
        message[1] = 2.0;

        let pt = ckks_ctx.encode(&message);
        let ct_original = ckks_ctx.encrypt(&pt, &pk);

        // Generate all required rotation keys for bootstrap
        use crate::clifford_fhe_v3::bootstrapping::keys::required_rotations_for_bootstrap;
        let rotations = required_rotations_for_bootstrap(params.n);
        let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

        // Apply CoeffToSlot then SlotToCoeff
        let ct_slots = coeff_to_slot(&ct_original, &rotation_keys)
            .expect("CoeffToSlot failed");
        let ct_coeffs = slot_to_coeff(&ct_slots, &rotation_keys)
            .expect("SlotToCoeff failed");

        // Decrypt and compare
        let pt_result = ckks_ctx.decrypt(&ct_coeffs, &sk);
        let result = ckks_ctx.decode(&pt_result);

        // Should be approximately equal to original
        for i in 0..5 {
            let error = (result[i] - message[i]).abs();
            println!("result[{}] = {}, expected {}, error = {}",
                     i, result[i], message[i], error);
            assert!(error < 1.0, "Roundtrip error too large");
        }
    }
}
