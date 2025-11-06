//! Diagonal Matrix Multiplication for CKKS
//!
//! Implements homomorphic diagonal matrix-vector multiplication.
//! This is a key primitive for bootstrapping operations like CoeffToSlot/SlotToCoeff.
//!
//! ## Theory
//!
//! A diagonal matrix D multiplied by a vector v encoded in slots:
//! ```text
//! [d₀ 0  0  ...] [v₀]   [d₀·v₀]
//! [0  d₁ 0  ...] [v₁] = [d₁·v₁]
//! [0  0  d₂ ...] [v₂]   [d₂·v₂]
//! [... ... ...]  [...]  [...]
//! ```
//!
//! In CKKS slot encoding, this becomes element-wise multiplication:
//! `result[i] = diag[i] * ct[i]` for each slot i.
//!
//! ## Implementation
//!
//! We use plaintext-ciphertext multiplication since the diagonal is known:
//! 1. Encode diagonal values as plaintext
//! 2. Multiply: ct_result = ct_input * pt_diagonal
//! 3. Result has diagonal applied to all slots

use crate::clifford_fhe_v2::params::CliffordFHEParams;
use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{Ciphertext, Plaintext};
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

/// Multiply ciphertext by a diagonal matrix
///
/// Applies element-wise multiplication of ciphertext slots by diagonal values.
///
/// # Arguments
///
/// * `ct` - Input ciphertext (slots encode vector)
/// * `diagonal` - Diagonal values (length must match number of slots)
/// * `params` - FHE parameters
/// * `key_ctx` - Key context (for relinearization after multiplication)
///
/// # Returns
///
/// Ciphertext with diagonal applied: result[i] = ct[i] * diagonal[i]
///
/// # Errors
///
/// Returns error if:
/// - Diagonal length doesn't match slot count
/// - Multiplication/relinearization fails
///
/// # Example
///
/// ```rust,ignore
/// // Multiply ciphertext by diagonal [2.0, 3.0, 4.0, ...]
/// let diagonal = vec![2.0, 3.0, 4.0, ...];
/// let ct_result = diagonal_mult(&ct_input, &diagonal, &params, &key_ctx)?;
/// // Now ct_result[0] = 2.0 * ct_input[0], ct_result[1] = 3.0 * ct_input[1], ...
/// ```
pub fn diagonal_mult(
    ct: &Ciphertext,
    diagonal: &[f64],
    params: &CliffordFHEParams,
    key_ctx: &KeyContext,
) -> Result<Ciphertext, String> {
    // Verify diagonal length matches slot count
    let num_slots = params.n / 2;  // CKKS uses N/2 complex slots
    if diagonal.len() != num_slots {
        return Err(format!(
            "Diagonal length {} doesn't match slot count {}",
            diagonal.len(),
            num_slots
        ));
    }

    // Encode diagonal as plaintext
    // For CKKS, we need to encode complex values in canonical embedding
    // For simplicity, we'll encode real values (imaginary parts = 0)
    let pt_diagonal = encode_diagonal_as_plaintext(diagonal, params)?;

    // Multiply ciphertext by plaintext diagonal
    // This is plaintext-ciphertext multiplication (no relinearization needed)
    multiply_by_plaintext(ct, &pt_diagonal, params)
}

/// Encode diagonal values as CKKS plaintext
///
/// Converts real-valued diagonal into CKKS plaintext encoding.
///
/// # Arguments
///
/// * `diagonal` - Real diagonal values
/// * `params` - FHE parameters (for scaling)
///
/// # Returns
///
/// Plaintext encoding of diagonal
fn encode_diagonal_as_plaintext(
    diagonal: &[f64],
    params: &CliffordFHEParams,
) -> Result<Plaintext, String> {
    // For CKKS encoding, we need to:
    // 1. Create coefficient representation of diagonal values
    // 2. Apply inverse DFT to get polynomial coefficients
    // 3. Scale by Δ (scaling factor)

    // For now, we'll use a simplified encoding
    // TODO: Implement proper canonical embedding for complex slots

    let n = params.n;
    let scale = params.scale;

    // Create polynomial coefficients
    // In coefficient form, we need to encode the slot values
    let mut coeffs = vec![0i64; n];

    // Simplified: Just scale the diagonal values
    // This is NOT correct CKKS encoding - needs proper implementation
    for (i, &val) in diagonal.iter().enumerate() {
        if i < n {
            coeffs[i] = (val * scale).round() as i64;
        }
    }

    // Create RNS representation
    use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

    let moduli = &params.moduli[..=0]; // level 0
    let mut rns_coeffs = Vec::with_capacity(n);

    for &coeff in &coeffs {
        let values: Vec<u64> = moduli.iter().map(|&q| {
            if coeff >= 0 {
                (coeff as u64) % q
            } else {
                let abs_val = (-coeff) as u64;
                let remainder = abs_val % q;
                if remainder == 0 { 0 } else { q - remainder }
            }
        }).collect();

        rns_coeffs.push(RnsRepresentation::new(values, moduli.to_vec()));
    }

    Ok(Plaintext {
        coeffs: rns_coeffs,
        scale,
        n,
        level: 0,
    })
}

/// Multiply ciphertext by plaintext (element-wise in slots)
///
/// Performs plaintext-ciphertext multiplication without relinearization.
///
/// # Arguments
///
/// * `ct` - Input ciphertext
/// * `pt` - Plaintext (diagonal encoded)
/// * `params` - FHE parameters
///
/// # Returns
///
/// Result ciphertext: ct_result = ct * pt (slot-wise)
pub fn multiply_by_plaintext(
    ct: &Ciphertext,
    pt: &Plaintext,
    params: &CliffordFHEParams,
) -> Result<Ciphertext, String> {
    // Plaintext-ciphertext multiplication is simpler than ciphertext-ciphertext
    // For ct = (c0, c1), result = (c0 * pt, c1 * pt)
    // No relinearization needed since degree doesn't increase

    use crate::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;
    use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

    // Get active moduli for this level
    let moduli: Vec<u64> = params.moduli[..=ct.level].to_vec();
    let n = ct.n;

    // Helper to multiply two RNS polynomials
    let multiply_rns_polys = |a: &[RnsRepresentation], b: &[RnsRepresentation]| -> Vec<RnsRepresentation> {
        let mut result = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.clone()); n];

        for (prime_idx, &q) in moduli.iter().enumerate() {
            let ntt_ctx = NttContext::new(n, q);

            let a_mod_q: Vec<u64> = a.iter().map(|rns| rns.values[prime_idx]).collect();
            let b_mod_q: Vec<u64> = b.iter().map(|rns| rns.values[prime_idx]).collect();

            let product_mod_q = ntt_ctx.multiply_polynomials(&a_mod_q, &b_mod_q);

            for i in 0..n {
                result[i].values[prime_idx] = product_mod_q[i];
            }
        }
        result
    };

    // Multiply c0 by pt
    let c0_result = multiply_rns_polys(&ct.c0, &pt.coeffs);

    // Multiply c1 by pt
    let c1_result = multiply_rns_polys(&ct.c1, &pt.coeffs);

    // New scale is product of scales
    let new_scale = ct.scale * pt.scale;

    Ok(Ciphertext {
        c0: c0_result,
        c1: c1_result,
        n: ct.n,
        level: ct.level,
        scale: new_scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{encode_vec, decode_vec};

    #[test]
    fn test_diagonal_mult_simple() {
        // Create test parameters
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, evk) = key_ctx.keygen();

        // Create test vector
        let vec = vec![1.0, 2.0, 3.0, 4.0];

        // Encrypt
        let ct = encode_vec(&vec, &pk, &params).unwrap();

        // Create diagonal [2.0, 3.0, 4.0, 5.0, ...]
        let num_slots = params.n / 2;
        let mut diagonal = vec![1.0; num_slots];
        diagonal[0] = 2.0;
        diagonal[1] = 3.0;
        diagonal[2] = 4.0;
        diagonal[3] = 5.0;

        // Apply diagonal multiplication
        let ct_result = diagonal_mult(&ct, &diagonal, &params, &key_ctx).unwrap();

        // Decrypt and check
        let result = decode_vec(&ct_result, &sk, &params).unwrap();

        // Expected: [1*2, 2*3, 3*4, 4*5] = [2, 6, 12, 20]
        assert!((result[0] - 2.0).abs() < 1.0, "slot 0: expected 2, got {}", result[0]);
        assert!((result[1] - 6.0).abs() < 1.0, "slot 1: expected 6, got {}", result[1]);
        assert!((result[2] - 12.0).abs() < 1.0, "slot 2: expected 12, got {}", result[2]);
        assert!((result[3] - 20.0).abs() < 1.0, "slot 3: expected 20, got {}", result[3]);
    }

    #[test]
    fn test_diagonal_mult_wrong_size() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, _, _) = key_ctx.keygen();

        let vec = vec![1.0, 2.0, 3.0];
        let ct = encode_vec(&vec, &pk, &params).unwrap();

        // Wrong diagonal size
        let diagonal = vec![1.0, 2.0, 3.0];  // Should be N/2 = 512 elements

        let result = diagonal_mult(&ct, &diagonal, &params, &key_ctx);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("doesn't match"));
    }
}
