//! Advanced CKKS operations for Clifford-FHE
//!
//! This module provides operations needed for homomorphic geometric product:
//! - Component product computation (multiply specific coefficients using rotation)
//! - Component packing (combine results into multivector)
//!
//! IMPORTANT: This uses rotation keys, not component extraction!

use crate::clifford_fhe::ckks::{multiply, multiply_by_plaintext, rotate, Ciphertext, Plaintext};
use crate::clifford_fhe::keys::{EvaluationKey, RotationKey};
use crate::clifford_fhe::params::CliffordFHEParams;

/// Compute product of component i from ct_a with component j from ct_b
///
/// Strategy:
/// 1. Mask ct_a to isolate component i (multiplication with selector polynomial)
/// 2. Mask ct_b to isolate component j
/// 3. Multiply the masked ciphertexts
/// 4. The result has the product at position (i+j) mod N
/// 5. Use rotation to move it to the desired position k
///
/// # Arguments
/// * `ct_a` - First ciphertext (encrypts multivector a)
/// * `i` - Component index from ct_a (0-7 for Cl(3,0))
/// * `ct_b` - Second ciphertext (encrypts multivector b)
/// * `j` - Component index from ct_b (0-7)
/// * `target_position` - Where to place the result (0-7)
/// * `evk` - Evaluation key for multiplication
/// * `rotk` - Rotation key for moving result to target position
/// * `params` - FHE parameters
///
/// # Returns
/// Ciphertext encrypting the product a[i] * b[j] at position `target_position`
pub fn compute_component_product(
    ct_a: &Ciphertext,
    i: usize,
    ct_b: &Ciphertext,
    j: usize,
    target_position: usize,
    evk: &EvaluationKey,
    rotk: &RotationKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    assert!(i < 8 && j < 8 && target_position < 8, "Invalid component index");

    // Create selector polynomials (1 at position i or j, 0 elsewhere)
    let mut selector_i = vec![0i64; params.n];
    selector_i[i] = params.scale as i64;
    let pt_i = Plaintext::new(selector_i, params.scale);

    let mut selector_j = vec![0i64; params.n];
    selector_j[j] = params.scale as i64;
    let pt_j = Plaintext::new(selector_j, params.scale);

    // Mask ct_a to select component i
    let ct_a_masked = multiply_by_plaintext(ct_a, &pt_i, params);

    // Mask ct_b to select component j
    let ct_b_masked = multiply_by_plaintext(ct_b, &pt_j, params);

    // Multiply the masked ciphertexts
    // Result has product at position (i+j) with negacyclic reduction
    let ct_product = multiply(&ct_a_masked, &ct_b_masked, evk, params);

    // Compute where the product ended up after polynomial multiplication
    let product_position = if i + j < params.n {
        i + j
    } else {
        (i + j) % params.n
    };

    // Rotate to move product from product_position to target_position
    let rotation_amount = if target_position >= product_position {
        (target_position - product_position) as isize
    } else {
        (params.n + target_position - product_position) as isize
    };

    if rotation_amount == 0 {
        // No rotation needed
        ct_product
    } else {
        rotate(&ct_product, rotation_amount, rotk, params)
    }
}

/// Add two ciphertexts (component-wise addition)
pub fn add_ciphertexts(
    ct1: &Ciphertext,
    ct2: &Ciphertext,
    params: &CliffordFHEParams,
) -> Ciphertext {
    use crate::clifford_fhe::ckks::add;
    add(ct1, ct2, params)
}

/// Multiply ciphertext by plaintext scalar
///
/// This is used for applying structure constant coefficients (+1 or -1)
pub fn multiply_by_scalar(ct: &Ciphertext, scalar: i64, params: &CliffordFHEParams) -> Ciphertext {
    let q = params.modulus_at_level(ct.level);

    let c0: Vec<i64> = ct.c0.iter().map(|&x| ((x * scalar) % q + q) % q).collect();
    let c1: Vec<i64> = ct.c1.iter().map(|&x| ((x * scalar) % q + q) % q).collect();

    Ciphertext::new(c0, c1, ct.level, ct.scale)
}

/// Negate a ciphertext (multiply by -1)
pub fn negate(ct: &Ciphertext, params: &CliffordFHEParams) -> Ciphertext {
    multiply_by_scalar(ct, -1, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe::encoding::{decode_multivector, encode_multivector};
    use crate::clifford_fhe::keys::keygen_with_rotation;
    use crate::clifford_fhe::ckks::{encrypt, decrypt};

    #[test]
    fn test_multiply_by_scalar() {
        let params = CliffordFHEParams::new_128bit();
        let (pk, sk, _evk, _rotk) = keygen_with_rotation(&params);

        // Create multivector [1, 2, 3, 4, 5, 6, 7, 8]
        let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let pt_coeffs = encode_multivector(&mv, params.scale, params.n);
        let pt = Plaintext::new(pt_coeffs, params.scale);
        let ct = encrypt(&pk, &pt, &params);

        // Multiply by -1
        let ct_neg = multiply_by_scalar(&ct, -1, &params);

        // Decrypt and check
        let pt_neg = decrypt(&sk, &ct_neg, &params);
        let mv_neg = decode_multivector(&pt_neg.coeffs, params.scale);

        // Check negation worked (within error tolerance)
        for i in 0..8 {
            let error = (mv_neg[i] + mv[i]).abs();
            assert!(
                error < 0.1,
                "Negation error too large at component {}: {} (expected ~0)",
                i,
                error
            );
        }
    }
}
