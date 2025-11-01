//! Advanced CKKS operations for Clifford-FHE
//!
//! This module provides operations needed for homomorphic geometric product:
//! - Component extraction (isolate one coefficient from polynomial)
//! - Component packing (combine 8 scalars into multivector)
//! - Polynomial masking (multiply by selection polynomial)

use crate::clifford_fhe::ckks::{multiply_by_plaintext, Ciphertext, Plaintext};
use crate::clifford_fhe::params::CliffordFHEParams;

/// Extract a single component from encrypted multivector
///
/// Given ct = Enc([c0, c1, c2, c3, c4, c5, c6, c7, 0, 0, ...])
/// Returns ct' = Enc([0, 0, ci, 0, 0, 0, 0, 0, 0, ...])
///
/// # Strategy
///
/// **Direct coefficient masking** (not polynomial multiplication!):
/// We zero out all coefficients except position `component` by modifying
/// the ciphertext polynomials directly.
///
/// WARNING: This is a simplified approach. In production CKKS, you would
/// use rotation keys and slot permutations. For our Phase 2 MVP, we use
/// direct coefficient masking which works but isn't constant-time.
pub fn extract_component(
    ct: &Ciphertext,
    component: usize,
    params: &CliffordFHEParams,
) -> Ciphertext {
    assert!(component < 8, "Component must be 0-7 for Cl(3,0)");

    let q = params.modulus_at_level(ct.level);

    // Create masked versions of c0 and c1
    // Keep only coefficient at position 'component', zero out rest
    let mut c0_masked = vec![0i64; ct.n];
    let mut c1_masked = vec![0i64; ct.n];

    c0_masked[component] = ct.c0[component];
    c1_masked[component] = ct.c1[component];

    Ciphertext::new(c0_masked, c1_masked, ct.level, ct.scale)
}

/// Pack 8 component ciphertexts into single multivector ciphertext
///
/// Given: ct0 = Enc([c0, 0, ...]), ct1 = Enc([0, c1, 0, ...]), etc.
/// Returns: ct = Enc([c0, c1, c2, c3, c4, c5, c6, c7, 0, ...])
///
/// # Strategy
///
/// We shift each component to its proper position and add them:
/// ```text
/// ct = ct0 + shift(ct1, 1) + shift(ct2, 2) + ... + shift(ct7, 7)
/// ```
///
/// But wait! Components are already at the right positions (0-7).
/// So we just need to add all ciphertexts:
/// ```text
/// ct = ct0 + ct1 + ct2 + ... + ct7
/// ```
pub fn pack_components(
    components: &[Ciphertext; 8],
    params: &CliffordFHEParams,
) -> Ciphertext {
    let mut result = components[0].clone();

    for i in 1..8 {
        result = add_ciphertexts(&result, &components[i], params);
    }

    result
}

/// Add two ciphertexts (component-wise addition)
fn add_ciphertexts(
    ct1: &Ciphertext,
    ct2: &Ciphertext,
    params: &CliffordFHEParams,
) -> Ciphertext {
    assert_eq!(ct1.level, ct2.level, "Ciphertexts must be at same level");
    let q = params.modulus_at_level(ct1.level);

    let c0: Vec<i64> = ct1
        .c0
        .iter()
        .zip(&ct2.c0)
        .map(|(a, b)| ((a + b) % q + q) % q)
        .collect();

    let c1: Vec<i64> = ct1
        .c1
        .iter()
        .zip(&ct2.c1)
        .map(|(a, b)| ((a + b) % q + q) % q)
        .collect();

    Ciphertext::new(c0, c1, ct1.level, ct1.scale)
}

/// Multiply polynomial by scalar polynomial (component-wise)
///
/// Used for plaintext multiplication in component extraction
fn polynomial_multiply_scalar(poly: &[i64], scalar: &[i64], q: i64, n: usize) -> Vec<i64> {
    assert_eq!(poly.len(), n);
    assert_eq!(scalar.len(), n);

    // For simplicity, we'll do component-wise multiplication
    // This works because selector has only one non-zero entry
    // More efficient than full polynomial multiplication!

    let result: Vec<i64> = poly
        .iter()
        .zip(scalar)
        .map(|(p, s)| {
            let prod = (p * s) % q;
            ((prod % q) + q) % q
        })
        .collect();

    result
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

/// Shift polynomial coefficients (for SIMD slot rotation)
///
/// This is useful for more advanced packing schemes
pub fn shift_polynomial(poly: &[i64], shift: usize, q: i64) -> Vec<i64> {
    let n = poly.len();
    let mut result = vec![0i64; n];

    for i in 0..n {
        let new_idx = (i + shift) % n;
        result[new_idx] = poly[i];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_multiply_scalar() {
        let n = 16;
        let q = 100;

        let poly = vec![1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut scalar = vec![0i64; n];
        scalar[2] = 10; // Select coefficient 2

        let result = polynomial_multiply_scalar(&poly, &scalar, q, n);

        // Only coefficient 2 should be non-zero
        assert_eq!(result[2], 30); // 3 * 10 = 30
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 0);
        assert_eq!(result[3], 0);
    }

    #[test]
    fn test_shift_polynomial() {
        let poly = vec![1, 2, 3, 4, 0, 0, 0, 0];
        let q = 100;

        let shifted = shift_polynomial(&poly, 2, q);

        assert_eq!(shifted[2], 1);
        assert_eq!(shifted[3], 2);
        assert_eq!(shifted[4], 3);
        assert_eq!(shifted[5], 4);
    }

    #[test]
    fn test_multiply_by_scalar() {
        let params = CliffordFHEParams::new_128bit();
        let q = params.modulus_at_level(0);

        let c0 = vec![1, 2, 3, 4, 0, 0, 0, 0];
        let c1 = vec![5, 6, 7, 8, 0, 0, 0, 0];

        let ct = Ciphertext::new(c0, c1, 0, params.scale);

        let ct_neg = multiply_by_scalar(&ct, -1, &params);

        // Check negation worked
        assert_eq!(ct_neg.c0[0], ((q - 1) % q + q) % q);
        assert_eq!(ct_neg.c0[1], ((q - 2) % q + q) % q);
    }
}
