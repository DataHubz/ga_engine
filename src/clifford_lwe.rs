//! Clifford-LWE Cryptosystem
//!
//! Learning With Errors (LWE) encryption using Clifford algebra elements.
//!
//! This module provides the core cryptographic primitives for Clifford-LWE:
//! - Key generation
//! - Encryption
//! - Decryption
//! - Homomorphic operations

use crate::clifford_ring_int::{CliffordPolynomialInt, CliffordRingElementInt};
use crate::lazy_reduction::LazyReductionContext;
use crate::ntt_optimized::OptimizedNTTContext;
use crate::ntt_clifford_optimized::multiply_ntt_optimized;
use crate::shake_poly::{discrete_poly_shake, error_poly_shake, generate_seed};

/// Clifford-LWE parameters
#[derive(Clone, Debug)]
pub struct CliffordLWEParams {
    pub n: usize,           // Polynomial degree
    pub q: i64,             // Modulus
    pub error_bound: i64,   // Error sampled from [-error_bound, error_bound]
}

impl Default for CliffordLWEParams {
    fn default() -> Self {
        Self {
            n: 32,
            q: 3329,
            error_bound: 2,
        }
    }
}

/// Public key for Clifford-LWE
#[derive(Clone)]
pub struct PublicKey {
    pub a: CliffordPolynomialInt,
    pub b: CliffordPolynomialInt,
}

/// Secret key for Clifford-LWE
#[derive(Clone)]
pub struct SecretKey {
    pub s: CliffordPolynomialInt,
}

/// Ciphertext for Clifford-LWE
#[derive(Clone)]
pub struct Ciphertext {
    pub u: CliffordPolynomialInt,  // First component
    pub v: CliffordPolynomialInt,  // Second component (contains message)
}

/// Generate a keypair for Clifford-LWE
pub fn keygen(
    params: &CliffordLWEParams,
    ntt: &OptimizedNTTContext,
    lazy: &LazyReductionContext,
) -> (PublicKey, SecretKey) {
    let seed_a = generate_seed();
    let seed_s = generate_seed();
    let seed_e = generate_seed();

    let a = discrete_poly_shake(&seed_a, params.n);
    let s = error_poly_shake(&seed_s, params.n, params.error_bound);
    let e = error_poly_shake(&seed_e, params.n, params.error_bound);

    // b = a ⊗ s + e
    let b = {
        let as_product = multiply_ntt_optimized(&a, &s, ntt, lazy);
        add_polynomials(&as_product, &e, params.q)
    };

    let pk = PublicKey { a, b };
    let sk = SecretKey { s };

    (pk, sk)
}

/// Encrypt a Clifford element (8-component vector) under public key
pub fn encrypt(
    ntt: &OptimizedNTTContext,
    pk: &PublicKey,
    message: &CliffordPolynomialInt,
    params: &CliffordLWEParams,
    lazy: &LazyReductionContext,
) -> Ciphertext {
    let seed_r = generate_seed();
    let seed_e1 = generate_seed();
    let seed_e2 = generate_seed();

    let r = error_poly_shake(&seed_r, params.n, params.error_bound);
    let e1 = error_poly_shake(&seed_e1, params.n, params.error_bound);
    let e2 = error_poly_shake(&seed_e2, params.n, params.error_bound);

    // u = a ⊗ r + e1
    let u = {
        let ar_product = multiply_ntt_optimized(&pk.a, &r, ntt, lazy);
        add_polynomials(&ar_product, &e1, params.q)
    };

    // v = b ⊗ r + e2 + message
    let v = {
        let br_product = multiply_ntt_optimized(&pk.b, &r, ntt, lazy);
        let br_plus_e2 = add_polynomials(&br_product, &e2, params.q);
        add_polynomials(&br_plus_e2, message, params.q)
    };

    Ciphertext { u, v }
}

/// Decrypt a ciphertext using secret key
pub fn decrypt(
    ntt: &OptimizedNTTContext,
    sk: &SecretKey,
    ct: &Ciphertext,
    params: &CliffordLWEParams,
    lazy: &LazyReductionContext,
) -> CliffordPolynomialInt {
    // message = v - s ⊗ u
    let su_product = multiply_ntt_optimized(&sk.s, &ct.u, ntt, lazy);
    sub_polynomials(&ct.v, &su_product, params.q)
}

/// Add two Clifford polynomials component-wise (mod q)
pub fn add_polynomials(
    a: &CliffordPolynomialInt,
    b: &CliffordPolynomialInt,
    q: i64,
) -> CliffordPolynomialInt {
    assert_eq!(a.coeffs.len(), b.coeffs.len());

    let mut result_coeffs = Vec::with_capacity(a.coeffs.len());
    for i in 0..a.coeffs.len() {
        let mut sum = [0i64; 8];
        for j in 0..8 {
            sum[j] = ((a.coeffs[i].coeffs[j] + b.coeffs[i].coeffs[j]) % q + q) % q;
        }
        result_coeffs.push(CliffordRingElementInt::from_multivector(sum));
    }

    CliffordPolynomialInt::new(result_coeffs)
}

/// Subtract two Clifford polynomials component-wise (mod q)
pub fn sub_polynomials(
    a: &CliffordPolynomialInt,
    b: &CliffordPolynomialInt,
    q: i64,
) -> CliffordPolynomialInt {
    assert_eq!(a.coeffs.len(), b.coeffs.len());

    let mut result_coeffs = Vec::with_capacity(a.coeffs.len());
    for i in 0..a.coeffs.len() {
        let mut diff = [0i64; 8];
        for j in 0..8 {
            diff[j] = ((a.coeffs[i].coeffs[j] - b.coeffs[i].coeffs[j]) % q + q) % q;
        }
        result_coeffs.push(CliffordRingElementInt::from_multivector(diff));
    }

    CliffordPolynomialInt::new(result_coeffs)
}

/// Scale a Clifford polynomial by a scalar (mod q)
pub fn scale_polynomial(
    a: &CliffordPolynomialInt,
    scalar: i64,
    q: i64,
) -> CliffordPolynomialInt {
    let mut result_coeffs = Vec::with_capacity(a.coeffs.len());
    for i in 0..a.coeffs.len() {
        let mut scaled = [0i64; 8];
        for j in 0..8 {
            scaled[j] = ((a.coeffs[i].coeffs[j] * scalar) % q + q) % q;
        }
        result_coeffs.push(CliffordRingElementInt::from_multivector(scaled));
    }

    CliffordPolynomialInt::new(result_coeffs)
}

/// Homomorphic addition: E(a) + E(b) = E(a + b)
pub fn homomorphic_add(
    ct1: &Ciphertext,
    ct2: &Ciphertext,
    q: i64,
) -> Ciphertext {
    Ciphertext {
        u: add_polynomials(&ct1.u, &ct2.u, q),
        v: add_polynomials(&ct1.v, &ct2.v, q),
    }
}

/// Homomorphic scalar multiplication: α · E(a) = E(α · a)
pub fn homomorphic_scale(
    ct: &Ciphertext,
    scalar: i64,
    q: i64,
) -> Ciphertext {
    Ciphertext {
        u: scale_polynomial(&ct.u, scalar, q),
        v: scale_polynomial(&ct.v, scalar, q),
    }
}

/// Homomorphic linear transformation
///
/// Apply an 8×8 matrix M to the encrypted Clifford element.
/// This enables homomorphic rotations, reflections, and other linear geometric operations.
///
/// **Key insight**: Geometric operations like R ⊗ v ⊗ R̃ are linear transformations
/// on the 8-component vector, so they can be applied homomorphically!
///
/// # Arguments
/// * `matrix` - 8×8 transformation matrix (e.g., rotation matrix)
/// * `ct` - Ciphertext to transform
/// * `q` - Modulus
///
/// # Returns
/// Transformed ciphertext E(M · v)
pub fn homomorphic_linear_transform(
    matrix: &[[i64; 8]; 8],
    ct: &Ciphertext,
    q: i64,
) -> Ciphertext {
    let n = ct.u.coeffs.len();

    // For each polynomial coefficient position, apply the matrix
    let mut u_result_coeffs = Vec::with_capacity(n);
    let mut v_result_coeffs = Vec::with_capacity(n);

    for pos in 0..n {
        let mut u_transformed = [0i64; 8];
        let mut v_transformed = [0i64; 8];

        // Matrix multiplication: result[i] = Σⱼ M[i][j] · input[j]
        for i in 0..8 {
            for j in 0..8 {
                u_transformed[i] = (u_transformed[i] + matrix[i][j] * ct.u.coeffs[pos].coeffs[j]) % q;
                v_transformed[i] = (v_transformed[i] + matrix[i][j] * ct.v.coeffs[pos].coeffs[j]) % q;
            }
            u_transformed[i] = (u_transformed[i] % q + q) % q;
            v_transformed[i] = (v_transformed[i] % q + q) % q;
        }

        u_result_coeffs.push(CliffordRingElementInt::from_multivector(u_transformed));
        v_result_coeffs.push(CliffordRingElementInt::from_multivector(v_transformed));
    }

    Ciphertext {
        u: CliffordPolynomialInt::new(u_result_coeffs),
        v: CliffordPolynomialInt::new(v_result_coeffs),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt() {
        let params = CliffordLWEParams::default();
        let ntt = OptimizedNTTContext::new_clifford_lwe();
        let lazy = LazyReductionContext::new(params.q);

        let (pk, sk) = keygen(&params, &ntt, &lazy);

        // Create test message (just scalar and e1 components)
        let mut msg_coeffs = Vec::new();
        for i in 0..params.n {
            let mut mv = [0i64; 8];
            mv[0] = if i == 0 { 1 } else { 0 };  // Scalar: 1 at position 0
            mv[1] = if i == 1 { 1 } else { 0 };  // e1: 1 at position 1
            msg_coeffs.push(CliffordRingElementInt::from_multivector(mv));
        }
        let message = CliffordPolynomialInt::new(msg_coeffs);

        let ct = encrypt(&ntt, &pk, &message, &params, &lazy);
        let decrypted = decrypt(&ntt, &sk, &ct, &params, &lazy);

        // Verify decryption matches original (allowing for small error)
        for i in 0..params.n {
            for j in 0..8 {
                let original = message.coeffs[i].coeffs[j];
                let recovered = decrypted.coeffs[i].coeffs[j];
                let error = ((original - recovered) % params.q + params.q) % params.q;

                // Error should be small (< q/4)
                assert!(error < params.q / 4 || error > 3 * params.q / 4,
                    "Large decryption error at pos {} component {}: {} vs {} (error {})",
                    i, j, original, recovered, error);
            }
        }
    }

    #[test]
    fn test_homomorphic_addition() {
        let params = CliffordLWEParams::default();
        let ntt = OptimizedNTTContext::new_clifford_lwe();
        let lazy = LazyReductionContext::new(params.q);

        let (pk, sk) = keygen(&params, &ntt, &lazy);

        // Create two simple messages
        let msg1_coeffs = vec![CliffordRingElementInt::from_multivector([1, 0, 0, 0, 0, 0, 0, 0]); params.n];
        let msg2_coeffs = vec![CliffordRingElementInt::from_multivector([2, 0, 0, 0, 0, 0, 0, 0]); params.n];

        let msg1 = CliffordPolynomialInt::new(msg1_coeffs);
        let msg2 = CliffordPolynomialInt::new(msg2_coeffs);

        let ct1 = encrypt(&ntt, &pk, &msg1, &params, &lazy);
        let ct2 = encrypt(&ntt, &pk, &msg2, &params, &lazy);

        // Homomorphic addition
        let ct_sum = homomorphic_add(&ct1, &ct2, params.q);
        let decrypted_sum = decrypt(&ntt, &sk, &ct_sum, &params, &lazy);

        // Should decrypt to msg1 + msg2 = 3 (with small error)
        let scalar_sum = decrypted_sum.coeffs[0].coeffs[0];
        assert!((scalar_sum - 3).abs() < 10, "Homomorphic addition failed: got {}", scalar_sum);
    }
}
