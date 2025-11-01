//! CKKS encryption scheme adapted for Clifford algebra
//!
//! Implements the core CKKS operations: encryption, decryption,
//! homomorphic addition, and homomorphic multiplication.

use crate::clifford_fhe::keys::{EvaluationKey, PublicKey, SecretKey};
use crate::clifford_fhe::params::CliffordFHEParams;

/// CKKS plaintext (polynomial representation)
#[derive(Debug, Clone)]
pub struct Plaintext {
    /// Polynomial coefficients
    pub coeffs: Vec<i64>,
    /// Scaling factor used for encoding
    pub scale: f64,
    /// Ring dimension
    pub n: usize,
}

impl Plaintext {
    /// Create plaintext from polynomial coefficients
    pub fn new(coeffs: Vec<i64>, scale: f64) -> Self {
        let n = coeffs.len();
        Self { coeffs, scale, n }
    }
}

/// CKKS ciphertext
///
/// A ciphertext is a pair (c0, c1) of polynomials in R_q
/// Decryption: m ≈ c0 + c1*s (mod q)
#[derive(Debug, Clone)]
pub struct Ciphertext {
    /// First component
    pub c0: Vec<i64>,
    /// Second component
    pub c1: Vec<i64>,
    /// Current level (determines modulus)
    pub level: usize,
    /// Scaling factor (carries through homomorphic operations)
    pub scale: f64,
    /// Ring dimension
    pub n: usize,
}

impl Ciphertext {
    /// Create new ciphertext
    pub fn new(c0: Vec<i64>, c1: Vec<i64>, level: usize, scale: f64) -> Self {
        let n = c0.len();
        assert_eq!(c1.len(), n, "Ciphertext components must have same length");
        Self {
            c0,
            c1,
            level,
            scale,
            n,
        }
    }
}

/// Encrypt plaintext using public key
///
/// CKKS encryption:
/// 1. Sample random r, e0, e1 from error distribution
/// 2. Compute: c0 = b*r + e0 + m
///            c1 = a*r + e1
/// where (a, b) is the public key
///
/// # Example
/// ```rust,ignore
/// let pt = Plaintext::new(coeffs, scale);
/// let ct = encrypt(&pk, &pt, &params);
/// ```
pub fn encrypt(pk: &PublicKey, pt: &Plaintext, params: &CliffordFHEParams) -> Ciphertext {
    use rand::{thread_rng, Rng};
    use rand_distr::{Distribution, Normal};

    let n = params.n;
    let q = params.modulus_at_level(0); // Fresh ciphertext at level 0
    let mut rng = thread_rng();

    // Sample random polynomial r from {-1, 0, 1}
    let r: Vec<i64> = (0..n)
        .map(|_| {
            let val: f64 = rng.gen();
            if val < 0.333 {
                -1
            } else if val < 0.666 {
                0
            } else {
                1
            }
        })
        .collect();

    // Sample errors e0, e1 from Gaussian
    let normal = Normal::new(0.0, params.error_std).unwrap();
    let e0: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();
    let e1: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();

    // Compute b*r (using our NTT - TODO: integrate properly)
    let br = polynomial_multiply_ntt(&pk.b, &r, q, n);

    // Compute a*r
    let ar = polynomial_multiply_ntt(&pk.a, &r, q, n);

    // c0 = b*r + e0 + m (mod q)
    let c0: Vec<i64> = br
        .iter()
        .zip(&e0)
        .zip(&pt.coeffs)
        .map(|((br_i, e0_i), m_i)| {
            let val = br_i + e0_i + m_i;
            ((val % q) + q) % q
        })
        .collect();

    // c1 = a*r + e1 (mod q)
    let c1: Vec<i64> = ar
        .iter()
        .zip(&e1)
        .map(|(ar_i, e1_i)| {
            let val = ar_i + e1_i;
            ((val % q) + q) % q
        })
        .collect();

    Ciphertext::new(c0, c1, 0, pt.scale)
}

/// Decrypt ciphertext using secret key
///
/// CKKS decryption:
/// m' = c0 + c1*s (mod q)
///
/// The result m' should be close to the original message m (with some noise)
pub fn decrypt(sk: &SecretKey, ct: &Ciphertext, params: &CliffordFHEParams) -> Plaintext {
    let n = ct.n;
    let q = params.modulus_at_level(ct.level);

    // Compute c1*s (mod q)
    let c1s = polynomial_multiply_ntt(&ct.c1, &sk.coeffs, q, n);

    // Compute m' = c0 + c1*s (mod q)
    let m_prime: Vec<i64> = ct
        .c0
        .iter()
        .zip(&c1s)
        .map(|(c0_i, c1s_i)| {
            let val = c0_i + c1s_i;
            let result = ((val % q) + q) % q;

            // Center around 0 for proper decoding
            if result > q / 2 {
                result - q
            } else {
                result
            }
        })
        .collect();

    Plaintext::new(m_prime, ct.scale)
}

/// Multiply ciphertext by plaintext polynomial
///
/// This is CKKS plaintext-ciphertext multiplication:
/// ct × pt = (c0 × pt, c1 × pt)
///
/// No relinearization needed! This is much faster than ciphertext-ciphertext multiplication.
/// Used for component extraction and other masking operations.
///
/// # Arguments
/// * `ct` - Ciphertext to multiply
/// * `pt` - Plaintext polynomial (e.g., selection mask)
/// * `params` - CKKS parameters
pub fn multiply_by_plaintext(
    ct: &Ciphertext,
    pt: &Plaintext,
    params: &CliffordFHEParams,
) -> Ciphertext {
    let q = params.modulus_at_level(ct.level);
    let n = ct.n;

    // Multiply each ciphertext component by plaintext
    // c0' = c0 × pt (mod q)
    let c0_new = polynomial_multiply_ntt(&ct.c0, &pt.coeffs, q, n);

    // c1' = c1 × pt (mod q)
    let c1_new = polynomial_multiply_ntt(&ct.c1, &pt.coeffs, q, n);

    // Scale is multiplied (CKKS scaling semantics)
    let new_scale = ct.scale * pt.scale / params.scale;

    Ciphertext::new(c0_new, c1_new, ct.level, new_scale)
}

/// Homomorphic addition
///
/// Add two ciphertexts component-wise:
/// (c0, c1) + (d0, d1) = (c0 + d0, c1 + d1)
///
/// Noise grows additively: error(ct1 + ct2) ≈ error(ct1) + error(ct2)
pub fn add(ct1: &Ciphertext, ct2: &Ciphertext, params: &CliffordFHEParams) -> Ciphertext {
    assert_eq!(ct1.n, ct2.n, "Ciphertexts must have same dimension");
    assert_eq!(
        ct1.level, ct2.level,
        "Ciphertexts must be at same level for addition"
    );

    let q = params.modulus_at_level(ct1.level);

    let c0: Vec<i64> = ct1
        .c0
        .iter()
        .zip(&ct2.c0)
        .map(|(a, b)| {
            let val = a + b;
            ((val % q) + q) % q
        })
        .collect();

    let c1: Vec<i64> = ct1
        .c1
        .iter()
        .zip(&ct2.c1)
        .map(|(a, b)| {
            let val = a + b;
            ((val % q) + q) % q
        })
        .collect();

    // Scale should be same for both inputs in CKKS
    assert!(
        (ct1.scale - ct2.scale).abs() < 1.0,
        "Scales must match for addition"
    );

    Ciphertext::new(c0, c1, ct1.level, ct1.scale)
}

/// Homomorphic multiplication (with relinearization)
///
/// This is the KEY operation that makes FHE possible!
///
/// Steps:
/// 1. Multiply ciphertexts (creates degree-2 ciphertext)
/// 2. Relinearize to convert back to degree-1
/// 3. Rescale to manage noise growth
///
/// Noise grows multiplicatively: error(ct1 * ct2) ≈ error(ct1) * error(ct2)
pub fn multiply(
    ct1: &Ciphertext,
    ct2: &Ciphertext,
    evk: &EvaluationKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    assert_eq!(ct1.n, ct2.n, "Ciphertexts must have same dimension");
    assert_eq!(
        ct1.level, ct2.level,
        "Ciphertexts must be at same level for multiplication"
    );

    let n = ct1.n;
    let q = params.modulus_at_level(ct1.level);

    // Step 1: Multiply ciphertexts
    // (c0, c1) * (d0, d1) = (c0*d0, c0*d1 + c1*d0, c1*d1)
    // This creates a degree-2 ciphertext (3 components)!

    let c0d0 = polynomial_multiply_ntt(&ct1.c0, &ct2.c0, q, n);
    let c0d1 = polynomial_multiply_ntt(&ct1.c0, &ct2.c1, q, n);
    let c1d0 = polynomial_multiply_ntt(&ct1.c1, &ct2.c0, q, n);
    let c1d1 = polynomial_multiply_ntt(&ct1.c1, &ct2.c1, q, n);

    // Combine middle terms
    let c_mid: Vec<i64> = c0d1
        .iter()
        .zip(&c1d0)
        .map(|(a, b)| {
            let val = a + b;
            ((val % q) + q) % q
        })
        .collect();

    // Step 2: Relinearization
    // Convert (c0d0, c_mid, c1d1) back to degree-1 using evaluation key
    let (new_c0, new_c1) = relinearize_degree2(&c0d0, &c_mid, &c1d1, evk, q, n);

    // Step 3: Rescaling (to manage noise growth)
    // In CKKS, we divide by scale and move to next level
    let new_scale = ct1.scale * ct2.scale / params.scale;
    let new_level = ct1.level + 1;

    if new_level >= params.max_level() {
        panic!("Multiplication depth exceeded! Need bootstrapping.");
    }

    Ciphertext::new(new_c0, new_c1, new_level, new_scale)
}

/// Relinearize degree-2 ciphertext to degree-1
///
/// Input: (d0, d1, d2) where decryption is d0 + d1*s + d2*s²
/// Output: (c0, c1) where decryption is c0 + c1*s
///
/// Uses evaluation key which encrypts s²
fn relinearize_degree2(
    d0: &[i64],
    d1: &[i64],
    d2: &[i64],
    evk: &EvaluationKey,
    q: i64,
    n: usize,
) -> (Vec<i64>, Vec<i64>) {
    // Decompose d2 into digits for faster relinearization
    // This is a simplified version - production would use proper digit decomposition

    // For now, simplified: use first relinearization key
    let (evk0, evk1) = &evk.relin_keys[0];

    // Multiply d2 by evaluation key components
    let d2_evk0 = polynomial_multiply_ntt(d2, evk0, q, n);
    let d2_evk1 = polynomial_multiply_ntt(d2, evk1, q, n);

    // New c0 = d0 + d2_evk0
    let c0: Vec<i64> = d0
        .iter()
        .zip(&d2_evk0)
        .map(|(a, b)| {
            let val = a + b;
            ((val % q) + q) % q
        })
        .collect();

    // New c1 = d1 + d2_evk1
    let c1: Vec<i64> = d1
        .iter()
        .zip(&d2_evk1)
        .map(|(a, b)| {
            let val = a + b;
            ((val % q) + q) % q
        })
        .collect();

    (c0, c1)
}

/// Polynomial multiplication using NTT
///
/// TODO: Replace this stub with our optimized NTT from ntt.rs
fn polynomial_multiply_ntt(a: &[i64], b: &[i64], q: i64, n: usize) -> Vec<i64> {
    // Temporary naive implementation
    let mut result = vec![0i64; n];

    for i in 0..n {
        for j in 0..n {
            let idx = i + j;
            if idx < n {
                result[idx] = (result[idx] + a[i] * b[j]) % q;
            } else {
                // x^n = -1 reduction
                let wrapped_idx = idx % n;
                result[wrapped_idx] = (result[wrapped_idx] - a[i] * b[j]) % q;
            }
        }
    }

    result.iter().map(|&x| ((x % q) + q) % q).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe::keys::keygen;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let params = CliffordFHEParams::new_128bit();
        let (pk, sk, _evk) = keygen(&params);

        // Create simple plaintext
        let mut coeffs = vec![0i64; params.n];
        coeffs[0] = 1000; // Scaled value
        let scale = params.scale;
        let pt = Plaintext::new(coeffs, scale);

        // Encrypt and decrypt
        let ct = encrypt(&pk, &pt, &params);
        let pt_decrypted = decrypt(&sk, &ct, &params);

        // Check first coefficient (should be close to original)
        let error = (pt_decrypted.coeffs[0] - pt.coeffs[0]).abs();
        assert!(
            error < 100,
            "Decryption error too large: {} (expected < 100)",
            error
        );
    }

    #[test]
    fn test_homomorphic_addition() {
        let params = CliffordFHEParams::new_128bit();
        let (pk, sk, _evk) = keygen(&params);

        let scale = params.scale;

        // Create two plaintexts: [1000, 0, ...] and [2000, 0, ...]
        let mut coeffs1 = vec![0i64; params.n];
        coeffs1[0] = 1000;
        let pt1 = Plaintext::new(coeffs1, scale);

        let mut coeffs2 = vec![0i64; params.n];
        coeffs2[0] = 2000;
        let pt2 = Plaintext::new(coeffs2, scale);

        // Encrypt both
        let ct1 = encrypt(&pk, &pt1, &params);
        let ct2 = encrypt(&pk, &pt2, &params);

        // Homomorphic addition
        let ct_sum = add(&ct1, &ct2, &params);

        // Decrypt result
        let pt_sum = decrypt(&sk, &ct_sum, &params);

        // Should be close to 3000
        let expected = 3000;
        let error = (pt_sum.coeffs[0] - expected).abs();
        assert!(
            error < 200,
            "Addition error too large: {} (expected < 200)",
            error
        );
    }
}
