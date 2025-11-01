//! Key structures for Clifford-FHE
//!
//! Implements CKKS key generation and management for geometric algebra operations.

use crate::clifford_fhe::params::CliffordFHEParams;

/// Secret key for CKKS
///
/// In CKKS, the secret key is a polynomial s(x) with small coefficients (typically {-1, 0, 1})
#[derive(Debug, Clone)]
pub struct SecretKey {
    /// Secret polynomial coefficients
    pub coeffs: Vec<i64>,
    /// Ring dimension
    pub n: usize,
}

impl SecretKey {
    /// Generate secret key from random ternary polynomial
    ///
    /// Coefficients drawn from {-1, 0, 1} with:
    /// - P(-1) = P(1) = 1/3
    /// - P(0) = 1/3
    pub fn generate(params: &CliffordFHEParams) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let coeffs: Vec<i64> = (0..params.n)
            .map(|_| {
                let r: f64 = rng.gen();
                if r < 0.333 {
                    -1
                } else if r < 0.666 {
                    0
                } else {
                    1
                }
            })
            .collect();

        Self {
            coeffs,
            n: params.n,
        }
    }
}

/// Public key for CKKS
///
/// Public key is a pair (b, a) where:
/// - a is random polynomial
/// - b = -a*s + e (where s is secret key, e is small error)
#[derive(Debug, Clone)]
pub struct PublicKey {
    /// First component (b = -a*s + e)
    pub b: Vec<i64>,
    /// Second component (random)
    pub a: Vec<i64>,
    /// Ring dimension
    pub n: usize,
    /// Current modulus
    pub q: i64,
}

/// Evaluation key (relinearization key)
///
/// Allows converting degree-2 ciphertexts (after multiplication) back to degree-1.
/// This is essential for homomorphic multiplication!
///
/// Structure: evk = (evk0, evk1) where evk encrypts s²
#[derive(Debug, Clone)]
pub struct EvaluationKey {
    /// Relinearization key components
    pub relin_keys: Vec<(Vec<i64>, Vec<i64>)>,
    /// Ring dimension
    pub n: usize,
}

/// Rotation key (for slot rotations in SIMD mode)
///
/// Allows rotating slots in SIMD-packed ciphertexts using Galois automorphisms.
/// Maps automorphism index k → rotation key pair (key0, key1).
#[derive(Debug, Clone)]
pub struct RotationKey {
    /// Keys for different automorphism indices
    /// Maps k → (rot_key_0, rot_key_1) where k is automorphism index
    pub keys: std::collections::HashMap<usize, (Vec<i64>, Vec<i64>)>,
    /// Ring dimension
    pub n: usize,
}

/// Generate all keys for Clifford-FHE
///
/// Returns: (public_key, secret_key, evaluation_key)
///
/// # Example
/// ```rust,ignore
/// let params = CliffordFHEParams::new_128bit();
/// let (pk, sk, evk) = keygen(&params);
/// ```
pub fn keygen(params: &CliffordFHEParams) -> (PublicKey, SecretKey, EvaluationKey) {
    // Generate secret key
    let sk = SecretKey::generate(params);

    // Generate public key
    let pk = generate_public_key(&sk, params);

    // Generate evaluation key (for relinearization)
    let evk = generate_evaluation_key(&sk, params);

    (pk, sk, evk)
}

/// Generate all keys including rotation keys for Clifford-FHE
///
/// Returns: (public_key, secret_key, evaluation_key, rotation_key)
///
/// # Example
/// ```rust,ignore
/// let params = CliffordFHEParams::new_128bit();
/// let (pk, sk, evk, rotk) = keygen_with_rotation(&params);
/// ```
pub fn keygen_with_rotation(
    params: &CliffordFHEParams,
) -> (PublicKey, SecretKey, EvaluationKey, RotationKey) {
    // Generate secret key
    let sk = SecretKey::generate(params);

    // Generate public key
    let pk = generate_public_key(&sk, params);

    // Generate evaluation key (for relinearization)
    let evk = generate_evaluation_key(&sk, params);

    // Generate rotation keys for SIMD slot operations
    // For geometric product on Cl(3,0) with 8 components in slots 0-7:
    // - Need to extract/place values at slots 0-7
    // - Requires rotations: -7, -6, ..., -1, 0, 1, ..., 6, 7
    // This allows moving any slot to position 0 and back
    let rotation_amounts: Vec<isize> = (-7..=7).collect();
    let rotk = generate_rotation_keys(&sk, &rotation_amounts, params);

    (pk, sk, evk, rotk)
}

/// Generate public key from secret key
fn generate_public_key(sk: &SecretKey, params: &CliffordFHEParams) -> PublicKey {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let n = params.n;
    let q = params.modulus_at_level(0); // Use largest modulus for fresh pk

    // Sample random polynomial 'a' uniformly from Z_q
    let a: Vec<i64> = (0..n).map(|_| rng.gen_range(0..q)).collect();

    // Sample error polynomial 'e' from Gaussian distribution
    let e: Vec<i64> = sample_error(n, params.error_std);

    // Compute b = -a*s + e (mod q)
    let a_times_s = polynomial_multiply_ntt(&a, &sk.coeffs, q, n);
    let b: Vec<i64> = a_times_s
        .iter()
        .zip(&e)
        .map(|(as_i, e_i)| {
            let val = -as_i + e_i;
            ((val % q) + q) % q
        })
        .collect();

    PublicKey { b, a, n, q }
}

/// Generate evaluation key for relinearization
fn generate_evaluation_key(sk: &SecretKey, params: &CliffordFHEParams) -> EvaluationKey {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let n = params.n;
    let q = params.modulus_at_level(0);

    // Compute s² (needed for relinearization)
    let s_squared = polynomial_multiply_ntt(&sk.coeffs, &sk.coeffs, q, n);

    // Number of decomposition levels (for digit decomposition)
    let num_levels = 3; // Typical value, trade-off between size and speed

    let mut relin_keys = Vec::new();

    for level in 0..num_levels {
        // Sample random polynomial
        let a: Vec<i64> = (0..n).map(|_| rng.gen_range(0..q)).collect();

        // Sample error
        let e: Vec<i64> = sample_error(n, params.error_std);

        // Compute evk_level = -a*s + e + 2^level * s² (mod q)
        let a_times_s = polynomial_multiply_ntt(&a, &sk.coeffs, q, n);

        let base = 1i64 << (level * 20); // Decomposition base (adjustable)

        let evk: Vec<i64> = a_times_s
            .iter()
            .zip(&e)
            .zip(&s_squared)
            .map(|((as_i, e_i), s2_i)| {
                let val = -as_i + e_i + base * s2_i;
                ((val % q) + q) % q
            })
            .collect();

        relin_keys.push((evk, a));
    }

    EvaluationKey { relin_keys, n }
}

/// Generate rotation keys for SIMD slot rotations using Galois automorphisms
///
/// For each rotation amount r, computes the corresponding Galois automorphism
/// index k = 5^r mod M and generates a key for that automorphism.
///
/// # Arguments
/// * `sk` - Secret key
/// * `rotation_amounts` - List of rotation amounts (positive = left, negative = right)
/// * `params` - FHE parameters
///
/// # Returns
/// Rotation key mapping automorphism indices to key pairs
fn generate_rotation_keys(
    sk: &SecretKey,
    rotation_amounts: &[isize],
    params: &CliffordFHEParams,
) -> RotationKey {
    use rand::Rng;
    use crate::clifford_fhe::automorphisms::{rotation_to_automorphism, apply_automorphism};
    use std::collections::HashMap;

    let mut rng = rand::thread_rng();
    let n = params.n;
    let q = params.modulus_at_level(0);

    let mut keys = HashMap::new();

    for &r in rotation_amounts {
        // Convert rotation amount to automorphism index
        let k = rotation_to_automorphism(r, n);

        // Skip if we already have a key for this automorphism
        if keys.contains_key(&k) {
            continue;
        }

        // Apply Galois automorphism σₖ to secret key: s(x) → s(x^k)
        let s_automorphed = apply_automorphism(&sk.coeffs, k, n);

        // Sample random polynomial
        let a: Vec<i64> = (0..n).map(|_| rng.gen_range(0..q)).collect();

        // Sample error
        let e: Vec<i64> = sample_error(n, params.error_std);

        // Compute rotation key: rot_key_0 = -a*s + e + s(x^k) (mod q)
        let a_times_s = polynomial_multiply_ntt(&a, &sk.coeffs, q, n);

        let rot_key_0: Vec<i64> = a_times_s
            .iter()
            .zip(&e)
            .zip(&s_automorphed)
            .map(|((as_i, e_i), sk_i)| {
                let val = -as_i + e_i + sk_i;
                ((val % q) + q) % q
            })
            .collect();

        keys.insert(k, (rot_key_0, a));
    }

    RotationKey { keys, n }
}

/// Sample error polynomial from discrete Gaussian distribution
///
/// Standard CKKS error sampling: coefficients ~ N(0, σ²) discretized to integers
fn sample_error(n: usize, sigma: f64) -> Vec<i64> {
    use rand::thread_rng;
    use rand_distr::{Distribution, Normal};

    let normal = Normal::new(0.0, sigma).unwrap();
    let mut rng = thread_rng();

    (0..n)
        .map(|_| {
            let sample = normal.sample(&mut rng);
            sample.round() as i64
        })
        .collect()
}

/// Polynomial multiplication using NTT (stub - will use our existing NTT)
///
/// TODO: Integrate with our existing NTT implementation from ntt.rs
fn polynomial_multiply_ntt(a: &[i64], b: &[i64], q: i64, n: usize) -> Vec<i64> {
    // Temporary naive implementation
    // Will replace with our optimized NTT code
    let mut result = vec![0i64; n];

    for i in 0..n {
        for j in 0..n {
            if i + j < n {
                result[i + j] = (result[i + j] + a[i] * b[j]) % q;
            } else {
                // Reduction by x^n + 1: x^n = -1
                let idx = (i + j) % n;
                result[idx] = (result[idx] - a[i] * b[j]) % q;
            }
        }
    }

    // Ensure positive coefficients
    result.iter().map(|&x| ((x % q) + q) % q).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_key_generation() {
        let params = CliffordFHEParams::new_128bit();
        let sk = SecretKey::generate(&params);

        assert_eq!(sk.coeffs.len(), params.n);

        // Check coefficients are in {-1, 0, 1}
        for &coeff in &sk.coeffs {
            assert!(coeff == -1 || coeff == 0 || coeff == 1);
        }
    }

    #[test]
    fn test_keygen() {
        let params = CliffordFHEParams::new_128bit();
        let (pk, sk, evk) = keygen(&params);

        assert_eq!(pk.n, params.n);
        assert_eq!(sk.n, params.n);
        assert_eq!(evk.n, params.n);

        assert_eq!(pk.a.len(), params.n);
        assert_eq!(pk.b.len(), params.n);
    }

    #[test]
    fn test_polynomial_multiply() {
        let n = 8;
        let q = 100;

        let a = vec![1, 0, 1, 0, 0, 0, 0, 0]; // 1 + x²
        let b = vec![1, 1, 0, 0, 0, 0, 0, 0]; // 1 + x

        let result = polynomial_multiply_ntt(&a, &b, q, n);

        // (1 + x²)(1 + x) = 1 + x + x² + x³
        assert_eq!(result[0], 1); // constant term
        assert_eq!(result[1], 1); // x
        assert_eq!(result[2], 1); // x²
        assert_eq!(result[3], 1); // x³
    }
}
