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
/// Allows rotating slots in SIMD-packed ciphertexts.
/// Useful for batch operations on multiple multivectors.
#[derive(Debug, Clone)]
pub struct RotationKey {
    /// Keys for different rotation amounts
    pub keys: Vec<(usize, Vec<i64>, Vec<i64>)>,
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

    // Generate rotation keys
    // For geometric product on Cl(3,0) with 8 components (positions 0-7):
    // - Products of component i and j land at position (i+j) mod N
    // - We need to rotate from position (i+j) to target position k
    // - Rotation amount = (k - (i+j) + N) % N
    // - Since i,j,k ∈ [0,7], we have (i+j) ∈ [0,14] and need rotations to move to [0,7]
    // - Maximum rotation: from 14 to 0 = N-14 or from 0 to 7 = 7
    // Actually, let's generate all rotations from 0 to N-1 for maximum flexibility
    // This is a one-time cost during key generation
    let mut rotation_amounts = Vec::new();
    for i in 0..params.n {
        rotation_amounts.push(i);
    }
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

/// Generate rotation keys for SIMD slot rotations
///
/// For each rotation amount r, generates a key that allows rotating
/// ciphertext slots by r positions.
///
/// # Arguments
/// * `sk` - Secret key
/// * `rotation_amounts` - List of rotation amounts to generate keys for
/// * `params` - FHE parameters
///
/// # Returns
/// Rotation key containing keys for all requested rotations
fn generate_rotation_keys(
    sk: &SecretKey,
    rotation_amounts: &[usize],
    params: &CliffordFHEParams,
) -> RotationKey {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let n = params.n;
    let q = params.modulus_at_level(0);

    let mut keys = Vec::new();

    for &r in rotation_amounts {
        // Compute s(x^(5^r)) - the secret key with automorphism applied
        // In CKKS, rotation by r slots corresponds to automorphism x → x^(5^r)
        // For simplicity, we'll use x → x^r for coefficient rotations
        let s_rotated = rotate_polynomial(&sk.coeffs, r, n);

        // Sample random polynomial
        let a: Vec<i64> = (0..n).map(|_| rng.gen_range(0..q)).collect();

        // Sample error
        let e: Vec<i64> = sample_error(n, params.error_std);

        // Compute rot_key = -a*s + e + s(x^r) (mod q)
        let a_times_s = polynomial_multiply_ntt(&a, &sk.coeffs, q, n);

        let rot_key_0: Vec<i64> = a_times_s
            .iter()
            .zip(&e)
            .zip(&s_rotated)
            .map(|((as_i, e_i), sr_i)| {
                let val = -as_i + e_i + sr_i;
                ((val % q) + q) % q
            })
            .collect();

        keys.push((r, rot_key_0, a));
    }

    RotationKey { keys, n }
}

/// Rotate polynomial coefficients by r positions
///
/// This implements the automorphism x → x^r in the polynomial ring
fn rotate_polynomial(poly: &[i64], r: usize, n: usize) -> Vec<i64> {
    let mut result = vec![0i64; n];

    for i in 0..n {
        let new_idx = (i * r) % (2 * n);
        if new_idx < n {
            result[new_idx] = poly[i];
        } else {
            // Negacyclic reduction: x^n = -1
            result[new_idx % n] = -poly[i];
        }
    }

    result
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
