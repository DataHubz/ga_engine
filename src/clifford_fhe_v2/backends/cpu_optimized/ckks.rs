//! V2 CKKS Encryption/Decryption with NTT Optimization
//!
//! **Optimizations over V1:**
//! - Uses Harvey NTT for O(n log n) polynomial operations
//! - Barrett reduction for fast modular arithmetic
//! - Precomputed NTT contexts for each prime
//! - Optimized RNS representation
//!
//! **Performance Target:** 10× faster encryption/decryption vs V1

use crate::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::{
    BarrettReducer, RnsContext, RnsRepresentation,
};
use crate::clifford_fhe_v2::params::CliffordFHEParams;

/// V2 Plaintext in RNS representation with NTT support
#[derive(Clone, Debug)]
pub struct Plaintext {
    /// Polynomial coefficients in RNS form
    /// coeffs[i] = polynomial mod q_i (for each prime q_i)
    pub coeffs: Vec<RnsRepresentation>,

    /// Scaling factor for fixed-point encoding
    pub scale: f64,

    /// Ring dimension (polynomial degree)
    pub n: usize,

    /// Current level (number of active primes)
    pub level: usize,
}

impl Plaintext {
    /// Create plaintext from RNS coefficients
    pub fn new(coeffs: Vec<RnsRepresentation>, scale: f64, level: usize) -> Self {
        let n = coeffs.len();
        Self {
            coeffs,
            scale,
            n,
            level,
        }
    }

    /// Encode a vector of floats into plaintext using CKKS encoding
    ///
    /// # Arguments
    /// * `values` - Float values to encode (length ≤ n/2 for CKKS)
    /// * `scale` - Scaling factor for fixed-point representation
    /// * `params` - CKKS parameters
    ///
    /// # Returns
    /// Encoded plaintext ready for encryption
    pub fn encode(values: &[f64], scale: f64, params: &CliffordFHEParams) -> Self {
        let n = params.n;
        let level = params.max_level();

        assert!(
            values.len() <= n / 2,
            "Too many values: {} > n/2 = {}",
            values.len(),
            n / 2
        );

        // Scale values to fixed-point integers
        let scaled: Vec<i64> = values
            .iter()
            .map(|&v| (v * scale).round() as i64)
            .collect();

        // For simplicity, put scaled values in first coefficients
        // In production CKKS, would use FFT/IFFT for slot encoding
        let mut coeffs_vec = vec![0i64; n];
        for (i, &val) in scaled.iter().enumerate() {
            coeffs_vec[i] = val;
        }

        // Convert to RNS representation for each prime
        let moduli: Vec<u64> = params.moduli[..=level].to_vec();
        let mut rns_coeffs = Vec::with_capacity(n);

        for &coeff in &coeffs_vec {
            // Handle negative coefficients by converting to positive mod q
            let values: Vec<u64> = moduli
                .iter()
                .map(|&q| {
                    if coeff >= 0 {
                        (coeff as u64) % q
                    } else {
                        let abs_coeff = (-coeff) as u64;
                        let remainder = abs_coeff % q;
                        if remainder == 0 {
                            0
                        } else {
                            q - remainder
                        }
                    }
                })
                .collect();

            rns_coeffs.push(RnsRepresentation::new(values, moduli.clone()));
        }

        Self::new(rns_coeffs, scale, level)
    }

    /// Decode plaintext to vector of floats
    ///
    /// # Arguments
    /// * `params` - CKKS parameters
    ///
    /// # Returns
    /// Decoded float values
    pub fn decode(&self, params: &CliffordFHEParams) -> Vec<f64> {
        // For simplicity, use the first (largest) prime for decoding
        // This avoids CRT overflow issues and is sufficient for testing
        // In production, would use proper multi-prime CRT or Garner's algorithm

        let q0 = params.moduli[0];
        let half_q0 = (q0 / 2) as i128;

        let mut coeffs_i128 = Vec::with_capacity(self.n);

        for rns_coeff in &self.coeffs {
            let val_mod_q0 = rns_coeff.values[0] as i128;

            // Convert to signed integer (centered lift)
            let signed_val = if val_mod_q0 > half_q0 {
                val_mod_q0 - (q0 as i128)
            } else {
                val_mod_q0
            };

            coeffs_i128.push(signed_val);
        }

        // Scale down to floats
        coeffs_i128
            .iter()
            .map(|&c| (c as f64) / self.scale)
            .collect()
    }
}

/// V2 Ciphertext in RNS representation with NTT support
#[derive(Clone, Debug)]
pub struct Ciphertext {
    /// First component c0 (RNS polynomial)
    pub c0: Vec<RnsRepresentation>,

    /// Second component c1 (RNS polynomial)
    pub c1: Vec<RnsRepresentation>,

    /// Current level (determines active primes)
    pub level: usize,

    /// Scaling factor
    pub scale: f64,

    /// Ring dimension
    pub n: usize,
}

impl Ciphertext {
    /// Create new ciphertext
    pub fn new(
        c0: Vec<RnsRepresentation>,
        c1: Vec<RnsRepresentation>,
        level: usize,
        scale: f64,
    ) -> Self {
        let n = c0.len();
        assert_eq!(c1.len(), n, "c0 and c1 must have same length");

        Self {
            c0,
            c1,
            level,
            scale,
            n,
        }
    }

    /// Add two ciphertexts (homomorphic addition)
    ///
    /// **Complexity:** O(n) per prime (no NTT needed for addition)
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n, "Ciphertexts must have same dimension");

        // Align levels if needed
        let (self_aligned, other_aligned) = if self.level != other.level {
            let min_level = self.level.min(other.level);
            let self_new = if self.level > min_level {
                self.mod_switch_to_level(min_level)
            } else {
                self.clone()
            };
            let other_new = if other.level > min_level {
                other.mod_switch_to_level(min_level)
            } else {
                other.clone()
            };
            (self_new, other_new)
        } else {
            (self.clone(), other.clone())
        };

        assert!(
            (self_aligned.scale - other_aligned.scale).abs() < 1.0,
            "Ciphertexts must have similar scale"
        );

        let c0: Vec<RnsRepresentation> = self_aligned
            .c0
            .iter()
            .zip(&other_aligned.c0)
            .map(|(a, b)| a.add(b))
            .collect();

        let c1: Vec<RnsRepresentation> = self_aligned
            .c1
            .iter()
            .zip(&other_aligned.c1)
            .map(|(a, b)| a.add(b))
            .collect();

        Self::new(c0, c1, self_aligned.level, self_aligned.scale)
    }

    /// Subtract two ciphertexts (homomorphic subtraction)
    ///
    /// **Complexity:** O(n) per prime
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n, "Ciphertexts must have same dimension");

        // Align levels if needed
        let (self_aligned, other_aligned) = if self.level != other.level {
            let min_level = self.level.min(other.level);
            let self_new = if self.level > min_level {
                self.mod_switch_to_level(min_level)
            } else {
                self.clone()
            };
            let other_new = if other.level > min_level {
                other.mod_switch_to_level(min_level)
            } else {
                other.clone()
            };
            (self_new, other_new)
        } else {
            (self.clone(), other.clone())
        };

        let c0: Vec<RnsRepresentation> = self_aligned
            .c0
            .iter()
            .zip(&other_aligned.c0)
            .map(|(a, b)| a.sub(b))
            .collect();

        let c1: Vec<RnsRepresentation> = self_aligned
            .c1
            .iter()
            .zip(&other_aligned.c1)
            .map(|(a, b)| a.sub(b))
            .collect();

        Self::new(c0, c1, self_aligned.level, self_aligned.scale)
    }

    /// Multiply ciphertext by scalar (homomorphic scalar multiplication)
    ///
    /// **Complexity:** O(n) per prime
    pub fn mul_scalar(&self, scalar: f64) -> Self {
        // For plaintext-ciphertext multiplication in CKKS:
        // - Ciphertext encrypts ⌊m * Δ⌉ with scale Δ
        // - We want result to encrypt ⌊(m * s) * Δ⌉ with SAME scale Δ
        //
        // Standard CKKS approach:
        // - Encode scalar s as plaintext polynomial (just the value, no scaling)
        // - Multiply: ct * s keeps scale Δ
        //
        // For special cases:
        // - Integer scalars: multiply RNS values directly
        // - 0.5: use modular inverse of 2

        let c0: Vec<RnsRepresentation>;
        let c1: Vec<RnsRepresentation>;

        if (scalar - 0.5).abs() < 1e-10 {
            // Special case: division by 2
            // Compute 2^(-1) mod q for each prime, then multiply
            c0 = self
                .c0
                .iter()
                .map(|rns| {
                    let values: Vec<u64> = rns
                        .values
                        .iter()
                        .zip(&rns.moduli)
                        .map(|(&val, &q)| {
                            // Compute 2^(-1) mod q = (q + 1) / 2
                            let inv2 = (q + 1) / 2;
                            ((val as u128 * inv2 as u128) % q as u128) as u64
                        })
                        .collect();
                    RnsRepresentation::new(values, rns.moduli.clone())
                })
                .collect();

            c1 = self
                .c1
                .iter()
                .map(|rns| {
                    let values: Vec<u64> = rns
                        .values
                        .iter()
                        .zip(&rns.moduli)
                        .map(|(&val, &q)| {
                            let inv2 = (q + 1) / 2;
                            ((val as u128 * inv2 as u128) % q as u128) as u64
                        })
                        .collect();
                    RnsRepresentation::new(values, rns.moduli.clone())
                })
                .collect();
        } else if scalar.abs() < 1e-10 {
            // Multiplication by 0 - return zero ciphertext
            let zero_rns = RnsRepresentation::from_u64(0, &self.c0[0].moduli);
            c0 = vec![zero_rns.clone(); self.n];
            c1 = vec![zero_rns; self.n];
        } else if (scalar - 1.0).abs() < 1e-10 {
            // Multiplication by 1 - return copy
            c0 = self.c0.clone();
            c1 = self.c1.clone();
        } else if (scalar - scalar.round()).abs() < 1e-10 {
            // Integer scalar: multiply directly
            let scalar_int = scalar.round() as u64;
            c0 = self.c0.iter().map(|rns| rns.mul_scalar(scalar_int)).collect();
            c1 = self.c1.iter().map(|rns| rns.mul_scalar(scalar_int)).collect();
        } else {
            // Fractional scalar (not 0.5): NOT IMPLEMENTED YET
            // This requires computing modular inverse which is complex for arbitrary fractions
            panic!("Multiplication by fractional scalar {} not implemented (only 0.5 supported)", scalar);
        }

        // Scale stays the same!
        Self::new(c0, c1, self.level, self.scale)
    }

    /// Add plaintext to ciphertext (homomorphic plaintext addition)
    pub fn add_plaintext(&self, pt: &Plaintext) -> Self {
        assert_eq!(self.n, pt.n, "Dimensions must match");
        assert_eq!(self.level, pt.level, "Levels must match");

        let c0: Vec<RnsRepresentation> = self
            .c0
            .iter()
            .zip(&pt.coeffs)
            .map(|(ct, pt)| ct.add(pt))
            .collect();

        Self::new(c0.clone(), self.c1.clone(), self.level, self.scale)
    }

    /// Modulus switch to a target level (drop primes without rescaling)
    ///
    /// This operation brings a ciphertext from a higher level (more primes)
    /// to a lower level (fewer primes) without changing the scale.
    /// Used to align levels before multiplication.
    ///
    /// # Arguments
    /// * `target_level` - The level to switch to (must be ≤ current level)
    ///
    /// # Returns
    /// Ciphertext at the target level with same scale
    pub fn mod_switch_to_level(&self, target_level: usize) -> Self {
        if target_level == self.level {
            return self.clone();
        }

        assert!(
            target_level < self.level,
            "Target level {} must be less than current level {}",
            target_level,
            self.level
        );

        // Drop RNS components to match target level
        // target_level is index of last prime to keep, so we want [0..=target_level]
        let new_c0: Vec<RnsRepresentation> = self
            .c0
            .iter()
            .map(|rns| {
                let new_values = rns.values[..=target_level].to_vec();
                let new_moduli = rns.moduli[..=target_level].to_vec();
                RnsRepresentation::new(new_values, new_moduli)
            })
            .collect();

        let new_c1: Vec<RnsRepresentation> = self
            .c1
            .iter()
            .map(|rns| {
                let new_values = rns.values[..=target_level].to_vec();
                let new_moduli = rns.moduli[..=target_level].to_vec();
                RnsRepresentation::new(new_values, new_moduli)
            })
            .collect();

        Self::new(new_c0, new_c1, target_level, self.scale)
    }
}

/// CKKS Context with precomputed NTT transforms
pub struct CkksContext {
    /// Parameters
    pub params: CliffordFHEParams,

    /// NTT contexts for each prime
    pub ntt_contexts: Vec<NttContext>,

    /// Barrett reducers for each prime
    pub reducers: Vec<BarrettReducer>,

    /// RNS context for CRT
    pub rns_context: RnsContext,
}

impl CkksContext {
    /// Create new CKKS context with precomputed NTT data
    pub fn new(params: CliffordFHEParams) -> Self {
        let moduli = params.moduli.clone();

        // Create NTT context for each prime
        let ntt_contexts: Vec<NttContext> = moduli
            .iter()
            .map(|&q| NttContext::new(params.n, q))
            .collect();

        // Create Barrett reducers
        let reducers: Vec<BarrettReducer> = moduli
            .iter()
            .map(|&q| BarrettReducer::new(q))
            .collect();

        // Create RNS context
        let rns_context = RnsContext::new(moduli);

        Self {
            params,
            ntt_contexts,
            reducers,
            rns_context,
        }
    }

    /// Encode float values to plaintext
    pub fn encode(&self, values: &[f64]) -> Plaintext {
        Plaintext::encode(values, self.params.scale, &self.params)
    }

    /// Decode plaintext to float values
    pub fn decode(&self, pt: &Plaintext) -> Vec<f64> {
        pt.decode(&self.params)
    }

    /// Encrypt plaintext to ciphertext using public key
    ///
    /// **Algorithm:**
    /// 1. Sample error polynomials e0, e1 ~ N(0, σ²)
    /// 2. Sample random u ~ {-1, 0, 1}
    /// 3. c0 = b*u + e0 + m (where b is from public key)
    /// 4. c1 = a*u + e1 (where a is from public key)
    ///
    /// **Security:** Relies on RLWE hardness
    ///
    /// # Arguments
    /// * `pt` - Plaintext to encrypt
    /// * `pk` - Public key (a, b) where b = -a*s + e
    ///
    /// # Returns
    /// Ciphertext (c0, c1) encrypting the plaintext
    pub fn encrypt(&self, pt: &Plaintext, pk: &crate::clifford_fhe_v2::backends::cpu_optimized::keys::PublicKey) -> Ciphertext {
        use rand::{thread_rng, Rng};
        use rand_distr::{Distribution, Normal};

        let n = self.params.n;
        let level = self.params.max_level();
        let moduli: Vec<u64> = self.params.moduli[..=level].to_vec();
        let mut rng = thread_rng();

        // Sample ternary random polynomial u ∈ {-1, 0, 1}^n
        let u_coeffs: Vec<i64> = (0..n)
            .map(|_| {
                let val: f64 = rng.gen();
                if val < 0.33 {
                    -1
                } else if val < 0.66 {
                    0
                } else {
                    1
                }
            })
            .collect();

        // Sample error polynomials e0, e1 from Gaussian distribution
        let normal = Normal::new(0.0, self.params.error_std).unwrap();
        let e0_coeffs: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();
        let e1_coeffs: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();

        // Convert to RNS representation
        let u = self.coeffs_to_rns(&u_coeffs, &moduli);
        let e0 = self.coeffs_to_rns(&e0_coeffs, &moduli);
        let e1 = self.coeffs_to_rns(&e1_coeffs, &moduli);

        // c0 = b*u + e0 + m
        let bu = self.multiply_polys_ntt(&pk.b, &u.coeffs, &moduli);
        let c0: Vec<RnsRepresentation> = bu
            .iter()
            .zip(&e0.coeffs)
            .zip(&pt.coeffs)
            .map(|((bu_i, e0_i), m_i)| bu_i.add(e0_i).add(m_i))
            .collect();

        // c1 = a*u + e1
        let au = self.multiply_polys_ntt(&pk.a, &u.coeffs, &moduli);
        let c1: Vec<RnsRepresentation> = au
            .iter()
            .zip(&e1.coeffs)
            .map(|(au_i, e1_i)| au_i.add(e1_i))
            .collect();

        Ciphertext::new(c0, c1, level, pt.scale)
    }

    /// Decrypt ciphertext to plaintext using secret key
    ///
    /// **Algorithm:**
    /// m = c0 + c1*s (mod q)
    ///
    /// **Decryption correctness:**
    /// m ≈ (b*u + e0 + m) + (a*u + e1)*s
    ///   = b*u + a*u*s + e_total + m
    ///   = (-a*s + e_pk)*u + a*u*s + e_total + m
    ///   = e_pk*u + e_total + m
    ///   ≈ m (if noise is small)
    ///
    /// # Arguments
    /// * `ct` - Ciphertext to decrypt
    /// * `sk` - Secret key s
    ///
    /// # Returns
    /// Plaintext approximating the encrypted message
    pub fn decrypt(&self, ct: &Ciphertext, sk: &crate::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey) -> Plaintext {
        let moduli: Vec<u64> = self.params.moduli[..=ct.level].to_vec();

        // Extract secret key coefficients at the ciphertext's level
        // Secret key was generated at max_level, but ciphertext may be at lower level
        let sk_at_level: Vec<RnsRepresentation> = sk
            .coeffs
            .iter()
            .map(|rns| {
                // Take only the RNS components for active primes at this level
                let values = rns.values[..=ct.level].to_vec();
                let moduli_at_level = rns.moduli[..=ct.level].to_vec();
                RnsRepresentation::new(values, moduli_at_level)
            })
            .collect();

        // Compute c1 * s
        let c1s = self.multiply_polys_ntt(&ct.c1, &sk_at_level, &moduli);

        // m = c0 + c1*s
        let m: Vec<RnsRepresentation> = ct
            .c0
            .iter()
            .zip(&c1s)
            .map(|(c0_i, c1s_i)| c0_i.add(c1s_i))
            .collect();

        Plaintext::new(m, ct.scale, ct.level)
    }

    /// Convert signed integer coefficients to RNS representation
    fn coeffs_to_rns(&self, coeffs: &[i64], moduli: &[u64]) -> Plaintext {
        let n = coeffs.len();
        let mut rns_coeffs = Vec::with_capacity(n);

        for &coeff in coeffs {
            let values: Vec<u64> = moduli
                .iter()
                .map(|&q| {
                    if coeff >= 0 {
                        (coeff as u64) % q
                    } else {
                        let abs_coeff = (-coeff) as u64;
                        let remainder = abs_coeff % q;
                        if remainder == 0 {
                            0
                        } else {
                            q - remainder
                        }
                    }
                })
                .collect();

            rns_coeffs.push(RnsRepresentation::new(values, moduli.to_vec()));
        }

        Plaintext::new(rns_coeffs, 1.0, moduli.len() - 1)
    }

    /// Multiply two polynomials using NTT (negacyclic convolution mod x^n + 1)
    ///
    /// Uses V2's fixed NTT implementation (twisted NTT for negacyclic convolution).
    fn multiply_polys_ntt(
        &self,
        a: &[RnsRepresentation],
        b: &[RnsRepresentation],
        moduli: &[u64],
    ) -> Vec<RnsRepresentation> {
        let n = a.len();
        let mut result = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.to_vec()); n];

        // For each prime modulus, multiply using V2's NTT
        for (prime_idx, &q) in moduli.iter().enumerate() {
            // Create NTT context for this prime
            let ntt_ctx = super::ntt::NttContext::new(n, q);

            // Extract coefficients for this prime (convert i64 to u64 mod q)
            let a_mod_q: Vec<u64> = a.iter().map(|rns| {
                let val = rns.values[prime_idx] as i64;
                let normalized = ((val % q as i64) + q as i64) % q as i64;
                normalized as u64
            }).collect();

            let b_mod_q: Vec<u64> = b.iter().map(|rns| {
                let val = rns.values[prime_idx] as i64;
                let normalized = ((val % q as i64) + q as i64) % q as i64;
                normalized as u64
            }).collect();

            // Multiply using V2's NTT (negacyclic convolution)
            let product_mod_q = ntt_ctx.multiply_polynomials(&a_mod_q, &b_mod_q);

            // Store result (convert u64 back to i64)
            for i in 0..n {
                result[i].values[prime_idx] = product_mod_q[i];
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plaintext_encode_decode() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = CkksContext::new(params.clone());

        let values = vec![1.5, 2.7, -3.2, 0.5];
        let pt = ctx.encode(&values);

        assert_eq!(pt.n, params.n);
        assert_eq!(pt.scale, params.scale);

        let decoded = ctx.decode(&pt);

        // Check first few values (rest should be ~0)
        for i in 0..values.len() {
            let error = (decoded[i] - values[i]).abs();
            assert!(
                error < 1e-6,
                "Value {} decode error: expected {}, got {} (error: {})",
                i,
                values[i],
                decoded[i],
                error
            );
        }
    }

    #[test]
    fn test_plaintext_encode_negative() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = CkksContext::new(params);

        let values = vec![-10.5, -20.3, -5.0];
        let pt = ctx.encode(&values);
        let decoded = ctx.decode(&pt);

        for i in 0..values.len() {
            let error = (decoded[i] - values[i]).abs();
            assert!(error < 1e-6, "Negative value decode error at index {}", i);
        }
    }

    #[test]
    fn test_ciphertext_addition() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let n = params.n;
        let level = params.max_level();
        let moduli = params.moduli[..=level].to_vec();

        // Create dummy ciphertexts with simple RNS values
        let c0_a: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64(i as u64, &moduli))
            .collect();
        let c1_a: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64((i + 1) as u64, &moduli))
            .collect();

        let c0_b: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64((2 * i) as u64, &moduli))
            .collect();
        let c1_b: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64((2 * i + 1) as u64, &moduli))
            .collect();

        let ct_a = Ciphertext::new(c0_a, c1_a, level, params.scale);
        let ct_b = Ciphertext::new(c0_b, c1_b, level, params.scale);

        let ct_sum = ct_a.add(&ct_b);

        // Check dimensions
        assert_eq!(ct_sum.n, n);
        assert_eq!(ct_sum.level, level);

        // Check first coefficient of c0: should be 0 + 0 = 0
        assert_eq!(ct_sum.c0[0].values[0], 0);

        // Check second coefficient of c0: should be 1 + 2 = 3
        assert_eq!(ct_sum.c0[1].values[0], 3);
    }

    #[test]
    fn test_ciphertext_subtraction() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let n = params.n;
        let level = params.max_level();
        let moduli = params.moduli[..=level].to_vec();

        let c0_a: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64((i + 10) as u64, &moduli))
            .collect();
        let c1_a: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64(i as u64, &moduli))
            .collect();

        let c0_b: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64(i as u64, &moduli))
            .collect();
        let c1_b: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64(i as u64, &moduli))
            .collect();

        let ct_a = Ciphertext::new(c0_a, c1_a, level, params.scale);
        let ct_b = Ciphertext::new(c0_b, c1_b, level, params.scale);

        let ct_diff = ct_a.sub(&ct_b);

        // First coefficient: (0+10) - 0 = 10
        assert_eq!(ct_diff.c0[0].values[0], 10);

        // Second coefficient: (1+10) - 1 = 10
        assert_eq!(ct_diff.c0[1].values[0], 10);
    }

    #[test]
    fn test_ciphertext_scalar_multiplication() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let n = params.n;
        let level = params.max_level();
        let moduli = params.moduli[..=level].to_vec();

        let c0: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64((i + 1) as u64, &moduli))
            .collect();
        let c1: Vec<RnsRepresentation> = (0..n)
            .map(|_| RnsRepresentation::from_u64(0, &moduli))
            .collect();

        let ct = Ciphertext::new(c0, c1, level, params.scale);
        let scalar = 3.0;
        let ct_scaled = ct.mul_scalar(scalar);

        // Scale should STAY THE SAME (plaintext-ciphertext multiplication)
        assert!((ct_scaled.scale - params.scale).abs() < 0.1);

        // Level should stay the same
        assert_eq!(ct_scaled.level, level);
    }

    #[test]
    fn test_ckks_context_creation() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = CkksContext::new(params.clone());

        assert_eq!(ctx.ntt_contexts.len(), params.moduli.len());
        assert_eq!(ctx.reducers.len(), params.moduli.len());
        assert_eq!(ctx.rns_context.moduli.len(), params.moduli.len());
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = CkksContext::new(params.clone());
        let key_ctx = KeyContext::new(params.clone());

        // Generate keys
        let (pk, sk, _evk) = key_ctx.keygen();

        // Create simple plaintext with known coefficients
        let level = params.max_level();
        let moduli: Vec<u64> = params.moduli[..=level].to_vec();
        let scale = params.scale;

        // Create plaintext with small integer values scaled up
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mut coeffs = vec![RnsRepresentation::from_u64(0, &moduli); params.n];
        for i in 0..values.len().min(params.n) {
            let scaled_val = (values[i] * scale) as u64;
            coeffs[i] = RnsRepresentation::from_u64(scaled_val, &moduli);
        }
        let pt = Plaintext::new(coeffs, scale, level);

        // Encrypt
        let ct = ctx.encrypt(&pt, &pk);

        // Decrypt
        let pt_decrypted = ctx.decrypt(&ct, &sk);

        // Check first few coefficients are approximately correct
        // Encryption adds noise, so we allow some tolerance
        for i in 0..values.len() {
            let original = pt.coeffs[i].values[0];
            let decrypted = pt_decrypted.coeffs[i].values[0];
            let q = moduli[0];

            // Convert to signed representation (centered lift)
            let to_signed = |val: u64| -> i128 {
                if val > q / 2 {
                    val as i128 - q as i128
                } else {
                    val as i128
                }
            };

            let original_signed = to_signed(original);
            let decrypted_signed = to_signed(decrypted);
            let error = (decrypted_signed - original_signed).abs();

            // Allow noise up to 5% of the scaled value (RLWE adds noise)
            let tolerance = (scale * 0.05) as i128;
            assert!(error < tolerance,
                "Coefficient {} error too large: {} vs {} (error: {}), q={}",
                i, decrypted_signed, original_signed, error, q);
        }
    }
}
