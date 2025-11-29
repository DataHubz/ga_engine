//! Metal GPU Relinearization Keys for CKKS
//!
//! **Purpose:** Enable degree reduction after ciphertext multiplication on Metal GPU.
//!
//! **Background:**
//! After multiplying two degree-1 ciphertexts (c₀, c₁) × (d₀, d₁), we get a degree-2 ciphertext:
//! - ct_mult = (c₀×d₀, c₀×d₁ + c₁×d₀, c₁×d₁)
//! - Decrypts as: c₀×d₀ + (c₀×d₁ + c₁×d₀)·s + c₁×d₁·s²
//!
//! We need to convert this back to degree-1: (c'₀, c'₁) where:
//! - c'₀ + c'₁·s ≈ c₀×d₀ + (c₀×d₁ + c₁×d₀)·s + c₁×d₁·s²
//!
//! **Solution: Relinearization**
//! Relinearization keys encode s² in an encrypted form, allowing us to transform c₂·s² → c'₁·s
//! using key switching with gadget decomposition to minimize noise growth.
//!
//! **Relinearization Key Structure:**
//! We store (rlk0[], rlk1[]) where:
//! - rlk0[i] = -a_i·s + e_i + B^i·s²  (for i = 0..num_digits)
//! - rlk1[i] = a_i
//! - B = 2^base_w (gadget base, typically w=16-20)
//!
//! **Memory Layout:**
//! For N=1024, 41 primes, base_w=16 (→3 digits):
//! - Total size: 2 × num_digits × N × num_primes × 8 bytes ≈ 2 MB
//! - Acceptable on Apple Silicon unified memory
//!
//! **Usage:**
//! ```rust,ignore
//! // Generate relinearization keys
//! let relin_keys = MetalRelinKeys::generate(
//!     device,
//!     &sk,
//!     &params,
//!     &ntt_contexts,
//!     16,  // base_w
//! )?;
//!
//! // Use for ciphertext multiplication
//! let ct_mult = ct1.multiply(&ct2, &relin_keys, &ctx)?;
//! ```
//!
//! **Reference:** Gentry, Halevi, Smart 2012 (BGV relinearization), Cheon et al. 2017 (CKKS adaptation)

use super::device::MetalDevice;
use super::ntt::MetalNttContext;
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use rand::Rng;
use std::sync::Arc;

/// Metal GPU Relinearization Keys (with Gadget Decomposition)
///
/// Stores key-switching keys for degree reduction using gadget decomposition.
/// This ensures minimal noise growth during relinearization (additive instead of multiplicative).
///
/// # Key Structure
///
/// We store (rlk0[], rlk1[]) where:
/// - rlk0[t] = -a_t·s + e_t + B^t·s²  (for t = 0..num_digits)
/// - rlk1[t] = a_t
/// - B = 2^base_w (gadget base)
///
/// # Memory Layout
///
/// Each component is Vec<u64> in flat RNS layout: [coeff0_mod_q0, coeff0_mod_q1, ...]
pub struct MetalRelinKeys {
    /// Metal device (shared)
    device: Arc<MetalDevice>,

    /// Relinearization keys in coefficient domain
    /// (rlk0[], rlk1[]) where each array has num_digits elements
    /// Each element is Vec<u64> in flat RNS layout
    rlk0_coeff: Vec<Vec<u64>>,
    rlk1_coeff: Vec<Vec<u64>>,

    /// Pre-computed NTT-transformed relinearization keys (per level)
    /// Vec indexed by level, containing (rlk0_ntt[], rlk1_ntt[])
    /// This eliminates runtime NTT transforms during multiplication
    rlk0_ntt: Vec<Vec<Vec<u64>>>,
    rlk1_ntt: Vec<Vec<Vec<u64>>>,

    /// Gadget base exponent (e.g., 16 → B = 2^16)
    base_w: u32,

    /// Number of decomposition digits
    num_digits: usize,

    /// Ring dimension N
    n: usize,

    /// Number of RNS primes
    num_primes: usize,

    /// Maximum level these keys support
    max_level: usize,
}

impl MetalRelinKeys {
    /// Generate relinearization keys for a secret key
    ///
    /// # Arguments
    /// * `device` - Metal device context
    /// * `sk` - Secret key
    /// * `params` - FHE parameters
    /// * `ntt_contexts` - NTT contexts for each RNS prime
    /// * `base_w` - Gadget base exponent (e.g., 16 for B = 2^16)
    pub fn generate(
        device: Arc<MetalDevice>,
        sk: &SecretKey,
        params: &CliffordFHEParams,
        ntt_contexts: &[MetalNttContext],
        base_w: u32,
    ) -> Result<Self, String> {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║        Generating Metal GPU Relinearization Keys             ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let n = params.n;
        let num_primes = params.moduli.len();
        let max_level = num_primes - 1;
        let moduli = &params.moduli;

        // Calculate number of gadget digits
        // dnum = ceil(log_B(Q)) where Q = product of all primes
        let total_bits: u32 = moduli.iter()
            .map(|&q| 64 - q.leading_zeros())
            .sum();
        let num_digits = ((total_bits + base_w - 1) / base_w) as usize;

        println!("Relinearization key parameters:");
        println!("  Ring dimension (N): {}", n);
        println!("  Number of RNS primes: {}", num_primes);
        println!("  Gadget base: B = 2^{}", base_w);
        println!("  Number of digits: {}", num_digits);
        println!("  Max level: {}", max_level);

        // Compute s² in RNS representation
        let s_squared = Self::multiply_secret_keys(sk, sk, moduli, n)?;

        println!("\nGenerating {} gadget components...", num_digits);

        let mut rlk0_coeff = Vec::with_capacity(num_digits);
        let mut rlk1_coeff = Vec::with_capacity(num_digits);

        let base_big = 1u128 << base_w;

        for digit_idx in 0..num_digits {
            // Compute B^digit_idx mod each q_i
            let power = base_big.pow(digit_idx as u32);

            // Sample random polynomial a_i
            let a_i = Self::sample_uniform_poly(n, moduli);

            // Compute b_i = -a_i·s + e_i + B^digit_idx·s²
            let b_i = Self::compute_rlk_component(
                &a_i,
                sk,
                &s_squared,
                power,
                moduli,
                n,
                params.error_std,
            )?;

            rlk0_coeff.push(b_i);
            rlk1_coeff.push(a_i);

            if (digit_idx + 1) % 5 == 0 || digit_idx + 1 == num_digits {
                println!("  Generated {}/{} components", digit_idx + 1, num_digits);
            }
        }

        println!("\nPrecomputing NTT-transformed keys for all levels...");

        // Precompute NTT transforms for each level
        let mut rlk0_ntt = Vec::with_capacity(max_level + 1);
        let mut rlk1_ntt = Vec::with_capacity(max_level + 1);

        for level in 0..=max_level {
            let num_primes_level = level + 1;
            let mut rlk0_ntt_level = Vec::with_capacity(num_digits);
            let mut rlk1_ntt_level = Vec::with_capacity(num_digits);

            for digit_idx in 0..num_digits {
                // Extract relevant primes for this level
                let rlk0_digit_ntt = Self::forward_ntt_flat(
                    &rlk0_coeff[digit_idx],
                    n,
                    num_primes_level,
                    ntt_contexts,
                )?;
                let rlk1_digit_ntt = Self::forward_ntt_flat(
                    &rlk1_coeff[digit_idx],
                    n,
                    num_primes_level,
                    ntt_contexts,
                )?;

                rlk0_ntt_level.push(rlk0_digit_ntt);
                rlk1_ntt_level.push(rlk1_digit_ntt);
            }

            rlk0_ntt.push(rlk0_ntt_level);
            rlk1_ntt.push(rlk1_ntt_level);

            if (level + 1) % 10 == 0 || level == max_level {
                println!("  Transformed {}/{} levels", level + 1, max_level + 1);
            }
        }

        let memory_mb = (2 * num_digits * n * num_primes * 8) as f64 / 1_048_576.0;
        println!("\n✅ Relinearization keys generated successfully!");
        println!("   Memory footprint: {:.2} MB", memory_mb);

        Ok(Self {
            device,
            rlk0_coeff,
            rlk1_coeff,
            rlk0_ntt,
            rlk1_ntt,
            base_w,
            num_digits,
            n,
            num_primes,
            max_level,
        })
    }

    /// Get NTT-transformed relinearization keys for a specific level
    pub fn get_ntt_keys(&self, level: usize) -> Result<(&[Vec<u64>], &[Vec<u64>]), String> {
        if level > self.max_level {
            return Err(format!("Level {} exceeds max level {}", level, self.max_level));
        }
        Ok((&self.rlk0_ntt[level], &self.rlk1_ntt[level]))
    }

    /// Get gadget decomposition parameters
    pub fn gadget_params(&self) -> (u32, usize) {
        (self.base_w, self.num_digits)
    }

    /// Multiply two secret keys in RNS representation: s1 · s2
    fn multiply_secret_keys(
        sk1: &SecretKey,
        sk2: &SecretKey,
        moduli: &[u64],
        n: usize,
    ) -> Result<Vec<RnsRepresentation>, String> {
        let mut result = Vec::with_capacity(n);

        for coeff_idx in 0..n {
            let mut product_rns = Vec::with_capacity(moduli.len());

            for (prime_idx, &q) in moduli.iter().enumerate() {
                let s1_val = sk1.coeffs[coeff_idx].values[prime_idx];
                let s2_val = sk2.coeffs[coeff_idx].values[prime_idx];

                // Multiply mod q
                let product = ((s1_val as u128 * s2_val as u128) % q as u128) as u64;
                product_rns.push(product);
            }

            result.push(RnsRepresentation::new(product_rns, moduli.to_vec()));
        }

        Ok(result)
    }

    /// Sample uniform random polynomial in flat RNS layout
    fn sample_uniform_poly(n: usize, moduli: &[u64]) -> Vec<u64> {
        let num_primes = moduli.len();
        let mut rng = rand::thread_rng();
        let mut poly = vec![0u64; n * num_primes];

        for coeff_idx in 0..n {
            for (prime_idx, &q) in moduli.iter().enumerate() {
                poly[coeff_idx * num_primes + prime_idx] = rng.gen::<u64>() % q;
            }
        }

        poly
    }

    /// Compute rlk component: b = -a·s + e + power·s²
    fn compute_rlk_component(
        a: &[u64],
        sk: &SecretKey,
        s_squared: &[RnsRepresentation],
        power: u128,
        moduli: &[u64],
        n: usize,
        error_std: f64,
    ) -> Result<Vec<u64>, String> {
        use rand_distr::{Distribution, Normal};

        let num_primes = moduli.len();
        let mut b = vec![0u64; n * num_primes];
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, error_std).unwrap();

        for coeff_idx in 0..n {
            for (prime_idx, &q) in moduli.iter().enumerate() {
                // -a·s
                let a_val = a[coeff_idx * num_primes + prime_idx];
                let s_val = sk.coeffs[coeff_idx].values[prime_idx];
                let neg_as = q - ((a_val as u128 * s_val as u128) % q as u128) as u64;

                // e (error)
                let e_float: f64 = normal.sample(&mut rng);
                let e = if e_float >= 0.0 {
                    (e_float.round() as u64) % q
                } else {
                    let abs_e = ((-e_float).round() as u64) % q;
                    if abs_e == 0 { 0 } else { q - abs_e }
                };

                // power·s² mod q
                let s2_val = s_squared[coeff_idx].values[prime_idx];
                let power_mod_q = (power % q as u128) as u64;
                let power_s2 = ((power_mod_q as u128 * s2_val as u128) % q as u128) as u64;

                // b = -a·s + e + power·s²
                let sum = (neg_as as u128 + e as u128 + power_s2 as u128) % q as u128;
                b[coeff_idx * num_primes + prime_idx] = sum as u64;
            }
        }

        Ok(b)
    }

    /// Forward NTT transform on flat RNS polynomial (CPU-side, for precomputation)
    fn forward_ntt_flat(
        poly_flat: &[u64],
        n: usize,
        num_primes_level: usize,
        ntt_contexts: &[MetalNttContext],
    ) -> Result<Vec<u64>, String> {
        let mut result = vec![0u64; n * num_primes_level];

        for prime_idx in 0..num_primes_level {
            let ntt_ctx = &ntt_contexts[prime_idx];

            // Extract coefficients for this prime
            let mut poly_prime = vec![0u64; n];
            for coeff_idx in 0..n {
                poly_prime[coeff_idx] = poly_flat[coeff_idx * poly_flat.len() / n + prime_idx];
            }

            // Forward NTT (CPU)
            ntt_ctx.forward(&mut poly_prime)?;

            // Store back to flat layout
            for coeff_idx in 0..n {
                result[coeff_idx * num_primes_level + prime_idx] = poly_prime[coeff_idx];
            }
        }

        Ok(result)
    }

    /// Get device reference
    pub fn device(&self) -> &Arc<MetalDevice> {
        &self.device
    }

    /// Get number of digits
    pub fn num_digits(&self) -> usize {
        self.num_digits
    }

    /// Get gadget base
    pub fn base_w(&self) -> u32 {
        self.base_w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_metal_relin_keys_generation() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let device = Arc::new(MetalDevice::new().expect("Failed to create Metal device"));

        // Generate secret key
        let key_ctx = KeyContext::new(params.clone());
        let (_, sk, _) = key_ctx.keygen();

        // Create NTT contexts
        let ntt_contexts: Vec<MetalNttContext> = params.moduli.iter()
            .map(|&q| MetalNttContext::new(params.n, q, device.clone()).unwrap())
            .collect();

        // Generate relinearization keys
        let result = MetalRelinKeys::generate(
            device.clone(),
            &sk,
            &params,
            &ntt_contexts,
            16,
        );

        assert!(result.is_ok(), "Failed to generate relinearization keys: {:?}", result.err());

        let relin_keys = result.unwrap();
        assert_eq!(relin_keys.n, params.n);
        assert_eq!(relin_keys.num_primes, params.moduli.len());
        assert!(relin_keys.num_digits > 0);
    }
}
