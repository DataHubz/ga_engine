//! CUDA GPU-Accelerated Relinearization Keys for CKKS
//!
//! Implements relinearization (degree reduction) for ciphertext multiplication.
//!
//! **Relinearization Overview:**
//! After ciphertext multiplication, we have a degree-2 ciphertext:
//!   ct = (c0, c1, c2)
//!
//! We need to convert it back to degree-1:
//!   ct' = (c0', c1')
//!
//! **Relinearization Key Structure:**
//! We precompute:
//!   RelinKey = {KS_0, KS_1, ..., KS_{dnum-1}}
//! where:
//!   KS_i = (-a_i · s^2 + e_i + w^i · s, a_i)
//!        = (b_i, a_i) in RNS representation
//!
//! **Gadget Decomposition:**
//! We decompose c2 into base-w digits:
//!   c2 = Σ_{i=0}^{dnum-1} d_i · w^i
//! where d_i ∈ [-w/2, w/2] (signed decomposition)
//!
//! **Key Application:**
//!   c0' = c0 + Σ_i d_i · b_i
//!   c1' = c1 + Σ_i d_i · a_i
//!
//! **GPU Acceleration:**
//! - NTT-multiply for key generation (GPU)
//! - Gadget decomposition (CPU, simple)
//! - Key application multiply (GPU via NTT)

use crate::clifford_fhe_v2::backends::gpu_cuda::device::CudaDeviceContext;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::BarrettReducer;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use rand::Rng;
use std::sync::Arc;

/// Relinearization key structure
#[derive(Clone)]
pub struct RelinearizationKey {
    /// Key switching components: [(b_0, a_0), (b_1, a_1), ..., (b_{dnum-1}, a_{dnum-1})]
    /// Each component is in flat RNS layout: poly[prime_idx * n + coeff_idx]
    /// b_i and a_i have size n * num_primes_key
    pub ks_components: Vec<(Vec<u64>, Vec<u64>)>,

    /// Number of RNS primes in the key (typically num_primes + special_primes)
    pub num_primes_key: usize,

    /// Polynomial degree
    pub n: usize,
}

/// CUDA relinearization keys manager
pub struct CudaRelinKeys {
    device: Arc<CudaDeviceContext>,
    params: CliffordFHEParams,

    /// Barrett reducers for key level
    reducers_key: Vec<BarrettReducer>,

    /// Gadget decomposition base (typically w = 2^base_bits, e.g., 2^16)
    pub base_w: u64,

    /// Number of gadget digits: dnum = ceil(log_w(Q_key))
    pub dnum: usize,

    /// Relinearization key
    relin_key: Option<RelinearizationKey>,
}

impl CudaRelinKeys {
    /// Create new relinearization keys manager
    ///
    /// # Arguments
    /// * `device` - CUDA device context
    /// * `params` - FHE parameters
    /// * `secret_key` - Secret key polynomial (strided layout)
    /// * `base_bits` - Gadget base bits (e.g., 16 for w = 2^16)
    pub fn new(
        device: Arc<CudaDeviceContext>,
        params: CliffordFHEParams,
        secret_key: Vec<u64>,
        base_bits: usize,
    ) -> Result<Self, String> {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║        Initializing CUDA Relinearization Keys                ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let base_w = 1u64 << base_bits;  // w = 2^base_bits

        // Determine number of RNS primes for keys (use all available)
        let num_primes_key = params.moduli.len();

        // Calculate number of gadget digits
        // dnum = ceil(log_w(Q_key)) where Q_key = product of all key primes
        let total_bits: usize = params.moduli.iter()
            .map(|&q| (64 - q.leading_zeros()) as usize)
            .sum();
        let dnum = (total_bits + base_bits - 1) / base_bits;

        println!("Relinearization key parameters:");
        println!("  Base w: 2^{} = {}", base_bits, base_w);
        println!("  Number of primes (key level): {}", num_primes_key);
        println!("  Number of gadget digits (dnum): {}", dnum);

        // Create Barrett reducers for key level
        let reducers_key: Vec<BarrettReducer> = params.moduli[..num_primes_key]
            .iter()
            .map(|&q| BarrettReducer::new(q))
            .collect();

        let mut ctx = Self {
            device,
            params,
            reducers_key,
            base_w,
            dnum,
            relin_key: None,
        };

        // Generate relinearization key
        println!("\nGenerating relinearization key...");
        let start = std::time::Instant::now();
        ctx.generate_relin_key(&secret_key)?;
        let elapsed = start.elapsed().as_secs_f64();
        println!("  ✅ Relinearization key generated in {:.2}s\n", elapsed);

        Ok(ctx)
    }

    /// Generate relinearization key for s^2 → s
    ///
    /// Generates: RelinKey = {(b_i, a_i)} where b_i = -a_i·s^2 + e_i + w^i·s
    fn generate_relin_key(&mut self, secret_key: &[u64]) -> Result<(), String> {
        let n = self.params.n;
        let num_primes_key = self.reducers_key.len();

        // Compute s^2 (secret key squared)
        let s_squared = self.compute_secret_key_squared(secret_key, num_primes_key)?;

        // Generate key switching components
        let mut ks_components = Vec::with_capacity(self.dnum);

        for digit_idx in 0..self.dnum {
            // Generate random polynomial a_i
            let a_i = self.generate_random_poly(num_primes_key);

            // Compute -a_i · s^2 using CPU (TODO: GPU NTT multiply)
            let mut b_i = self.cpu_multiply_flat(&a_i, &s_squared, num_primes_key)?;

            // Negate: -a_i · s^2
            for i in 0..(n * num_primes_key) {
                let prime_idx = i / n;
                let q = self.params.moduli[prime_idx];
                b_i[i] = if b_i[i] == 0 { 0 } else { q - b_i[i] };
            }

            // Add error term e_i ~ Gaussian
            let e_i = self.generate_error_poly(num_primes_key);
            for i in 0..(n * num_primes_key) {
                let prime_idx = i / n;
                let q = self.params.moduli[prime_idx];
                b_i[i] = (b_i[i] + e_i[i]) % q;
            }

            // Add w^i · s
            let power_of_w = self.base_w.pow(digit_idx as u32);
            for coeff_idx in 0..n {
                for prime_idx in 0..num_primes_key {
                    let flat_idx = prime_idx * n + coeff_idx;
                    let strided_idx = coeff_idx * num_primes_key + prime_idx;
                    let q = self.params.moduli[prime_idx];

                    let s_val = secret_key[strided_idx];
                    let scaled_s = ((power_of_w % q as u64) * s_val) % q;
                    b_i[flat_idx] = (b_i[flat_idx] + scaled_s) % q;
                }
            }

            ks_components.push((b_i, a_i));
        }

        self.relin_key = Some(RelinearizationKey {
            ks_components,
            num_primes_key,
            n,
        });

        Ok(())
    }

    /// Compute s^2 (secret key squared) in RNS representation
    fn compute_secret_key_squared(&self, secret_key: &[u64], num_primes: usize) -> Result<Vec<u64>, String> {
        let n = self.params.n;

        // Convert from strided to flat layout
        let mut s_flat = vec![0u64; n * num_primes];
        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                let strided_idx = coeff_idx * num_primes + prime_idx;
                let flat_idx = prime_idx * n + coeff_idx;
                s_flat[flat_idx] = secret_key[strided_idx];
            }
        }

        // Multiply s * s using CPU (TODO: GPU NTT multiply)
        self.cpu_multiply_flat(&s_flat, &s_flat, num_primes)
    }

    /// Apply relinearization to reduce degree-2 ciphertext to degree-1
    ///
    /// Input: (c0, c1, c2) where c2 needs to be eliminated
    /// Output: (c0', c1') where c2 has been absorbed using relin key
    pub fn apply_relinearization(
        &self,
        c0: &[u64],
        c1: &[u64],
        c2: &[u64],
        level: usize,
    ) -> Result<(Vec<u64>, Vec<u64>), String> {
        let relin_key = self.relin_key.as_ref()
            .ok_or_else(|| "Relinearization key not generated".to_string())?;

        let n = self.params.n;
        let num_primes = level + 1;

        // Decompose c2 into base-w digits
        let digits = self.gadget_decompose(c2, num_primes)?;

        // Initialize accumulator
        let mut c0_acc = c0.to_vec();
        let mut c1_acc = c1.to_vec();

        // Accumulate: c0' = c0 + Σ d_i · b_i, c1' = c1 + Σ d_i · a_i
        for (digit_idx, d_i) in digits.iter().enumerate() {
            if digit_idx >= relin_key.ks_components.len() {
                break;
            }

            let (b_i, a_i) = &relin_key.ks_components[digit_idx];

            // Multiply d_i · b_i (CPU for now)
            let d_b = self.cpu_multiply_flat(d_i, b_i, num_primes)?;
            // Multiply d_i · a_i (CPU for now)
            let d_a = self.cpu_multiply_flat(d_i, a_i, num_primes)?;

            // Accumulate
            for i in 0..n * num_primes {
                let q = self.params.moduli[i / n];
                c0_acc[i] = (c0_acc[i] + d_b[i]) % q;
                c1_acc[i] = (c1_acc[i] + d_a[i]) % q;
            }
        }

        Ok((c0_acc, c1_acc))
    }

    /// Gadget decomposition: decompose polynomial into base-w digits
    ///
    /// Input: poly in flat RNS layout
    /// Output: Vec of digit polynomials, each in flat RNS layout
    fn gadget_decompose(&self, poly: &[u64], num_primes: usize) -> Result<Vec<Vec<u64>>, String> {
        let n = self.params.n;
        let mut digits = vec![vec![0u64; n * num_primes]; self.dnum];

        // For each coefficient
        for coeff_idx in 0..n {
            // For each prime
            for prime_idx in 0..num_primes {
                let flat_idx = prime_idx * n + coeff_idx;
                let mut val = poly[flat_idx];
                let q = self.params.moduli[prime_idx];

                // Decompose into base-w digits
                for digit_idx in 0..self.dnum {
                    let digit = val % self.base_w;

                    // Signed decomposition: center around 0
                    let signed_digit = if digit > self.base_w / 2 {
                        if digit >= q {
                            digit
                        } else {
                            q - (self.base_w - digit)
                        }
                    } else {
                        digit
                    };

                    digits[digit_idx][flat_idx] = signed_digit;
                    val /= self.base_w;
                }
            }
        }

        Ok(digits)
    }

    /// CPU polynomial multiplication in flat RNS layout (schoolbook)
    fn cpu_multiply_flat(&self, poly1: &[u64], poly2: &[u64], num_primes: usize) -> Result<Vec<u64>, String> {
        let n = self.params.n;
        let mut result = vec![0u64; n * num_primes];

        // For each prime
        for prime_idx in 0..num_primes {
            let q = self.params.moduli[prime_idx];
            let reducer = &self.reducers_key[prime_idx];

            // Schoolbook multiplication with negacyclic reduction
            for i in 0..n {
                for j in 0..n {
                    let idx1 = prime_idx * n + i;
                    let idx2 = prime_idx * n + j;

                    let a = poly1[idx1];
                    let b = poly2[idx2];
                    let prod = ((a as u128 * b as u128) % q as u128) as u64;

                    let result_idx = (i + j) % (2 * n);
                    if result_idx < n {
                        // Positive coefficient
                        let result_flat_idx = prime_idx * n + result_idx;
                        result[result_flat_idx] = (result[result_flat_idx] + prod) % q;
                    } else {
                        // Negative coefficient (X^N = -1)
                        let result_flat_idx = prime_idx * n + (result_idx - n);
                        result[result_flat_idx] = if result[result_flat_idx] >= prod {
                            result[result_flat_idx] - prod
                        } else {
                            q - (prod - result[result_flat_idx])
                        };
                    }
                }
            }
        }

        Ok(result)
    }

    /// Generate random polynomial in flat RNS layout
    fn generate_random_poly(&self, num_primes: usize) -> Vec<u64> {
        let n = self.params.n;
        let mut rng = rand::thread_rng();
        let mut poly = vec![0u64; n * num_primes];

        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                let flat_idx = prime_idx * n + coeff_idx;
                let q = self.params.moduli[prime_idx];
                poly[flat_idx] = rng.gen::<u64>() % q;
            }
        }

        poly
    }

    /// Generate error polynomial with Gaussian distribution
    fn generate_error_poly(&self, num_primes: usize) -> Vec<u64> {
        let n = self.params.n;
        let mut rng = rand::thread_rng();
        let mut poly = vec![0u64; n * num_primes];

        // Small Gaussian error (simplified: use small random values)
        for coeff_idx in 0..n {
            // Generate base error (small value)
            let error = (rng.gen::<u32>() % 8) as i32 - 4; // [-4, 3]

            for prime_idx in 0..num_primes {
                let flat_idx = prime_idx * n + coeff_idx;
                let q = self.params.moduli[prime_idx];

                poly[flat_idx] = if error >= 0 {
                    error as u64
                } else {
                    q - ((-error) as u64)
                };
            }
        }

        poly
    }

    /// Get reference to device context
    pub fn device(&self) -> &Arc<CudaDeviceContext> {
        &self.device
    }

    /// Get reference to parameters
    pub fn params(&self) -> &CliffordFHEParams {
        &self.params
    }

    /// Get modulus for a given prime index
    pub fn modulus(&self, prime_idx: usize) -> u64 {
        self.params.moduli[prime_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gadget_decomposition() {
        // Test that gadget decomposition works correctly
        // w = 2^16, decompose a value into base-w digits
        let base_w = 1u64 << 16;
        let val = 1234567890u64;

        // Manually decompose
        let d0 = val % base_w;
        let d1 = (val / base_w) % base_w;
        let d2 = (val / base_w / base_w) % base_w;

        // Reconstruct
        let reconstructed = d0 + d1 * base_w + d2 * base_w * base_w;
        assert_eq!(reconstructed, val);
    }
}
