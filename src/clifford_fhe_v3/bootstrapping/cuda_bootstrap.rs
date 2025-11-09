//! V3 CUDA GPU Bootstrap Implementation
//!
//! Full homomorphic encryption bootstrap using CUDA GPU acceleration.
//!
//! **Bootstrap Pipeline:**
//! 1. **Modulus Raise**: Extend ciphertext to higher modulus level
//! 2. **CoeffToSlot (C2S)**: Transform coefficients to slots using rotations
//! 3. **EvalMod**: Evaluate modular reduction (removes noise)
//! 4. **SlotToCoeff (S2C)**: Transform slots back to coefficients
//! 5. **Modulus Switch**: Reduce back to original modulus level
//!
//! **GPU Acceleration:**
//! - Rotation operations use GPU Galois kernel
//! - NTT operations use GPU kernels
//! - Rescaling uses GPU RNS kernel
//! - Key switching uses rotation keys
//!
//! **Performance Target:**
//! - RTX 5090: ~20-25s full bootstrap (3× faster than Metal M3 Max)

use crate::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCkksContext;
use crate::clifford_fhe_v2::backends::gpu_cuda::rotation::CudaRotationContext;
use crate::clifford_fhe_v2::backends::gpu_cuda::rotation_keys::CudaRotationKeys;
use crate::clifford_fhe_v2::backends::gpu_cuda::relin_keys::CudaRelinKeys;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use crate::clifford_fhe_v3::bootstrapping::BootstrapParams;
use crate::clifford_fhe_v3::bootstrapping::cuda_coeff_to_slot::cuda_coeff_to_slot;
use crate::clifford_fhe_v3::bootstrapping::cuda_slot_to_coeff::cuda_slot_to_coeff;
use crate::clifford_fhe_v3::bootstrapping::cuda_eval_mod::cuda_eval_mod;
use std::sync::Arc;

/// CUDA GPU bootstrap context
pub struct CudaBootstrapContext {
    /// CKKS context for basic operations
    ckks_ctx: Arc<CudaCkksContext>,

    /// Rotation context for Galois automorphisms
    rotation_ctx: Arc<CudaRotationContext>,

    /// Rotation keys for key switching
    rotation_keys: Arc<CudaRotationKeys>,

    /// Relinearization keys for ciphertext multiplication
    relin_keys: Arc<CudaRelinKeys>,

    /// V3 bootstrap parameters
    bootstrap_params: BootstrapParams,

    /// Base FHE parameters
    params: CliffordFHEParams,
}

impl CudaBootstrapContext {
    /// Create new CUDA bootstrap context
    pub fn new(
        ckks_ctx: Arc<CudaCkksContext>,
        rotation_ctx: Arc<CudaRotationContext>,
        rotation_keys: Arc<CudaRotationKeys>,
        relin_keys: Arc<CudaRelinKeys>,
        bootstrap_params: BootstrapParams,
        params: CliffordFHEParams,
    ) -> Result<Self, String> {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║         CUDA GPU Bootstrap Context Initialized               ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        Ok(Self {
            ckks_ctx,
            rotation_ctx,
            rotation_keys,
            relin_keys,
            bootstrap_params,
            params,
        })
    }

    /// Perform full bootstrap operation
    ///
    /// Input: Noisy ciphertext at low level
    /// Output: Refreshed ciphertext with reduced noise
    pub fn bootstrap(
        &self,
        ct_in: &CudaCiphertext,
    ) -> Result<CudaCiphertext, String> {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║              CUDA GPU Bootstrap Pipeline                     ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let total_start = std::time::Instant::now();

        // Step 1: Modulus raise (extend to max level)
        println!("Step 1: Modulus raise");
        let step1_start = std::time::Instant::now();
        let ct_raised = self.modulus_raise(ct_in)?;
        println!("  ✅ Modulus raised in {:.2}s\n", step1_start.elapsed().as_secs_f64());

        // Step 2: CoeffToSlot transformation
        println!("Step 2: CoeffToSlot transformation");
        let step2_start = std::time::Instant::now();
        let ct_slots = self.coeff_to_slot(&ct_raised)?;
        println!("  ✅ CoeffToSlot completed in {:.2}s\n", step2_start.elapsed().as_secs_f64());

        // Step 3: EvalMod (sine evaluation for modular reduction)
        println!("Step 3: EvalMod (modular reduction)");
        let step3_start = std::time::Instant::now();
        let ct_evalmod = self.eval_mod(&ct_slots)?;
        println!("  ✅ EvalMod completed in {:.2}s\n", step3_start.elapsed().as_secs_f64());

        // Step 4: SlotToCoeff transformation
        println!("Step 4: SlotToCoeff transformation");
        let step4_start = std::time::Instant::now();
        let ct_coeffs = self.slot_to_coeff(&ct_evalmod)?;
        println!("  ✅ SlotToCoeff completed in {:.2}s\n", step4_start.elapsed().as_secs_f64());

        // Step 5: Modulus switch (reduce back to original level)
        println!("Step 5: Modulus switch");
        let step5_start = std::time::Instant::now();
        let ct_out = self.modulus_switch(&ct_coeffs, ct_in.level)?;
        println!("  ✅ Modulus switched in {:.2}s\n", step5_start.elapsed().as_secs_f64());

        let total_time = total_start.elapsed().as_secs_f64();
        println!("═══════════════════════════════════════════════════════════════");
        println!("Bootstrap completed in {:.2}s", total_time);
        println!("═══════════════════════════════════════════════════════════════\n");

        Ok(ct_out)
    }

    /// Step 1: Modulus raise - extend ciphertext to higher modulus
    fn modulus_raise(&self, ct: &CudaCiphertext) -> Result<CudaCiphertext, String> {
        let target_level = self.params.moduli.len() - 1;

        if ct.level >= target_level {
            return Ok(ct.clone());
        }

        // Simply extend RNS representation with zeros for higher primes
        let n = ct.n;
        let mut c0_raised = vec![0u64; n * (target_level + 1)];
        let mut c1_raised = vec![0u64; n * (target_level + 1)];

        // Copy existing coefficients (strided layout)
        for coeff_idx in 0..n {
            for prime_idx in 0..=ct.level {
                c0_raised[coeff_idx * (target_level + 1) + prime_idx] =
                    ct.c0[coeff_idx * (ct.level + 1) + prime_idx];
                c1_raised[coeff_idx * (target_level + 1) + prime_idx] =
                    ct.c1[coeff_idx * (ct.level + 1) + prime_idx];
            }
            // Higher primes get value 0 (already initialized)
        }

        Ok(CudaCiphertext {
            c0: c0_raised,
            c1: c1_raised,
            n: ct.n,
            num_primes: target_level + 1,
            level: target_level,
            scale: ct.scale,
        })
    }

    /// Step 2: CoeffToSlot - transform coefficient encoding to slot encoding
    ///
    /// Uses FFT-like butterfly algorithm with rotations
    fn coeff_to_slot(&self, ct: &CudaCiphertext) -> Result<CudaCiphertext, String> {
        cuda_coeff_to_slot(ct, &self.rotation_keys, &self.ckks_ctx)
    }

    /// Step 3: EvalMod - evaluate modular reduction using sine approximation
    ///
    /// Removes noise by evaluating: f(x) = x - q/2π · sin(2πx/q)
    fn eval_mod(&self, ct: &CudaCiphertext) -> Result<CudaCiphertext, String> {
        // Use the top-level modulus for EvalMod
        let q = self.params.moduli[ct.level];
        let sin_degree = self.bootstrap_params.sin_degree;
        cuda_eval_mod(ct, q, sin_degree, &self.ckks_ctx, Some(&self.relin_keys))
    }

    /// Step 4: SlotToCoeff - transform slot encoding back to coefficient encoding
    ///
    /// Inverse of CoeffToSlot
    fn slot_to_coeff(&self, ct: &CudaCiphertext) -> Result<CudaCiphertext, String> {
        cuda_slot_to_coeff(ct, &self.rotation_keys, &self.ckks_ctx)
    }

    /// Step 5: Modulus switch - reduce ciphertext to target level
    fn modulus_switch(&self, ct: &CudaCiphertext, target_level: usize) -> Result<CudaCiphertext, String> {
        if ct.level <= target_level {
            return Ok(ct.clone());
        }

        // Apply GPU rescaling repeatedly to drop primes
        let mut current_ct = ct.clone();

        for _ in target_level..ct.level {
            // Use GPU rescaling
            let c0_rescaled = self.ckks_ctx.exact_rescale_gpu(&current_ct.c0, current_ct.level)?;
            let c1_rescaled = self.ckks_ctx.exact_rescale_gpu(&current_ct.c1, current_ct.level)?;

            let new_level = current_ct.level - 1;
            let new_scale = current_ct.scale / self.params.moduli[current_ct.level] as f64;

            current_ct = CudaCiphertext {
                c0: c0_rescaled,
                c1: c1_rescaled,
                n: current_ct.n,
                num_primes: new_level + 1,
                level: new_level,
                scale: new_scale,
            };
        }

        Ok(current_ct)
    }
}

/// CUDA ciphertext structure
#[derive(Clone)]
pub struct CudaCiphertext {
    pub c0: Vec<u64>,
    pub c1: Vec<u64>,
    pub n: usize,
    pub num_primes: usize,
    pub level: usize,
    pub scale: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modulus_raise() {
        // Test that modulus raise extends ciphertext correctly
        let params = CliffordFHEParams::new_test_ntt_1024();
        let n = params.n;

        let ct_low = CudaCiphertext {
            c0: vec![1u64; n * 2],  // Level 1 (2 primes)
            c1: vec![2u64; n * 2],
            n,
            num_primes: 2,
            level: 1,
            scale: 1e10,
        };

        // Modulus raise should extend to level 2 (3 primes)
        // Result should have original values in first 2 primes, zeros in prime 3
    }
}
