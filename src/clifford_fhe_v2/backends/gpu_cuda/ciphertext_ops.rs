///! Extension methods for CudaCiphertext to match Metal API
///!
///! Implements methods like add(), rotate_by_steps(), multiply_plain()
///! directly on CudaCiphertext to provide the same API as MetalCiphertext.
///!
///! This allows V4 code to work seamlessly with CUDA backend.

use super::ckks::{CudaCiphertext, CudaCkksContext, CudaPlaintext};
use super::rotation_keys::CudaRotationKeys;

impl CudaCiphertext {
    /// Add two ciphertexts (component-wise polynomial addition)
    ///
    /// Mirrors MetalCiphertext::add() API
    pub fn add(&self, other: &Self, ctx: &CudaCkksContext) -> Result<Self, String> {
        // Delegate to CudaCkksContext::add
        ctx.add(self, other)
    }

    /// Subtract two ciphertexts
    ///
    /// Mirrors MetalCiphertext::subtract() API
    pub fn subtract(&self, other: &Self, ctx: &CudaCkksContext) -> Result<Self, String> {
        assert_eq!(self.n, other.n, "Dimensions must match");
        assert_eq!(self.level, other.level, "Levels must match");
        assert_eq!(self.num_primes, other.num_primes, "Number of primes must match");

        // Use subtract_polynomials_gpu for component-wise subtraction
        let num_primes_active = self.level + 1;

        let c0_sub = ctx.subtract_polynomials_gpu(&self.c0, &other.c0, num_primes_active)?;
        let c1_sub = ctx.subtract_polynomials_gpu(&self.c1, &other.c1, num_primes_active)?;

        Ok(Self {
            c0: c0_sub,
            c1: c1_sub,
            n: self.n,
            num_primes: self.num_primes,
            level: self.level,
            scale: self.scale,
        })
    }

    /// Multiply ciphertext by plaintext
    ///
    /// Mirrors MetalCiphertext::multiply_plain() API
    pub fn multiply_plain(
        &self,
        plaintext: &CudaPlaintext,
        ctx: &CudaCkksContext,
    ) -> Result<Self, String> {
        let num_primes_active = self.level + 1;

        if plaintext.num_primes < num_primes_active {
            return Err(format!(
                "Plaintext has {} primes but ciphertext is at level {} (needs {} primes)",
                plaintext.num_primes, self.level, num_primes_active
            ));
        }

        // Multiply c0 and c1 by plaintext
        let c0_mult = ctx.pointwise_multiply_polynomials_gpu_strided(
            &self.c0,
            &plaintext.poly,
            num_primes_active,
        )?;

        let c1_mult = ctx.pointwise_multiply_polynomials_gpu_strided(
            &self.c1,
            &plaintext.poly,
            num_primes_active,
        )?;

        // Rescale to maintain scale
        let c0_rescaled = ctx.exact_rescale_gpu_strided(&c0_mult, self.level)?;
        let c1_rescaled = ctx.exact_rescale_gpu_strided(&c1_mult, self.level)?;

        Ok(Self {
            c0: c0_rescaled,
            c1: c1_rescaled,
            n: self.n,
            num_primes: self.num_primes,
            level: self.level.saturating_sub(1),
            scale: self.scale * plaintext.scale / ctx.params().scale,
        })
    }

    /// Rotate ciphertext by given number of slots
    ///
    /// Mirrors MetalCiphertext::rotate_by_steps() API (3 parameters)
    pub fn rotate_by_steps(
        &self,
        step: i32,
        rot_keys: &CudaRotationKeys,
        ctx: &CudaCkksContext,
    ) -> Result<Self, String> {
        // Access rotation context through rotation keys (like V3 CUDA bootstrap)
        let rot_ctx = rot_keys.rotation_context();

        let n = self.n;
        let num_primes_active = self.level + 1;

        // Compute Galois element for this rotation
        let galois_elt = rotation_step_to_galois_element(step, n)?;

        // Apply Galois automorphism to c0 and c1
        let rotated_c0 = rot_ctx.rotate_gpu(&self.c0, step, num_primes_active)?;
        let rotated_c1 = rot_ctx.rotate_gpu(&self.c1, step, num_primes_active)?;

        // Apply key switching using rotation keys
        rot_keys.apply_key_switch_gpu(
            &rotated_c0,
            &rotated_c1,
            galois_elt,
            self.level,
            self.scale,
            ctx.ntt_contexts(),
        )
    }

    /// Batch rotation with hoisting (multiple rotations optimized)
    ///
    /// Mirrors MetalCiphertext::rotate_batch_with_hoisting() API (3 parameters)
    pub fn rotate_batch_with_hoisting(
        &self,
        steps: &[i32],
        rot_keys: &CudaRotationKeys,
        ctx: &CudaCkksContext,
    ) -> Result<Vec<Self>, String> {
        if steps.is_empty() {
            return Ok(vec![]);
        }

        // For now, implement as sequential rotations
        // TODO: Implement true hoisting optimization later
        let mut results = Vec::with_capacity(steps.len());

        for &step in steps {
            let rotated = self.rotate_by_steps(step, rot_keys, ctx)?;
            results.push(rotated);
        }

        Ok(results)
    }
}

/// Compute Galois element for rotation by k slots
///
/// For cyclotomic ring Z[X]/(X^N + 1), galois_elt = 5^k mod 2N
fn rotation_step_to_galois_element(rotation_steps: i32, n: usize) -> Result<usize, String> {
    let two_n = 2 * n;
    let k = if rotation_steps >= 0 {
        rotation_steps as usize % (n / 2)
    } else {
        let abs_steps = (-rotation_steps) as usize % (n / 2);
        (n / 2) - abs_steps
    };

    // Compute 5^k mod 2N using modular exponentiation
    let mut result = 1usize;
    let mut base = 5usize;
    let mut exp = k;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % two_n;
        }
        base = (base * base) % two_n;
        exp >>= 1;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_galois_element_computation() {
        // For N=1024, test common rotations
        assert_eq!(rotation_step_to_galois_element(1, 1024).unwrap(), 5);
        assert_eq!(rotation_step_to_galois_element(2, 1024).unwrap(), 25);
        assert_eq!(rotation_step_to_galois_element(0, 1024).unwrap(), 1);

        // Test negative rotation
        let neg_1 = rotation_step_to_galois_element(-1, 1024).unwrap();
        assert!(neg_1 > 1 && neg_1 < 2048); // Should be valid Galois element
    }
}
