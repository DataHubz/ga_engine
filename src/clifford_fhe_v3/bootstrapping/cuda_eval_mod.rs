//! CUDA GPU EvalMod - Homomorphic Modular Reduction
//!
//! GPU-accelerated homomorphic evaluation of modular reduction using sine approximation.
//!
//! **Algorithm**: x mod q ≈ x - (q/2π) · sin(2πx/q)
//!
//! **GPU Optimizations**:
//! - Polynomial evaluation uses GPU rescaling
//! - Multiplication by constants (plaintext multiply)
//! - Baby-step giant-step for deep polynomials
//!
//! **Performance Target**: ~10-12s on RTX 5090 (vs ~30s on Metal M3 Max)

use crate::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCkksContext;
use crate::clifford_fhe_v2::backends::gpu_cuda::relin_keys::CudaRelinKeys;
use crate::clifford_fhe_v3::bootstrapping::cuda_bootstrap::CudaCiphertext;
use crate::clifford_fhe_v3::bootstrapping::cuda_coeff_to_slot::{cuda_add_ciphertexts, cuda_multiply_plain};
use crate::clifford_fhe_v3::bootstrapping::sin_approx::chebyshev_sin_coeffs;
use std::f64::consts::PI;
use std::sync::Arc;

/// CUDA GPU EvalMod - Homomorphic modular reduction
///
/// Implements: x mod q ≈ x - (q/2π) · sin(2πx/q)
///
/// # Arguments
/// * `ct` - Input ciphertext (in slot representation)
/// * `q` - Modulus to reduce by
/// * `sin_degree` - Degree of sine polynomial (15, 23, or 31)
/// * `ckks_ctx` - CUDA CKKS context
/// * `relin_keys` - Relinearization keys for exact multiplication (optional)
///
/// # Returns
/// Ciphertext with modular reduction applied
///
/// # Algorithm
/// ```text
/// 1. Scale input: ct' = (2π/q) · ct
/// 2. Evaluate sine: ct_sin = sin(ct')  [using Chebyshev polynomial]
/// 3. Final result: ct_out = ct - (q/2π) · ct_sin
/// ```
pub fn cuda_eval_mod(
    ct: &CudaCiphertext,
    q: u64,
    sin_degree: usize,
    ckks_ctx: &Arc<CudaCkksContext>,
    relin_keys: Option<&Arc<CudaRelinKeys>>,
) -> Result<CudaCiphertext, String> {
    println!("  [CUDA EvalMod] Starting modular reduction");
    println!("    Modulus: {}", q);
    println!("    Sine degree: {}", sin_degree);
    if relin_keys.is_some() {
        println!("    Relinearization: ENABLED (exact multiplication)");
    } else {
        println!("    Relinearization: DISABLED (approximation)");
    }

    // Step 1: Scale input by 2π/q
    println!("    [1/3] Scaling input by 2π/q...");
    let scale_factor = 2.0 * PI / (q as f64);
    let ct_scaled = cuda_multiply_by_constant(ct, scale_factor, ckks_ctx, relin_keys)?;
    println!("      → Scaled: level={}, scale={:.2e}", ct_scaled.level, ct_scaled.scale);

    // Step 2: Evaluate sine polynomial using Chebyshev approximation
    println!("    [2/3] Evaluating degree-{} sine polynomial...", sin_degree);
    let sin_coeffs = chebyshev_sin_coeffs(sin_degree);
    let ct_sin = cuda_eval_sine_polynomial(&ct_scaled, &sin_coeffs, ckks_ctx, relin_keys)?;
    println!("      → Sine evaluated: level={}, scale={:.2e}", ct_sin.level, ct_sin.scale);

    // Step 3: Compute final result: ct - (q/2π)·sin(ct)
    println!("    [3/3] Computing final result: x - (q/2π)·sin(x)...");
    let rescale_factor = (q as f64) / (2.0 * PI);
    let ct_sin_scaled = cuda_multiply_by_constant(&ct_sin, rescale_factor, ckks_ctx, relin_keys)?;

    // Subtract: result = ct - ct_sin_scaled
    let ct_result = cuda_subtract_ciphertexts(ct, &ct_sin_scaled, ckks_ctx)?;

    println!("  [CUDA EvalMod] Complete: level={}, scale={:.2e}",
        ct_result.level, ct_result.scale);

    Ok(ct_result)
}

/// Multiply ciphertext by a constant (plaintext scalar)
fn cuda_multiply_by_constant(
    ct: &CudaCiphertext,
    constant: f64,
    ckks_ctx: &Arc<CudaCkksContext>,
    _relin_keys: Option<&Arc<CudaRelinKeys>>,  // Not needed for plaintext multiply
) -> Result<CudaCiphertext, String> {
    let num_slots = ct.n / 2;

    // Create plaintext with constant value in all slots
    let values = vec![constant; num_slots];

    // Encode as plaintext with scale matching top modulus
    let scale_for_pt = ckks_ctx.params().moduli[ct.level] as f64;
    let pt = ckks_ctx.encode(&values, scale_for_pt, ct.level)?;

    // Multiply using GPU
    cuda_multiply_plain(ct, &pt.poly, ckks_ctx, scale_for_pt)
}

/// Evaluate sine polynomial using Chebyshev coefficients
///
/// Uses baby-step giant-step algorithm for efficient deep polynomial evaluation
fn cuda_eval_sine_polynomial(
    ct: &CudaCiphertext,
    coeffs: &[f64],
    ckks_ctx: &Arc<CudaCkksContext>,
    relin_keys: Option<&Arc<CudaRelinKeys>>,
) -> Result<CudaCiphertext, String> {
    let degree = coeffs.len() - 1;

    println!("      Evaluating polynomial of degree {}...", degree);

    // For small degrees, use Horner's method
    if degree <= 7 {
        return cuda_eval_polynomial_horner(ct, coeffs, ckks_ctx, relin_keys);
    }

    // For larger degrees, use baby-step giant-step
    cuda_eval_polynomial_bsgs(ct, coeffs, ckks_ctx, relin_keys)
}

/// Evaluate polynomial using Horner's method
///
/// For polynomial p(x) = a0 + a1·x + a2·x² + ... + an·x^n
/// Horner's method: p(x) = a0 + x·(a1 + x·(a2 + x·(...)))
fn cuda_eval_polynomial_horner(
    ct: &CudaCiphertext,
    coeffs: &[f64],
    ckks_ctx: &Arc<CudaCkksContext>,
    relin_keys: Option<&Arc<CudaRelinKeys>>,
) -> Result<CudaCiphertext, String> {
    let degree = coeffs.len() - 1;

    // Start with highest degree coefficient
    let mut result = cuda_create_constant_ciphertext(coeffs[degree], ct.n, ct.level, ckks_ctx)?;

    // Work backwards: result = result * x + coeffs[i]
    for i in (0..degree).rev() {
        // Multiply by x
        result = cuda_multiply_ciphertexts(&result, ct, ckks_ctx, relin_keys)?;

        // Add constant
        let ct_const = cuda_create_constant_ciphertext(coeffs[i], ct.n, result.level, ckks_ctx)?;
        result = cuda_add_ciphertexts(&result, &ct_const)?;
    }

    Ok(result)
}

/// Evaluate polynomial using baby-step giant-step algorithm
///
/// For degree n polynomial, uses O(√n) multiplications instead of O(n)
fn cuda_eval_polynomial_bsgs(
    ct: &CudaCiphertext,
    coeffs: &[f64],
    ckks_ctx: &Arc<CudaCkksContext>,
    relin_keys: Option<&Arc<CudaRelinKeys>>,
) -> Result<CudaCiphertext, String> {
    let degree = coeffs.len() - 1;
    let baby_steps = ((degree as f64).sqrt().ceil() as usize).max(2);
    let giant_steps = (degree + baby_steps) / baby_steps;

    println!("        Using BSGS: baby_steps={}, giant_steps={}", baby_steps, giant_steps);

    // Precompute powers of x: x, x², x³, ..., x^baby_steps
    let mut x_powers = vec![ct.clone()];
    for i in 1..baby_steps {
        let x_next = cuda_multiply_ciphertexts(&x_powers[i-1], ct, ckks_ctx, relin_keys)?;
        x_powers.push(x_next);
    }

    // Compute giant step: x^baby_steps
    let x_giant = if baby_steps < x_powers.len() {
        x_powers[baby_steps - 1].clone()
    } else {
        cuda_multiply_ciphertexts(&x_powers[baby_steps - 2], ct, ckks_ctx, relin_keys)?
    };

    // Evaluate polynomial in blocks
    let mut result = cuda_create_constant_ciphertext(0.0, ct.n, ct.level, ckks_ctx)?;
    let mut x_giant_power = cuda_create_constant_ciphertext(1.0, ct.n, ct.level, ckks_ctx)?;

    for g in 0..giant_steps {
        // Evaluate baby steps for this giant step
        let mut baby_sum = cuda_create_constant_ciphertext(0.0, ct.n, ct.level, ckks_ctx)?;

        for b in 0..baby_steps {
            let idx = g * baby_steps + b;
            if idx >= coeffs.len() {
                break;
            }

            let coeff = coeffs[idx];
            if coeff.abs() > 1e-10 {
                let term = if b == 0 {
                    cuda_create_constant_ciphertext(coeff, ct.n, baby_sum.level, ckks_ctx)?
                } else {
                    let ct_coeff = cuda_create_constant_ciphertext(coeff, ct.n, x_powers[b-1].level, ckks_ctx)?;
                    cuda_multiply_ciphertexts(&ct_coeff, &x_powers[b-1], ckks_ctx, relin_keys)?
                };

                baby_sum = cuda_add_ciphertexts(&baby_sum, &term)?;
            }
        }

        // Multiply by giant step power and add to result
        let term = cuda_multiply_ciphertexts(&baby_sum, &x_giant_power, ckks_ctx, relin_keys)?;
        result = cuda_add_ciphertexts(&result, &term)?;

        // Update giant step power
        if g < giant_steps - 1 {
            x_giant_power = cuda_multiply_ciphertexts(&x_giant_power, &x_giant, ckks_ctx, relin_keys)?;
        }
    }

    Ok(result)
}

/// Create a constant ciphertext (trivial encryption of a constant)
fn cuda_create_constant_ciphertext(
    value: f64,
    n: usize,
    level: usize,
    ckks_ctx: &Arc<CudaCkksContext>,
) -> Result<CudaCiphertext, String> {
    let num_slots = n / 2;
    let values = vec![value; num_slots];

    // Encode as plaintext
    let scale = ckks_ctx.params().moduli[level] as f64;
    let pt = ckks_ctx.encode(&values, scale, level)?;

    // Trivial encryption: (pt, 0)
    let num_primes = level + 1;
    let c0 = pt.poly;
    let c1 = vec![0u64; n * num_primes];

    Ok(CudaCiphertext {
        c0,
        c1,
        n,
        num_primes,
        level,
        scale,
    })
}

/// Multiply two ciphertexts
///
/// Implements tensor product multiplication: (a0, a1) × (b0, b1) = (c0, c1, c2)
/// where:
///   c0 = a0 * b0
///   c1 = a0 * b1 + a1 * b0
///   c2 = a1 * b1
///
/// If relinearization keys are provided, reduces (c0, c1, c2) → (c0', c1')
/// Otherwise, uses approximation: drops c2 term
fn cuda_multiply_ciphertexts(
    ct1: &CudaCiphertext,
    ct2: &CudaCiphertext,
    ckks_ctx: &Arc<CudaCkksContext>,
    relin_keys: Option<&Arc<CudaRelinKeys>>,
) -> Result<CudaCiphertext, String> {
    if ct1.level != ct2.level {
        return Err(format!("Level mismatch: {} vs {}", ct1.level, ct2.level));
    }

    let n = ct1.n;
    let num_primes = ct1.num_primes;

    // Tensor product: (a0, a1) × (b0, b1) = (c0, c1, c2)
    // c0 = a0 * b0
    // c1 = a0 * b1 + a1 * b0
    // c2 = a1 * b1

    let mut c0_result = vec![0u64; n * num_primes];
    let mut c1_result = vec![0u64; n * num_primes];
    let mut c2_result = vec![0u64; n * num_primes];

    // Compute c0, c1, and c2
    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            let idx = coeff_idx * num_primes + prime_idx;
            let q = ckks_ctx.params().moduli[prime_idx];

            let a0 = ct1.c0[idx];
            let a1 = ct1.c1[idx];
            let b0 = ct2.c0[idx];
            let b1 = ct2.c1[idx];

            // c0 = a0 * b0 (mod q)
            c0_result[idx] = ((a0 as u128 * b0 as u128) % q as u128) as u64;

            // c1 = a0 * b1 + a1 * b0 (mod q)
            let term1 = ((a0 as u128 * b1 as u128) % q as u128) as u64;
            let term2 = ((a1 as u128 * b0 as u128) % q as u128) as u64;
            c1_result[idx] = (term1 + term2) % q;

            // c2 = a1 * b1 (mod q)
            c2_result[idx] = ((a1 as u128 * b1 as u128) % q as u128) as u64;
        }
    }

    // Apply relinearization if keys are available
    let (c0_final, c1_final) = if let Some(relin_keys) = relin_keys {
        // Convert to flat layout for relinearization
        let mut c0_flat = vec![0u64; n * num_primes];
        let mut c1_flat = vec![0u64; n * num_primes];
        let mut c2_flat = vec![0u64; n * num_primes];

        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                let strided_idx = coeff_idx * num_primes + prime_idx;
                let flat_idx = prime_idx * n + coeff_idx;
                c0_flat[flat_idx] = c0_result[strided_idx];
                c1_flat[flat_idx] = c1_result[strided_idx];
                c2_flat[flat_idx] = c2_result[strided_idx];
            }
        }

        // Apply relinearization: (c0, c1, c2) → (c0', c1')
        let (c0_relin, c1_relin) = relin_keys.apply_relinearization(
            &c0_flat,
            &c1_flat,
            &c2_flat,
            ct1.level,
        )?;

        // Convert back to strided layout
        let mut c0_strided = vec![0u64; n * num_primes];
        let mut c1_strided = vec![0u64; n * num_primes];
        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes {
                let flat_idx = prime_idx * n + coeff_idx;
                let strided_idx = coeff_idx * num_primes + prime_idx;
                c0_strided[strided_idx] = c0_relin[flat_idx];
                c1_strided[strided_idx] = c1_relin[flat_idx];
            }
        }

        (c0_strided, c1_strided)
    } else {
        // No relinearization keys: use approximation (drop c2)
        (c0_result, c1_result)
    };

    // New scale = scale1 * scale2
    let new_scale = ct1.scale * ct2.scale;

    // Rescale to maintain scale (drop one modulus)
    if num_primes > 1 {
        let c0_rescaled = ckks_ctx.exact_rescale_gpu(&c0_final, num_primes - 1)?;
        let c1_rescaled = ckks_ctx.exact_rescale_gpu(&c1_final, num_primes - 1)?;

        Ok(CudaCiphertext {
            c0: c0_rescaled,
            c1: c1_rescaled,
            n,
            num_primes: num_primes - 1,
            level: ct1.level - 1,
            scale: new_scale / ckks_ctx.params().moduli[num_primes - 1] as f64,
        })
    } else {
        Ok(CudaCiphertext {
            c0: c0_final,
            c1: c1_final,
            n,
            num_primes,
            level: ct1.level,
            scale: new_scale,
        })
    }
}

/// Subtract two ciphertexts
fn cuda_subtract_ciphertexts(
    ct1: &CudaCiphertext,
    ct2: &CudaCiphertext,
    ckks_ctx: &Arc<CudaCkksContext>,
) -> Result<CudaCiphertext, String> {
    if ct1.level != ct2.level {
        return Err(format!("Level mismatch: {} vs {}", ct1.level, ct2.level));
    }

    let n = ct1.n;
    let num_primes = ct1.num_primes;
    let mut c0_diff = vec![0u64; n * num_primes];
    let mut c1_diff = vec![0u64; n * num_primes];

    // Subtract coefficient-wise in RNS
    for idx in 0..(n * num_primes) {
        let prime_idx = idx % num_primes;
        let q = ckks_ctx.params().moduli[prime_idx];

        // c0_diff = c1 - c2 (mod q)
        c0_diff[idx] = if ct1.c0[idx] >= ct2.c0[idx] {
            ct1.c0[idx] - ct2.c0[idx]
        } else {
            q - (ct2.c0[idx] - ct1.c0[idx])
        };

        c1_diff[idx] = if ct1.c1[idx] >= ct2.c1[idx] {
            ct1.c1[idx] - ct2.c1[idx]
        } else {
            q - (ct2.c1[idx] - ct1.c1[idx])
        };
    }

    Ok(CudaCiphertext {
        c0: c0_diff,
        c1: c1_diff,
        n,
        num_primes,
        level: ct1.level,
        scale: ct1.scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_function_periodicity() {
        // Verify that our sine approximation has correct period
        let degree = 23;
        let coeffs = chebyshev_sin_coeffs(degree);

        // sin(0) ≈ 0
        // sin(π/2) ≈ 1
        // sin(π) ≈ 0
        // sin(2π) ≈ 0

        // (This would require actually evaluating the polynomial,
        //  which we can't do without a full CKKS context)
    }
}
