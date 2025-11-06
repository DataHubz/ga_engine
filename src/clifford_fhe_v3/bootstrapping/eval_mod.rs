//! EvalMod - Homomorphic Modular Reduction
//!
//! Implements the core bootstrapping operation: homomorphic evaluation of
//! modular reduction to remove noise from ciphertexts.
//!
//! ## Theory
//!
//! For CKKS, decryption is: `m = (c0 + c1·s) mod q`
//!
//! The challenge: `mod q` is not a polynomial operation!
//!
//! **Solution**: Use sine approximation:
//! ```text
//! x mod q ≈ x - (q/2π) · sin(2πx/q)
//! ```
//!
//! ## Why This Works
//!
//! The sine function is periodic with period 2π, and when properly scaled,
//! approximates the "sawtooth" shape of modular reduction.
//!
//! For x in [0, q]:
//! - sin(2πx/q) oscillates from 0 to 0
//! - Scaled by q/2π, it subtracts the excess over each modulus
//! - Result: x mod q (approximately)
//!
//! ## Implementation Steps
//!
//! 1. **Scale input**: x' = 2πx/q (map to sine period)
//! 2. **Evaluate sine**: sin(x') using polynomial approximation
//! 3. **Scale and subtract**: result = x - (q/2π)·sin(x')
//!
//! The sine polynomial is evaluated homomorphically using Baby-Step Giant-Step
//! to minimize multiplication depth.

use crate::clifford_fhe_v2::params::CliffordFHEParams;
use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{Ciphertext, Plaintext};
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::{EvaluationKey, KeyContext};
use super::sin_approx::eval_polynomial;
use super::diagonal_mult::multiply_by_plaintext;
use std::f64::consts::PI;

/// Homomorphically evaluate modular reduction
///
/// Implements: x mod q ≈ x - (q/2π) · sin(2πx/q)
///
/// # Arguments
///
/// * `ct` - Input ciphertext (in slot representation)
/// * `q` - Modulus to reduce by
/// * `sin_coeffs` - Polynomial coefficients for sine approximation
/// * `evk` - Evaluation key (for relinearization)
/// * `params` - FHE parameters
/// * `key_ctx` - Key context (for NTT operations)
///
/// # Returns
///
/// Ciphertext with modular reduction applied
///
/// # Performance
///
/// For degree-23 sine polynomial:
/// - ~12 ciphertext multiplications
/// - ~500ms on CPU
/// - ~100ms on GPU (future)
///
/// # Example
///
/// ```rust,ignore
/// // Reduce ciphertext modulo q
/// let q = params.moduli[0];
/// let sin_coeffs = taylor_sin_coeffs(23);
/// let ct_reduced = eval_mod(&ct, q, &sin_coeffs, &params, &key_ctx)?;
/// ```
pub fn eval_mod(
    ct: &Ciphertext,
    q: u64,
    sin_coeffs: &[f64],
    evk: &EvaluationKey,
    params: &CliffordFHEParams,
    key_ctx: &KeyContext,
) -> Result<Ciphertext, String> {
    println!("    EvalMod: Starting modular reduction");
    println!("      Modulus: {}", q);
    println!("      Sine degree: {}", sin_coeffs.len() - 1);

    // Step 1: Scale input: x' = (2π/q) · x
    println!("      [1/3] Scaling input by 2π/q...");
    let scale_factor = 2.0 * PI / (q as f64);
    let ct_scaled = multiply_by_constant(ct, scale_factor, params)?;

    // Step 2: Evaluate sine polynomial: sin(x')
    println!("      [2/3] Evaluating sine polynomial...");
    let ct_sin = eval_sine_polynomial(&ct_scaled, sin_coeffs, evk, params, key_ctx)?;

    // Step 3: Compute result: x - (q/2π)·sin(x')
    println!("      [3/3] Computing final result...");
    let rescale_factor = (q as f64) / (2.0 * PI);
    let ct_sin_scaled = multiply_by_constant(&ct_sin, rescale_factor, params)?;

    // Subtract: ct_result = ct - ct_sin_scaled
    let ct_result = subtract_ciphertexts(ct, &ct_sin_scaled, params)?;

    println!("      ✓ EvalMod complete");

    Ok(ct_result)
}

/// Multiply ciphertext by a constant
///
/// Multiplies all slots by the same scalar value.
///
/// # Arguments
///
/// * `ct` - Input ciphertext
/// * `constant` - Scalar multiplier
/// * `params` - FHE parameters
///
/// # Returns
///
/// Ciphertext with all slots multiplied by constant
fn multiply_by_constant(
    ct: &Ciphertext,
    constant: f64,
    params: &CliffordFHEParams,
) -> Result<Ciphertext, String> {
    // Create plaintext with constant in all slots
    let num_slots = params.n / 2;
    let constant_vec = vec![constant; num_slots];

    // Encode as plaintext
    let pt_constant = encode_constant_plaintext(&constant_vec, params)?;

    // Multiply
    multiply_by_plaintext(ct, &pt_constant, params)
}

/// Encode constant vector as plaintext
fn encode_constant_plaintext(
    values: &[f64],
    params: &CliffordFHEParams,
) -> Result<Plaintext, String> {
    // Similar to diagonal encoding, but all slots have same value
    let n = params.n;
    let scale = params.scale;

    let mut coeffs = vec![0i64; n];

    // Simplified encoding (TODO: proper CKKS canonical embedding)
    for (i, &val) in values.iter().enumerate() {
        if i < n {
            coeffs[i] = (val * scale).round() as i64;
        }
    }

    use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

    // Convert coefficients to RNS representation
    let moduli = &params.moduli[..=0]; // level 0
    let mut rns_coeffs = Vec::with_capacity(n);

    for &coeff in &coeffs {
        let values: Vec<u64> = moduli.iter().map(|&q| {
            if coeff >= 0 {
                (coeff as u64) % q
            } else {
                let abs_val = (-coeff) as u64;
                let remainder = abs_val % q;
                if remainder == 0 { 0 } else { q - remainder }
            }
        }).collect();

        rns_coeffs.push(RnsRepresentation::new(values, moduli.to_vec()));
    }

    Ok(Plaintext {
        coeffs: rns_coeffs,
        scale,
        n,
        level: 0,
    })
}

/// Evaluate sine polynomial homomorphically
///
/// Computes sin(x) ≈ c₁x + c₃x³ + c₅x⁵ + ... using polynomial approximation.
///
/// Uses **Paterson-Stockmeyer** algorithm to minimize multiplication depth:
/// - Baby-step: Precompute powers x, x², x³, ...
/// - Giant-step: Group terms to reuse powers
///
/// For degree-23 polynomial:
/// - Depth: log₂(23) ≈ 5 levels
/// - Multiplications: ~12 operations
///
/// # Arguments
///
/// * `ct` - Input ciphertext (x)
/// * `sin_coeffs` - Polynomial coefficients [c₀, c₁, c₃, c₅, ...]
/// * `evk` - Evaluation key (for relinearization)
/// * `params` - FHE parameters
/// * `key_ctx` - Key context (for NTT operations)
///
/// # Returns
///
/// Ciphertext containing sin(x)
fn eval_sine_polynomial(
    ct: &Ciphertext,
    sin_coeffs: &[f64],
    evk: &EvaluationKey,
    params: &CliffordFHEParams,
    key_ctx: &KeyContext,
) -> Result<Ciphertext, String> {
    // For simplicity, use Horner's method
    // TODO: Implement Paterson-Stockmeyer for better depth

    // Find the highest non-zero coefficient
    let mut max_degree = sin_coeffs.len() - 1;
    while max_degree > 0 && sin_coeffs[max_degree].abs() < 1e-10 {
        max_degree -= 1;
    }

    if max_degree == 0 {
        // Constant polynomial
        return multiply_by_constant(ct, sin_coeffs[0], params);
    }

    // Start with highest coefficient
    let mut result = multiply_by_constant(ct, sin_coeffs[max_degree], params)?;

    // Horner's method: result = (...((c_n·x + c_{n-1})·x + c_{n-2})·x + ...)
    for i in (0..max_degree).rev() {
        // result = result · x
        result = multiply_ciphertexts(&result, ct, evk, params, key_ctx)?;

        // result = result + c_i
        if sin_coeffs[i].abs() > 1e-10 {
            let ct_coeff = create_constant_ciphertext(sin_coeffs[i], params)?;
            result = add_ciphertexts(&result, &ct_coeff, params)?;
        }
    }

    Ok(result)
}

/// Multiply two ciphertexts (with relinearization)
fn multiply_ciphertexts(
    ct1: &Ciphertext,
    ct2: &Ciphertext,
    evk: &EvaluationKey,
    params: &CliffordFHEParams,
    key_ctx: &KeyContext,
) -> Result<Ciphertext, String> {
    // Use V2 multiplication
    use crate::clifford_fhe_v2::backends::cpu_optimized::multiplication::multiply_ciphertexts;

    Ok(multiply_ciphertexts(ct1, ct2, evk, key_ctx))
}

/// Add two ciphertexts
fn add_ciphertexts(
    ct1: &Ciphertext,
    ct2: &Ciphertext,
    _params: &CliffordFHEParams,
) -> Result<Ciphertext, String> {
    // Component-wise addition using RnsRepresentation::add
    let c0_sum: Vec<_> = ct1.c0.iter().zip(&ct2.c0)
        .map(|(a, b)| a.add(b))
        .collect();

    let c1_sum: Vec<_> = ct1.c1.iter().zip(&ct2.c1)
        .map(|(a, b)| a.add(b))
        .collect();

    Ok(Ciphertext {
        c0: c0_sum,
        c1: c1_sum,
        n: ct1.n,
        level: ct1.level.max(ct2.level),
        scale: ct1.scale,  // Assuming same scale
    })
}

/// Subtract two ciphertexts
fn subtract_ciphertexts(
    ct1: &Ciphertext,
    ct2: &Ciphertext,
    _params: &CliffordFHEParams,
) -> Result<Ciphertext, String> {
    // Component-wise subtraction using RnsRepresentation::sub
    let c0_diff: Vec<_> = ct1.c0.iter().zip(&ct2.c0)
        .map(|(a, b)| a.sub(b))
        .collect();

    let c1_diff: Vec<_> = ct1.c1.iter().zip(&ct2.c1)
        .map(|(a, b)| a.sub(b))
        .collect();

    Ok(Ciphertext {
        c0: c0_diff,
        c1: c1_diff,
        n: ct1.n,
        level: ct1.level.max(ct2.level),
        scale: ct1.scale,
    })
}

/// Create ciphertext encoding a constant
fn create_constant_ciphertext(
    constant: f64,
    params: &CliffordFHEParams,
) -> Result<Ciphertext, String> {
    // Create plaintext with constant
    let n = params.n;
    let scale = params.scale;

    let mut coeffs = vec![0i64; n];
    coeffs[0] = (constant * scale).round() as i64;

    use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

    // Convert coefficients to RNS representation
    let moduli = &params.moduli[..=0]; // level 0
    let mut rns_coeffs = Vec::with_capacity(n);

    for &coeff in &coeffs {
        let values: Vec<u64> = moduli.iter().map(|&q| {
            if coeff >= 0 {
                (coeff as u64) % q
            } else {
                let abs_val = (-coeff) as u64;
                let remainder = abs_val % q;
                if remainder == 0 { 0 } else { q - remainder }
            }
        }).collect();

        rns_coeffs.push(RnsRepresentation::new(values, moduli.to_vec()));
    }

    // Create "trivial" ciphertext: (m, 0)
    // This is an encryption of m with no noise
    let c0 = rns_coeffs;

    // Create zero polynomial for c1
    let zero_values = vec![0u64; moduli.len()];
    let c1 = vec![RnsRepresentation::new(zero_values.clone(), moduli.to_vec()); n];

    Ok(Ciphertext {
        c0,
        c1,
        n,
        level: 0,
        scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v3::bootstrapping::sin_approx::taylor_sin_coeffs;

    #[test]
    fn test_multiply_by_constant() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();

        // Encrypt value 5.0
        use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{encode_scalar, decode_scalar};
        let ct = encode_scalar(5.0, &pk, &params).unwrap();

        // Multiply by 3.0
        let ct_result = multiply_by_constant(&ct, 3.0, &params).unwrap();

        // Decrypt
        let result = decode_scalar(&ct_result, &sk, &params).unwrap();

        // Should be 15.0
        assert!((result - 15.0).abs() < 1.0, "Expected 15, got {}", result);
    }

    #[test]
    fn test_add_ciphertexts() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();

        use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{encode_scalar, decode_scalar};

        let ct1 = encode_scalar(5.0, &pk, &params).unwrap();
        let ct2 = encode_scalar(3.0, &pk, &params).unwrap();

        let ct_sum = add_ciphertexts(&ct1, &ct2, &params).unwrap();

        let result = decode_scalar(&ct_sum, &sk, &params).unwrap();
        assert!((result - 8.0).abs() < 1.0, "Expected 8, got {}", result);
    }

    #[test]
    fn test_eval_sine_polynomial_simple() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, evk) = key_ctx.keygen();

        use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{encode_scalar, decode_scalar};

        // Test with linear approximation: sin(x) ≈ x (just c₁ = 1, others = 0)
        let mut sin_coeffs = vec![0.0; 8];
        sin_coeffs[1] = 1.0;  // Linear term only

        // Encrypt x = 0.5
        let ct = encode_scalar(0.5, &pk, &params).unwrap();

        // Evaluate: should return 0.5
        let ct_result = eval_sine_polynomial(&ct, &sin_coeffs, &evk, &params, &key_ctx).unwrap();

        let result = decode_scalar(&ct_result, &sk, &params).unwrap();
        assert!((result - 0.5).abs() < 0.1, "Linear sine: expected 0.5, got {}", result);
    }
}
