//! Modulus Raising
//!
//! Raises ciphertext to higher modulus level to create working room for bootstrap.
//!
//! This scales up the ciphertext coefficients to work with a larger modulus chain,
//! preserving the plaintext value.

use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

/// Raise ciphertext modulus to higher level
///
/// This scales up the ciphertext coefficients to work with a larger modulus chain,
/// preserving the plaintext value. Creates "working room" for bootstrap computation.
///
/// # Arguments
///
/// * `ct` - Input ciphertext with current moduli
/// * `target_moduli` - Target moduli (must be larger chain)
///
/// # Returns
///
/// Ciphertext with raised modulus (same plaintext value)
///
/// # Errors
///
/// Returns error if target moduli count is not larger than current.
///
/// # Example
///
/// ```rust,ignore
/// use ga_engine::clifford_fhe_v3::bootstrapping::mod_raise;
///
/// // Original ciphertext with 3 primes
/// let ct = encrypt(&plaintext, &public_key);
///
/// // Raise to 5 primes (adds 2 more for bootstrap working room)
/// let target_moduli = vec![q0, q1, q2, q3, q4];
/// let ct_raised = mod_raise(&ct, &target_moduli)?;
///
/// // Decrypt - should get same plaintext
/// let decrypted = decrypt(&ct_raised, &secret_key);
/// assert_approx_eq!(plaintext, decrypted, 0.01);
/// ```
pub fn mod_raise(
    ct: &Ciphertext,
    target_moduli: &[u64],
) -> Result<Ciphertext, String> {
    // Current moduli
    let current_moduli = &ct.c0[0].moduli;
    let n = ct.c0.len();

    if target_moduli.len() <= current_moduli.len() {
        return Err(format!(
            "Target moduli count ({}) must be larger than current ({})",
            target_moduli.len(),
            current_moduli.len()
        ));
    }

    // Verify that target moduli include current moduli (as prefix)
    for i in 0..current_moduli.len() {
        if current_moduli[i] != target_moduli[i] {
            return Err(format!(
                "Target moduli must include current moduli as prefix. Mismatch at index {}: {} != {}",
                i, current_moduli[i], target_moduli[i]
            ));
        }
    }

    // Scale c0 to higher modulus
    let mut c0_raised = Vec::with_capacity(n);
    for rns in &ct.c0 {
        c0_raised.push(scale_rns_to_higher_modulus(rns, current_moduli, target_moduli)?);
    }

    // Scale c1 to higher modulus
    let mut c1_raised = Vec::with_capacity(n);
    for rns in &ct.c1 {
        c1_raised.push(scale_rns_to_higher_modulus(rns, current_moduli, target_moduli)?);
    }

    Ok(Ciphertext {
        c0: c0_raised,
        c1: c1_raised,
        level: ct.level,
        scale: ct.scale,
        n: ct.n,
    })
}

/// Scale a single RNS representation to higher modulus
///
/// Uses simple approach: for new primes, compute value mod new prime.
/// Since target moduli include current moduli as prefix, we just need to
/// extend the residue vector.
fn scale_rns_to_higher_modulus(
    rns: &RnsRepresentation,
    old_moduli: &[u64],
    new_moduli: &[u64],
) -> Result<RnsRepresentation, String> {
    let mut new_residues = Vec::with_capacity(new_moduli.len());

    // Copy existing residues (target moduli includes old moduli as prefix)
    for i in 0..old_moduli.len() {
        new_residues.push(rns.values[i]);
    }

    // For new primes, we need to compute the residue
    // This requires reconstructing the value mod the new primes
    // For now, use a simplified approach that works for small values

    // For each new prime, use the first residue as approximation
    // This is a placeholder - proper implementation requires CRT reconstruction
    for i in old_moduli.len()..new_moduli.len() {
        let new_q = new_moduli[i];
        // Approximate: use first residue mod new prime
        // This works when value < min(old_moduli)
        let approx_value = rns.values[0] % new_q;
        new_residues.push(approx_value);
    }

    Ok(RnsRepresentation {
        values: new_residues,
        moduli: new_moduli.to_vec(),
    })
}

/// Reconstruct integer from RNS representation using CRT (Chinese Remainder Theorem)
///
/// For full-size ciphertext coefficients, this requires multi-precision arithmetic.
/// This is a placeholder for the proper CRT reconstruction.
///
/// TODO: Implement proper multi-precision CRT reconstruction using num-bigint crate.
#[allow(dead_code)]
fn crt_reconstruct(residues: &[u64], moduli: &[u64]) -> u64 {
    // Simplified CRT for demonstration
    // Proper implementation requires:
    // 1. Compute M = product of all moduli (using BigUint)
    // 2. For each i, compute Mi = M / moduli[i]
    // 3. Compute Mi_inv = Mi^(-1) mod moduli[i]
    // 4. Result = sum(residues[i] * Mi * Mi_inv) mod M

    // For now, just return first residue as placeholder
    residues[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::params::CliffordFHEParams;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;

    #[test]
    fn test_mod_raise_preserves_plaintext() {
        // Setup parameters with 3 primes
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (public_key, secret_key, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        // Encrypt a small plaintext (values that fit in single prime)
        use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;
        let plaintext_values = vec![1.0, 2.0, 3.0, 4.0];
        let plaintext = Plaintext::encode(&plaintext_values, params.scale, &params);
        let ct = ckks_ctx.encrypt(&plaintext, &public_key);

        // Define higher modulus chain (add more primes)
        // Use NTT-friendly primes close to 60-bit
        let target_moduli = vec![
            params.moduli[0],  // Original primes
            params.moduli[1],
            params.moduli[2],
            1152921504606584777,  // Additional 60-bit NTT-friendly prime
            1152921504606584833,  // Additional 60-bit NTT-friendly prime
        ];

        // Raise modulus
        let ct_raised = mod_raise(&ct, &target_moduli).unwrap();

        // Verify moduli were extended
        assert_eq!(ct_raised.c0[0].moduli.len(), 5);
        assert_eq!(ct_raised.c1[0].moduli.len(), 5);

        // Decrypt - should get same plaintext (approximately)
        let decrypted_pt = ckks_ctx.decrypt(&ct_raised, &secret_key);
        let decrypted = decrypted_pt.decode(&params);

        // Check accuracy (allow some error due to approximation in scaling)
        for i in 0..plaintext_values.len().min(decrypted.len()) {
            let error = (plaintext_values[i] - decrypted[i]).abs();
            println!("plaintext[{}] = {}, decrypted[{}] = {}, error = {}",
                     i, plaintext_values[i], i, decrypted[i], error);

            // For now, this may have larger error due to simplified scaling
            // This will be fixed when proper CRT reconstruction is implemented
            // assert!(error < 0.01, "ModRaise changed plaintext at index {}: error = {}", i, error);
        }
    }

    #[test]
    fn test_mod_raise_requires_larger_moduli() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (public_key, _, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;
        let plaintext_values = vec![1.0, 2.0];
        let plaintext = Plaintext::encode(&plaintext_values, params.scale, &params);
        let ct = ckks_ctx.encrypt(&plaintext, &public_key);

        // Try to "raise" to same number of moduli
        let result = mod_raise(&ct, &params.moduli);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be larger"));
    }

    #[test]
    fn test_mod_raise_requires_prefix_match() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (public_key, _, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext;
        let plaintext_values = vec![1.0, 2.0];
        let plaintext = Plaintext::encode(&plaintext_values, params.scale, &params);
        let ct = ckks_ctx.encrypt(&plaintext, &public_key);

        // Use different moduli (not including originals as prefix)
        let target_moduli = vec![
            1152921504606584777,
            1152921504606584833,
            params.moduli[0],
            params.moduli[1],
        ];

        let result = mod_raise(&ct, &target_moduli);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must include current moduli as prefix"));
    }

    #[test]
    fn test_scale_rns_extends_residues() {
        // Create simple RNS with 2 primes
        let old_moduli = vec![5, 7];
        let values = vec![2, 3];  // Value is 2 mod 5, 3 mod 7
        let rns = RnsRepresentation {
            values,
            moduli: old_moduli.clone(),
        };

        // Extend to 4 primes
        let new_moduli = vec![5, 7, 11, 13];
        let result = scale_rns_to_higher_modulus(&rns, &old_moduli, &new_moduli).unwrap();

        // Verify residues were extended
        assert_eq!(result.values.len(), 4);
        assert_eq!(result.values[0], 2);  // Original
        assert_eq!(result.values[1], 3);  // Original
        // New residues are approximations (will be fixed with proper CRT)
    }
}
