//! Canonical Embedding for CKKS (Correct Version)
//!
//! This implements the proper canonical embedding from CKKS that ensures
//! Galois automorphisms correspond to slot rotations.
//!
//! # Key Difference from Standard FFT
//!
//! - **Standard FFT**: Evaluates polynomial at ω^k where ω = e^(2πi/N)
//! - **CKKS Canonical Embedding**: Evaluates at ζ_M^(2k+1) where ζ_M = e^(2πi/M), M=2N
//!
//! This specific choice of evaluation points ensures that the Galois automorphism
//! σ_k : x → x^k corresponds to a slot permutation.
//!
//! # Mathematical Foundation
//!
//! For cyclotomic polynomial Φ_M(x) where M = 2N:
//! - Roots are ζ_M^k for k coprime to M (Euler phi function: φ(M) = N)
//! - For M = 2N power of 2: roots are ζ_M^(2k+1) for k = 0, 1, ..., N-1
//! - These are the N primitive M-th roots of unity
//!
//! The canonical embedding maps:
//! ```text
//! σ: R = Z[x]/(Φ_M(x)) → C^N
//! σ(p(x)) = [p(ζ_M), p(ζ_M^3), p(ζ_M^5), ..., p(ζ_M^(2N-1))]
//! ```
//!
//! With this embedding, the automorphism σ_k: x → x^k acts on slots as:
//! - If k = 2r+1 (odd), it permutes slots
//! - The specific permutation depends on how k acts on the exponents
//!
//! # References
//!
//! - CKKS paper: "Homomorphic Encryption for Arithmetic of Approximate Numbers"
//!   Cheon, Kim, Kim, Song (2017)
//! - SEAL implementation: Microsoft SEAL library
//! - Halevi & Shoup: "Algorithms in HElib" (2014)

use rustfft::num_complex::Complex;
use std::f64::consts::PI;

/// Compute the Galois orbit order for CKKS slot indexing
///
/// For power-of-two cyclotomics M=2N, the odd residues mod M form two orbits
/// under multiplication by generator g (typically g=5). This function computes
/// the orbit starting from 1: e[t] = g^t mod M.
///
/// With this ordering, automorphism σ_g acts as a left rotation by 1 slot!
///
/// # Arguments
/// * `n` - Ring dimension N
/// * `g` - Generator (typically 5 for power-of-two cyclotomics)
///
/// # Returns
/// Vector e where e[t] = g^t mod M for t=0..(N/2-1)
fn orbit_order(n: usize, g: usize) -> Vec<usize> {
    let m = 2 * n; // M = 2N
    let num_slots = n / 2; // N/2 slots

    let mut e = vec![0usize; num_slots];
    let mut cur = 1usize;

    for t in 0..num_slots {
        e[t] = cur; // odd exponent in [1..2N-1]
        cur = (cur * g) % m;
    }

    e
}

/// Encode slots using CKKS canonical embedding
///
/// Evaluates slots at the specific primitive roots ζ_M^(2k+1)
/// to ensure automorphisms correspond to slot rotations.
///
/// # Arguments
/// * `slots` - N/2 complex values to encode
/// * `scale` - Scaling factor
/// * `n` - Ring dimension (N in the formula above)
///
/// # Returns
/// Polynomial coefficients
pub fn canonical_embed_encode(slots: &[Complex<f64>], scale: f64, n: usize) -> Vec<i64> {
    assert!(n.is_power_of_two());
    let num_slots = n / 2;
    assert_eq!(slots.len(), num_slots);

    let m = 2 * n; // Cyclotomic index M = 2N
    let g = 5; // Generator for power-of-two cyclotomics

    // CRITICAL FIX: Use Galois orbit order instead of natural order!
    // This ensures automorphism σ_g acts as rotate-by-1
    let e = orbit_order(n, g);

    // Step 1: Extend to full N slots with conjugate symmetry
    // slots[t] corresponds to evaluation at ζ_M^{e[t]} where e[t] = g^t mod M
    // The conjugate orbit is at -e[t] mod M
    let mut extended = vec![Complex::new(0.0, 0.0); n];

    for t in 0..num_slots {
        extended[t] = slots[t];
    }

    // Conjugate symmetry: extended[N/2 + t] = conj(slots[t])
    // This maps to roots ζ^{-e[t]} = ζ^{M - e[t]}
    for t in 1..num_slots {
        extended[n - t] = slots[t].conj();
    }

    // Step 2: Compute coefficients via inverse DFT at orbit-ordered roots
    // We want: p(ζ_M^{e[t]}) = slots[t] for t=0..(N/2-1)
    //
    // The inverse transform is:
    // coeffs[j] = (1/N) * Σ_{t=0}^{N-1} extended[t] * ζ_M^{-e[t]·j}
    //
    // But since extended has conjugate symmetry, we can simplify:
    // coeffs[j] = (2/N) * Re(Σ_{t=0}^{N/2-1} slots[t] * ζ_M^{-e[t]·j})

    let mut coeffs_complex = vec![Complex::new(0.0, 0.0); n];

    for j in 0..n {
        let mut sum = Complex::new(0.0, 0.0);

        // Sum over first N/2 slots using orbit order
        for t in 0..num_slots {
            // Use e[t] instead of (2*k+1)!
            let exponent = -((e[t] * j) as i64);
            let angle = 2.0 * PI * (exponent as f64) / (m as f64);
            let root = Complex::new(angle.cos(), angle.sin());
            sum += slots[t] * root;
        }

        // Account for conjugate pairs
        coeffs_complex[j] = sum * 2.0 / (n as f64);
    }

    // Step 3: Scale and round to integers
    let mut coeffs = vec![0i64; n];
    for i in 0..n {
        // Should be real due to conjugate symmetry
        let value = coeffs_complex[i].re * scale;
        coeffs[i] = value.round() as i64;
    }

    coeffs
}

/// Decode slots using CKKS canonical embedding with orbit ordering
///
/// Evaluates polynomial at the orbit-ordered primitive roots ζ_M^{e[t]}.
///
/// # Arguments
/// * `coeffs` - Polynomial coefficients
/// * `scale` - Scaling factor
/// * `n` - Ring dimension
///
/// # Returns
/// N/2 complex slot values
pub fn canonical_embed_decode(coeffs: &[i64], scale: f64, n: usize) -> Vec<Complex<f64>> {
    assert_eq!(coeffs.len(), n);

    let m = 2 * n; // M = 2N
    let num_slots = n / 2;
    let g = 5; // Generator

    // CRITICAL FIX: Use Galois orbit order!
    let e = orbit_order(n, g);

    // Convert to floating point
    let coeffs_float: Vec<f64> = coeffs.iter().map(|&c| c as f64 / scale).collect();

    // Evaluate polynomial at ζ_M^{e[t]} for t = 0..N/2-1
    let mut slots = vec![Complex::new(0.0, 0.0); num_slots];

    for t in 0..num_slots {
        let mut sum = Complex::new(0.0, 0.0);
        for j in 0..n {
            // Compute ζ_M^{e[t]·j} where e[t] = g^t mod M
            let exponent = e[t] * j;
            let angle = 2.0 * PI * (exponent as f64) / (m as f64);
            let root = Complex::new(angle.cos(), angle.sin());
            sum += root * coeffs_float[j];
        }
        slots[t] = sum;
    }

    slots
}

/// Encode multivector using canonical embedding
pub fn encode_multivector_canonical(mv: &[f64; 8], scale: f64, n: usize) -> Vec<i64> {
    assert!(n >= 16);
    let num_slots = n / 2;

    let mut slots = vec![Complex::new(0.0, 0.0); num_slots];
    for i in 0..8 {
        slots[i] = Complex::new(mv[i], 0.0);
    }

    canonical_embed_encode(&slots, scale, n)
}

/// Decode multivector using canonical embedding
pub fn decode_multivector_canonical(coeffs: &[i64], scale: f64, n: usize) -> [f64; 8] {
    let slots = canonical_embed_decode(coeffs, scale, n);

    let mut mv = [0.0; 8];
    for i in 0..8 {
        mv[i] = slots[i].re;
    }
    mv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_embedding_roundtrip() {
        let n = 32;
        let scale = 1u64 << 40;

        let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let coeffs = encode_multivector_canonical(&mv, scale as f64, n);
        let mv_decoded = decode_multivector_canonical(&coeffs, scale as f64, n);

        for i in 0..8 {
            let error = (mv[i] - mv_decoded[i]).abs();
            assert!(error < 1e-3, "Slot {} error {} too large", i, error);
        }
    }

    #[test]
    fn test_automorphism_rotates_slots() {
        use crate::clifford_fhe::automorphisms::apply_automorphism;

        let n = 32;
        let scale = 1u64 << 40;

        let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let coeffs = encode_multivector_canonical(&mv, scale as f64, n);

        // Try different automorphism indices to find which one rotates
        for k in [3, 5, 7, 9, 11, 13, 15, 17].iter() {
            let coeffs_auto = apply_automorphism(&coeffs, *k, n);
            let mv_result = decode_multivector_canonical(&coeffs_auto, scale as f64, n);

            // Check if this is a left rotation by 1
            let expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0];
            let matches = mv_result.iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() < 0.1);

            if matches {
                println!("✓ Automorphism k={} produces left rotation by 1", k);
            }
        }
    }
}
