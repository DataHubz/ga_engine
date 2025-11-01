//! Deterministic RNG using SHAKE128 for fast polynomial generation
//!
//! **Key optimization**: Instead of generating random values one-by-one,
//! expand a seed using SHAKE128 to generate all polynomial coefficients at once.
//!
//! **Expected speedup**: 2-3× over rand::thread_rng() (RNG is ~30% of encryption time)
//! **Time savings**: ~4 µs per encryption
//!
//! **Approach** (Kyber-style):
//! 1. Generate one 32-byte random seed
//! 2. Expand seed using SHAKE128 to desired length
//! 3. Parse bytes into ternary {-1, 0, 1} or small error values
//!
//! This is the same approach used by Kyber-512 for deterministic polynomial generation.

use sha3::{Shake128, digest::{Update, ExtendableOutput, XofReader}};
use rand::{Rng, RngCore};

/// Sample a ternary polynomial from SHAKE128
///
/// **Input**: 32-byte seed
/// **Output**: N coefficients, each in {-1, 0, 1}
///
/// **Sampling method**: Rejection sampling
/// - Read 2 bits per coefficient
/// - 00 → -1, 01 → 0, 10 → 1, 11 → reject (resample)
/// - Expected 1.33 bytes per coefficient (due to rejection)
pub fn sample_ternary_shake128(seed: &[u8; 32], n: usize) -> Vec<i64> {
    let mut shake = Shake128::default();
    shake.update(seed);
    let mut reader = shake.finalize_xof();

    let mut coeffs = Vec::with_capacity(n);
    let mut byte_buffer = [0u8; 1];

    let mut i = 0;
    while i < n {
        reader.read(&mut byte_buffer);
        let byte = byte_buffer[0];

        // Process 4 samples per byte (2 bits each)
        for shift in (0..8).step_by(2) {
            if i >= n {
                break;
            }

            let sample = (byte >> shift) & 0b11;
            match sample {
                0b00 => {
                    coeffs.push(-1);
                    i += 1;
                }
                0b01 => {
                    coeffs.push(0);
                    i += 1;
                }
                0b10 => {
                    coeffs.push(1);
                    i += 1;
                }
                _ => {
                    // 0b11 → Reject and resample (will get from next bits)
                }
            }
        }
    }

    coeffs
}

/// Sample a small error polynomial from SHAKE128
///
/// **Input**: 32-byte seed
/// **Output**: N coefficients, each in {-bound, ..., bound}
///
/// **For bound=2**: coefficients in {-2, -1, 0, 1, 2}
///
/// **Sampling method**: 3 bits per coefficient
/// - 000 → -2, 001 → -1, 010 → 0, 011 → 1, 100 → 2
/// - 101, 110, 111 → reject (resample)
pub fn sample_small_error_shake128(seed: &[u8; 32], n: usize, bound: i64) -> Vec<i64> {
    let mut shake = Shake128::default();
    shake.update(seed);
    let mut reader = shake.finalize_xof();

    let mut coeffs = Vec::with_capacity(n);
    let mut byte_buffer = [0u8; 1];
    let mut bit_buffer = 0u16;
    let mut bits_available = 0;

    let num_values = (2 * bound + 1) as u16;

    while coeffs.len() < n {
        // Ensure we have at least 3 bits
        while bits_available < 3 {
            reader.read(&mut byte_buffer);
            bit_buffer |= (byte_buffer[0] as u16) << bits_available;
            bits_available += 8;
        }

        // Extract 3 bits
        let sample = bit_buffer & 0b111;
        bit_buffer >>= 3;
        bits_available -= 3;

        // Accept if sample < num_values
        if (sample as i64) < num_values as i64 {
            coeffs.push((sample as i64) - bound);
        }
        // Otherwise reject and resample
    }

    coeffs
}

/// Sample a uniform polynomial from SHAKE128
///
/// **Input**: 32-byte seed
/// **Output**: N coefficients, each in [0, q)
///
/// **Sampling method**: Rejection sampling from 16-bit values
pub fn sample_uniform_shake128(seed: &[u8; 32], n: usize, q: i64) -> Vec<i64> {
    let mut shake = Shake128::default();
    shake.update(seed);
    let mut reader = shake.finalize_xof();

    let mut coeffs = Vec::with_capacity(n);
    let mut byte_buffer = [0u8; 2];

    while coeffs.len() < n {
        reader.read(&mut byte_buffer);
        let value = u16::from_le_bytes(byte_buffer) as i64;

        // Accept if value < q
        if value < q {
            coeffs.push(value);
        }
    }

    coeffs
}

/// Generate a random 32-byte seed
///
/// Uses system RNG to generate one seed - much faster than calling RNG N times
#[inline]
pub fn generate_seed() -> [u8; 32] {
    let mut seed = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut seed);
    seed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_sampling() {
        let seed = [42u8; 32];
        let coeffs = sample_ternary_shake128(&seed, 32);

        assert_eq!(coeffs.len(), 32);

        // All coefficients should be in {-1, 0, 1}
        for &c in &coeffs {
            assert!(c >= -1 && c <= 1, "Coefficient {} not in {{-1,0,1}}", c);
        }
    }

    #[test]
    fn test_small_error_sampling() {
        let seed = [99u8; 32];
        let coeffs = sample_small_error_shake128(&seed, 32, 2);

        assert_eq!(coeffs.len(), 32);

        // All coefficients should be in {-2, -1, 0, 1, 2}
        for &c in &coeffs {
            assert!(c >= -2 && c <= 2, "Coefficient {} not in {{-2,-1,0,1,2}}", c);
        }
    }

    #[test]
    fn test_deterministic() {
        let seed = [123u8; 32];

        let coeffs1 = sample_ternary_shake128(&seed, 32);
        let coeffs2 = sample_ternary_shake128(&seed, 32);

        // Same seed should produce same coefficients
        assert_eq!(coeffs1, coeffs2);
    }

    #[test]
    fn test_different_seeds() {
        let seed1 = [1u8; 32];
        let seed2 = [2u8; 32];

        let coeffs1 = sample_ternary_shake128(&seed1, 32);
        let coeffs2 = sample_ternary_shake128(&seed2, 32);

        // Different seeds should (almost certainly) produce different coefficients
        assert_ne!(coeffs1, coeffs2);
    }

    #[test]
    fn test_uniform_sampling() {
        let seed = [77u8; 32];
        let q = 3329;
        let coeffs = sample_uniform_shake128(&seed, 32, q);

        assert_eq!(coeffs.len(), 32);

        // All coefficients should be in [0, q)
        for &c in &coeffs {
            assert!(c >= 0 && c < q, "Coefficient {} not in [0, {})", c, q);
        }
    }
}
