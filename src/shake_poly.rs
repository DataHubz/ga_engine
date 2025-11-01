//! Fast polynomial generation using SHAKE128
//!
//! **Key insight**: Generate ONE seed, expand to all coefficients at once
//! This is the correct Kyber-style approach

use sha3::{Shake128, digest::{Update, ExtendableOutput, XofReader}};
use crate::clifford_ring_int::{CliffordPolynomialInt, CliffordRingElementInt};
use rand::RngCore;

/// Generate discrete Clifford polynomial using SHAKE128
///
/// **Input**: One 32-byte seed
/// **Output**: N Clifford ring elements with ternary components {-1, 0, 1}
///
/// **Approach**: Expand seed to N×8 ternary values at once
pub fn discrete_poly_shake(seed: &[u8; 32], n: usize) -> CliffordPolynomialInt {
    let mut shake = Shake128::default();
    shake.update(seed);
    let mut reader = shake.finalize_xof();

    let total_values = n * 8;  // N coefficients × 8 components each
    let mut values = Vec::with_capacity(total_values);

    let mut byte_buffer = [0u8; 1];

    while values.len() < total_values {
        reader.read(&mut byte_buffer);
        let byte = byte_buffer[0];

        // Process 4 samples per byte (2 bits each)
        for shift in (0..8).step_by(2) {
            if values.len() >= total_values {
                break;
            }

            let sample = (byte >> shift) & 0b11;
            match sample {
                0b00 => values.push(-1),
                0b01 => values.push(0),
                0b10 => values.push(1),
                _ => {} // 0b11 → reject, resample
            }
        }
    }

    // Pack into Clifford polynomial
    let mut coeffs = Vec::with_capacity(n);
    for i in 0..n {
        let mut mv = [0i64; 8];
        for j in 0..8 {
            mv[j] = values[i * 8 + j];
        }
        coeffs.push(CliffordRingElementInt::from_multivector(mv));
    }

    CliffordPolynomialInt::new(coeffs)
}

/// Generate error Clifford polynomial using SHAKE128
///
/// **Input**: One 32-byte seed
/// **Output**: N Clifford ring elements with small error components
pub fn error_poly_shake(seed: &[u8; 32], n: usize, bound: i64) -> CliffordPolynomialInt {
    let mut shake = Shake128::default();
    shake.update(seed);
    let mut reader = shake.finalize_xof();

    let total_values = n * 8;
    let mut values = Vec::with_capacity(total_values);

    let mut byte_buffer = [0u8; 1];
    let mut bit_buffer = 0u16;
    let mut bits_available = 0;

    let num_values = (2 * bound + 1) as u16;

    while values.len() < total_values {
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
            values.push((sample as i64) - bound);
        }
    }

    // Pack into Clifford polynomial
    let mut coeffs = Vec::with_capacity(n);
    for i in 0..n {
        let mut mv = [0i64; 8];
        for j in 0..8 {
            mv[j] = values[i * 8 + j];
        }
        coeffs.push(CliffordRingElementInt::from_multivector(mv));
    }

    CliffordPolynomialInt::new(coeffs)
}

/// Generate a random 32-byte seed (call once per polynomial)
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
    fn test_discrete_poly() {
        let seed = [42u8; 32];
        let poly = discrete_poly_shake(&seed, 32);

        assert_eq!(poly.coeffs.len(), 32);

        // All coefficients should be ternary
        for coeff in &poly.coeffs {
            for &c in &coeff.coeffs {
                assert!(c >= -1 && c <= 1);
            }
        }
    }

    #[test]
    fn test_error_poly() {
        let seed = [99u8; 32];
        let poly = error_poly_shake(&seed, 32, 2);

        assert_eq!(poly.coeffs.len(), 32);

        // All coefficients should be in {-2, -1, 0, 1, 2}
        for coeff in &poly.coeffs {
            for &c in &coeff.coeffs {
                assert!(c >= -2 && c <= 2);
            }
        }
    }

    #[test]
    fn test_deterministic() {
        let seed = [123u8; 32];

        let poly1 = discrete_poly_shake(&seed, 32);
        let poly2 = discrete_poly_shake(&seed, 32);

        // Same seed should produce same polynomial
        for i in 0..32 {
            assert_eq!(poly1.coeffs[i].coeffs, poly2.coeffs[i].coeffs);
        }
    }
}
