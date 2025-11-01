//! SIMD-accelerated NTT for batched component processing
//!
//! This module provides SIMD-optimized NTT that processes multiple Clifford
//! components in parallel using ARM NEON intrinsics.
//!
//! Key optimization: Process 2 NTT operations simultaneously using int64x2_t.
//!
//! Expected speedup: 1.3-1.8× on NTT operations (40% of total time)
//! Target: 3 µs savings → 22.86 µs → ~20 µs

use crate::ntt::NTTContext;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Batched forward NTT for 2 components using SIMD
///
/// Processes two NTT transformations in parallel using ARM NEON int64x2_t.
///
/// # Safety
/// Uses unsafe SIMD intrinsics. Caller must ensure:
/// - `a` and `b` have length == ntt.n
/// - Platform supports ARM NEON (aarch64)
#[cfg(target_arch = "aarch64")]
pub unsafe fn ntt_forward_batch2(
    a: &mut [i64],
    b: &mut [i64],
    ntt: &NTTContext,
) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, ntt.n);
    let q = ntt.q;

    // Bit-reversal permutation (scalar - not worth SIMD for swaps)
    bit_reverse_permutation(a);
    bit_reverse_permutation(b);

    // Cooley-Tukey butterfly with SIMD
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = 2 * n / len;

        for start in (0..n).step_by(len) {
            let mut k = 0;
            for j in start..(start + half) {
                // Load pairs of values into SIMD registers
                let u_ab = vld1q_s64([a[j], b[j]].as_ptr());

                // Multiply by twiddle factor
                let twiddle = ntt.psi[k];
                let v_a = (a[j + half] * twiddle) % q;
                let v_b = (b[j + half] * twiddle) % q;
                let v_ab = vld1q_s64([v_a, v_b].as_ptr());

                // Butterfly: u + v and u - v (in parallel for both a and b!)
                let sum_ab = vaddq_s64(u_ab, v_ab);
                let diff_ab = vsubq_s64(u_ab, v_ab);

                // Store results (need to do modular reduction)
                let mut sum_out = [0i64; 2];
                let mut diff_out = [0i64; 2];
                vst1q_s64(sum_out.as_mut_ptr(), sum_ab);
                vst1q_s64(diff_out.as_mut_ptr(), diff_ab);

                a[j] = ((sum_out[0] % q) + q) % q;
                b[j] = ((sum_out[1] % q) + q) % q;
                a[j + half] = ((diff_out[0] % q) + q) % q;
                b[j + half] = ((diff_out[1] % q) + q) % q;

                k += step;
            }
        }

        len *= 2;
    }
}

/// Batched inverse NTT for 2 components using SIMD
#[cfg(target_arch = "aarch64")]
pub unsafe fn ntt_inverse_batch2(
    a: &mut [i64],
    b: &mut [i64],
    ntt: &NTTContext,
) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, ntt.n);
    let q = ntt.q;

    // Bit-reversal permutation
    bit_reverse_permutation(a);
    bit_reverse_permutation(b);

    // Inverse butterfly with SIMD
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = 2 * n / len;

        for start in (0..n).step_by(len) {
            let mut k = 0;
            for j in start..(start + half) {
                // Load pairs
                let u_ab = vld1q_s64([a[j], b[j]].as_ptr());

                // Multiply by inverse twiddle factor
                let twiddle_inv = ntt.psi_inv[k];
                let v_a = (a[j + half] * twiddle_inv) % q;
                let v_b = (b[j + half] * twiddle_inv) % q;
                let v_ab = vld1q_s64([v_a, v_b].as_ptr());

                // Butterfly
                let sum_ab = vaddq_s64(u_ab, v_ab);
                let diff_ab = vsubq_s64(u_ab, v_ab);

                // Store with modular reduction
                let mut sum_out = [0i64; 2];
                let mut diff_out = [0i64; 2];
                vst1q_s64(sum_out.as_mut_ptr(), sum_ab);
                vst1q_s64(diff_out.as_mut_ptr(), diff_ab);

                a[j] = ((sum_out[0] % q) + q) % q;
                b[j] = ((sum_out[1] % q) + q) % q;
                a[j + half] = ((diff_out[0] % q) + q) % q;
                b[j + half] = ((diff_out[1] % q) + q) % q;

                k += step;
            }
        }

        len *= 2;
    }

    // Normalize by N^(-1) (in parallel)
    let n_inv = ntt.n_inv;
    for i in 0..n {
        let prod_ab = vld1q_s64([a[i] * n_inv, b[i] * n_inv].as_ptr());
        let mut out = [0i64; 2];
        vst1q_s64(out.as_mut_ptr(), prod_ab);

        a[i] = ((out[0] % q) + q) % q;
        b[i] = ((out[1] % q) + q) % q;
    }
}

// Helper function for bit-reversal (scalar version - not SIMD worth it)
fn bit_reverse_permutation(a: &mut [i64]) {
    let n = a.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j >= bit {
            j -= bit;
            bit >>= 1;
        }
        j += bit;

        if i < j {
            a.swap(i, j);
        }
    }
}

// Fallback for non-ARM platforms
#[cfg(not(target_arch = "aarch64"))]
pub fn ntt_forward_batch2(
    a: &mut [i64],
    b: &mut [i64],
    ntt: &NTTContext,
) {
    // Fallback: just call standard NTT twice
    ntt.forward(a);
    ntt.forward(b);
}

#[cfg(not(target_arch = "aarch64"))]
pub fn ntt_inverse_batch2(
    a: &mut [i64],
    b: &mut [i64],
    ntt: &NTTContext,
) {
    // Fallback: just call standard NTT twice
    ntt.inverse(a);
    ntt.inverse(b);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_ntt_equivalence() {
        let ntt = NTTContext::new_clifford_lwe();

        // Test data
        let mut a1 = vec![1, 2, 3, 5, 7, 11, 13, 17, 0, 0, 0, 0, 0, 0, 0, 0,
                          19, 23, 29, 31, 37, 41, 43, 47, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut a2 = a1.clone();

        let mut b1 = vec![2, 3, 5, 7, 11, 13, 17, 19, 0, 0, 0, 0, 0, 0, 0, 0,
                          23, 29, 31, 37, 41, 43, 47, 53, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut b2 = b1.clone();

        // Standard NTT (reference)
        ntt.forward(&mut a1);
        ntt.forward(&mut b1);

        // Batched SIMD NTT
        #[cfg(target_arch = "aarch64")]
        unsafe {
            ntt_forward_batch2(&mut a2, &mut b2, &ntt);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            ntt_forward_batch2(&mut a2, &mut b2, &ntt);
        }

        // Compare results
        for i in 0..32 {
            assert_eq!(a1[i], a2[i], "Mismatch in component A at index {}", i);
            assert_eq!(b1[i], b2[i], "Mismatch in component B at index {}", i);
        }
    }

    #[test]
    fn test_simd_ntt_roundtrip() {
        let ntt = NTTContext::new_clifford_lwe();

        let a_orig = vec![1, 2, 3, 5, 7, 11, 13, 17, 0, 0, 0, 0, 0, 0, 0, 0,
                          19, 23, 29, 31, 37, 41, 43, 47, 0, 0, 0, 0, 0, 0, 0, 0];
        let b_orig = vec![2, 3, 5, 7, 11, 13, 17, 19, 0, 0, 0, 0, 0, 0, 0, 0,
                          23, 29, 31, 37, 41, 43, 47, 53, 0, 0, 0, 0, 0, 0, 0, 0];

        let mut a = a_orig.clone();
        let mut b = b_orig.clone();

        // Forward + Inverse (SIMD)
        #[cfg(target_arch = "aarch64")]
        unsafe {
            ntt_forward_batch2(&mut a, &mut b, &ntt);
            ntt_inverse_batch2(&mut a, &mut b, &ntt);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            ntt_forward_batch2(&mut a, &mut b, &ntt);
            ntt_inverse_batch2(&mut a, &mut b, &ntt);
        }

        // Should match original
        for i in 0..32 {
            assert_eq!(a[i], a_orig[i], "Round-trip failed for A at index {}", i);
            assert_eq!(b[i], b_orig[i], "Round-trip failed for B at index {}", i);
        }
    }
}
