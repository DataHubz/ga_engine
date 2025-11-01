//! SIMD-batched NTT for Clifford ring polynomials
//!
//! This module extends NTT to Clifford ring polynomials using SIMD batching.
//! Processes multiple components in parallel for better performance.
//!
//! Expected speedup: 1.3-1.8× over standard component-wise NTT
//! Target: 3 µs savings → 22.86 µs → ~20 µs

use crate::clifford_ring_int::{CliffordPolynomialInt, CliffordRingElementInt};
use crate::lazy_reduction::LazyReductionContext;
use crate::ntt::NTTContext;
use crate::ntt_simd::{ntt_forward_batch2, ntt_inverse_batch2};

/// Multiply two Clifford polynomials using SIMD-batched NTT
///
/// Key optimization: Process components in pairs using SIMD!
/// - Components 0-1: Batched SIMD
/// - Components 2-3: Batched SIMD
/// - Components 4-5: Batched SIMD
/// - Components 6-7: Batched SIMD
///
/// This gives us 4 batched operations instead of 8 sequential ones.
pub fn multiply_ntt_simd(
    a: &CliffordPolynomialInt,
    b: &CliffordPolynomialInt,
    ntt: &NTTContext,
    lazy: &LazyReductionContext,
) -> CliffordPolynomialInt {
    let n = ntt.n;
    assert_eq!(a.coeffs.len(), n, "Polynomial a length must equal NTT size");
    assert_eq!(b.coeffs.len(), n, "Polynomial b length must equal NTT size");

    // Step 1: Extract components and apply forward NTT (SIMD-batched!)
    let mut a_ntt = vec![vec![0i64; n]; 8];
    let mut b_ntt = vec![vec![0i64; n]; 8];

    // Extract all components first
    for component in 0..8 {
        for i in 0..n {
            a_ntt[component][i] = a.coeffs[i].coeffs[component];
            b_ntt[component][i] = b.coeffs[i].coeffs[component];
        }
    }

    // Forward NTT: Process in pairs using SIMD!
    // Process A components
    for i in (0..8).step_by(2) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let (first, second) = a_ntt.split_at_mut(i + 1);
            ntt_forward_batch2(&mut first[i], &mut second[0], ntt);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let (first, second) = a_ntt.split_at_mut(i + 1);
            ntt_forward_batch2(&mut first[i], &mut second[0], ntt);
        }
    }

    // Process B components
    for i in (0..8).step_by(2) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let (first, second) = b_ntt.split_at_mut(i + 1);
            ntt_forward_batch2(&mut first[i], &mut second[0], ntt);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let (first, second) = b_ntt.split_at_mut(i + 1);
            ntt_forward_batch2(&mut first[i], &mut second[0], ntt);
        }
    }

    // Step 2: Point-wise geometric product in frequency domain
    let mut c_ntt = vec![vec![0i64; n]; 8];

    // Pre-allocate result buffer for in-place operations
    let mut c_elem = CliffordRingElementInt::zero();

    for k in 0..n {
        // At frequency index k, compute geometric product
        let a_elem = CliffordRingElementInt::from_multivector([
            a_ntt[0][k], a_ntt[1][k], a_ntt[2][k], a_ntt[3][k],
            a_ntt[4][k], a_ntt[5][k], a_ntt[6][k], a_ntt[7][k],
        ]);

        let b_elem = CliffordRingElementInt::from_multivector([
            b_ntt[0][k], b_ntt[1][k], b_ntt[2][k], b_ntt[3][k],
            b_ntt[4][k], b_ntt[5][k], b_ntt[6][k], b_ntt[7][k],
        ]);

        // Geometric product
        a_elem.geometric_product_lazy_inplace(&b_elem, lazy, &mut c_elem);

        // Store result components
        for component in 0..8 {
            c_ntt[component][k] = c_elem.coeffs[component];
        }
    }

    // Step 3: Inverse NTT for each component (SIMD-batched!)
    for i in (0..8).step_by(2) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let (first, second) = c_ntt.split_at_mut(i + 1);
            ntt_inverse_batch2(&mut first[i], &mut second[0], ntt);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let (first, second) = c_ntt.split_at_mut(i + 1);
            ntt_inverse_batch2(&mut first[i], &mut second[0], ntt);
        }
    }

    // Step 4: Reconstruct Clifford polynomial from components
    let mut result_coeffs = Vec::with_capacity(n);
    for i in 0..n {
        let elem = CliffordRingElementInt::from_multivector([
            c_ntt[0][i], c_ntt[1][i], c_ntt[2][i], c_ntt[3][i],
            c_ntt[4][i], c_ntt[5][i], c_ntt[6][i], c_ntt[7][i],
        ]);
        result_coeffs.push(elem);
    }

    CliffordPolynomialInt::new(result_coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_clifford_ntt_multiply() {
        let ntt = NTTContext::new_clifford_lwe();
        let lazy = LazyReductionContext::new(3329);

        // Create two simple Clifford polynomials
        // a(x) = 1 + e1·x
        let mut a_coeffs = vec![CliffordRingElementInt::zero(); 32];
        a_coeffs[0] = CliffordRingElementInt::from_multivector([1, 0, 0, 0, 0, 0, 0, 0]); // 1
        a_coeffs[1] = CliffordRingElementInt::from_multivector([0, 1, 0, 0, 0, 0, 0, 0]); // e1
        let a = CliffordPolynomialInt::new(a_coeffs);

        // b(x) = 2 + e2·x
        let mut b_coeffs = vec![CliffordRingElementInt::zero(); 32];
        b_coeffs[0] = CliffordRingElementInt::from_multivector([2, 0, 0, 0, 0, 0, 0, 0]); // 2
        b_coeffs[1] = CliffordRingElementInt::from_multivector([0, 0, 1, 0, 0, 0, 0, 0]); // e2
        let b = CliffordPolynomialInt::new(b_coeffs);

        // SIMD NTT multiplication
        let mut result_simd = multiply_ntt_simd(&a, &b, &ntt, &lazy);

        // Reference: Standard NTT multiplication
        use crate::ntt_clifford::multiply_ntt;
        let mut result_standard = multiply_ntt(&a, &b, &ntt, &lazy);

        // Both should have same length after reduction
        result_simd.reduce_modulo_xn_minus_1_lazy(32, &lazy);
        result_standard.reduce_modulo_xn_minus_1_lazy(32, &lazy);

        // Compare results
        assert_eq!(result_simd.coeffs.len(), result_standard.coeffs.len());

        for i in 0..32 {
            for j in 0..8 {
                assert_eq!(
                    result_simd.coeffs[i].coeffs[j],
                    result_standard.coeffs[i].coeffs[j],
                    "Mismatch at coeff[{}].coeffs[{}]: SIMD={}, Standard={}",
                    i,
                    j,
                    result_simd.coeffs[i].coeffs[j],
                    result_standard.coeffs[i].coeffs[j]
                );
            }
        }
    }
}
