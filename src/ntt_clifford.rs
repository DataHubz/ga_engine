//! NTT for Clifford Ring Polynomials
//!
//! Extends NTT to work with polynomials whose coefficients are Clifford algebra elements.
//!
//! **Approach**: Component-wise NTT
//! 1. Each Clifford element has 8 components (scalar, e1, e2, e3, e12, e13, e23, e123)
//! 2. Apply NTT to each component independently
//! 3. In frequency domain, do point-wise geometric products
//! 4. Apply inverse NTT to each component

use crate::clifford_ring_int::{CliffordPolynomialInt, CliffordRingElementInt};
use crate::ntt::NTTContext;
use crate::lazy_reduction::LazyReductionContext;

/// Multiply two Clifford polynomials using NTT
///
/// **Algorithm**:
/// 1. Transform each polynomial's components to frequency domain
/// 2. Point-wise multiply using geometric product
/// 3. Transform back to coefficient domain
///
/// **Complexity**: O(8 × N log N + 8 × N × 64) = O(N log N) for geometric products
/// vs O(N^1.585 × 64) for Karatsuba
pub fn multiply_ntt(
    a: &CliffordPolynomialInt,
    b: &CliffordPolynomialInt,
    ntt: &NTTContext,
    lazy: &LazyReductionContext,
) -> CliffordPolynomialInt {
    let n = ntt.n;
    assert_eq!(a.coeffs.len(), n, "Polynomial a length must equal NTT size");
    assert_eq!(b.coeffs.len(), n, "Polynomial b length must equal NTT size");

    // Step 1: Extract components and apply forward NTT
    let mut a_ntt = vec![vec![0i64; n]; 8]; // 8 components, each with N frequency coeffs
    let mut b_ntt = vec![vec![0i64; n]; 8];

    for component in 0..8 {
        // Extract component from all polynomial coefficients
        for i in 0..n {
            a_ntt[component][i] = a.coeffs[i].coeffs[component];
            b_ntt[component][i] = b.coeffs[i].coeffs[component];
        }

        // Forward NTT for this component
        ntt.forward(&mut a_ntt[component]);
        ntt.forward(&mut b_ntt[component]);
    }

    // Step 2: Point-wise geometric product in frequency domain
    let mut c_ntt = vec![vec![0i64; n]; 8];

    for k in 0..n {
        // At frequency index k, compute geometric product of a_ntt[k] ⊗ b_ntt[k]
        let a_elem = CliffordRingElementInt::from_multivector([
            a_ntt[0][k], a_ntt[1][k], a_ntt[2][k], a_ntt[3][k],
            a_ntt[4][k], a_ntt[5][k], a_ntt[6][k], a_ntt[7][k],
        ]);

        let b_elem = CliffordRingElementInt::from_multivector([
            b_ntt[0][k], b_ntt[1][k], b_ntt[2][k], b_ntt[3][k],
            b_ntt[4][k], b_ntt[5][k], b_ntt[6][k], b_ntt[7][k],
        ]);

        // Geometric product in frequency domain
        let c_elem = a_elem.geometric_product_lazy(&b_elem, lazy);

        // Store result components
        for component in 0..8 {
            c_ntt[component][k] = c_elem.coeffs[component];
        }
    }

    // Step 3: Inverse NTT for each component
    for component in 0..8 {
        ntt.inverse(&mut c_ntt[component]);
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
    fn test_ntt_clifford_multiply() {
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

        // NTT multiplication
        let mut result_ntt = multiply_ntt(&a, &b, &ntt, &lazy);

        // Reference: Karatsuba multiplication
        let result_karatsuba = a.multiply_karatsuba_lazy(&b, &lazy);

        // Both should have same length after reduction
        result_ntt.reduce_modulo_xn_minus_1_lazy(32, &lazy);
        let mut result_karatsuba_reduced = result_karatsuba;
        result_karatsuba_reduced.reduce_modulo_xn_minus_1_lazy(32, &lazy);

        // Compare results
        assert_eq!(result_ntt.coeffs.len(), result_karatsuba_reduced.coeffs.len());

        for i in 0..32 {
            for j in 0..8 {
                assert_eq!(
                    result_ntt.coeffs[i].coeffs[j],
                    result_karatsuba_reduced.coeffs[i].coeffs[j],
                    "Mismatch at coeff[{}].component[{}]: NTT={}, Karatsuba={}",
                    i, j,
                    result_ntt.coeffs[i].coeffs[j],
                    result_karatsuba_reduced.coeffs[i].coeffs[j]
                );
            }
        }
    }

    #[test]
    fn test_ntt_random_polynomials() {
        use rand::Rng;

        let ntt = NTTContext::new_clifford_lwe();
        let lazy = LazyReductionContext::new(3329);

        let mut rng = rand::thread_rng();

        // Generate random Clifford polynomials
        let mut a_coeffs = Vec::with_capacity(32);
        let mut b_coeffs = Vec::with_capacity(32);

        for _ in 0..32 {
            let mut a_mv = [0i64; 8];
            let mut b_mv = [0i64; 8];

            for j in 0..8 {
                a_mv[j] = rng.gen_range(-1..=1); // Ternary for simplicity
                b_mv[j] = rng.gen_range(-1..=1);
            }

            a_coeffs.push(CliffordRingElementInt::from_multivector(a_mv));
            b_coeffs.push(CliffordRingElementInt::from_multivector(b_mv));
        }

        let a = CliffordPolynomialInt::new(a_coeffs);
        let b = CliffordPolynomialInt::new(b_coeffs);

        // NTT multiplication
        let mut result_ntt = multiply_ntt(&a, &b, &ntt, &lazy);
        result_ntt.reduce_modulo_xn_minus_1_lazy(32, &lazy);

        // Karatsuba multiplication
        let mut result_karatsuba = a.multiply_karatsuba_lazy(&b, &lazy);
        result_karatsuba.reduce_modulo_xn_minus_1_lazy(32, &lazy);

        // Compare
        for i in 0..32 {
            for j in 0..8 {
                assert_eq!(
                    result_ntt.coeffs[i].coeffs[j],
                    result_karatsuba.coeffs[i].coeffs[j],
                    "Mismatch at coeff[{}].component[{}]",
                    i, j
                );
            }
        }
    }
}
