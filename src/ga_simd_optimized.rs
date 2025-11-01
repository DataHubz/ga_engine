//! Optimized SIMD geometric product using explicit unrolled computation
//!
//! Instead of using a lookup table (which doesn't vectorize well), we explicitly
//! compute each output component as a linear combination of input products.
//!
//! This allows the compiler to better vectorize and optimize the code.

/// Optimized geometric product that's easier for LLVM to auto-vectorize
///
/// Key insight: Each output component out[k] is a sum of sign * a[i] * b[j] terms.
/// By explicitly writing these sums, LLVM can auto-vectorize better than with
/// a lookup table approach.
#[inline(always)]
pub fn geometric_product_full_optimized(a: &[f64; 8], b: &[f64; 8], out: &mut [f64; 8]) {
    // These formulas are generated from the GP_PAIRS lookup table
    // but written explicitly so LLVM can vectorize them

    // Formulas generated from GP_PAIRS table with orientation correction
    // Each output component is a sum of ±a[i]*b[j] terms

    // Component 0 (scalar)
    out[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
           - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];

    // Component 1 (e1)
    out[1] = a[0]*b[1] + a[1]*b[0] - a[2]*b[6] + a[3]*b[5]
           - a[4]*b[7] - a[5]*b[3] + a[6]*b[2] - a[7]*b[4];

    // Component 2 (e2)
    out[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[0] - a[3]*b[4]
           + a[4]*b[3] - a[5]*b[7] - a[6]*b[1] - a[7]*b[5];

    // Component 3 (e3)
    out[3] = a[0]*b[3] - a[1]*b[5] + a[2]*b[4] + a[3]*b[0]
           - a[4]*b[2] + a[5]*b[1] - a[6]*b[7] - a[7]*b[6];

    // Component 4 (e23)
    out[4] = a[0]*b[4] + a[1]*b[7] + a[2]*b[3] - a[3]*b[2]
           + a[4]*b[0] - a[5]*b[6] + a[6]*b[5] + a[7]*b[1];

    // Component 5 (e31)
    out[5] = a[0]*b[5] - a[1]*b[3] + a[2]*b[7] + a[3]*b[1]
           + a[4]*b[6] + a[5]*b[0] - a[6]*b[4] + a[7]*b[2];

    // Component 6 (e12)
    out[6] = a[0]*b[6] + a[1]*b[2] - a[2]*b[1] + a[3]*b[7]
           - a[4]*b[5] + a[5]*b[4] + a[6]*b[0] + a[7]*b[3];

    // Component 7 (e123)
    out[7] = a[0]*b[7] + a[1]*b[4] + a[2]*b[5] + a[3]*b[6]
           + a[4]*b[1] + a[5]*b[2] + a[6]*b[3] + a[7]*b[0];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ga::geometric_product_full;

    #[test]
    fn test_optimized_vs_reference() {
        // Test that optimized version matches reference implementation
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let mut out_ref = [0.0; 8];
        let mut out_opt = [0.0; 8];

        geometric_product_full(&a, &b, &mut out_ref);
        geometric_product_full_optimized(&a, &b, &mut out_opt);

        for i in 0..8 {
            assert!((out_ref[i] - out_opt[i]).abs() < 1e-10,
                "Mismatch at index {}: ref={}, opt={}",
                i, out_ref[i], out_opt[i]);
        }
    }

    #[test]
    fn test_basis_products() {
        // Test e1 * e2 = e12
        let e1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let e2 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut e12 = [0.0; 8];

        geometric_product_full_optimized(&e1, &e2, &mut e12);

        assert!((e12[6] - 1.0).abs() < 1e-10);
        for i in 0..8 {
            if i != 6 {
                assert!(e12[i].abs() < 1e-10);
            }
        }
    }
}
