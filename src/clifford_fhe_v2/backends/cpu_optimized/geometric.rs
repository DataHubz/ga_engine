//! V2 Geometric Operations with NTT Optimization
//!
//! **Optimizations over V1:**
//! - Uses NTT for O(n log n) ciphertext multiplication
//! - Optimized component-wise operations
//! - Precomputed structure constants
//!
//! **Performance Target:** 10-20× faster geometric operations vs V1
//!
//! **Status:**
//! - ✅ Structure constants (Cl3StructureConstants)
//! - ✅ Basic operations: reverse, add, sub, scalar mul
//! - 🚧 Ciphertext multiplication (needs relinearization)
//! - 🚧 Geometric product (needs NTT-based ct multiplication)
//! - 🚧 Wedge product
//! - 🚧 Inner product
//! - 🚧 Rotation
//! - 🚧 Projection
//! - 🚧 Rejection

use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::{EvaluationKey, KeyContext};
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use crate::clifford_fhe_v2::params::CliffordFHEParams;

/// Multivector ciphertext in Cl(3,0) - 8 encrypted components
///
/// Components represent:
/// - [0]: scalar (grade 0)
/// - [1,2,3]: vectors e₁, e₂, e₃ (grade 1)
/// - [4,5,6]: bivectors e₁₂, e₁₃, e₂₃ (grade 2)
/// - [7]: trivector e₁₂₃ (grade 3)
///
/// **Note:** Component ordering matches V1 for compatibility
pub type MultivectorCiphertext = [Ciphertext; 8];

/// Cl(3,0) structure constants for geometric product
///
/// For each output component, stores list of (coefficient, input_a_idx, input_b_idx)
/// This encodes the Clifford algebra multiplication table
pub struct Cl3StructureConstants {
    pub products: Vec<Vec<(i64, usize, usize)>>,
}

impl Cl3StructureConstants {
    /// Create structure constants for Cl(3,0)
    ///
    /// Basis: {1, e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃}
    /// Signature: e₁²=e₂²=e₃²=1
    pub fn new() -> Self {
        let mut products = vec![Vec::new(); 8];

        // Component 0 (scalar): vectors square to +1, bivectors to -1
        products[0] = vec![
            (1, 0, 0),   // 1⊗1
            (1, 1, 1),   // e₁⊗e₁
            (1, 2, 2),   // e₂⊗e₂
            (1, 3, 3),   // e₃⊗e₃
            (-1, 4, 4),  // e₁₂⊗e₁₂
            (-1, 5, 5),  // e₁₃⊗e₁₃
            (-1, 6, 6),  // e₂₃⊗e₂₃
            (-1, 7, 7),  // e₁₂₃⊗e₁₂₃
        ];

        // Component 1 (e₁)
        products[1] = vec![
            (1, 0, 1),   // 1⊗e₁
            (1, 1, 0),   // e₁⊗1
            (1, 2, 4),   // e₂⊗e₁₂
            (-1, 4, 2),  // e₁₂⊗e₂
            (1, 3, 5),   // e₃⊗e₁₃
            (-1, 5, 3),  // e₁₃⊗e₃
            (-1, 6, 7),  // e₂₃⊗e₁₂₃
            (1, 7, 6),   // e₁₂₃⊗e₂₃
        ];

        // Component 2 (e₂)
        products[2] = vec![
            (1, 0, 2),   // 1⊗e₂
            (1, 2, 0),   // e₂⊗1
            (-1, 1, 4),  // e₁⊗e₁₂
            (1, 4, 1),   // e₁₂⊗e₁
            (1, 3, 6),   // e₃⊗e₂₃
            (-1, 6, 3),  // e₂₃⊗e₃
            (-1, 5, 7),  // e₁₃⊗e₁₂₃
            (1, 7, 5),   // e₁₂₃⊗e₁₃
        ];

        // Component 3 (e₃)
        products[3] = vec![
            (1, 0, 3),   // 1⊗e₃
            (1, 3, 0),   // e₃⊗1
            (-1, 1, 5),  // e₁⊗e₁₃
            (1, 5, 1),   // e₁₃⊗e₁
            (-1, 2, 6),  // e₂⊗e₂₃
            (1, 6, 2),   // e₂₃⊗e₂
            (-1, 4, 7),  // e₁₂⊗e₁₂₃
            (1, 7, 4),   // e₁₂₃⊗e₁₂
        ];

        // Component 4 (e₁₂)
        products[4] = vec![
            (1, 0, 4),   // 1⊗e₁₂
            (1, 4, 0),   // e₁₂⊗1
            (1, 1, 2),   // e₁⊗e₂
            (-1, 2, 1),  // e₂⊗e₁
            (1, 3, 7),   // e₃⊗e₁₂₃
            (-1, 7, 3),  // e₁₂₃⊗e₃
            (1, 5, 6),   // e₁₃⊗e₂₃
            (-1, 6, 5),  // e₂₃⊗e₁₃
        ];

        // Component 5 (e₁₃)
        products[5] = vec![
            (1, 0, 5),   // 1⊗e₁₃
            (1, 5, 0),   // e₁₃⊗1
            (1, 1, 3),   // e₁⊗e₃
            (-1, 3, 1),  // e₃⊗e₁
            (-1, 2, 7),  // e₂⊗e₁₂₃
            (1, 7, 2),   // e₁₂₃⊗e₂
            (-1, 4, 6),  // e₁₂⊗e₂₃
            (1, 6, 4),   // e₂₃⊗e₁₂
        ];

        // Component 6 (e₂₃)
        products[6] = vec![
            (1, 0, 6),   // 1⊗e₂₃
            (1, 6, 0),   // e₂₃⊗1
            (1, 2, 3),   // e₂⊗e₃
            (-1, 3, 2),  // e₃⊗e₂
            (1, 1, 7),   // e₁⊗e₁₂₃
            (-1, 7, 1),  // e₁₂₃⊗e₁
            (1, 4, 5),   // e₁₂⊗e₁₃
            (-1, 5, 4),  // e₁₃⊗e₁₂
        ];

        // Component 7 (e₁₂₃)
        products[7] = vec![
            (1, 0, 7),   // 1⊗e₁₂₃
            (1, 7, 0),   // e₁₂₃⊗1
            (1, 1, 6),   // e₁⊗e₂₃
            (-1, 6, 1),  // e₂₃⊗e₁
            (-1, 2, 5),  // e₂⊗e₁₃
            (1, 5, 2),   // e₁₃⊗e₂
            (1, 3, 4),   // e₃⊗e₁₂
            (-1, 4, 3),  // e₁₂⊗e₃
        ];

        Cl3StructureConstants { products }
    }
}

/// Geometric algebra context for homomorphic operations
pub struct GeometricContext {
    /// Key context for polynomial operations
    pub key_ctx: KeyContext,

    /// Parameters
    pub params: CliffordFHEParams,
}

impl GeometricContext {
    /// Create new geometric context
    pub fn new(params: CliffordFHEParams) -> Self {
        let key_ctx = KeyContext::new(params.clone());
        Self { key_ctx, params }
    }

    /// Reverse operation: ã = [a₀, a₁, a₂, a₃, -a₄, -a₅, -a₆, a₇]
    ///
    /// **Complexity:** O(n) - just negates bivector components
    pub fn reverse(&self, ct: &MultivectorCiphertext) -> MultivectorCiphertext {
        let moduli: Vec<u64> = self.params.moduli[..=ct[0].level].to_vec();

        [
            ct[0].clone(),                                 // scalar (unchanged)
            ct[1].clone(),                                 // e₁ (unchanged)
            ct[2].clone(),                                 // e₂ (unchanged)
            ct[3].clone(),                                 // e₃ (unchanged)
            self.negate_ciphertext(&ct[4], &moduli),      // -e₂₃
            self.negate_ciphertext(&ct[5], &moduli),      // -e₃₁
            self.negate_ciphertext(&ct[6], &moduli),      // -e₁₂
            ct[7].clone(),                                 // trivector (unchanged)
        ]
    }

    /// Negate a ciphertext: compute -ct
    fn negate_ciphertext(&self, ct: &Ciphertext, moduli: &[u64]) -> Ciphertext {
        let neg_c0 = self.negate_polynomial(&ct.c0, moduli);
        let neg_c1 = self.negate_polynomial(&ct.c1, moduli);

        Ciphertext::new(neg_c0, neg_c1, ct.level, ct.scale)
    }

    /// Negate polynomial: -a mod q for each coefficient
    fn negate_polynomial(
        &self,
        a: &[RnsRepresentation],
        moduli: &[u64],
    ) -> Vec<RnsRepresentation> {
        a.iter()
            .map(|rns| {
                let negated_values: Vec<u64> = rns
                    .values
                    .iter()
                    .zip(moduli)
                    .map(|(&val, &q)| if val == 0 { 0 } else { q - val })
                    .collect();
                RnsRepresentation::new(negated_values, moduli.to_vec())
            })
            .collect()
    }

    /// Add two ciphertexts component-wise
    pub fn add_ciphertexts(&self, a: &Ciphertext, b: &Ciphertext) -> Ciphertext {
        a.add(b)
    }

    /// Subtract two ciphertexts component-wise
    pub fn sub_ciphertexts(&self, a: &Ciphertext, b: &Ciphertext) -> Ciphertext {
        a.sub(b)
    }

    /// Add two multivectors component-wise
    pub fn add_multivectors(
        &self,
        a: &MultivectorCiphertext,
        b: &MultivectorCiphertext,
    ) -> MultivectorCiphertext {
        [
            self.add_ciphertexts(&a[0], &b[0]),
            self.add_ciphertexts(&a[1], &b[1]),
            self.add_ciphertexts(&a[2], &b[2]),
            self.add_ciphertexts(&a[3], &b[3]),
            self.add_ciphertexts(&a[4], &b[4]),
            self.add_ciphertexts(&a[5], &b[5]),
            self.add_ciphertexts(&a[6], &b[6]),
            self.add_ciphertexts(&a[7], &b[7]),
        ]
    }

    /// Subtract two multivectors component-wise
    pub fn sub_multivectors(
        &self,
        a: &MultivectorCiphertext,
        b: &MultivectorCiphertext,
    ) -> MultivectorCiphertext {
        [
            self.sub_ciphertexts(&a[0], &b[0]),
            self.sub_ciphertexts(&a[1], &b[1]),
            self.sub_ciphertexts(&a[2], &b[2]),
            self.sub_ciphertexts(&a[3], &b[3]),
            self.sub_ciphertexts(&a[4], &b[4]),
            self.sub_ciphertexts(&a[5], &b[5]),
            self.sub_ciphertexts(&a[6], &b[6]),
            self.sub_ciphertexts(&a[7], &b[7]),
        ]
    }

    /// Multiply ciphertext by scalar
    pub fn mul_scalar(&self, ct: &Ciphertext, scalar: f64) -> Ciphertext {
        ct.mul_scalar(scalar)
    }

    /// Multiply multivector by scalar
    pub fn mul_multivector_scalar(
        &self,
        mv: &MultivectorCiphertext,
        scalar: f64,
    ) -> MultivectorCiphertext {
        [
            self.mul_scalar(&mv[0], scalar),
            self.mul_scalar(&mv[1], scalar),
            self.mul_scalar(&mv[2], scalar),
            self.mul_scalar(&mv[3], scalar),
            self.mul_scalar(&mv[4], scalar),
            self.mul_scalar(&mv[5], scalar),
            self.mul_scalar(&mv[6], scalar),
            self.mul_scalar(&mv[7], scalar),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey;

    fn create_test_ciphertext(params: &CliffordFHEParams, value: f64) -> Ciphertext {
        let level = params.max_level();
        let moduli: Vec<u64> = params.moduli[..=level].to_vec();
        let n = params.n;

        // Create simple ciphertext with value in first coefficient
        let c0: Vec<RnsRepresentation> = (0..n)
            .map(|i| {
                let val = if i == 0 {
                    (value * params.scale) as u64
                } else {
                    0
                };
                RnsRepresentation::from_u64(val, &moduli)
            })
            .collect();

        let c1: Vec<RnsRepresentation> =
            vec![RnsRepresentation::from_u64(0, &moduli); n];

        Ciphertext::new(c0, c1, level, params.scale)
    }

    fn create_test_multivector(params: &CliffordFHEParams) -> MultivectorCiphertext {
        [
            create_test_ciphertext(params, 1.0),  // scalar
            create_test_ciphertext(params, 2.0),  // e₁
            create_test_ciphertext(params, 3.0),  // e₂
            create_test_ciphertext(params, 4.0),  // e₃
            create_test_ciphertext(params, 5.0),  // e₂₃
            create_test_ciphertext(params, 6.0),  // e₃₁
            create_test_ciphertext(params, 7.0),  // e₁₂
            create_test_ciphertext(params, 8.0),  // e₁₂₃
        ]
    }

    #[test]
    fn test_geometric_context_creation() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = GeometricContext::new(params.clone());

        assert_eq!(ctx.params.n, params.n);
    }

    #[test]
    fn test_reverse_operation() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = GeometricContext::new(params.clone());

        let mv = create_test_multivector(&params);
        let reversed = ctx.reverse(&mv);

        // Components 0, 1, 2, 3, 7 should be unchanged
        assert_eq!(reversed[0].c0[0].values[0], mv[0].c0[0].values[0]);
        assert_eq!(reversed[1].c0[0].values[0], mv[1].c0[0].values[0]);
        assert_eq!(reversed[2].c0[0].values[0], mv[2].c0[0].values[0]);
        assert_eq!(reversed[3].c0[0].values[0], mv[3].c0[0].values[0]);
        assert_eq!(reversed[7].c0[0].values[0], mv[7].c0[0].values[0]);

        // Components 4, 5, 6 (bivectors) should be negated
        let q = params.moduli[0];
        let original_val_4 = mv[4].c0[0].values[0];
        let reversed_val_4 = reversed[4].c0[0].values[0];

        // -val mod q should equal q - val (if val != 0)
        if original_val_4 != 0 {
            assert_eq!(reversed_val_4, q - original_val_4);
        }
    }

    #[test]
    fn test_add_multivectors() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = GeometricContext::new(params.clone());

        let mv_a = create_test_multivector(&params);
        let mv_b = create_test_multivector(&params);

        let sum = ctx.add_multivectors(&mv_a, &mv_b);

        // First component should be sum of scalars (1.0 + 1.0 = 2.0)
        // In RNS: (1 * scale) + (1 * scale) = 2 * scale
        let expected = (2.0 * params.scale) as u64;
        assert_eq!(sum[0].c0[0].values[0], expected % params.moduli[0]);
    }

    #[test]
    fn test_sub_multivectors() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = GeometricContext::new(params.clone());

        let mv_a = create_test_multivector(&params);
        let mv_b = create_test_multivector(&params);

        let diff = ctx.sub_multivectors(&mv_a, &mv_b);

        // Subtracting same values should give 0
        assert_eq!(diff[0].c0[0].values[0], 0);
        assert_eq!(diff[1].c0[0].values[0], 0);
    }

    #[test]
    fn test_mul_scalar() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = GeometricContext::new(params.clone());

        let mv = create_test_multivector(&params);
        let scaled = ctx.mul_multivector_scalar(&mv, 2.0);

        // Scalar component: 1.0 * 2.0 = 2.0
        // But scale also increases, so we just check it's non-zero and different
        assert!(scaled[0].c0[0].values[0] > 0);
        assert!(scaled[0].scale > mv[0].scale);
    }
}
