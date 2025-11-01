//! Homomorphic Geometric Product for Clifford-FHE
//!
//! This is THE KEY innovation that makes Clifford-FHE unique!
//!
//! # Problem
//!
//! Standard CKKS only knows about polynomial multiplication:
//! ```text
//! Enc(a(x)) × Enc(b(x)) = Enc(a(x) · b(x))
//! ```
//!
//! But geometric product is NOT polynomial multiplication!
//! ```text
//! (1 + 2e1) ⊗ (3 + 4e2) = 3 + 4e2 + 6e1 + 8e12
//! ```
//!
//! # Solution
//!
//! Encode Cl(3,0) structure constants into homomorphic operations:
//! ```text
//! a = [a0, a1, a2, a3, a4, a5, a6, a7]  (components)
//! b = [b0, b1, b2, b3, b4, b5, b6, b7]
//!
//! c = a ⊗ b = structure_constants_apply(a, b)
//!
//! Each c[i] is a linear combination of products a[j] * b[k]
//! ```
//!
//! We compute this HOMOMORPHICALLY using CKKS multiplication!

use crate::clifford_fhe::ckks::Ciphertext;
use crate::clifford_fhe::keys::{EvaluationKey, RotationKey};
use crate::clifford_fhe::operations::{add_ciphertexts, compute_component_product, multiply_by_scalar};
use crate::clifford_fhe::params::CliffordFHEParams;

/// Cl(3,0) basis elements
///
/// ```text
/// 0: 1     (scalar)
/// 1: e1    (vector)
/// 2: e2    (vector)
/// 3: e3    (vector)
/// 4: e12   (bivector)
/// 5: e13   (bivector)
/// 6: e23   (bivector)
/// 7: e123  (pseudoscalar)
/// ```
#[derive(Debug, Clone, Copy)]
pub enum BasisBlade {
    Scalar = 0,
    E1 = 1,
    E2 = 2,
    E3 = 3,
    E12 = 4,
    E13 = 5,
    E23 = 6,
    E123 = 7,
}

/// Structure constants for Cl(3,0) geometric product
///
/// STRUCTURE[i][j] = (coefficient, result_basis)
/// Meaning: e_i ⊗ e_j = coefficient * e_result
///
/// Example: e1 ⊗ e1 = 1 (scalar)
///          e1 ⊗ e2 = e12
///          e12 ⊗ e1 = -e2
///
/// This is precomputed and used for all geometric products!
pub struct StructureConstants {
    /// For each pair (i, j), stores list of (coefficient, target_basis, source_a, source_b)
    /// where result[target] += coefficient * a[source_a] * b[source_b]
    pub products: Vec<Vec<(i32, usize, usize, usize)>>,
}

impl StructureConstants {
    /// Generate structure constants for Cl(3,0)
    ///
    /// This is computed ONCE and reused for all geometric products
    pub fn new_cl30() -> Self {
        let mut products = vec![Vec::new(); 8];

        // We'll build this systematically using the multiplication table
        // For each target basis element, we compute which products contribute to it

        // Component 0 (scalar):
        // 1⊗1=1, e1⊗e1=1, e2⊗e2=1, e3⊗e3=1, e12⊗e12=-1, e13⊗e13=-1, e23⊗e23=-1, e123⊗e123=-1
        products[0] = vec![
            (1, 0, 0, 0),   // 1 * a[0] * b[0]
            (1, 0, 1, 1),   // 1 * a[1] * b[1] (e1⊗e1)
            (1, 0, 2, 2),   // 1 * a[2] * b[2] (e2⊗e2)
            (1, 0, 3, 3),   // 1 * a[3] * b[3] (e3⊗e3)
            (-1, 0, 4, 4),  // -1 * a[4] * b[4] (e12⊗e12)
            (-1, 0, 5, 5),  // -1 * a[5] * b[5] (e13⊗e13)
            (-1, 0, 6, 6),  // -1 * a[6] * b[6] (e23⊗e23)
            (-1, 0, 7, 7),  // -1 * a[7] * b[7] (e123⊗e123)
        ];

        // Component 1 (e1):
        // 1⊗e1=e1, e1⊗1=e1, e2⊗e12=e1, e12⊗e2=-e1, e3⊗e13=e1, e13⊗e3=-e1, e23⊗e123=-e1, e123⊗e23=-e1
        products[1] = vec![
            (1, 1, 0, 1),   // 1 * a[0] * b[1]
            (1, 1, 1, 0),   // 1 * a[1] * b[0]
            (-1, 1, 2, 4),  // -1 * a[2] * b[4] (e2⊗e12)
            (1, 1, 3, 5),   // 1 * a[3] * b[5] (e3⊗e13)
            (1, 1, 4, 2),   // 1 * a[4] * b[2] (e12⊗e2)
            (-1, 1, 5, 3),  // -1 * a[5] * b[3] (e13⊗e3)
            (-1, 1, 6, 7),  // -1 * a[6] * b[7] (e23⊗e123)
            (-1, 1, 7, 6),  // -1 * a[7] * b[6] (e123⊗e23)
        ];

        // Component 2 (e2):
        // 1⊗e2=e2, e2⊗1=e2, e1⊗e12=e2, e12⊗e1=-e2, e3⊗e23=e2, e23⊗e3=-e2, e13⊗e123=e2, e123⊗e13=e2
        products[2] = vec![
            (1, 2, 0, 2),   // 1 * a[0] * b[2]
            (1, 2, 1, 4),   // 1 * a[1] * b[4] (e1⊗e12)
            (1, 2, 2, 0),   // 1 * a[2] * b[0]
            (-1, 2, 3, 6),  // -1 * a[3] * b[6] (e3⊗e23)
            (-1, 2, 4, 1),  // -1 * a[4] * b[1] (e12⊗e1)
            (1, 2, 5, 7),   // 1 * a[5] * b[7] (e13⊗e123)
            (1, 2, 6, 3),   // 1 * a[6] * b[3] (e23⊗e3)
            (1, 2, 7, 5),   // 1 * a[7] * b[5] (e123⊗e13)
        ];

        // Component 3 (e3):
        // 1⊗e3=e3, e3⊗1=e3, e1⊗e13=e3, e13⊗e1=-e3, e2⊗e23=e3, e23⊗e2=-e3, e12⊗e123=e3, e123⊗e12=-e3
        products[3] = vec![
            (1, 3, 0, 3),   // 1 * a[0] * b[3]
            (-1, 3, 1, 5),  // -1 * a[1] * b[5] (e1⊗e13)
            (1, 3, 2, 6),   // 1 * a[2] * b[6] (e2⊗e23)
            (1, 3, 3, 0),   // 1 * a[3] * b[0]
            (-1, 3, 4, 7),  // -1 * a[4] * b[7] (e12⊗e123)
            (-1, 3, 5, 1),  // -1 * a[5] * b[1] (e13⊗e1)
            (-1, 3, 6, 2),  // -1 * a[6] * b[2] (e23⊗e2)
            (-1, 3, 7, 4),  // -1 * a[7] * b[4] (e123⊗e12)
        ];

        // Component 4 (e12):
        // 1⊗e12=e12, e12⊗1=e12, e1⊗e2=e12, e2⊗e1=-e12, e3⊗e123=e12, e123⊗e3=e12, e13⊗e23=-e12, e23⊗e13=e12
        products[4] = vec![
            (1, 4, 0, 4),   // 1 * a[0] * b[4]
            (1, 4, 1, 2),   // 1 * a[1] * b[2] (e1⊗e2)
            (-1, 4, 2, 1),  // -1 * a[2] * b[1] (e2⊗e1)
            (1, 4, 3, 7),   // 1 * a[3] * b[7] (e3⊗e123)
            (1, 4, 4, 0),   // 1 * a[4] * b[0]
            (-1, 4, 5, 6),  // -1 * a[5] * b[6] (e13⊗e23)
            (1, 4, 6, 5),   // 1 * a[6] * b[5] (e23⊗e13)
            (1, 4, 7, 3),   // 1 * a[7] * b[3] (e123⊗e3)
        ];

        // Component 5 (e13):
        // 1⊗e13=e13, e13⊗1=e13, e1⊗e3=e13, e3⊗e1=-e13, e2⊗e123=e13, e123⊗e2=-e13, e12⊗e23=e13, e23⊗e12=-e13
        products[5] = vec![
            (1, 5, 0, 5),   // 1 * a[0] * b[5]
            (1, 5, 1, 3),   // 1 * a[1] * b[3] (e1⊗e3)
            (-1, 5, 2, 7),  // -1 * a[2] * b[7] (e2⊗e123)
            (-1, 5, 3, 1),  // -1 * a[3] * b[1] (e3⊗e1)
            (1, 5, 4, 6),   // 1 * a[4] * b[6] (e12⊗e23)
            (1, 5, 5, 0),   // 1 * a[5] * b[0]
            (-1, 5, 6, 4),  // -1 * a[6] * b[4] (e23⊗e12)
            (-1, 5, 7, 2),  // -1 * a[7] * b[2] (e123⊗e2)
        ];

        // Component 6 (e23):
        // 1⊗e23=e23, e23⊗1=e23, e2⊗e3=e23, e3⊗e2=-e23, e1⊗e123=-e23, e123⊗e1=e23, e12⊗e13=e23, e13⊗e12=-e23
        products[6] = vec![
            (1, 6, 0, 6),   // 1 * a[0] * b[6]
            (-1, 6, 1, 7),  // -1 * a[1] * b[7] (e1⊗e123)
            (1, 6, 2, 3),   // 1 * a[2] * b[3] (e2⊗e3)
            (-1, 6, 3, 2),  // -1 * a[3] * b[2] (e3⊗e2)
            (-1, 6, 4, 5),  // -1 * a[4] * b[5] (e12⊗e13)
            (1, 6, 5, 4),   // 1 * a[5] * b[4] (e13⊗e12)
            (1, 6, 6, 0),   // 1 * a[6] * b[0]
            (1, 6, 7, 1),   // 1 * a[7] * b[1] (e123⊗e1)
        ];

        // Component 7 (e123):
        // 1⊗e123=e123, e123⊗1=e123, e1⊗e23=e123, e23⊗e1=-e123, e2⊗e13=-e123, e13⊗e2=e123, e3⊗e12=e123, e12⊗e3=-e123
        products[7] = vec![
            (1, 7, 0, 7),   // 1 * a[0] * b[7]
            (1, 7, 1, 6),   // 1 * a[1] * b[6] (e1⊗e23)
            (-1, 7, 2, 5),  // -1 * a[2] * b[5] (e2⊗e13)
            (1, 7, 3, 4),   // 1 * a[3] * b[4] (e3⊗e12)
            (1, 7, 4, 3),   // 1 * a[4] * b[3] (e12⊗e3)
            (-1, 7, 5, 2),  // -1 * a[5] * b[2] (e13⊗e2)
            (1, 7, 6, 1),   // 1 * a[6] * b[1] (e23⊗e1)
            (1, 7, 7, 0),   // 1 * a[7] * b[0]
        ];

        Self { products }
    }

    /// Get products contributing to a specific result component
    pub fn get_products_for(&self, component: usize) -> &[(i32, usize, usize, usize)] {
        &self.products[component]
    }
}

/// Homomorphic geometric product using structure constants
///
/// This is THE KEY OPERATION that makes Clifford-FHE unique!
///
/// # Algorithm
///
/// For each result component c[i]:
/// ```text
/// c[i] = Σ coeff[j,k] * a[j] * b[k]
/// ```
///
/// We compute this homomorphically:
/// ```text
/// Enc(c[i]) = Σ coeff[j,k] * Enc(a[j]) * Enc(b[k])
/// ```
///
/// Uses rotation keys to move products to correct positions!
///
/// # Example
///
/// ```rust,ignore
/// let ct_a = encrypt(&pk, &mv_a, &params);
/// let ct_b = encrypt(&pk, &mv_b, &params);
///
/// // Homomorphic geometric product!
/// let ct_c = geometric_product_homomorphic(&ct_a, &ct_b, &evk, &rotk, &params);
///
/// let mv_c = decrypt(&sk, &ct_c, &params);
/// // mv_c should equal mv_a ⊗ mv_b !
/// ```
pub fn geometric_product_homomorphic(
    ct_a: &Ciphertext,
    ct_b: &Ciphertext,
    evk: &EvaluationKey,
    rotk: &RotationKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    let structure = StructureConstants::new_cl30();

    // Array to collect result for each component position
    // We'll build up the full result by computing each component's contributions
    let mut result: Option<Ciphertext> = None;

    // For each output component (0-7)
    for target in 0..8 {
        let products = structure.get_products_for(target);

        // Accumulate all products contributing to this component
        for &(coeff, _target, src_a, src_b) in products {
            // Compute ct_a[src_a] * ct_b[src_b] and place result at position target
            let product_ct = compute_component_product(
                ct_a, src_a, ct_b, src_b, target, evk, rotk, params,
            );

            // Apply coefficient (+1 or -1)
            let scaled = if coeff == 1 {
                product_ct
            } else {
                multiply_by_scalar(&product_ct, coeff as i64, params)
            };

            // Add to global result
            result = Some(match result {
                None => scaled,
                Some(acc) => add_ciphertexts(&acc, &scaled, params),
            });
        }
    }

    result.expect("Geometric product produced empty result")
}

/// Negate a ciphertext (multiply by -1)
fn negate(ct: &Ciphertext, params: &CliffordFHEParams) -> Ciphertext {
    let q = params.modulus_at_level(ct.level);

    let c0: Vec<i64> = ct.c0.iter().map(|&x| ((q - x) % q + q) % q).collect();
    let c1: Vec<i64> = ct.c1.iter().map(|&x| ((q - x) % q + q) % q).collect();

    Ciphertext::new(c0, c1, ct.level, ct.scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_constants_scalar() {
        let sc = StructureConstants::new_cl30();

        // Component 0 (scalar) should have 8 products
        let products = sc.get_products_for(0);
        assert_eq!(products.len(), 8);

        // Check a few specific products
        // 1⊗1 = 1
        assert!(products.contains(&(1, 0, 0, 0)));
        // e1⊗e1 = 1
        assert!(products.contains(&(1, 0, 1, 1)));
        // e12⊗e12 = -1
        assert!(products.contains(&(-1, 0, 4, 4)));
    }

    #[test]
    fn test_structure_constants_e1() {
        let sc = StructureConstants::new_cl30();

        // Component 1 (e1) should have 8 products
        let products = sc.get_products_for(1);
        assert_eq!(products.len(), 8);

        // 1⊗e1 = e1
        assert!(products.contains(&(1, 1, 0, 1)));
        // e1⊗1 = e1
        assert!(products.contains(&(1, 1, 1, 0)));
    }

    #[test]
    fn test_structure_constants_all_components() {
        let sc = StructureConstants::new_cl30();

        // Each component should have exactly 8 products
        for i in 0..8 {
            let products = sc.get_products_for(i);
            assert_eq!(
                products.len(),
                8,
                "Component {} should have 8 products",
                i
            );
        }
    }
}
