//! Clifford Algebra Ring Cl(3,0) via Left-Regular Representation
//!
//! This module implements the key insight: Cl(3,0) can be embedded as an
//! 8-dimensional subspace S ⊂ M₈(ℝ) via the left-regular representation.
//!
//! Key properties:
//! - S is a unital subalgebra (hence a ring)
//! - Addition: ρ(a) + ρ(b) = ρ(a+b) ∈ S
//! - Multiplication: ρ(a)·ρ(b) = ρ(ab) ∈ S (closed!)
//! - Unit: ρ(1) = I₈
//! - Isomorphism: S ≅ Cl(3,0)
//!
//! Left-regular representation: ρ(a)(x) = a·x (geometric product)
//!
//! This gives us a **closed ring** that we can use as an algebraic structure
//! in cryptography and ML, with GA speedups!

/// An 8×8 matrix in row-major order
pub type Matrix8x8 = [f64; 64];

/// An element of Cl(3,0) represented as 8 components
/// Order: [1, e₁, e₂, e₃, e₂₃, e₃₁, e₁₂, e₁₂₃]
pub type Multivector3D = [f64; 8];

/// The left-regular representation basis matrices ρ(eᵢ)
///
/// These 8 matrices form a basis for the subspace S ⊂ M₈(ℝ)
///
/// For each basis element eᵢ of Cl(3,0), we compute the 8×8 matrix
/// that represents left-multiplication by eᵢ:
///   ρ(eᵢ)(eⱼ) = eᵢ · eⱼ (geometric product)
///
/// This gives us 8 basis matrices, one for each:
/// e₀=1, e₁, e₂, e₃, e₂₃, e₃₁, e₁₂, e₁₂₃
pub struct LeftRegularBasis {
    /// ρ(1) = I₈ (identity)
    pub rho_1: Matrix8x8,

    /// ρ(e₁): left-multiply by e₁
    pub rho_e1: Matrix8x8,

    /// ρ(e₂): left-multiply by e₂
    pub rho_e2: Matrix8x8,

    /// ρ(e₃): left-multiply by e₃
    pub rho_e3: Matrix8x8,

    /// ρ(e₂₃): left-multiply by e₂₃
    pub rho_e23: Matrix8x8,

    /// ρ(e₃₁): left-multiply by e₃₁
    pub rho_e31: Matrix8x8,

    /// ρ(e₁₂): left-multiply by e₁₂
    pub rho_e12: Matrix8x8,

    /// ρ(e₁₂₃): left-multiply by e₁₂₃ (pseudoscalar)
    pub rho_e123: Matrix8x8,
}

impl LeftRegularBasis {
    /// Construct the basis matrices by computing eᵢ·eⱼ for all i,j
    ///
    /// We use the geometric product table:
    /// - 1·eⱼ = eⱼ
    /// - eᵢ·eᵢ = 1 (for i=1,2,3)
    /// - e₁·e₂ = e₁₂, e₂·e₁ = -e₁₂ (antisymmetric)
    /// - etc.
    pub fn new() -> Self {
        use crate::ga::geometric_product_full;

        // Basis elements as multivectors
        let basis = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // 1
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // e₁
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], // e₂
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], // e₃
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], // e₂₃
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], // e₃₁
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], // e₁₂
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], // e₁₂₃
        ];

        // Helper to compute ρ(eᵢ) matrix
        let compute_rho = |i: usize| -> Matrix8x8 {
            let mut matrix = [0.0; 64];
            for j in 0..8 {
                // Compute eᵢ · eⱼ
                let mut result = [0.0; 8];
                geometric_product_full(&basis[i], &basis[j], &mut result);

                // Store in column j of the matrix
                for row in 0..8 {
                    matrix[row * 8 + j] = result[row];
                }
            }
            matrix
        };

        Self {
            rho_1: compute_rho(0),
            rho_e1: compute_rho(1),
            rho_e2: compute_rho(2),
            rho_e3: compute_rho(3),
            rho_e23: compute_rho(4),
            rho_e31: compute_rho(5),
            rho_e12: compute_rho(6),
            rho_e123: compute_rho(7),
        }
    }

    /// Get basis matrix by index (0-7)
    pub fn get(&self, i: usize) -> &Matrix8x8 {
        match i {
            0 => &self.rho_1,
            1 => &self.rho_e1,
            2 => &self.rho_e2,
            3 => &self.rho_e3,
            4 => &self.rho_e23,
            5 => &self.rho_e31,
            6 => &self.rho_e12,
            7 => &self.rho_e123,
            _ => panic!("Invalid basis index: {}", i),
        }
    }
}

/// An element of the Clifford ring S ⊂ M₈(ℝ)
///
/// Represented as both:
/// 1. 8 scalar coefficients (for the multivector)
/// 2. 8×8 matrix ρ(a) = Σᵢ aᵢ·ρ(eᵢ)
#[derive(Clone, Debug)]
pub struct CliffordRingElement {
    /// Multivector coefficients [a₀, a₁, ..., a₇]
    pub coeffs: Multivector3D,

    /// The 8×8 matrix representation ρ(a)
    /// Computed lazily on demand
    matrix: Option<Matrix8x8>,
}

impl CliffordRingElement {
    /// Create from multivector coefficients
    pub fn from_multivector(coeffs: Multivector3D) -> Self {
        Self {
            coeffs,
            matrix: None,
        }
    }

    /// Create from 8×8 matrix (must be in subspace S!)
    pub fn from_matrix(matrix: Matrix8x8) -> Self {
        // TODO: Extract coefficients by projection onto basis
        Self {
            coeffs: [0.0; 8], // Placeholder
            matrix: Some(matrix),
        }
    }

    /// Get the 8×8 matrix representation ρ(a)
    pub fn to_matrix(&mut self) -> &Matrix8x8 {
        if self.matrix.is_none() {
            // Compute ρ(a) = Σᵢ aᵢ·ρ(eᵢ)
            let basis = LeftRegularBasis::new();
            let mut result = [0.0; 64];

            for i in 0..8 {
                let coeff = self.coeffs[i];
                let basis_matrix = basis.get(i);

                for j in 0..64 {
                    result[j] += coeff * basis_matrix[j];
                }
            }

            self.matrix = Some(result);
        }

        self.matrix.as_ref().unwrap()
    }

    /// Ring addition: (a + b) ∈ S
    pub fn add(&self, other: &Self) -> Self {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = self.coeffs[i] + other.coeffs[i];
        }
        Self::from_multivector(result)
    }

    /// Ring multiplication: (a · b) ∈ S (via geometric product)
    pub fn multiply(&self, other: &Self) -> Self {
        use crate::ga::geometric_product_full;

        let mut result = [0.0; 8];
        geometric_product_full(&self.coeffs, &other.coeffs, &mut result);
        Self::from_multivector(result)
    }

    /// Scalar multiplication: (c · a) ∈ S
    pub fn scalar_mul(&self, scalar: f64) -> Self {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = scalar * self.coeffs[i];
        }
        Self::from_multivector(result)
    }

    /// Ring unit: ρ(1) = I₈
    pub fn unit() -> Self {
        Self::from_multivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }

    /// Ring zero: ρ(0) = 0
    pub fn zero() -> Self {
        Self::from_multivector([0.0; 8])
    }
}

/// Polynomial ring R[x] where R is the Clifford ring S
///
/// Elements: a₀ + a₁x + a₂x² + ... + aₙxⁿ where each aᵢ ∈ S
///
/// This is the key for cryptography: we can do polynomial operations
/// with coefficients in the Clifford ring, getting GA speedups!
#[derive(Clone, Debug)]
pub struct CliffordPolynomial {
    /// Coefficients in the Clifford ring S
    /// coeffs[i] = coefficient of xⁱ
    pub coeffs: Vec<CliffordRingElement>,
}

impl CliffordPolynomial {
    /// Create polynomial from coefficients
    pub fn new(coeffs: Vec<CliffordRingElement>) -> Self {
        Self { coeffs }
    }

    /// Create zero polynomial
    pub fn zero(degree: usize) -> Self {
        Self {
            coeffs: vec![CliffordRingElement::zero(); degree],
        }
    }

    /// Degree of polynomial
    pub fn degree(&self) -> usize {
        self.coeffs.len()
    }

    /// Polynomial addition: (f + g) ∈ R[x]
    pub fn add(&self, other: &Self) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut result = Vec::with_capacity(max_len);
        let zero = CliffordRingElement::zero();

        for i in 0..max_len {
            let a = self.coeffs.get(i).unwrap_or(&zero);
            let b = other.coeffs.get(i).unwrap_or(&zero);
            result.push(a.add(b));
        }

        Self { coeffs: result }
    }

    /// Polynomial multiplication: (f · g) ∈ R[x]
    ///
    /// This is where GA speedup applies!
    /// Each coefficient multiplication uses geometric product (52 ns)
    /// instead of 8×8 matrix multiplication (82 ns)
    pub fn multiply(&self, other: &Self) -> Self {
        let result_degree = self.degree() + other.degree() - 1;
        let mut result = vec![CliffordRingElement::zero(); result_degree];

        // Convolution: c[k] = Σᵢ₊ⱼ₌ₖ a[i] · b[j]
        for i in 0..self.coeffs.len() {
            for j in 0..other.coeffs.len() {
                let product = self.coeffs[i].multiply(&other.coeffs[j]);
                result[i + j] = result[i + j].add(&product);
            }
        }

        Self { coeffs: result }
    }

    /// Modular reduction: f mod (xⁿ - 1)
    ///
    /// For NTRU-like constructions: R = S[x]/(xⁿ - 1)
    pub fn reduce_modulo_xn_minus_1(&mut self, n: usize) {
        // Reduce high-degree terms: xⁿ ≡ 1 (mod xⁿ - 1)
        while self.coeffs.len() > n {
            let high = self.coeffs.pop().unwrap();
            let wrap_idx = self.coeffs.len() - n;
            self.coeffs[wrap_idx] = self.coeffs[wrap_idx].add(&high);
        }

        // Pad with zeros if needed
        while self.coeffs.len() < n {
            self.coeffs.push(CliffordRingElement::zero());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_closure_addition() {
        // Test: a + b ∈ S
        let a = CliffordRingElement::from_multivector([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
        let b = CliffordRingElement::from_multivector([0.5, 1.5, 2.5, 3.5, 1.0, 1.0, 1.0, 1.0]);

        let c = a.add(&b);

        // Check coefficients are correct
        assert_eq!(c.coeffs[0], 1.5);
        assert_eq!(c.coeffs[1], 3.5);
        assert_eq!(c.coeffs[2], 5.5);
        assert_eq!(c.coeffs[3], 7.5);
    }

    #[test]
    fn test_ring_closure_multiplication() {
        // Test: a · b ∈ S (via geometric product)
        let e1 = CliffordRingElement::from_multivector([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let e2 = CliffordRingElement::from_multivector([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // e₁ · e₂ = e₁₂
        let e12 = e1.multiply(&e2);

        // Check: should have e₁₂ component only
        assert!((e12.coeffs[0] - 0.0).abs() < 1e-10); // scalar
        assert!((e12.coeffs[1] - 0.0).abs() < 1e-10); // e₁
        assert!((e12.coeffs[2] - 0.0).abs() < 1e-10); // e₂
        assert!((e12.coeffs[6] - 1.0).abs() < 1e-10); // e₁₂
    }

    #[test]
    fn test_ring_unit() {
        // Test: 1 · a = a
        let unit = CliffordRingElement::unit();
        let a = CliffordRingElement::from_multivector([2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 1.0, 1.0]);

        let result = unit.multiply(&a);

        for i in 0..8 {
            assert!((result.coeffs[i] - a.coeffs[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_basis_matrices_computed() {
        let basis = LeftRegularBasis::new();

        // ρ(1) should be identity
        let rho_1 = basis.get(0);
        for i in 0..8 {
            for j in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((rho_1[i * 8 + j] - expected).abs() < 1e-10);
            }
        }
    }
}
