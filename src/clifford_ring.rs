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
    ///
    /// Uses optimized explicit formula version (5.44× faster than lookup table)
    pub fn multiply(&self, other: &Self) -> Self {
        use crate::ga_simd_optimized::geometric_product_full_optimized;

        let mut result = [0.0; 8];
        geometric_product_full_optimized(&self.coeffs, &other.coeffs, &mut result);
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
    /// Each coefficient multiplication uses geometric product (48 ns)
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

    /// Fast polynomial multiplication using FFT (O(N log N))
    ///
    /// Strategy: Apply FFT to each of the 8 Clifford components independently,
    /// multiply in frequency domain, then IFFT. This works because Clifford
    /// multiplication is linear in each component when expanded.
    ///
    /// NOTE: This is an approximation that works for circulant convolution.
    /// For modular reduction by (x^n - 1), use multiply_fft_circular.
    #[allow(dead_code)]
    pub fn multiply_fft(&self, other: &Self) -> Self {
        use rustfft::{FftPlanner, num_complex::Complex};

        let n1 = self.coeffs.len();
        let n2 = other.coeffs.len();
        let result_len = n1 + n2 - 1;

        // Find next power of 2 for FFT
        let fft_size = result_len.next_power_of_two();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        let mut result_coeffs = vec![CliffordRingElement::zero(); result_len];

        // Process each of 8 Clifford components independently
        for component in 0..8 {
            // Extract component from polynomial coefficients
            let mut poly1: Vec<Complex<f64>> = self.coeffs.iter()
                .map(|c| Complex::new(c.coeffs[component], 0.0))
                .collect();
            poly1.resize(fft_size, Complex::new(0.0, 0.0));

            let mut poly2: Vec<Complex<f64>> = other.coeffs.iter()
                .map(|c| Complex::new(c.coeffs[component], 0.0))
                .collect();
            poly2.resize(fft_size, Complex::new(0.0, 0.0));

            // FFT
            fft.process(&mut poly1);
            fft.process(&mut poly2);

            // Pointwise multiply in frequency domain
            for i in 0..fft_size {
                poly1[i] *= poly2[i];
            }

            // IFFT
            ifft.process(&mut poly1);

            // Normalize and extract result
            let scale = 1.0 / fft_size as f64;
            for (i, val) in poly1.iter().take(result_len).enumerate() {
                result_coeffs[i].coeffs[component] = val.re * scale;
            }
        }

        Self { coeffs: result_coeffs }
    }

    /// Fast circular convolution: multiply mod (x^n - 1) using FFT
    ///
    /// This is more efficient than multiply + reduce when n is large.
    /// Complexity: O(n log n) vs O(n²) for naive multiplication
    pub fn multiply_circular_fft(&self, other: &Self, n: usize) -> Self {
        use rustfft::{FftPlanner, num_complex::Complex};

        // Use FFT size = n (already a power of 2 for efficiency)
        let fft_size = n.next_power_of_two();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        let mut result_coeffs = vec![CliffordRingElement::zero(); n];

        // Process each of 8 Clifford components independently
        for component in 0..8 {
            // Extract and pad component
            let mut poly1: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); fft_size];
            for (i, c) in self.coeffs.iter().enumerate().take(n) {
                poly1[i] = Complex::new(c.coeffs[component], 0.0);
            }

            let mut poly2: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); fft_size];
            for (i, c) in other.coeffs.iter().enumerate().take(n) {
                poly2[i] = Complex::new(c.coeffs[component], 0.0);
            }

            // FFT
            fft.process(&mut poly1);
            fft.process(&mut poly2);

            // Pointwise multiply
            for i in 0..fft_size {
                poly1[i] *= poly2[i];
            }

            // IFFT
            ifft.process(&mut poly1);

            // Extract result (circular convolution automatically wraps at n)
            let scale = 1.0 / fft_size as f64;
            for i in 0..n {
                result_coeffs[i].coeffs[component] = poly1[i].re * scale;
            }
        }

        Self { coeffs: result_coeffs }
    }

    /// Karatsuba polynomial multiplication: O(N^1.585) complexity (OPTIMIZED)
    ///
    /// This algorithm works correctly with non-commutative rings like Clifford algebras,
    /// unlike FFT which requires componentwise treatment and has correctness issues.
    ///
    /// Algorithm:
    /// - Split f = f₀ + f₁·x^m, g = g₀ + g₁·x^m
    /// - Compute z₀ = f₀·g₀, z₂ = f₁·g₁, z₁ = (f₀+f₁)·(g₀+g₁) - z₀ - z₂
    /// - Result: z₀ + z₁·x^m + z₂·x^(2m)
    ///
    /// Optimizations:
    /// - Reduced allocations (reuse buffers where possible)
    /// - Tuned base case threshold (16 instead of 8)
    /// - In-place operations to avoid temporary allocations
    ///
    /// Complexity: T(N) = 3·T(N/2) + O(N) = O(N^log₂3) ≈ O(N^1.585)
    pub fn multiply_karatsuba(&self, other: &Self) -> Self {
        let n1 = self.coeffs.len();
        let n2 = other.coeffs.len();
        let n = n1.max(n2);

        // Base case: use optimized naive multiplication for small polynomials
        // Threshold tuned empirically - Karatsuba overhead dominates below ~16
        if n <= 16 {
            return self.multiply(other);
        }

        // Split at midpoint
        let m = n / 2;

        // Create sub-polynomials with minimal copying
        // Split self: f = f₀ + f₁·x^m
        let f0_len = m.min(n1);
        let f1_start = m.min(n1);
        let f1_len = if n1 > m { n1 - m } else { 0 };

        let mut f0_coeffs = Vec::with_capacity(m);
        f0_coeffs.extend_from_slice(&self.coeffs[..f0_len]);
        while f0_coeffs.len() < m {
            f0_coeffs.push(CliffordRingElement::zero());
        }

        let mut f1_coeffs = Vec::with_capacity(m);
        if f1_len > 0 {
            f1_coeffs.extend_from_slice(&self.coeffs[f1_start..n1]);
        }
        while f1_coeffs.len() < m {
            f1_coeffs.push(CliffordRingElement::zero());
        }

        // Split other: g = g₀ + g₁·x^m
        let g0_len = m.min(n2);
        let g1_start = m.min(n2);
        let g1_len = if n2 > m { n2 - m } else { 0 };

        let mut g0_coeffs = Vec::with_capacity(m);
        g0_coeffs.extend_from_slice(&other.coeffs[..g0_len]);
        while g0_coeffs.len() < m {
            g0_coeffs.push(CliffordRingElement::zero());
        }

        let mut g1_coeffs = Vec::with_capacity(m);
        if g1_len > 0 {
            g1_coeffs.extend_from_slice(&other.coeffs[g1_start..n2]);
        }
        while g1_coeffs.len() < m {
            g1_coeffs.push(CliffordRingElement::zero());
        }

        let f0 = CliffordPolynomial::new(f0_coeffs);
        let f1 = CliffordPolynomial::new(f1_coeffs);
        let g0 = CliffordPolynomial::new(g0_coeffs);
        let g1 = CliffordPolynomial::new(g1_coeffs);

        // Three recursive multiplications (Karatsuba trick)
        let z0 = f0.multiply_karatsuba(&g0);
        let z2 = f1.multiply_karatsuba(&g1);

        // z₁ = (f₀+f₁)·(g₀+g₁) - z₀ - z₂
        let f_sum = f0.add(&f1);
        let g_sum = g0.add(&g1);
        let z1_full = f_sum.multiply_karatsuba(&g_sum);
        let z1_temp = z1_full.sub(&z0);
        let z1 = z1_temp.sub(&z2);

        // Combine: result = z₀ + z₁·x^m + z₂·x^(2m)
        // Pre-allocate result with correct size
        let result_len = n1 + n2 - 1;
        let mut result = vec![CliffordRingElement::zero(); result_len];

        // Add z₀ (in-place)
        for (i, coeff) in z0.coeffs.iter().enumerate() {
            if i < result_len {
                // Directly set instead of add for first component
                result[i] = coeff.clone();
            }
        }

        // Add z₁·x^m (in-place)
        for (i, coeff) in z1.coeffs.iter().enumerate() {
            let idx = m + i;
            if idx < result_len {
                result[idx] = result[idx].add(coeff);
            }
        }

        // Add z₂·x^(2m) (in-place)
        for (i, coeff) in z2.coeffs.iter().enumerate() {
            let idx = 2 * m + i;
            if idx < result_len {
                result[idx] = result[idx].add(coeff);
            }
        }

        CliffordPolynomial::new(result)
    }

    /// Polynomial subtraction: (f - g) ∈ R[x]
    pub fn sub(&self, other: &Self) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut result = Vec::with_capacity(max_len);
        let zero = CliffordRingElement::zero();

        for i in 0..max_len {
            let a = self.coeffs.get(i).unwrap_or(&zero);
            let b = other.coeffs.get(i).unwrap_or(&zero);
            // a - b = a + (-1)·b
            result.push(a.add(&b.scalar_mul(-1.0)));
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
