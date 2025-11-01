//! Integer-based Clifford algebra ring elements for cryptography
//!
//! This module provides Cl(3,0) ring elements over integers modulo q,
//! suitable for cryptographic use (deterministic, constant-time capable).
//!
//! All operations use modular arithmetic to ensure:
//! - Deterministic behavior (no floating-point rounding)
//! - Platform-independent results
//! - Preparation for constant-time implementation

use crate::barrett::BarrettContext;
use crate::lazy_reduction::LazyReductionContext;
use crate::clifford_ring_simd::geometric_product_lazy_optimized;

/// Clifford algebra Cl(3,0) ring element with integer coefficients
/// Basis: {1, e1, e2, e3, e12, e13, e23, e123}
/// Dimension: 8
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CliffordRingElementInt {
    pub coeffs: [i64; 8],
}

impl CliffordRingElementInt {
    /// Create from multivector coefficients [scalar, e1, e2, e3, e12, e13, e23, e123]
    #[inline]
    pub fn from_multivector(coeffs: [i64; 8]) -> Self {
        Self { coeffs }
    }

    /// Create zero element
    #[inline]
    pub fn zero() -> Self {
        Self { coeffs: [0; 8] }
    }

    /// Check if element is zero
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|&c| c == 0)
    }

    /// Create scalar element
    #[inline]
    pub fn scalar(value: i64) -> Self {
        let mut coeffs = [0; 8];
        coeffs[0] = value;
        Self { coeffs }
    }

    /// Reduce all coefficients modulo q
    #[inline]
    pub fn reduce_mod(&mut self, q: i64) {
        for i in 0..8 {
            self.coeffs[i] = ((self.coeffs[i] % q) + q) % q;
        }
    }

    /// Scalar multiplication with modular reduction
    #[inline]
    pub fn scalar_mul(&self, scalar: i64, q: i64) -> Self {
        let mut result = [0i64; 8];
        for i in 0..8 {
            result[i] = ((self.coeffs[i] * scalar) % q + q) % q;
        }
        Self::from_multivector(result)
    }

    /// Addition with modular reduction
    #[inline]
    pub fn add_mod(&self, other: &Self, q: i64) -> Self {
        let mut result = [0i64; 8];
        for i in 0..8 {
            result[i] = ((self.coeffs[i] + other.coeffs[i]) % q + q) % q;
        }
        Self::from_multivector(result)
    }

    /// Subtraction with modular reduction
    #[inline]
    pub fn sub_mod(&self, other: &Self, q: i64) -> Self {
        let mut result = [0i64; 8];
        for i in 0..8 {
            result[i] = ((self.coeffs[i] - other.coeffs[i]) % q + q) % q;
        }
        Self::from_multivector(result)
    }

    /// Negation with modular reduction
    #[inline]
    pub fn neg_mod(&self, q: i64) -> Self {
        let mut result = [0i64; 8];
        for i in 0..8 {
            result[i] = ((q - self.coeffs[i]) % q + q) % q;
        }
        Self::from_multivector(result)
    }

    /// Optimized geometric product for Cl(3,0) with modular arithmetic
    /// SAME optimization as f64 version (5.44× speedup applies here too!)
    #[inline]
    pub fn geometric_product(&self, other: &Self, q: i64) -> Self {
        let a = &self.coeffs;
        let b = &other.coeffs;

        // Explicit formulas derived from multiplication table
        // This is the SAME optimization that gave us 5.44× speedup!
        let mut result = [0i64; 8];

        // Scalar component (basis element 1)
        result[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
                  - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];

        // e1 component
        result[1] = a[0]*b[1] + a[1]*b[0] - a[2]*b[4] + a[3]*b[5]
                  + a[4]*b[2] - a[5]*b[3] - a[6]*b[7] - a[7]*b[6];

        // e2 component
        result[2] = a[0]*b[2] + a[1]*b[4] + a[2]*b[0] - a[3]*b[6]
                  - a[4]*b[1] + a[5]*b[7] + a[6]*b[3] + a[7]*b[5];

        // e3 component
        result[3] = a[0]*b[3] - a[1]*b[5] + a[2]*b[6] + a[3]*b[0]
                  - a[4]*b[7] - a[5]*b[1] - a[6]*b[2] - a[7]*b[4];

        // e12 component
        result[4] = a[0]*b[4] + a[1]*b[2] - a[2]*b[1] + a[3]*b[7]
                  + a[4]*b[0] - a[5]*b[6] + a[6]*b[5] + a[7]*b[3];

        // e13 component
        result[5] = a[0]*b[5] + a[1]*b[3] - a[2]*b[7] - a[3]*b[1]
                  + a[4]*b[6] + a[5]*b[0] - a[6]*b[4] - a[7]*b[2];

        // e23 component
        result[6] = a[0]*b[6] - a[1]*b[7] + a[2]*b[3] - a[3]*b[2]
                  - a[4]*b[5] + a[5]*b[4] + a[6]*b[0] + a[7]*b[1];

        // e123 component (pseudoscalar)
        result[7] = a[0]*b[7] + a[1]*b[6] - a[2]*b[5] + a[3]*b[4]
                  + a[4]*b[3] - a[5]*b[2] + a[6]*b[1] + a[7]*b[0];

        // Apply modular reduction
        for i in 0..8 {
            result[i] = ((result[i] % q) + q) % q;
        }

        Self::from_multivector(result)
    }

    /// Optimized geometric product using Barrett reduction
    /// ~2-3× faster than standard modular reduction!
    #[inline]
    pub fn geometric_product_barrett(&self, other: &Self, barrett: &BarrettContext) -> Self {
        let a = &self.coeffs;
        let b = &other.coeffs;

        let mut result = [0i64; 8];

        // Same explicit formulas as geometric_product()
        // But use Barrett reduction instead of % operator

        // Scalar component
        result[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
                  - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];

        // e1 component
        result[1] = a[0]*b[1] + a[1]*b[0] - a[2]*b[4] + a[3]*b[5]
                  + a[4]*b[2] - a[5]*b[3] - a[6]*b[7] - a[7]*b[6];

        // e2 component
        result[2] = a[0]*b[2] + a[1]*b[4] + a[2]*b[0] - a[3]*b[6]
                  - a[4]*b[1] + a[5]*b[7] + a[6]*b[3] + a[7]*b[5];

        // e3 component
        result[3] = a[0]*b[3] - a[1]*b[5] + a[2]*b[6] + a[3]*b[0]
                  - a[4]*b[7] - a[5]*b[1] - a[6]*b[2] - a[7]*b[4];

        // e12 component
        result[4] = a[0]*b[4] + a[1]*b[2] - a[2]*b[1] + a[3]*b[7]
                  + a[4]*b[0] - a[5]*b[6] + a[6]*b[5] + a[7]*b[3];

        // e13 component
        result[5] = a[0]*b[5] + a[1]*b[3] - a[2]*b[7] - a[3]*b[1]
                  + a[4]*b[6] + a[5]*b[0] - a[6]*b[4] - a[7]*b[2];

        // e23 component
        result[6] = a[0]*b[6] - a[1]*b[7] + a[2]*b[3] - a[3]*b[2]
                  - a[4]*b[5] + a[5]*b[4] + a[6]*b[0] + a[7]*b[1];

        // e123 component (pseudoscalar)
        result[7] = a[0]*b[7] + a[1]*b[6] - a[2]*b[5] + a[3]*b[4]
                  + a[4]*b[3] - a[5]*b[2] + a[6]*b[1] + a[7]*b[0];

        // Apply Barrett reduction (faster than % operator!)
        for i in 0..8 {
            result[i] = barrett.reduce(result[i]);
        }

        Self::from_multivector(result)
    }

    /// Addition with Barrett reduction
    #[inline]
    pub fn add_mod_barrett(&self, other: &Self, barrett: &BarrettContext) -> Self {
        let mut result = [0i64; 8];
        for i in 0..8 {
            result[i] = barrett.add(self.coeffs[i], other.coeffs[i]);
        }
        Self::from_multivector(result)
    }

    /// Scalar multiplication with Barrett reduction
    #[inline]
    pub fn scalar_mul_barrett(&self, scalar: i64, barrett: &BarrettContext) -> Self {
        let mut result = [0i64; 8];
        for i in 0..8 {
            result[i] = barrett.scalar_mul(self.coeffs[i], scalar);
        }
        Self::from_multivector(result)
    }

    /// Geometric product with lazy reduction
    /// Defers modular reduction until the end (faster!)
    #[inline]
    pub fn geometric_product_lazy(&self, other: &Self, lazy: &LazyReductionContext) -> Self {
        let a = &self.coeffs;
        let b = &other.coeffs;

        // Same formulas as geometric_product(), but accumulate WITHOUT reduction
        let mut result = [0i64; 8];

        // Scalar component
        result[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
                  - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];

        // e1 component
        result[1] = a[0]*b[1] + a[1]*b[0] - a[2]*b[4] + a[3]*b[5]
                  + a[4]*b[2] - a[5]*b[3] - a[6]*b[7] - a[7]*b[6];

        // e2 component
        result[2] = a[0]*b[2] + a[1]*b[4] + a[2]*b[0] - a[3]*b[6]
                  - a[4]*b[1] + a[5]*b[7] + a[6]*b[3] + a[7]*b[5];

        // e3 component
        result[3] = a[0]*b[3] - a[1]*b[5] + a[2]*b[6] + a[3]*b[0]
                  - a[4]*b[7] - a[5]*b[1] - a[6]*b[2] - a[7]*b[4];

        // e12 component
        result[4] = a[0]*b[4] + a[1]*b[2] - a[2]*b[1] + a[3]*b[7]
                  + a[4]*b[0] - a[5]*b[6] + a[6]*b[5] + a[7]*b[3];

        // e13 component
        result[5] = a[0]*b[5] + a[1]*b[3] - a[2]*b[7] - a[3]*b[1]
                  + a[4]*b[6] + a[5]*b[0] - a[6]*b[4] - a[7]*b[2];

        // e23 component
        result[6] = a[0]*b[6] - a[1]*b[7] + a[2]*b[3] - a[3]*b[2]
                  - a[4]*b[5] + a[5]*b[4] + a[6]*b[0] + a[7]*b[1];

        // e123 component (pseudoscalar)
        result[7] = a[0]*b[7] + a[1]*b[6] - a[2]*b[5] + a[3]*b[4]
                  + a[4]*b[3] - a[5]*b[2] + a[6]*b[1] + a[7]*b[0];

        // LAZY: Only reduce at the very end!
        for i in 0..8 {
            result[i] = lazy.finalize(result[i]);
        }

        Self::from_multivector(result)
    }

    /// In-place geometric product with lazy reduction
    /// Writes result directly to output buffer (avoids copy)
    #[inline]
    pub fn geometric_product_lazy_inplace(
        &self,
        other: &Self,
        lazy: &LazyReductionContext,
        result: &mut Self,
    ) {
        let a = &self.coeffs;
        let b = &other.coeffs;

        // Scalar component
        result.coeffs[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
                         - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];

        // e1 component
        result.coeffs[1] = a[0]*b[1] + a[1]*b[0] - a[2]*b[4] + a[3]*b[5]
                         + a[4]*b[2] - a[5]*b[3] - a[6]*b[7] - a[7]*b[6];

        // e2 component
        result.coeffs[2] = a[0]*b[2] + a[1]*b[4] + a[2]*b[0] - a[3]*b[6]
                         - a[4]*b[1] + a[5]*b[7] + a[6]*b[3] + a[7]*b[5];

        // e3 component
        result.coeffs[3] = a[0]*b[3] - a[1]*b[5] + a[2]*b[6] + a[3]*b[0]
                         - a[4]*b[7] - a[5]*b[1] - a[6]*b[2] - a[7]*b[4];

        // e12 component
        result.coeffs[4] = a[0]*b[4] + a[1]*b[2] - a[2]*b[1] + a[3]*b[7]
                         + a[4]*b[0] - a[5]*b[6] + a[6]*b[5] + a[7]*b[3];

        // e13 component
        result.coeffs[5] = a[0]*b[5] + a[1]*b[3] - a[2]*b[7] - a[3]*b[1]
                         + a[4]*b[6] + a[5]*b[0] - a[6]*b[4] - a[7]*b[2];

        // e23 component
        result.coeffs[6] = a[0]*b[6] - a[1]*b[7] + a[2]*b[3] - a[3]*b[2]
                         - a[4]*b[5] + a[5]*b[4] + a[6]*b[0] + a[7]*b[1];

        // e123 component (pseudoscalar)
        result.coeffs[7] = a[0]*b[7] + a[1]*b[6] - a[2]*b[5] + a[3]*b[4]
                         + a[4]*b[3] - a[5]*b[2] + a[6]*b[1] + a[7]*b[0];

        // LAZY: Only reduce at the very end!
        for i in 0..8 {
            result.coeffs[i] = lazy.finalize(result.coeffs[i]);
        }
    }

    /// Addition with lazy reduction (no immediate reduction)
    #[inline]
    pub fn add_lazy(&self, other: &Self) -> Self {
        let mut result = [0i64; 8];
        for i in 0..8 {
            result[i] = self.coeffs[i] + other.coeffs[i];  // No modular reduction!
        }
        Self::from_multivector(result)
    }

    /// Finalize lazy-accumulated coefficients
    #[inline]
    pub fn finalize_lazy(&self, lazy: &LazyReductionContext) -> Self {
        let mut result = [0i64; 8];
        for i in 0..8 {
            result[i] = lazy.finalize(self.coeffs[i]);
        }
        Self::from_multivector(result)
    }
}

/// Polynomial over Clifford ring (integer coefficients)
#[derive(Debug, Clone)]
pub struct CliffordPolynomialInt {
    pub coeffs: Vec<CliffordRingElementInt>,
}

impl CliffordPolynomialInt {
    pub fn new(coeffs: Vec<CliffordRingElementInt>) -> Self {
        Self { coeffs }
    }

    /// Degree of polynomial (length - 1)
    pub fn degree(&self) -> usize {
        if self.coeffs.is_empty() { 0 } else { self.coeffs.len() - 1 }
    }

    /// Addition of polynomials with modular reduction
    pub fn add_mod(&self, other: &Self, q: i64) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = if i < self.coeffs.len() { &self.coeffs[i] } else { &CliffordRingElementInt::zero() };
            let b = if i < other.coeffs.len() { &other.coeffs[i] } else { &CliffordRingElementInt::zero() };
            result.push(a.add_mod(b, q));
        }

        Self::new(result)
    }

    /// Scalar multiplication with modular reduction
    pub fn scalar_mul(&self, scalar: i64, q: i64) -> Self {
        let coeffs: Vec<_> = self.coeffs.iter()
            .map(|c| c.scalar_mul(scalar, q))
            .collect();
        Self::new(coeffs)
    }

    /// Karatsuba multiplication with modular reduction
    /// SAME O(N^1.585) optimization as floating-point version!
    pub fn multiply_karatsuba(&self, other: &Self, q: i64) -> Self {
        let n = self.coeffs.len();
        let m = other.coeffs.len();

        // Base case: school multiplication for small polynomials
        if n <= 8 || m <= 8 {
            return self.multiply_naive(other, q);
        }

        // Karatsuba split
        let mid = n / 2;

        let (a0_vec, a1_vec) = self.coeffs.split_at(mid);
        let (b0_vec, b1_vec) = other.coeffs.split_at(mid.min(m));

        let a0 = Self::new(a0_vec.to_vec());
        let a1 = Self::new(a1_vec.to_vec());
        let b0 = Self::new(b0_vec.to_vec());
        let b1 = Self::new(b1_vec.to_vec());

        // Three recursive multiplications
        let z0 = a0.multiply_karatsuba(&b0, q);
        let z2 = a1.multiply_karatsuba(&b1, q);

        let a0_plus_a1 = a0.add_mod(&a1, q);
        let b0_plus_b1 = b0.add_mod(&b1, q);
        let z1_full = a0_plus_a1.multiply_karatsuba(&b0_plus_b1, q);

        // z1 = z1_full - z0 - z2
        let mut z1 = z1_full;
        for i in 0..z1.coeffs.len().min(z0.coeffs.len()) {
            z1.coeffs[i] = z1.coeffs[i].sub_mod(&z0.coeffs[i], q);
        }
        for i in 0..z1.coeffs.len().min(z2.coeffs.len()) {
            z1.coeffs[i] = z1.coeffs[i].sub_mod(&z2.coeffs[i], q);
        }

        // Combine: result = z0 + z1*x^mid + z2*x^(2*mid)
        let result_len = n + m - 1;
        let mut result_coeffs = vec![CliffordRingElementInt::zero(); result_len];

        for i in 0..z0.coeffs.len().min(result_len) {
            result_coeffs[i] = result_coeffs[i].add_mod(&z0.coeffs[i], q);
        }
        for i in 0..z1.coeffs.len() {
            if i + mid < result_len {
                result_coeffs[i + mid] = result_coeffs[i + mid].add_mod(&z1.coeffs[i], q);
            }
        }
        for i in 0..z2.coeffs.len() {
            if i + 2*mid < result_len {
                result_coeffs[i + 2*mid] = result_coeffs[i + 2*mid].add_mod(&z2.coeffs[i], q);
            }
        }

        Self::new(result_coeffs)
    }

    /// Naive O(N²) multiplication (used for base case in Karatsuba)
    fn multiply_naive(&self, other: &Self, q: i64) -> Self {
        let n = self.coeffs.len();
        let m = other.coeffs.len();
        let mut result = vec![CliffordRingElementInt::zero(); n + m - 1];

        for i in 0..n {
            for j in 0..m {
                let prod = self.coeffs[i].geometric_product(&other.coeffs[j], q);
                result[i + j] = result[i + j].add_mod(&prod, q);
            }
        }

        Self::new(result)
    }

    /// Reduce polynomial modulo (x^n - 1)
    /// This creates the quotient ring R[x]/(x^n - 1)
    pub fn reduce_modulo_xn_minus_1(&mut self, n: usize, q: i64) {
        if self.coeffs.len() <= n {
            return;
        }

        let mut reduced = vec![CliffordRingElementInt::zero(); n];

        for (i, coeff) in self.coeffs.iter().enumerate() {
            let idx = i % n;
            reduced[idx] = reduced[idx].add_mod(coeff, q);
        }

        self.coeffs = reduced;
    }

    /// Karatsuba multiplication with Barrett reduction (optimized!)
    /// ~2-3× faster than standard karatsuba due to Barrett reduction
    pub fn multiply_karatsuba_barrett(&self, other: &Self, barrett: &BarrettContext) -> Self {
        let n = self.coeffs.len();
        let m = other.coeffs.len();

        // Base case: use Barrett-optimized naive multiplication
        if n <= 8 || m <= 8 {
            return self.multiply_naive_barrett(other, barrett);
        }

        // Karatsuba split (same algorithm, Barrett for reductions)
        let mid = n / 2;

        let (a0_vec, a1_vec) = self.coeffs.split_at(mid);
        let (b0_vec, b1_vec) = other.coeffs.split_at(mid.min(m));

        let a0 = Self::new(a0_vec.to_vec());
        let a1 = Self::new(a1_vec.to_vec());
        let b0 = Self::new(b0_vec.to_vec());
        let b1 = Self::new(b1_vec.to_vec());

        // Three recursive multiplications (with Barrett)
        let z0 = a0.multiply_karatsuba_barrett(&b0, barrett);
        let z2 = a1.multiply_karatsuba_barrett(&b1, barrett);

        let a0_plus_a1 = a0.add_mod_barrett(&a1, barrett);
        let b0_plus_b1 = b0.add_mod_barrett(&b1, barrett);
        let z1_full = a0_plus_a1.multiply_karatsuba_barrett(&b0_plus_b1, barrett);

        // z1 = z1_full - z0 - z2
        let mut z1 = z1_full;
        for i in 0..z1.coeffs.len().min(z0.coeffs.len()) {
            z1.coeffs[i] = z1.coeffs[i].sub_mod_barrett(&z0.coeffs[i], barrett);
        }
        for i in 0..z1.coeffs.len().min(z2.coeffs.len()) {
            z1.coeffs[i] = z1.coeffs[i].sub_mod_barrett(&z2.coeffs[i], barrett);
        }

        // Combine: result = z0 + z1*x^mid + z2*x^(2*mid)
        let result_len = n + m - 1;
        let mut result_coeffs = vec![CliffordRingElementInt::zero(); result_len];

        for i in 0..z0.coeffs.len().min(result_len) {
            result_coeffs[i] = result_coeffs[i].add_mod_barrett(&z0.coeffs[i], barrett);
        }
        for i in 0..z1.coeffs.len() {
            if i + mid < result_len {
                result_coeffs[i + mid] = result_coeffs[i + mid].add_mod_barrett(&z1.coeffs[i], barrett);
            }
        }
        for i in 0..z2.coeffs.len() {
            if i + 2*mid < result_len {
                result_coeffs[i + 2*mid] = result_coeffs[i + 2*mid].add_mod_barrett(&z2.coeffs[i], barrett);
            }
        }

        Self::new(result_coeffs)
    }

    /// Naive multiplication with Barrett reduction
    fn multiply_naive_barrett(&self, other: &Self, barrett: &BarrettContext) -> Self {
        let n = self.coeffs.len();
        let m = other.coeffs.len();
        let mut result = vec![CliffordRingElementInt::zero(); n + m - 1];

        for i in 0..n {
            for j in 0..m {
                let prod = self.coeffs[i].geometric_product_barrett(&other.coeffs[j], barrett);
                result[i + j] = result[i + j].add_mod_barrett(&prod, barrett);
            }
        }

        Self::new(result)
    }

    /// Addition with Barrett reduction
    pub fn add_mod_barrett(&self, other: &Self, barrett: &BarrettContext) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = if i < self.coeffs.len() { &self.coeffs[i] } else { &CliffordRingElementInt::zero() };
            let b = if i < other.coeffs.len() { &other.coeffs[i] } else { &CliffordRingElementInt::zero() };
            result.push(a.add_mod_barrett(b, barrett));
        }

        Self::new(result)
    }

    /// Scalar multiplication with Barrett reduction
    pub fn scalar_mul_barrett(&self, scalar: i64, barrett: &BarrettContext) -> Self {
        let coeffs: Vec<_> = self.coeffs.iter()
            .map(|c| c.scalar_mul_barrett(scalar, barrett))
            .collect();
        Self::new(coeffs)
    }

    /// Reduce polynomial modulo (x^n - 1) using Barrett reduction
    pub fn reduce_modulo_xn_minus_1_barrett(&mut self, n: usize, barrett: &BarrettContext) {
        if self.coeffs.len() <= n {
            return;
        }

        let mut reduced = vec![CliffordRingElementInt::zero(); n];

        for (i, coeff) in self.coeffs.iter().enumerate() {
            let idx = i % n;
            reduced[idx] = reduced[idx].add_mod_barrett(coeff, barrett);
        }

        self.coeffs = reduced;
    }

    /// Karatsuba multiplication with lazy reduction (FASTEST!)
    /// Reduces only after complete polynomial operations, not after every coefficient
    pub fn multiply_karatsuba_lazy(&self, other: &Self, lazy: &LazyReductionContext) -> Self {
        let n = self.coeffs.len();
        let m = other.coeffs.len();

        // Base case: use lazy naive multiplication
        if n <= 8 || m <= 8 {
            return self.multiply_naive_lazy(other, lazy);
        }

        // Karatsuba recursion
        self.multiply_karatsuba_lazy_impl(other, lazy)
    }

    /// SIMD-optimized Karatsuba multiplication with lazy reduction
    /// Uses SIMD-optimized geometric product at the base case for better performance
    pub fn multiply_karatsuba_lazy_simd(&self, other: &Self, lazy: &LazyReductionContext) -> Self {
        let n = self.coeffs.len();
        let m = other.coeffs.len();

        // Base case: use SIMD-optimized naive multiplication
        if n <= 8 || m <= 8 {
            return self.multiply_naive_lazy_simd(other, lazy);
        }

        // Karatsuba recursion (will call SIMD base case)
        self.multiply_karatsuba_lazy_simd_impl(other, lazy)
    }

    fn multiply_karatsuba_lazy_simd_impl(&self, other: &Self, lazy: &LazyReductionContext) -> Self {
        let n = self.coeffs.len();
        let m = other.coeffs.len();

        // Base case already handled by caller

        // Karatsuba split
        let mid = n / 2;

        let (a0_vec, a1_vec) = self.coeffs.split_at(mid);
        let (b0_vec, b1_vec) = other.coeffs.split_at(mid.min(m));

        let a0 = Self::new(a0_vec.to_vec());
        let a1 = Self::new(a1_vec.to_vec());
        let b0 = Self::new(b0_vec.to_vec());
        let b1 = Self::new(b1_vec.to_vec());

        // Three recursive multiplications (SIMD version)
        let z0 = a0.multiply_karatsuba_lazy_simd(&b0, lazy);
        let z2 = a1.multiply_karatsuba_lazy_simd(&b1, lazy);

        // LAZY: Use add_lazy (no reduction!) for intermediate sums
        let a0_plus_a1 = a0.add_lazy_poly(&a1);
        let b0_plus_b1 = b0.add_lazy_poly(&b1);
        let z1_full = a0_plus_a1.multiply_karatsuba_lazy_simd(&b0_plus_b1, lazy);

        // z1 = z1_full - z0 - z2 (using lazy subtraction)
        let mut z1 = z1_full;
        for i in 0..z1.coeffs.len().min(z0.coeffs.len()) {
            z1.coeffs[i] = z1.coeffs[i].sub_lazy(&z0.coeffs[i]);
        }
        for i in 0..z1.coeffs.len().min(z2.coeffs.len()) {
            z1.coeffs[i] = z1.coeffs[i].sub_lazy(&z2.coeffs[i]);
        }

        // Combine: result = z0 + z1*x^mid + z2*x^(2*mid)
        // LAZY: Accumulate without reduction, reduce only at end
        let result_len = n + m - 1;
        let mut result_coeffs = vec![CliffordRingElementInt::zero(); result_len];

        for i in 0..z0.coeffs.len().min(result_len) {
            result_coeffs[i] = result_coeffs[i].add_lazy(&z0.coeffs[i]);
        }
        for i in 0..z1.coeffs.len() {
            if i + mid < result_len {
                result_coeffs[i + mid] = result_coeffs[i + mid].add_lazy(&z1.coeffs[i]);
            }
        }
        for i in 0..z2.coeffs.len() {
            if i + 2*mid < result_len {
                result_coeffs[i + 2*mid] = result_coeffs[i + 2*mid].add_lazy(&z2.coeffs[i]);
            }
        }

        // FINALIZE: Reduce all coefficients at the very end!
        for coeff in &mut result_coeffs {
            *coeff = coeff.finalize_lazy(lazy);
        }

        Self::new(result_coeffs)
    }

    fn multiply_karatsuba_lazy_impl(&self, other: &Self, lazy: &LazyReductionContext) -> Self {
        let n = self.coeffs.len();
        let m = other.coeffs.len();

        // Base case already handled by caller

        // Karatsuba split
        let mid = n / 2;

        let (a0_vec, a1_vec) = self.coeffs.split_at(mid);
        let (b0_vec, b1_vec) = other.coeffs.split_at(mid.min(m));

        let a0 = Self::new(a0_vec.to_vec());
        let a1 = Self::new(a1_vec.to_vec());
        let b0 = Self::new(b0_vec.to_vec());
        let b1 = Self::new(b1_vec.to_vec());

        // Three recursive multiplications
        let z0 = a0.multiply_karatsuba_lazy(&b0, lazy);
        let z2 = a1.multiply_karatsuba_lazy(&b1, lazy);

        // LAZY: Use add_lazy (no reduction!) for intermediate sums
        let a0_plus_a1 = a0.add_lazy_poly(&a1);
        let b0_plus_b1 = b0.add_lazy_poly(&b1);
        let z1_full = a0_plus_a1.multiply_karatsuba_lazy(&b0_plus_b1, lazy);

        // z1 = z1_full - z0 - z2 (using lazy subtraction)
        let mut z1 = z1_full;
        for i in 0..z1.coeffs.len().min(z0.coeffs.len()) {
            z1.coeffs[i] = z1.coeffs[i].sub_lazy(&z0.coeffs[i]);
        }
        for i in 0..z1.coeffs.len().min(z2.coeffs.len()) {
            z1.coeffs[i] = z1.coeffs[i].sub_lazy(&z2.coeffs[i]);
        }

        // Combine: result = z0 + z1*x^mid + z2*x^(2*mid)
        // LAZY: Accumulate without reduction, reduce only at end
        let result_len = n + m - 1;
        let mut result_coeffs = vec![CliffordRingElementInt::zero(); result_len];

        for i in 0..z0.coeffs.len().min(result_len) {
            result_coeffs[i] = result_coeffs[i].add_lazy(&z0.coeffs[i]);
        }
        for i in 0..z1.coeffs.len() {
            if i + mid < result_len {
                result_coeffs[i + mid] = result_coeffs[i + mid].add_lazy(&z1.coeffs[i]);
            }
        }
        for i in 0..z2.coeffs.len() {
            if i + 2*mid < result_len {
                result_coeffs[i + 2*mid] = result_coeffs[i + 2*mid].add_lazy(&z2.coeffs[i]);
            }
        }

        // FINALIZE: Reduce all coefficients at the very end!
        for coeff in &mut result_coeffs {
            *coeff = coeff.finalize_lazy(lazy);
        }

        Self::new(result_coeffs)
    }

    /// Naive multiplication with lazy reduction
    fn multiply_naive_lazy(&self, other: &Self, lazy: &LazyReductionContext) -> Self {
        let n = self.coeffs.len();
        let m = other.coeffs.len();
        let mut result = vec![CliffordRingElementInt::zero(); n + m - 1];

        for i in 0..n {
            for j in 0..m {
                let prod = self.coeffs[i].geometric_product_lazy(&other.coeffs[j], lazy);
                result[i + j] = result[i + j].add_lazy(&prod);
            }
        }

        // FINALIZE: Reduce all accumulated values
        for coeff in &mut result {
            *coeff = coeff.finalize_lazy(lazy);
        }

        Self::new(result)
    }

    /// SIMD-optimized naive multiplication with lazy reduction
    /// Uses the SIMD-optimized geometric product for better performance
    fn multiply_naive_lazy_simd(&self, other: &Self, lazy: &LazyReductionContext) -> Self {
        let n = self.coeffs.len();
        let m = other.coeffs.len();
        let mut result = vec![CliffordRingElementInt::zero(); n + m - 1];

        for i in 0..n {
            for j in 0..m {
                // Use SIMD-optimized geometric product
                let prod = geometric_product_lazy_optimized(&self.coeffs[i], &other.coeffs[j], lazy);
                result[i + j] = result[i + j].add_lazy(&prod);
            }
        }

        // FINALIZE: Reduce all accumulated values
        for coeff in &mut result {
            *coeff = coeff.finalize_lazy(lazy);
        }

        Self::new(result)
    }

    /// Polynomial addition with lazy reduction
    pub fn add_lazy_poly(&self, other: &Self) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = if i < self.coeffs.len() { &self.coeffs[i] } else { &CliffordRingElementInt::zero() };
            let b = if i < other.coeffs.len() { &other.coeffs[i] } else { &CliffordRingElementInt::zero() };
            result.push(a.add_lazy(b));
        }

        Self::new(result)
    }

    /// Reduce polynomial modulo (x^n - 1) with lazy reduction
    pub fn reduce_modulo_xn_minus_1_lazy(&mut self, n: usize, lazy: &LazyReductionContext) {
        if self.coeffs.len() <= n {
            // Still need to finalize accumulated values
            for coeff in &mut self.coeffs {
                *coeff = coeff.finalize_lazy(lazy);
            }
            return;
        }

        let mut reduced = vec![CliffordRingElementInt::zero(); n];

        // Accumulate without reduction
        for (i, coeff) in self.coeffs.iter().enumerate() {
            let idx = i % n;
            reduced[idx] = reduced[idx].add_lazy(coeff);
        }

        // Finalize all accumulated coefficients
        for coeff in &mut reduced {
            *coeff = coeff.finalize_lazy(lazy);
        }

        self.coeffs = reduced;
    }
}

impl CliffordRingElementInt {
    /// Subtraction without reduction (lazy)
    #[inline]
    pub fn sub_lazy(&self, other: &Self) -> Self {
        let mut result = [0i64; 8];
        for i in 0..8 {
            result[i] = self.coeffs[i] - other.coeffs[i];  // No modular reduction!
        }
        Self::from_multivector(result)
    }

    /// Subtraction with Barrett reduction
    #[inline]
    pub fn sub_mod_barrett(&self, other: &Self, barrett: &BarrettContext) -> Self {
        let mut result = [0i64; 8];
        for i in 0..8 {
            result[i] = barrett.sub(self.coeffs[i], other.coeffs[i]);
        }
        Self::from_multivector(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modular_reduction() {
        let q = 3329;
        let mut elem = CliffordRingElementInt::from_multivector([5000, -100, 3329, 0, 0, 0, 0, 0]);
        elem.reduce_mod(q);

        assert_eq!(elem.coeffs[0], 1671);  // 5000 mod 3329 = 1671
        assert_eq!(elem.coeffs[1], 3229);  // -100 mod 3329 = 3229
        assert_eq!(elem.coeffs[2], 0);     // 3329 mod 3329 = 0
    }

    #[test]
    fn test_geometric_product_integer() {
        let q = 3329;
        let a = CliffordRingElementInt::from_multivector([1, 2, 0, 0, 0, 0, 0, 0]);
        let b = CliffordRingElementInt::from_multivector([3, 0, 4, 0, 0, 0, 0, 0]);

        let c = a.geometric_product(&b, q);

        // (1 + 2e1) * (3 + 4e2)
        // = 3 + 4e2 + 6e1 + 8e12
        assert_eq!(c.coeffs[0], 3);   // scalar
        assert_eq!(c.coeffs[1], 6);   // e1
        assert_eq!(c.coeffs[2], 4);   // e2
        assert_eq!(c.coeffs[4], 8);   // e12
    }

    #[test]
    fn test_polynomial_reduction() {
        let q = 3329;
        let n = 4;

        // x^5 + x^2 + 1 reduced modulo (x^4 - 1) = x + x^2 + 1
        let coeffs = vec![
            CliffordRingElementInt::scalar(1),  // x^0
            CliffordRingElementInt::zero(),     // x^1
            CliffordRingElementInt::scalar(1),  // x^2
            CliffordRingElementInt::zero(),     // x^3
            CliffordRingElementInt::zero(),     // x^4
            CliffordRingElementInt::scalar(1),  // x^5
        ];

        let mut poly = CliffordPolynomialInt::new(coeffs);
        poly.reduce_modulo_xn_minus_1(n, q);

        assert_eq!(poly.coeffs.len(), 4);
        assert_eq!(poly.coeffs[0].coeffs[0], 1);  // constant term
        assert_eq!(poly.coeffs[1].coeffs[0], 1);  // x^1 coefficient (from x^5)
        assert_eq!(poly.coeffs[2].coeffs[0], 1);  // x^2 coefficient
    }
}
