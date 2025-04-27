// src/nd/vecn.rs
//! N-dimensional Euclidean vector type and basic operations.

use crate::nd::types::Scalar;
use std::ops::{Add, Mul, Neg, Sub};

/// An N-dimensional Euclidean vector.
#[derive(Debug, Clone, PartialEq)]
pub struct VecN<const N: usize> {
    /// Underlying components array
    pub data: [Scalar; N],
}

impl<const N: usize> VecN<N> {
    /// Construct from an array of length N.
    #[inline(always)]
    pub fn new(data: [Scalar; N]) -> Self {
        Self { data }
    }

    /// Dot product of two VecN.
    #[inline]
    pub fn dot(&self, other: &Self) -> Scalar {
        let mut sum: Scalar = 0.0;
        for i in 0..N {
            sum += self.data[i] * other.data[i];
        }
        sum
    }

    /// Euclidean norm (length).
    #[inline]
    pub fn norm(&self) -> Scalar {
        self.dot(self).sqrt()
    }

    /// Scale the vector by a scalar.
    #[inline]
    pub fn scale(&self, s: Scalar) -> Self {
        let data = self.data.map(|x| x * s);
        Self { data }
    }
}

// Arithmetic operators
impl<const N: usize> Add for VecN<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let mut sum = self.data;
        for (s, r) in sum.iter_mut().zip(rhs.data.iter()) {
            *s += *r;
        }
        Self { data: sum }
    }
}

impl<const N: usize> Sub for VecN<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut diff = self.data;
        for (d, r) in diff.iter_mut().zip(rhs.data.iter()) {
            *d -= *r;
        }
        Self { data: diff }
    }
}

impl<const N: usize> Mul<Scalar> for VecN<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Scalar) -> Self::Output {
        let data = self.data.map(|v| v * rhs);
        Self { data }
    }
}

impl<const N: usize> Neg for VecN<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        let mut negated = self.data;
        for v in &mut negated {
            *v = -*v;
        }
        Self { data: negated }
    }
}
