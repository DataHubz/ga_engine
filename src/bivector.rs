// src/bivector.rs
use crate::vector::Vec3;

/// A grade-2 multivector in 3-D: e23, e31, e12.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Bivector3 {
    /// e23 component
    pub xy: f64,
    /// e31 component
    pub yz: f64,
    /// e12 component
    pub zx: f64,
}

impl Bivector3 {
    pub fn new(xy: f64, yz: f64, zx: f64) -> Self {
        Self { xy, yz, zx }
    }

    /// a ∧ b
    pub fn from_wedge(a: Vec3, b: Vec3) -> Self {
        Self {
            xy: a.y * b.z - a.z * b.y,
            yz: a.z * b.x - a.x * b.z,
            zx: a.x * b.y - a.y * b.x,
        }
    }
}
