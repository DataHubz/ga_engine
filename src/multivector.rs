// src/multivector.rs
use crate::{vector::Vec3, bivector::Bivector3};

#[derive(Clone, Debug, PartialEq)]
pub struct Multivector3 {
    pub scalar:   f64,
    pub vector:   Vec3,
    pub bivector: Bivector3,
    pub pseudo:   f64,
}

impl Multivector3 {
    pub fn zero() -> Self {
        Self {
            scalar:   0.0,
            vector:   Vec3::new(0.0, 0.0, 0.0),
            bivector: Bivector3::new(0.0, 0.0, 0.0),
            pseudo:   0.0,
        }
    }

    pub fn from_scalar(s: f64) -> Self {
        Self { scalar: s, vector: Vec3::new(0.0,0.0,0.0), bivector: Bivector3::new(0.0,0.0,0.0), pseudo: 0.0 }
    }

    pub fn from_vector(v: Vec3) -> Self {
        Self { scalar: 0.0, vector: v, bivector: Bivector3::new(0.0,0.0,0.0), pseudo: 0.0 }
    }

    pub fn gp(&self, other: &Self) -> Self {
        let mut out = [0.0; 8];
        // flatten each multivector into [f64;8] and call your geometric_product_full
        super::ga::geometric_product_full(
            &[
                self.scalar,
                self.vector.x, self.vector.y, self.vector.z,
                self.bivector.xy, self.bivector.yz, self.bivector.zx,
                self.pseudo,
            ],
            &[
                other.scalar,
                other.vector.x, other.vector.y, other.vector.z,
                other.bivector.xy, other.bivector.yz, other.bivector.zx,
                other.pseudo,
            ],
            &mut out,
        );
        Multivector3 {
            scalar:   out[0],
            vector:   Vec3::new(out[1], out[2], out[3]),
            bivector: Bivector3::new(out[4], out[5], out[6]),
            pseudo:   out[7],
        }
    }

    pub fn reverse(&self) -> Self {
        // grades 2 and 3 change sign
        Self {
            scalar:   self.scalar,
            vector:   self.vector,
            bivector: Bivector3::new(-self.bivector.xy, -self.bivector.yz, -self.bivector.zx),
            pseudo:   -self.pseudo,
        }
    }
}
