// src/vector.rs

use std::fmt;

/// A 3-D Euclidean vector.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
    pub fn norm(&self) -> f64 { self.dot(self).sqrt() }
    pub fn scale(&self, s: f64) -> Self { Self::new(self.x * s, self.y * s, self.z * s) }
}

use std::ops::{Add, Sub, Mul};

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 { Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z) }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 { Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z) }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f64) -> Vec3 { Vec3::new(self.x * rhs, self.y * rhs, self.z * rhs) }
}

/// A tiny wrapper for printing a Vec3 rounded to `decimals` places.
pub struct Rounded<'a>(pub &'a Vec3, pub usize);

impl<'a> fmt::Display for Rounded<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Rounded(v, dec) = *self;
        write!(
            f,
            "Vec3 {{ x: {x:.dec$}, y: {y:.dec$}, z: {z:.dec$} }}",
            x = v.x,
            y = v.y,
            z = v.z,
            dec = dec
        )
    }
}

impl<'a> Rounded<'a> {
  /// Wrap a `&Vec3` for pretty‐printing with `decimals` digits.
  pub fn new(v: &'a Vec3, decimals: usize) -> Self {
      Rounded(v, decimals)
  }
}
