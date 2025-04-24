// src/transform.rs
//! Classical ↔ GA transformation adapters.

use crate::vector::Vec3;

/// Apply a 3×3 row-major matrix to a `Vec3`.
///
/// # Example
/// ```
/// # use ga_engine::{Vec3, apply_matrix3};
/// # const EPS: f64 = 1e-12;
/// let v = Vec3::new(1.0, 0.0, 0.0);
/// let m = [
///     0.0, -1.0, 0.0,
///     1.0,  0.0, 0.0,
///     0.0,  0.0, 1.0,
/// ];
/// let rotated = apply_matrix3(&m, v);
/// assert!((rotated.x - 0.0).abs() < EPS);
/// assert!((rotated.y - 1.0).abs() < EPS);
/// assert!((rotated.z - 0.0).abs() < EPS);
/// ```
pub fn apply_matrix3(m: &[f64; 9], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[3] * v.x + m[4] * v.y + m[5] * v.z,
        m[6] * v.x + m[7] * v.y + m[8] * v.z,
    )
}
