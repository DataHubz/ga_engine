#![doc = include_str!("../README.md")]

pub mod classical;
pub mod vector;
pub mod bivector;
pub mod multivector;
pub mod rotor;
pub mod transform;
pub mod ga;
pub mod prelude;

pub mod ops;

pub use vector::{Vec3, Rounded};
pub use bivector::Bivector3;
pub use multivector::Multivector3;
pub use rotor::Rotor3;
pub use transform::apply_matrix3;
pub use classical::multiply_matrices;
pub use ga::{geometric_product, geometric_product_full};

pub use crate::ops::projection::*;      // Vec3Projection
pub use crate::ops::reflection::*;      // Vec3Reflect + reflect_ga
pub use crate::ops::motor::Motor3;

