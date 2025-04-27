//! # GAEngine Quickstart
//!
//! ```rust
//! use ga_engine::prelude::*;
//!
//! // Rotate (1,0,0) 90° about the Z axis
//! let v = Vec3::new(1.0, 0.0, 0.0);
//! let r = Rotor3::from_axis_angle(
//!     Vec3::new(0.0, 0.0, 1.0),
//!     std::f64::consts::FRAC_PI_2,
//! );
//! let v_rot = r.rotate_fast(v);
//!
//! // Should end up at (0,1,0)
//! const EPS: f64 = 1e-12;
//! assert!((v_rot.x.abs()) < EPS);
//! assert!((v_rot.y - 1.0).abs() < EPS);
//! assert!((v_rot.z).abs() < EPS);
//! ```
//!
#![doc = include_str!("../README.md")]

// Core modules
pub mod bivector;
pub mod classical;
pub mod ga;
pub mod multivector;
pub mod ops;
pub mod prelude;
pub mod rotor;
pub mod transform;
pub mod vector;

// N-dimensional GA support
pub mod nd;

// --- Public API exports ---

// 3D types and operations
pub use bivector::Bivector3;
pub use classical::multiply_matrices;
pub use ga::{geometric_product, geometric_product_full};
pub use multivector::Multivector3;
pub use rotor::Rotor3;
pub use transform::apply_matrix3;
pub use vector::{Rounded, Vec3};

// High-level GA ops
pub use ops::motor::Motor3;
pub use ops::projection::*;
pub use ops::reflection::*;

// N-dimensional types
pub use nd::multivector::Multivector;
pub use nd::vecn::VecN;
