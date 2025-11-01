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
pub mod barrett;  // Barrett reduction for fast modular arithmetic
pub mod bivector;
pub mod classical;
pub mod clifford_ring;
pub mod clifford_ring_int;  // Integer arithmetic for cryptography
pub mod clifford_ring_simd;  // SIMD-optimized geometric product (ARM NEON)
pub mod ntt;  // Number Theoretic Transform for O(N log N) polynomial multiplication
pub mod ntt_clifford;  // NTT extended to Clifford ring polynomials
pub mod fast_rng;
pub mod shake_rng;  // SHAKE128-based deterministic RNG (Kyber-style)
pub mod shake_poly;  // Fast polynomial generation using SHAKE128
pub mod ga;
pub mod ga_simd_optimized;
pub mod lazy_reduction;  // Lazy modular reduction for performance
pub mod multivector;
pub mod numerical_checks;
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

// Specialized numerical types
pub use numerical_checks::multivector2::Multivector2;
