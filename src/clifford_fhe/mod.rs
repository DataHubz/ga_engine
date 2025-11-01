//! Clifford-FHE: Fully Homomorphic Encryption for Geometric Algebra
//!
//! This module implements a CKKS-based FHE scheme optimized for Clifford algebra
//! operations. Unlike the LWE-based scheme (clifford_lwe), this supports:
//!
//! - Homomorphic geometric product: Enc(a) ⊗ Enc(b) = Enc(a ⊗ b)
//! - Homomorphic rotations via rotors
//! - Arbitrary depth computations (with bootstrapping)
//!
//! # Architecture
//!
//! ```text
//! Multivector (Cl(3,0))
//!     ↓ encoding
//! Polynomial in R = Z[x]/(x^N + 1)
//!     ↓ CKKS encryption
//! Ciphertext (c0, c1)
//!     ↓ homomorphic ops
//! Result ciphertext
//!     ↓ CKKS decryption
//! Polynomial
//!     ↓ decoding
//! Multivector result
//! ```
//!
//! # Example (Future)
//!
//! ```rust,ignore
//! use clifford_fhe::*;
//!
//! // Generate keys
//! let (pk, sk, evk) = keygen(SecurityLevel::Bit128);
//!
//! // Encrypt multivectors
//! let mv_a = Multivector::from([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
//! let ct_a = encrypt(&pk, &mv_a);
//!
//! let mv_b = Multivector::from([3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
//! let ct_b = encrypt(&pk, &mv_b);
//!
//! // Homomorphic geometric product!
//! let ct_c = geometric_product(&evk, &ct_a, &ct_b);
//!
//! // Decrypt
//! let mv_c = decrypt(&sk, &ct_c);
//! // mv_c should equal mv_a ⊗ mv_b
//! ```

pub mod ckks;
pub mod encoding;
pub mod geometric_product;
pub mod keys;
pub mod operations;
pub mod params;

// Re-exports
pub use ckks::{multiply_by_plaintext, Ciphertext, Plaintext};
pub use encoding::{decode_multivector, encode_multivector};
pub use geometric_product::geometric_product_homomorphic;
pub use keys::{EvaluationKey, PublicKey, SecretKey};
pub use params::CliffordFHEParams;
