//! Parameter sets for Clifford-FHE
//!
//! Defines security levels and corresponding CKKS parameters optimized
//! for geometric algebra operations.

/// Security levels following NIST standards
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityLevel {
    /// ~128-bit security (NIST Level 1)
    Bit128,
    /// ~192-bit security (NIST Level 3)
    Bit192,
    /// ~256-bit security (NIST Level 5)
    Bit256,
}

/// CKKS parameters for Clifford-FHE
#[derive(Debug, Clone)]
pub struct CliffordFHEParams {
    /// Ring dimension (polynomial degree)
    /// Must be power of 2 for NTT
    pub n: usize,

    /// Ciphertext modulus chain (for leveled FHE)
    /// Each level uses a different modulus for rescaling
    pub moduli: Vec<i64>,

    /// Scaling factor (determines precision)
    /// Larger = more precision but more noise
    pub scale: f64,

    /// Standard deviation for error distribution
    pub error_std: f64,

    /// Security level
    pub security: SecurityLevel,
}

impl CliffordFHEParams {
    /// Minimal parameters for testing (NOT SECURE!)
    ///
    /// Use this only for development/testing. Much faster key generation.
    pub fn new_test() -> Self {
        Self {
            n: 64, // Very small for fast testing
            moduli: vec![
                // Just a few levels for testing
                40, // Level 0
                40, // Level 1
                40, // Level 2
            ]
            .iter()
            .map(|bits| Self::generate_prime(*bits))
            .collect(),
            scale: 2f64.powi(20), // Smaller scale for testing
            error_std: 3.2,
            security: SecurityLevel::Bit128, // Claimed, not actual!
        }
    }

    /// Parameters for 128-bit security
    ///
    /// Optimized for:
    /// - Multivector encryption (8 components)
    /// - ~10 levels of homomorphic operations
    /// - Geometric product + rotations
    pub fn new_128bit() -> Self {
        Self {
            n: 8192, // Smaller than typical CKKS (usually 2^14-2^16)
            // because we pack efficiently
            moduli: vec![
                // Level 0 (fresh ciphertext): Large modulus
                60,  // 60-bit prime
                50, 50, 50, // Intermediate levels
                50, 50, 50, // More levels
                50, 50, 50, // For deep circuits
                40, // Final level
            ]
            .iter()
            .map(|bits| Self::generate_prime(*bits))
            .collect(),
            scale: 2f64.powi(40), // 40-bit precision (~12 decimal digits)
            error_std: 3.2,       // Standard CKKS error
            security: SecurityLevel::Bit128,
        }
    }

    /// Parameters for 192-bit security
    pub fn new_192bit() -> Self {
        Self {
            n: 16384,
            moduli: vec![60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 40]
                .iter()
                .map(|bits| Self::generate_prime(*bits))
                .collect(),
            scale: 2f64.powi(40),
            error_std: 3.2,
            security: SecurityLevel::Bit192,
        }
    }

    /// Parameters for 256-bit security
    pub fn new_256bit() -> Self {
        Self {
            n: 32768,
            moduli: vec![60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 40]
                .iter()
                .map(|bits| Self::generate_prime(*bits))
                .collect(),
            scale: 2f64.powi(40),
            error_std: 3.2,
            security: SecurityLevel::Bit256,
        }
    }

    /// Generate NTT-friendly prime of given bit-length
    ///
    /// Prime must satisfy: q ≡ 1 (mod 2N) for NTT to work
    fn generate_prime(bits: u32) -> i64 {
        // For now, use precomputed primes
        // In production, would generate dynamically
        match bits {
            40 => 1_099_511_627_689, // 40-bit NTT-friendly prime
            50 => 1_125_899_906_826_241, // 50-bit
            60 => 1_152_921_504_598_630_401, // 60-bit (close to 2^60)
            _ => panic!("Unsupported bit length: {}", bits),
        }
    }

    /// Get current level modulus
    pub fn modulus_at_level(&self, level: usize) -> i64 {
        if level >= self.moduli.len() {
            panic!("Level {} exceeds maximum {}", level, self.moduli.len() - 1);
        }
        self.moduli[level]
    }

    /// Number of levels (depth) available
    pub fn max_level(&self) -> usize {
        self.moduli.len() - 1
    }

    /// Get product of all moduli up to given level
    /// This is the effective modulus for ciphertexts at that level
    pub fn modulus_product_up_to(&self, level: usize) -> f64 {
        self.moduli[..=level]
            .iter()
            .map(|&q| q as f64)
            .product()
    }
}

impl Default for CliffordFHEParams {
    fn default() -> Self {
        Self::new_128bit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_creation() {
        let params = CliffordFHEParams::new_128bit();
        assert_eq!(params.n, 8192);
        assert_eq!(params.security, SecurityLevel::Bit128);
        assert!(params.max_level() >= 10);
    }

    #[test]
    fn test_modulus_at_level() {
        let params = CliffordFHEParams::new_128bit();
        let q0 = params.modulus_at_level(0);
        assert!(q0 > 0);
        assert!(q0 % (2 * params.n as i64) == 1); // NTT-friendly check
    }
}
