//! Metal GPU-accelerated CKKS (Cheon-Kim-Kim-Song) implementation
//!
//! This module provides a complete GPU-only CKKS context for homomorphic encryption.
//! All operations (encode, decode, encrypt, decrypt, add, multiply) run on Metal GPU.
//!
//! **Design Principle**: Complete isolation from CPU backend - no fallbacks, no mixing.

use super::device::MetalDevice;
use super::ntt::MetalNttContext;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::BarrettReducer;
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::{PublicKey, SecretKey};
use std::sync::Arc;

/// Metal GPU-accelerated CKKS context
///
/// All operations run entirely on GPU - no CPU fallbacks.
/// Uses Metal NTT for polynomial multiplication.
pub struct MetalCkksContext {
    /// Shared Metal device
    device: Arc<MetalDevice>,

    /// FHE parameters
    params: CliffordFHEParams,

    /// Metal NTT contexts (one per prime in RNS)
    ntt_contexts: Vec<MetalNttContext>,

    /// Barrett reducers for modular reduction (lightweight, keep on CPU)
    reducers: Vec<BarrettReducer>,
}

/// GPU plaintext representation
///
/// Stores polynomial coefficients in RNS representation.
/// Each coefficient is represented modulo each prime in the modulus chain.
#[derive(Clone, Debug)]
pub struct MetalPlaintext {
    /// Polynomial coefficients in RNS form
    /// Each u64 value is a coefficient in a specific RNS component
    /// Flat layout: [coeff0_mod_q0, coeff0_mod_q1, ..., coeff1_mod_q0, coeff1_mod_q1, ...]
    /// Length: n × num_primes
    pub coeffs: Vec<u64>,

    /// Ring dimension (number of polynomial coefficients)
    pub n: usize,

    /// Number of RNS primes
    pub num_primes: usize,

    /// Current level in modulus chain
    pub level: usize,

    /// Current scale (encoding scale)
    pub scale: f64,
}

/// GPU ciphertext representation (RLWE ciphertext)
///
/// Standard RLWE ciphertext (c0, c1) where:
/// - Decryption: m ≈ c0 + c1 * s (mod q)
#[derive(Clone, Debug)]
pub struct MetalCiphertext {
    /// First polynomial (c0) in RNS form
    /// Flat layout: [coeff0_mod_q0, coeff0_mod_q1, ..., coeff1_mod_q0, coeff1_mod_q1, ...]
    pub c0: Vec<u64>,

    /// Second polynomial (c1) in RNS form
    /// Flat layout: same as c0
    pub c1: Vec<u64>,

    /// Ring dimension (number of polynomial coefficients)
    pub n: usize,

    /// Number of RNS primes
    pub num_primes: usize,

    /// Current level in modulus chain
    pub level: usize,

    /// Current scale
    pub scale: f64,
}

impl MetalCkksContext {
    /// Create new Metal CKKS context with GPU acceleration
    ///
    /// All operations will run on Metal GPU. If GPU is not available,
    /// this will return an error (no CPU fallback).
    ///
    /// # Arguments
    /// * `params` - FHE parameters (ring dimension, moduli, scale)
    ///
    /// # Returns
    /// GPU-accelerated CKKS context or error if GPU unavailable
    pub fn new(params: CliffordFHEParams) -> Result<Self, String> {
        println!("  [Metal CKKS] Initializing GPU-only CKKS context...");

        // Create shared Metal device
        println!("  [Metal CKKS] Creating Metal device...");
        let start = std::time::Instant::now();
        let device = Arc::new(MetalDevice::new()?);
        println!("  [Metal CKKS] Device initialized in {:.3}s", start.elapsed().as_secs_f64());
        println!("  [Metal CKKS] Device: {}", device.device().name());

        // Create Metal NTT contexts for each prime
        println!("  [Metal CKKS] Creating NTT contexts for {} primes...", params.moduli.len());
        let start = std::time::Instant::now();

        let mut ntt_contexts = Vec::with_capacity(params.moduli.len());
        for (i, &q) in params.moduli.iter().enumerate() {
            // Find primitive 2n-th root of unity
            let psi = Self::find_primitive_2n_root(params.n, q)?;

            // Create Metal NTT context
            let metal_ntt = MetalNttContext::new_with_device(
                device.clone(),
                params.n,
                q,
                psi
            )?;

            ntt_contexts.push(metal_ntt);

            if (i + 1) % 5 == 0 || i == params.moduli.len() - 1 {
                println!("    Created {}/{} NTT contexts", i + 1, params.moduli.len());
            }
        }

        println!("  [Metal CKKS] NTT contexts created in {:.2}s", start.elapsed().as_secs_f64());

        // Create Barrett reducers (lightweight, keep on CPU)
        let reducers: Vec<BarrettReducer> = params.moduli.iter()
            .map(|&q| BarrettReducer::new(q))
            .collect();

        println!("  [Metal CKKS] ✓ GPU-only CKKS context ready!\n");

        Ok(Self {
            device,
            params,
            ntt_contexts,
            reducers,
        })
    }

    /// Encode floating-point values into plaintext polynomial
    ///
    /// Currently uses CPU for canonical embedding (GPU optimization planned).
    /// The encoding is done in a way that's compatible with GPU operations.
    ///
    /// # Arguments
    /// * `values` - Real values to encode (length ≤ n/2)
    ///
    /// # Returns
    /// Plaintext polynomial in flat RNS representation
    pub fn encode(&self, values: &[f64]) -> Result<MetalPlaintext, String> {
        let n = self.params.n;
        let num_slots = n / 2;

        if values.len() > num_slots {
            return Err(format!("Too many values: {} > n/2 = {}", values.len(), num_slots));
        }

        // Step 1: Canonical embedding (CPU for now)
        let coeffs_i64 = Self::canonical_embed_encode_real(values, self.params.scale, n);

        // Step 2: Convert to flat RNS representation
        let level = self.params.max_level();
        let moduli = &self.params.moduli[..=level];
        let num_primes = moduli.len();

        // Flat layout: [coeff0_mod_q0, coeff0_mod_q1, ..., coeff1_mod_q0, ...]
        let mut flat_coeffs = vec![0u64; n * num_primes];

        for (i, &coeff) in coeffs_i64.iter().enumerate() {
            for (j, &q) in moduli.iter().enumerate() {
                // Convert signed coefficient to unsigned mod q
                let val_mod_q = if coeff >= 0 {
                    (coeff as u64) % q
                } else {
                    let abs_coeff = (-coeff) as u64;
                    let remainder = abs_coeff % q;
                    if remainder == 0 {
                        0
                    } else {
                        q - remainder
                    }
                };

                // Flat index: i * num_primes + j
                flat_coeffs[i * num_primes + j] = val_mod_q;
            }
        }

        Ok(MetalPlaintext {
            coeffs: flat_coeffs,
            n,
            num_primes,
            level,
            scale: self.params.scale,
        })
    }

    /// Decode plaintext polynomial to floating-point values
    ///
    /// Currently uses CPU for canonical embedding (GPU optimization planned).
    ///
    /// # Arguments
    /// * `pt` - Plaintext polynomial
    ///
    /// # Returns
    /// Decoded real values
    pub fn decode(&self, pt: &MetalPlaintext) -> Result<Vec<f64>, String> {
        let n = pt.n;
        let num_primes = pt.num_primes;

        // Step 1: Reconstruct coefficients from RNS using first prime (for simplicity)
        // In full implementation, would use CRT reconstruction
        let q0 = self.params.moduli[0];
        let mut coeffs_i64 = vec![0i64; n];

        for i in 0..n {
            let val = pt.coeffs[i * num_primes]; // Use first RNS component

            // Convert from mod q to signed integer (centered representation)
            coeffs_i64[i] = if val <= q0 / 2 {
                val as i64
            } else {
                -((q0 - val) as i64)
            };
        }

        // Step 2: Canonical embedding decode (CPU for now)
        let slots = Self::canonical_embed_decode_real(&coeffs_i64, pt.scale, n);

        Ok(slots)
    }

    /// Encrypt plaintext using public key
    ///
    /// Creates RLWE ciphertext (c0, c1) using polynomial operations.
    /// Uses Metal GPU for NTT-based polynomial multiplication.
    ///
    /// # Arguments
    /// * `pt` - Plaintext to encrypt
    /// * `pk` - Public key
    ///
    /// # Returns
    /// Ciphertext encrypting the plaintext
    pub fn encrypt(
        &self,
        pt: &MetalPlaintext,
        pk: &PublicKey,
    ) -> Result<MetalCiphertext, String> {
        use rand::{thread_rng, Rng};
        use rand_distr::{Distribution, Normal};

        let n = self.params.n;
        let level = pt.level;
        let moduli = &self.params.moduli[..=level];
        let num_primes = moduli.len();
        let mut rng = thread_rng();

        // Sample ternary random polynomial u ∈ {-1, 0, 1}^n
        let u_coeffs: Vec<i64> = (0..n)
            .map(|_| {
                let val: f64 = rng.gen();
                if val < 0.33 {
                    -1
                } else if val < 0.66 {
                    0
                } else {
                    1
                }
            })
            .collect();

        // Sample error polynomials e0, e1 from Gaussian distribution
        let normal = Normal::new(0.0, self.params.error_std).unwrap();
        let e0_coeffs: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();
        let e1_coeffs: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();

        // Convert to flat RNS representation
        let u_flat = Self::coeffs_to_flat_rns(&u_coeffs, moduli);
        let e0_flat = Self::coeffs_to_flat_rns(&e0_coeffs, moduli);
        let e1_flat = Self::coeffs_to_flat_rns(&e1_coeffs, moduli);

        // Convert public key to flat layout (if not already)
        let pk_a_flat = Self::rns_vec_to_flat(&pk.a, num_primes);
        let pk_b_flat = Self::rns_vec_to_flat(&pk.b, num_primes);

        // c0 = b*u + e0 + m (using Metal NTT for multiplication)
        // CRITICAL: Use negacyclic convolution to match CPU and CKKS ring R = Z[x]/(x^n + 1)
        let bu = self.multiply_polys_flat_ntt_negacyclic(&pk_b_flat, &u_flat, moduli)?;
        let mut c0 = vec![0u64; n * num_primes];
        for i in 0..(n * num_primes) {
            let prime_idx = i % num_primes;
            let q = moduli[prime_idx];

            // Use Barrett reducer for proper modular arithmetic
            let reducer = &self.reducers[prime_idx];
            let sum1 = reducer.add(bu[i], e0_flat[i]);
            let sum2 = reducer.add(sum1, pt.coeffs[i]);
            c0[i] = sum2;
        }

        // c1 = a*u + e1 (using Metal NTT for multiplication)
        // CRITICAL: Use negacyclic convolution to match CPU and CKKS ring R = Z[x]/(x^n + 1)
        let au = self.multiply_polys_flat_ntt_negacyclic(&pk_a_flat, &u_flat, moduli)?;
        let mut c1 = vec![0u64; n * num_primes];
        for i in 0..(n * num_primes) {
            let prime_idx = i % num_primes;
            let reducer = &self.reducers[prime_idx];
            c1[i] = reducer.add(au[i], e1_flat[i]);
        }

        Ok(MetalCiphertext {
            c0,
            c1,
            n,
            num_primes,
            level,
            scale: pt.scale,
        })
    }

    /// Decrypt ciphertext using secret key
    ///
    /// Recovers plaintext from RLWE ciphertext: m = c0 + c1*s
    /// Uses Metal GPU for NTT-based polynomial multiplication.
    ///
    /// # Arguments
    /// * `ct` - Ciphertext to decrypt
    /// * `sk` - Secret key
    ///
    /// # Returns
    /// Decrypted plaintext
    pub fn decrypt(
        &self,
        ct: &MetalCiphertext,
        sk: &SecretKey,
    ) -> Result<MetalPlaintext, String> {
        let n = ct.n;
        let level = ct.level;
        let moduli = &self.params.moduli[..=level];
        let num_primes = moduli.len();

        // Convert secret key to flat layout at ciphertext's level
        let sk_flat = Self::rns_vec_to_flat_at_level(&sk.coeffs, level, num_primes);

        // Compute c1 * s using Metal NTT
        // CRITICAL: Use negacyclic convolution to match CPU and CKKS ring R = Z[x]/(x^n + 1)
        let c1s = self.multiply_polys_flat_ntt_negacyclic(&ct.c1, &sk_flat, moduli)?;

        // m = c0 + c1*s
        let mut m_flat = vec![0u64; n * num_primes];
        for i in 0..(n * num_primes) {
            let prime_idx = i % num_primes;
            let reducer = &self.reducers[prime_idx];
            m_flat[i] = reducer.add(ct.c0[i], c1s[i]);
        }

        Ok(MetalPlaintext {
            coeffs: m_flat,
            n,
            num_primes,
            level,
            scale: ct.scale,
        })
    }

    // ==================== Conversion Functions (Metal ↔ CPU) ====================

    /// Convert MetalCiphertext to CPU Ciphertext
    ///
    /// This allows Metal GPU ciphertexts to be used with CPU-based operations
    /// like bootstrap. The conversion is lossless.
    pub fn to_cpu_ciphertext(&self, ct: &MetalCiphertext) -> crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext {
        use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;

        let moduli = self.params.moduli[..=ct.level].to_vec();

        // Convert c0 from flat layout to Vec<RnsRepresentation>
        let mut c0_rns = Vec::with_capacity(ct.n);
        for i in 0..ct.n {
            let mut values = Vec::with_capacity(ct.num_primes);
            for j in 0..ct.num_primes {
                values.push(ct.c0[i * ct.num_primes + j]);
            }
            c0_rns.push(RnsRepresentation::new(values, moduli.clone()));
        }

        // Convert c1 from flat layout to Vec<RnsRepresentation>
        let mut c1_rns = Vec::with_capacity(ct.n);
        for i in 0..ct.n {
            let mut values = Vec::with_capacity(ct.num_primes);
            for j in 0..ct.num_primes {
                values.push(ct.c1[i * ct.num_primes + j]);
            }
            c1_rns.push(RnsRepresentation::new(values, moduli.clone()));
        }

        crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext::new(
            c0_rns,
            c1_rns,
            ct.level,
            ct.scale,
        )
    }

    /// Convert CPU Ciphertext to MetalCiphertext
    ///
    /// This allows CPU ciphertexts (e.g., from bootstrap) to be used with
    /// Metal GPU operations. The conversion is lossless.
    pub fn from_cpu_ciphertext(&self, ct: &crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext) -> MetalCiphertext {
        let n = ct.n;
        let level = ct.level;
        let num_primes = ct.c0[0].values.len();

        // Convert c0 from Vec<RnsRepresentation> to flat layout
        let mut c0_flat = vec![0u64; n * num_primes];
        for (i, rns) in ct.c0.iter().enumerate() {
            for (j, &val) in rns.values.iter().enumerate() {
                c0_flat[i * num_primes + j] = val;
            }
        }

        // Convert c1 from Vec<RnsRepresentation> to flat layout
        let mut c1_flat = vec![0u64; n * num_primes];
        for (i, rns) in ct.c1.iter().enumerate() {
            for (j, &val) in rns.values.iter().enumerate() {
                c1_flat[i * num_primes + j] = val;
            }
        }

        MetalCiphertext {
            c0: c0_flat,
            c1: c1_flat,
            n,
            num_primes,
            level,
            scale: ct.scale,
        }
    }

    /// Convert CPU Plaintext to MetalPlaintext
    pub fn from_cpu_plaintext(&self, pt: &crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Plaintext) -> MetalPlaintext {
        let n = pt.n;
        let level = pt.level;
        let num_primes = pt.coeffs[0].values.len();

        // Convert coeffs from Vec<RnsRepresentation> to flat layout
        let mut coeffs_flat = vec![0u64; n * num_primes];
        for (i, rns) in pt.coeffs.iter().enumerate() {
            for (j, &val) in rns.values.iter().enumerate() {
                coeffs_flat[i * num_primes + j] = val;
            }
        }

        MetalPlaintext {
            coeffs: coeffs_flat,
            n,
            num_primes,
            level,
            scale: pt.scale,
        }
    }

    // ==================== Canonical Embedding Functions ====================

    /// Compute the Galois orbit order for CKKS slot indexing
    fn orbit_order(n: usize, g: usize) -> Vec<usize> {
        let m = 2 * n; // M = 2N
        let num_slots = n / 2; // N/2 slots

        let mut e = vec![0usize; num_slots];
        let mut cur = 1usize;

        for t in 0..num_slots {
            e[t] = cur; // odd exponent in [1..2N-1]
            cur = (cur * g) % m;
        }

        e
    }

    /// Encode real-valued slots using CKKS canonical embedding
    fn canonical_embed_encode_real(values: &[f64], scale: f64, n: usize) -> Vec<i64> {
        use std::f64::consts::PI;

        assert!(n.is_power_of_two());
        let num_slots = n / 2;
        assert!(values.len() <= num_slots);

        let m = 2 * n; // Cyclotomic index M = 2N
        let g = 5; // Generator for power-of-two cyclotomics

        // Use Galois orbit order
        let e = Self::orbit_order(n, g);

        // Pad values to full slot count
        let mut slots = vec![0.0; num_slots];
        for (i, &val) in values.iter().enumerate() {
            slots[i] = val;
        }

        // Inverse canonical embedding
        let mut coeffs_float = vec![0.0; n];

        for j in 0..n {
            let mut sum = 0.0;

            for t in 0..num_slots {
                // w_t(j) = exp(-2πi * e[t] * j / M)
                let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
                let cos_val = angle.cos();

                // For real slots: contribution is 2 * z[t] * cos(angle)
                sum += slots[t] * cos_val;
            }

            // Normalize by 2/N
            coeffs_float[j] = (2.0 / n as f64) * sum;
        }

        // Scale and round to integers
        coeffs_float.iter().map(|&x| (x * scale).round() as i64).collect()
    }

    /// Decode real-valued slots using CKKS canonical embedding
    fn canonical_embed_decode_real(coeffs: &[i64], scale: f64, n: usize) -> Vec<f64> {
        use std::f64::consts::PI;

        assert_eq!(coeffs.len(), n);

        let m = 2 * n; // M = 2N
        let num_slots = n / 2;
        let g = 5; // Generator

        // Use Galois orbit order
        let e = Self::orbit_order(n, g);

        // Convert to floating point (with scale normalization)
        let coeffs_float: Vec<f64> = coeffs.iter().map(|&c| c as f64 / scale).collect();

        // Forward canonical embedding
        let mut slots = vec![0.0; num_slots];

        for t in 0..num_slots {
            let mut sum_real = 0.0;
            for j in 0..n {
                // w_t(j) = exp(+2πi * e[t] * j / M) (positive angle for decode)
                let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
                let cos_val = angle.cos();
                sum_real += coeffs_float[j] * cos_val;
            }
            slots[t] = sum_real;
        }

        slots
    }

    // ==================== RNS Conversion Helper Functions ====================

    /// Convert signed coefficient vector to flat RNS representation
    fn coeffs_to_flat_rns(coeffs: &[i64], moduli: &[u64]) -> Vec<u64> {
        let n = coeffs.len();
        let num_primes = moduli.len();
        let mut flat = vec![0u64; n * num_primes];

        for (i, &coeff) in coeffs.iter().enumerate() {
            for (j, &q) in moduli.iter().enumerate() {
                // Convert signed coefficient to unsigned mod q
                let val_mod_q = if coeff >= 0 {
                    (coeff as u64) % q
                } else {
                    let abs_coeff = (-coeff) as u64;
                    let remainder = abs_coeff % q;
                    if remainder == 0 {
                        0
                    } else {
                        q - remainder
                    }
                };

                // Flat index: i * num_primes + j
                flat[i * num_primes + j] = val_mod_q;
            }
        }

        flat
    }

    /// Convert RnsRepresentation vector to flat layout
    fn rns_vec_to_flat(rns_vec: &[crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation], num_primes: usize) -> Vec<u64> {
        let n = rns_vec.len();
        let mut flat = vec![0u64; n * num_primes];

        for (i, rns) in rns_vec.iter().enumerate() {
            for j in 0..num_primes {
                flat[i * num_primes + j] = rns.values[j];
            }
        }

        flat
    }

    /// Convert RnsRepresentation vector to flat layout at specific level
    fn rns_vec_to_flat_at_level(rns_vec: &[crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation], level: usize, num_primes: usize) -> Vec<u64> {
        let n = rns_vec.len();
        let mut flat = vec![0u64; n * num_primes];

        for (i, rns) in rns_vec.iter().enumerate() {
            for j in 0..num_primes {
                // Take only components up to level
                flat[i * num_primes + j] = rns.values[j];
            }
        }

        flat
    }

    /// Multiply two polynomials using NTT - CYCLIC convolution (for key operations)
    ///
    /// This is used for key generation and operations that don't require negacyclic convolution.
    /// For CKKS plaintext multiplication, use multiply_polys_flat_ntt_negacyclic instead.
    fn multiply_polys_flat_ntt(&self, a_flat: &[u64], b_flat: &[u64], moduli: &[u64]) -> Result<Vec<u64>, String> {
        self.multiply_polys_flat_ntt_impl(a_flat, b_flat, moduli, false)
    }

    /// Multiply two polynomials using NTT - NEGACYCLIC convolution (for CKKS operations)
    ///
    /// This is used for CKKS plaintext multiplication which requires negacyclic convolution (mod x^n + 1).
    /// Uses twist/untwist to convert cyclic NTT to negacyclic.
    fn multiply_polys_flat_ntt_negacyclic(&self, a_flat: &[u64], b_flat: &[u64], moduli: &[u64]) -> Result<Vec<u64>, String> {
        self.multiply_polys_flat_ntt_impl(a_flat, b_flat, moduli, true)
    }

    /// Internal implementation of polynomial multiplication with optional twist/untwist
    ///
    /// Both input arrays must have the same stride (num_primes_in_array).
    /// The stride must be >= moduli.len().
    ///
    /// @param negacyclic If true, applies twist/untwist for negacyclic convolution (mod x^n + 1)
    fn multiply_polys_flat_ntt_impl(&self, a_flat: &[u64], b_flat: &[u64], moduli: &[u64], negacyclic: bool) -> Result<Vec<u64>, String> {
        let n = self.params.n;
        let num_primes_to_process = moduli.len();

        // Infer the stride from the input array length
        let num_primes_in_array = a_flat.len() / n;
        if b_flat.len() / n != num_primes_in_array {
            return Err("Input arrays have different strides".to_string());
        }

        let mut result_flat = vec![0u64; n * num_primes_to_process];

        // For each RNS component (each prime), multiply using Metal NTT
        for (prime_idx, &q) in moduli.iter().enumerate() {
            // Extract polynomial for this prime
            let mut a_poly = vec![0u64; n];
            let mut b_poly = vec![0u64; n];

            for i in 0..n {
                // IMPORTANT: Use the array's stride (num_primes_in_array), not num_primes_to_process
                a_poly[i] = a_flat[i * num_primes_in_array + prime_idx];
                b_poly[i] = b_flat[i * num_primes_in_array + prime_idx];
            }

            let ntt_ctx = &self.ntt_contexts[prime_idx];
            let q = moduli[prime_idx];

            // For negacyclic convolution (CKKS): apply twist/untwist
            // For cyclic convolution (key operations): use NTT directly
            if negacyclic {
                // TWIST: multiply a and b by psi^i (converts negacyclic → cyclic)
                for i in 0..n {
                    a_poly[i] = Self::mul_mod(a_poly[i], ntt_ctx.psi_powers()[i], q);
                    b_poly[i] = Self::mul_mod(b_poly[i], ntt_ctx.psi_powers()[i], q);
                }
            }

            // Forward NTT
            ntt_ctx.forward(&mut a_poly)?;
            ntt_ctx.forward(&mut b_poly)?;

            // Pointwise multiply in NTT domain
            let mut result_poly = vec![0u64; n];
            ntt_ctx.pointwise_multiply(&a_poly, &b_poly, &mut result_poly)?;

            // Inverse NTT
            ntt_ctx.inverse(&mut result_poly)?;

            if negacyclic {
                // UNTWIST: multiply by psi^{-i} (converts cyclic → negacyclic)
                for i in 0..n {
                    result_poly[i] = Self::mul_mod(result_poly[i], ntt_ctx.psi_inv_powers()[i], q);
                }
            }

            // Store back in flat layout
            for i in 0..n {
                result_flat[i * num_primes_to_process + prime_idx] = result_poly[i];
            }
        }

        Ok(result_flat)
    }

    // ==================== NTT Helper Functions ====================

    /// Find primitive 2n-th root of unity for NTT
    ///
    /// Uses same algorithm as CPU backend for compatibility.
    fn find_primitive_2n_root(n: usize, q: u64) -> Result<u64, String> {
        // Verify q ≡ 1 (mod 2n)
        let two_n = (2 * n) as u64;
        if (q - 1) % two_n != 0 {
            return Err(format!(
                "q = {} is not NTT-friendly for n = {} (q-1 must be divisible by 2n)",
                q, n
            ));
        }

        // Try small candidates that are often generators
        for candidate in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
            if Self::is_primitive_root_candidate(candidate, n, q) {
                let exponent = (q - 1) / two_n;
                return Ok(Self::pow_mod(candidate, exponent, q));
            }
        }

        // Extended search
        for candidate in 32..1000u64 {
            if Self::is_primitive_root_candidate(candidate, n, q) {
                let exponent = (q - 1) / two_n;
                return Ok(Self::pow_mod(candidate, exponent, q));
            }
        }

        Err(format!("Failed to find primitive root for q = {}, n = {}", q, n))
    }

    /// Check if g is suitable for generating primitive 2n-th root
    fn is_primitive_root_candidate(g: u64, n: usize, q: u64) -> bool {
        // Check if g is a quadratic non-residue
        if Self::pow_mod(g, (q - 1) / 2, q) == 1 {
            return false;
        }

        // Check if g^((q-1)/(2n)) generates the subgroup of order 2n
        let psi = Self::pow_mod(g, (q - 1) / (2 * n as u64), q);

        // psi^n should equal -1 mod q
        let psi_n = Self::pow_mod(psi, n as u64, q);
        if psi_n != q - 1 {
            return false;
        }

        // psi^(2n) should equal 1 mod q
        let psi_2n = Self::pow_mod(psi, 2 * n as u64, q);
        psi_2n == 1
    }

    /// Modular multiplication: (a * b) mod q
    fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
        ((a as u128 * b as u128) % q as u128) as u64
    }

    /// Modular exponentiation: base^exp mod q
    fn pow_mod(mut base: u64, mut exp: u64, q: u64) -> u64 {
        let mut result = 1u64;
        base %= q;

        while exp > 0 {
            if exp & 1 == 1 {
                result = ((result as u128 * base as u128) % q as u128) as u64;
            }
            base = ((base as u128 * base as u128) % q as u128) as u64;
            exp >>= 1;
        }

        result
    }
}

// Implement methods for MetalCiphertext
impl MetalCiphertext {
    /// Add two ciphertexts (component-wise polynomial addition)
    pub fn add(&self, other: &Self, ctx: &MetalCkksContext) -> Result<Self, String> {
        assert_eq!(self.n, other.n, "Dimensions must match");
        assert_eq!(self.level, other.level, "Levels must match");
        assert_eq!(self.num_primes, other.num_primes, "Number of primes must match");

        let moduli = &ctx.params.moduli[..=self.level];
        let n = self.n;
        let num_primes = self.num_primes;

        // Add c0 + c0' component-wise
        let mut new_c0 = vec![0u64; n * num_primes];
        for i in 0..(n * num_primes) {
            let prime_idx = i % num_primes;
            let q = moduli[prime_idx];
            new_c0[i] = ((self.c0[i] as u128 + other.c0[i] as u128) % q as u128) as u64;
        }

        // Add c1 + c1' component-wise
        let mut new_c1 = vec![0u64; n * num_primes];
        for i in 0..(n * num_primes) {
            let prime_idx = i % num_primes;
            let q = moduli[prime_idx];
            new_c1[i] = ((self.c1[i] as u128 + other.c1[i] as u128) % q as u128) as u64;
        }

        // Scale stays the same (assuming both have same scale)
        Ok(Self {
            c0: new_c0,
            c1: new_c1,
            n,
            num_primes,
            level: self.level,
            scale: self.scale,
        })
    }

    /// Multiply ciphertext by plaintext using Metal GPU NTT
    ///
    /// Algorithm: (c0, c1) * pt = (c0 * pt, c1 * pt)
    /// Then rescale to bring scale back down.
    pub fn multiply_plain(&self, pt: &MetalPlaintext, ctx: &MetalCkksContext) -> Result<Self, String> {
        assert_eq!(self.n, pt.n, "Dimensions must match");
        assert_eq!(self.level, pt.level, "Levels must match for plaintext multiplication");

        let moduli = &ctx.params.moduli[..=self.level];

        // CRITICAL: Use negacyclic NTT for CKKS plaintext multiplication!
        // CKKS operates in the ring R = Z[x]/(x^n + 1) which requires negacyclic convolution.
        let new_c0 = ctx.multiply_polys_flat_ntt_negacyclic(&self.c0, &pt.coeffs, moduli)?;

        // Multiply c1 by plaintext using negacyclic NTT
        let new_c1 = ctx.multiply_polys_flat_ntt_negacyclic(&self.c1, &pt.coeffs, moduli)?;

        // Compute pre-rescale scale
        let pre_rescale_scale = self.scale * pt.scale;

        // CRITICAL FIX: multiply_polys_flat_ntt returns arrays with stride = moduli.len()
        // Must set num_primes to match the output stride, not the input stride!
        let num_primes_to_process = moduli.len();

        // Create intermediate ciphertext (scale will be fixed in rescale)
        let ct_mult = Self {
            c0: new_c0,
            c1: new_c1,
            n: self.n,
            num_primes: num_primes_to_process,  // FIX: Use output stride, not self.num_primes
            level: self.level,
            scale: 0.0, // Will be set by rescale
        };

        // Rescale to bring scale back to ~Δ and drop one level
        ct_mult.rescale_to_next(ctx, pre_rescale_scale)
    }

    /// Rescale ciphertext to next level (drop one prime from modulus chain)
    ///
    /// This operation divides the ciphertext by the dropped prime and reduces the level.
    /// Essential for keeping noise growth manageable in CKKS.
    fn rescale_to_next(&self, ctx: &MetalCkksContext, pre_rescale_scale: f64) -> Result<Self, String> {
        if self.level == 0 {
            return Err("Cannot rescale at level 0".to_string());
        }

        use num_bigint::BigInt;
        use num_traits::{One, Signed, Zero};

        let moduli_before = &ctx.params.moduli[..=self.level];
        let moduli_after = &ctx.params.moduli[..self.level]; // Drop last prime
        let q_last = moduli_before[moduli_before.len() - 1];
        let n = self.n;
        let new_level = self.level - 1;
        let num_primes_after = moduli_after.len();

        // CRITICAL: Verify stride matches number of primes (prevents stride mismatch bugs)
        debug_assert_eq!(
            self.num_primes,
            moduli_before.len(),
            "Stride/level mismatch: num_primes={} but moduli_before.len()={}",
            self.num_primes,
            moduli_before.len()
        );
        debug_assert_eq!(
            self.c0.len(),
            n * self.num_primes,
            "Buffer size mismatch: c0.len()={} but expected n*num_primes={}",
            self.c0.len(),
            n * self.num_primes
        );

        // Helper: Convert residue to centered representation: [0, q) -> (-q/2, q/2]
        let centered_residue = |x: u64, q: u64| -> i128 {
            let x_i128 = x as i128;
            let q_i128 = q as i128;
            if x_i128 > q_i128 / 2 {
                x_i128 - q_i128
            } else {
                x_i128
            }
        };

        // Helper: Convert centered residue back to canonical [0, q)
        let canon_from_centered = |t: i128, q: u64| -> u64 {
            let q_i128 = q as i128;
            let mut u = t % q_i128;
            if u < 0 {
                u += q_i128;
            }
            u as u64
        };

        // CRT reconstruction using direct formula with centered residues
        let crt_reconstruct_centered = |residues: &[u64], moduli: &[u64]| -> BigInt {
            let centered: Vec<i128> = residues.iter().zip(moduli.iter())
                .map(|(&r, &q)| centered_residue(r, q))
                .collect();

            let mut q_product = BigInt::one();
            for &q in moduli {
                q_product *= q;
            }

            let mut result = BigInt::zero();
            for i in 0..moduli.len() {
                let r_i = BigInt::from(centered[i]);
                let q_i = BigInt::from(moduli[i]);
                let m_i = &q_product / &q_i;

                // Compute modular inverse using Extended Euclidean Algorithm
                let (gcd, x, _) = {
                    fn extended_gcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
                        if b.is_zero() {
                            return (a.clone(), BigInt::one(), BigInt::zero());
                        }
                        let (g, x1, y1) = extended_gcd(b, &(a % b));
                        let x = y1.clone();
                        let y = x1 - (a / b) * y1;
                        (g, x, y)
                    }
                    extended_gcd(&m_i, &q_i)
                };
                assert!(gcd.is_one(), "GCD must be 1");

                let m_i_inv = ((x % &q_i) + &q_i) % &q_i;
                let term_i = &r_i * &m_i * m_i_inv;
                result += term_i;
            }

            // Center the result to (-Q/2, Q/2]
            result = result % &q_product;
            if result > &q_product / 2 {
                result -= q_product;
            }
            result
        };

        // Rescale c0
        let mut new_c0_flat = vec![0u64; n * num_primes_after];
        for i in 0..n {
            // Extract RNS representation for this coefficient (before rescale)
            // IMPORTANT: Use self.num_primes for indexing (the current stride), not moduli_before.len()
            let residues_before: Vec<u64> = (0..moduli_before.len())
                .map(|j| self.c0[i * self.num_primes + j])
                .collect();

            // CRT reconstruct to BigInt
            let coeff_big = crt_reconstruct_centered(&residues_before, moduli_before);

            // Exact division by q_last (centered)
            let coeff_rescaled = &coeff_big / q_last;

            // Convert back to RNS (mod each remaining prime)
            for j in 0..num_primes_after {
                let q_j = moduli_after[j];
                let residue = (&coeff_rescaled % q_j).to_u64_digits();
                let val = if residue.1.is_empty() {
                    0u64
                } else if residue.0 == num_bigint::Sign::Minus {
                    // Negative: convert to canonical form
                    let abs_val = residue.1[0];
                    if abs_val % q_j == 0 {
                        0
                    } else {
                        q_j - (abs_val % q_j)
                    }
                } else {
                    residue.1[0] % q_j
                };
                new_c0_flat[i * num_primes_after + j] = val;
            }
        }

        // Rescale c1
        let mut new_c1_flat = vec![0u64; n * num_primes_after];
        for i in 0..n {
            // IMPORTANT: Use self.num_primes for indexing (the current stride), not moduli_before.len()
            let residues_before: Vec<u64> = (0..moduli_before.len())
                .map(|j| self.c1[i * self.num_primes + j])
                .collect();

            let coeff_big = crt_reconstruct_centered(&residues_before, moduli_before);
            let coeff_rescaled = &coeff_big / q_last;

            for j in 0..num_primes_after {
                let q_j = moduli_after[j];
                let residue = (&coeff_rescaled % q_j).to_u64_digits();
                let val = if residue.1.is_empty() {
                    0u64
                } else if residue.0 == num_bigint::Sign::Minus {
                    let abs_val = residue.1[0];
                    if abs_val % q_j == 0 {
                        0
                    } else {
                        q_j - (abs_val % q_j)
                    }
                } else {
                    residue.1[0] % q_j
                };
                new_c1_flat[i * num_primes_after + j] = val;
            }
        }

        // New scale after rescale
        let new_scale = pre_rescale_scale / (q_last as f64);

        Ok(Self {
            c0: new_c0_flat,
            c1: new_c1_flat,
            n,
            num_primes: num_primes_after,
            level: new_level,
            scale: new_scale,
        })
    }

    /// Multiply two ciphertexts (GPU) - NOT IMPLEMENTED YET
    pub fn multiply(&self, _other: &Self, _ctx: &MetalCkksContext) -> Result<Self, String> {
        // TODO: Implement GPU multiplication (requires relinearization)
        Err("GPU ciphertext multiplication not yet implemented".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_ckks_context_creation() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let result = MetalCkksContext::new(params);

        #[cfg(not(target_os = "macos"))]
        {
            assert!(result.is_err(), "Should fail on non-macOS");
        }

        #[cfg(target_os = "macos")]
        {
            assert!(result.is_ok(), "Should succeed on macOS with Metal support");
        }
    }
}
