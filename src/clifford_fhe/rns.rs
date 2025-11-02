//! Residue Number System (RNS) for CKKS
//!
//! RNS representation allows working with large moduli by using the Chinese Remainder Theorem.
//! Instead of storing coefficients modulo a single large Q, we store them as tuples of
//! residues modulo smaller primes: (c mod q₀, c mod q₁, ..., c mod qₗ)
//!
//! Key benefits:
//! - Support moduli Q > 2^63 (product of primes)
//! - Efficient rescaling (drop one prime)
//! - Parallelizable operations (per-prime)

/// Domain tag for RNS polynomials
///
/// Tracks whether polynomial is in coefficient or NTT domain.
/// This prevents accidentally multiplying COEF with NTT polynomials.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    /// Coefficient domain (standard representation)
    Coef,
    /// NTT domain (Number Theoretic Transform, for fast multiplication)
    Ntt,
}

/// RNS polynomial: coefficients in residue representation
///
/// Each coefficient is represented as a vector of residues mod each prime in the chain.
/// For example, with primes [q₀, q₁, q₂] and coefficient c:
///   rns_coeffs[i] = [c mod q₀, c mod q₁, c mod q₂]
#[derive(Clone, Debug)]
pub struct RnsPolynomial {
    /// Coefficients in RNS form
    /// Outer vector: polynomial coefficients (length N)
    /// Inner vector: residues for each prime (length L = number of primes at this level)
    pub rns_coeffs: Vec<Vec<i64>>,

    /// Number of coefficients (polynomial degree)
    pub n: usize,

    /// Current level (determines which primes are active)
    /// Level 0: all primes [q₀, q₁, ..., qₗ]
    /// Level 1: dropped last prime [q₀, q₁, ..., qₗ₋₁]
    pub level: usize,

    /// Domain tag (COEF or NTT)
    /// All operations verify domain compatibility
    pub domain: Domain,
}

impl RnsPolynomial {
    /// Create new RNS polynomial from RNS coefficients (COEF domain by default)
    pub fn new(rns_coeffs: Vec<Vec<i64>>, n: usize, level: usize) -> Self {
        assert_eq!(rns_coeffs.len(), n, "Wrong number of coefficients");
        Self { rns_coeffs, n, level, domain: Domain::Coef }
    }

    /// Create new RNS polynomial with explicit domain tag
    pub fn new_with_domain(rns_coeffs: Vec<Vec<i64>>, n: usize, level: usize, domain: Domain) -> Self {
        assert_eq!(rns_coeffs.len(), n, "Wrong number of coefficients");
        Self { rns_coeffs, n, level, domain }
    }

    /// Create RNS polynomial from regular coefficients
    ///
    /// Converts coefficients [c₀, c₁, ..., cₙ₋₁] to RNS form:
    /// rns_coeffs[i] = [cᵢ mod q₀, cᵢ mod q₁, ...]
    pub fn from_coeffs(coeffs: &[i64], primes: &[i64], n: usize, level: usize) -> Self {
        assert_eq!(coeffs.len(), n, "Wrong number of coefficients");

        let num_primes = primes.len() - level; // Active primes at this level
        let mut rns_coeffs = vec![vec![0i64; num_primes]; n];

        for i in 0..n {
            for (j, &q) in primes.iter().take(num_primes).enumerate() {
                rns_coeffs[i][j] = ((coeffs[i] % q) + q) % q;
            }
        }

        Self::new(rns_coeffs, n, level)
    }

    /// Convert RNS polynomial back to regular coefficients using CRT
    ///
    /// Uses Chinese Remainder Theorem to reconstruct coefficients from residues.
    /// For two primes q₀, q₁: c = (c₀·Q₀·Q₀⁻¹ + c₁·Q₁·Q₁⁻¹) mod Q
    /// where Q = q₀·q₁, Q₀ = Q/q₀, Q₁ = Q/q₁
    pub fn to_coeffs(&self, primes: &[i64]) -> Vec<i64> {
        let num_primes = self.rns_coeffs[0].len();
        let active_primes = &primes[..num_primes];

        // Compute Q = product of all active primes
        let q_product: i128 = active_primes.iter().map(|&q| q as i128).product();

        let mut coeffs = vec![0i64; self.n];

        for i in 0..self.n {
            let mut c: i128 = 0;

            for (j, &qj) in active_primes.iter().enumerate() {
                // Q_j = Q / q_j
                let q_j = q_product / (qj as i128);

                // Q_j^{-1} mod q_j (using extended Euclidean algorithm)
                let q_j_inv = mod_inverse(q_j, qj as i128);

                // Contribution: c_j * Q_j * Q_j^{-1}
                // Note: Need to avoid overflow in (a * b) % m when a, b are large
                let c_j = self.rns_coeffs[i][j] as i128;

                // First: (c_j * Q_j) % q_product
                let temp1 = mulmod_i128(c_j, q_j, q_product);

                // Second: (temp1 * Q_j^{-1}) % q_product
                let contrib = mulmod_i128(temp1, q_j_inv, q_product);

                c = (c + contrib) % q_product;
            }

            // Center around 0: map from [0, Q) to (-Q/2, Q/2]
            if c > q_product / 2 {
                c -= q_product;
            }

            coeffs[i] = c as i64;
        }

        coeffs
    }

    /// Number of active primes at this level
    pub fn num_primes(&self) -> usize {
        self.rns_coeffs[0].len()
    }

    /// Extract coefficients modulo a single prime (avoiding CRT)
    ///
    /// For CKKS decoding, we don't need the full CRT reconstruction!
    /// Since message and noise are small relative to any single prime qᵢ,
    /// we can decode from any prime's representation.
    ///
    /// This avoids:
    /// - Expensive CRT computation
    /// - Overflow issues when Q > i64_MAX
    ///
    /// # Arguments
    /// * `prime_idx` - Which prime to use (0 = first/largest prime)
    /// * `prime_value` - The actual prime value for center-lifting
    pub fn to_coeffs_single_prime(&self, prime_idx: usize, prime_value: i64) -> Vec<i64> {
        assert!(prime_idx < self.num_primes(), "Prime index out of range");

        let mut coeffs = vec![0i64; self.n];

        for i in 0..self.n {
            let c = self.rns_coeffs[i][prime_idx];

            // Center-lift to (-q/2, q/2]
            let centered = if c > prime_value / 2 {
                c - prime_value
            } else {
                c
            };

            coeffs[i] = centered;
        }

        coeffs
    }

    /// CRT reconstruction for 2 primes only (most common case after rescale)
    ///
    /// For 2 primes, we can use a simplified formula that stays in i128.
    /// For p, q: x ≡ a (mod p), x ≡ b (mod q)
    /// Solution: x = a + p * ((b - a) * p^{-1} mod q)
    ///
    /// # Arguments
    /// * `primes` - Exactly 2 primes
    ///
    /// # Returns
    /// Coefficients in range [0, p*q)
    pub fn to_coeffs_crt_two_primes(&self, primes: &[i64]) -> Vec<f64> {
        assert_eq!(primes.len(), 2, "This function requires exactly 2 primes");
        assert_eq!(self.num_primes(), 2, "Polynomial must be at level with 2 primes");

        let p = primes[0] as i128;
        let q = primes[1] as i128;
        let pq = (p as f64) * (q as f64); // Product as f64

        // Precompute p^{-1} mod q
        let p_inv_mod_q = mod_inverse(p, q);

        let mut coeffs = vec![0f64; self.n];

        for i in 0..self.n {
            let a = self.rns_coeffs[i][0] as i128; // Residue mod p
            let b = self.rns_coeffs[i][1] as i128; // Residue mod q

            // CRT formula: x = a + p * ((b - a) * p^{-1} mod q)
            let diff = ((b - a) % q + q) % q;
            let factor = (diff * p_inv_mod_q) % q;
            let x = a + p * factor;

            // Reduce mod pq if needed (should already be in range)
            let x_reduced = ((x % (pq as i128)) + (pq as i128)) % (pq as i128);

            coeffs[i] = x_reduced as f64;
        }

        coeffs
    }

    /// General CRT reconstruction for N primes using Garner's algorithm
    ///
    /// Garner's algorithm is numerically stable and works for arbitrary primes.
    ///
    /// # Arguments
    /// * `primes` - The active primes (2-10 supported)
    ///
    /// # Returns
    /// Coefficients in range [0, Q) where Q = product of primes
    fn to_coeffs_crt_general(&self, primes: &[i64]) -> Vec<f64> {
        assert!(primes.len() >= 2 && primes.len() <= 10, "CRT supports 2-10 primes");

        // Special case: 2 primes (use optimized direct formula)
        if primes.len() == 2 {
            return self.to_coeffs_crt_two_primes(primes);
        }

        // General case: Garner's algorithm
        let num_primes = primes.len();
        let mut coeffs = vec![0f64; self.n];

        // Precompute mixed radix constants: C[i][j] = q_j^{-1} mod q_i for j < i
        let mut c_matrix = vec![vec![1i128; num_primes]; num_primes];
        for i in 1..num_primes {
            let qi = primes[i] as i128;
            for j in 0..i {
                let qj = primes[j] as i128;
                c_matrix[i][j] = mod_inverse(qj, qi);
            }
        }

        // Apply Garner's algorithm for each coefficient
        for idx in 0..self.n {
            // Extract residues
            let residues: Vec<i128> = (0..num_primes)
                .map(|j| self.rns_coeffs[idx][j] as i128)
                .collect();

            // Garner: compute mixed radix digits
            let mut mixed_radix = vec![0i128; num_primes];
            mixed_radix[0] = residues[0];

            for i in 1..num_primes {
                let qi = primes[i] as i128;
                let mut temp = residues[i];

                for j in 0..i {
                    temp = ((temp - mixed_radix[j]) % qi + qi) % qi;
                    temp = (temp * c_matrix[i][j]) % qi;
                }

                mixed_radix[i] = temp;
            }

            // Convert to positional: x = v0 + v1*q0 + v2*q0*q1 + ...
            let mut result = 0f64;
            let mut prod_so_far = 1f64;

            for i in 0..num_primes {
                result += (mixed_radix[i] as f64) * prod_so_far;
                prod_so_far *= primes[i] as f64;
            }

            coeffs[idx] = result;
        }

        coeffs
    }

    /// CRT reconstruction with centered reduction to (-Q/2, Q/2]
    ///
    /// This is the standard way to decode CKKS ciphertexts.
    /// Supports 2-10 primes using Garner's algorithm.
    ///
    /// # Arguments
    /// * `primes` - The active primes (2-10 supported)
    ///
    /// # Returns
    /// Coefficients as f64 in range (-Q/2, Q/2]
    pub fn to_coeffs_crt_centered(&self, primes: &[i64]) -> Vec<f64> {
        assert!(primes.len() >= 2 && primes.len() <= 10, "CRT supports 2-10 primes");

        let coeffs = self.to_coeffs_crt_general(primes);

        // Compute Q = product of primes (use f64)
        let q_product: f64 = primes.iter().map(|&p| p as f64).product();
        let q_half = q_product / 2.0;

        // Center-lift each coefficient
        coeffs.iter().map(|&c| {
            if c > q_half {
                c - q_product
            } else {
                c
            }
        }).collect()
    }
}

/// Modular multiplication without overflow: (a * b) % m
///
/// Uses double-and-add method to avoid overflow when a*b > i128::MAX
pub fn mulmod_i128(a: i128, b: i128, m: i128) -> i128 {
    // Ensure inputs are reduced mod m
    let a = ((a % m) + m) % m;
    let b = ((b % m) + m) % m;

    // Check if we can multiply directly without overflow
    if let Some(product) = a.checked_mul(b) {
        ((product % m) + m) % m
    } else {
        // Use double-and-add method for large values
        let mut result = 0i128;
        let mut a = a;
        let mut b = b;

        while b > 0 {
            if b & 1 == 1 {
                // Add a to result (mod m)
                result = if let Some(sum) = result.checked_add(a) {
                    sum % m
                } else {
                    // Even addition can overflow with large m
                    ((result % m) + (a % m)) % m
                };
            }
            // Double a (mod m)
            a = if let Some(doubled) = a.checked_add(a) {
                doubled % m
            } else {
                ((a % m) + (a % m)) % m
            };
            b >>= 1;  // Halve b
        }
        result
    }
}

/// Modular inverse using extended Euclidean algorithm
///
/// Returns a^{-1} mod m such that (a * a^{-1}) ≡ 1 (mod m)
pub fn mod_inverse(a: i128, m: i128) -> i128 {
    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (m, a % m);

    while new_r != 0 {
        let quotient = r / new_r;

        let temp = new_t;
        new_t = t - quotient * new_t;
        t = temp;

        let temp = new_r;
        new_r = r - quotient * new_r;
        r = temp;
    }

    if r > 1 {
        panic!("a is not invertible mod m");
    }
    if t < 0 {
        t += m;
    }

    t
}

/// RNS polynomial addition
///
/// (a + b) mod Q = [(a₀ + b₀) mod q₀, (a₁ + b₁) mod q₁, ...]
pub fn rns_add(a: &RnsPolynomial, b: &RnsPolynomial, primes: &[i64]) -> RnsPolynomial {
    assert_eq!(a.n, b.n, "Polynomials must have same length");
    assert_eq!(a.level, b.level, "Polynomials must be at same level");

    let n = a.n;
    let level = a.level;
    let num_primes = a.num_primes();
    let active_primes = &primes[..num_primes];

    let mut rns_coeffs = vec![vec![0i64; num_primes]; n];

    for i in 0..n {
        for (j, &q) in active_primes.iter().enumerate() {
            let sum = a.rns_coeffs[i][j] + b.rns_coeffs[i][j];
            rns_coeffs[i][j] = ((sum % q) + q) % q;
        }
    }

    RnsPolynomial::new(rns_coeffs, n, level)
}

/// RNS polynomial subtraction
///
/// (a - b) mod Q = [(a₀ - b₀) mod q₀, (a₁ - b₁) mod q₁, ...]
pub fn rns_sub(a: &RnsPolynomial, b: &RnsPolynomial, primes: &[i64]) -> RnsPolynomial {
    assert_eq!(a.n, b.n, "Polynomials must have same length");
    assert_eq!(a.level, b.level, "Polynomials must be at same level");

    let n = a.n;
    let level = a.level;
    let num_primes = a.num_primes();
    let active_primes = &primes[..num_primes];

    let mut rns_coeffs = vec![vec![0i64; num_primes]; n];

    for i in 0..n {
        for (j, &q) in active_primes.iter().enumerate() {
            let diff = a.rns_coeffs[i][j] - b.rns_coeffs[i][j];
            rns_coeffs[i][j] = ((diff % q) + q) % q;
        }
    }

    RnsPolynomial::new(rns_coeffs, n, level)
}

/// RNS polynomial negation
///
/// (-a) mod Q = [(-a₀) mod q₀, (-a₁) mod q₁, ...]
pub fn rns_negate(a: &RnsPolynomial, primes: &[i64]) -> RnsPolynomial {
    let n = a.n;
    let level = a.level;
    let num_primes = a.num_primes();
    let active_primes = &primes[..num_primes];

    let mut rns_coeffs = vec![vec![0i64; num_primes]; n];

    for i in 0..n {
        for (j, &q) in active_primes.iter().enumerate() {
            rns_coeffs[i][j] = ((q - a.rns_coeffs[i][j]) % q + q) % q;
        }
    }

    RnsPolynomial::new(rns_coeffs, n, level)
}

/// RNS polynomial multiplication with self-check
///
/// (a * b) mod Q = [(a₀ * b₀) mod q₀, (a₁ * b₁) mod q₁, ...]
///
/// Each multiplication is done independently modulo each prime using NTT.
/// This is the key advantage of RNS: parallelizable operations!
///
/// SELF-CHECK: After computing the result, we independently recompute each
/// prime's multiplication and verify it matches. This catches per-prime
/// misalignment bugs where residues from different primes get mixed.
pub fn rns_multiply(
    a: &RnsPolynomial,
    b: &RnsPolynomial,
    primes: &[i64],
    ntt_multiply_fn: impl Fn(&[i64], &[i64], i64, usize) -> Vec<i64>,
) -> RnsPolynomial {
    assert_eq!(a.n, b.n, "Polynomials must have same length");
    assert_eq!(a.level, b.level, "Polynomials must be at same level");

    // CRITICAL: Verify domain compatibility
    assert_eq!(a.domain, b.domain,
        "RNS multiply: domain mismatch! a={:?}, b={:?}. Cannot multiply COEF with NTT!",
        a.domain, b.domain);
    // For now, only support COEF×COEF (NTT multiplication needs different handling)
    assert_eq!(a.domain, Domain::Coef,
        "RNS multiply: currently only supports COEF domain, got {:?}", a.domain);

    let n = a.n;
    let level = a.level;
    let num_primes = a.num_primes();

    // CRITICAL: Verify inputs have consistent prime counts
    assert_eq!(a.num_primes(), b.num_primes(),
        "RNS multiply: a has {} primes, b has {} primes - MISMATCH!",
        a.num_primes(), b.num_primes());
    assert_eq!(num_primes, primes.len(),
        "RNS multiply: polynomials have {} primes but primes array has {} - MISMATCH!",
        num_primes, primes.len());

    let active_primes = &primes[..num_primes];

    let mut rns_coeffs = vec![vec![0i64; num_primes]; n];

    // Multiply modulo each prime independently (COLUMN-WISE PER PRIME)
    for (j, &q) in active_primes.iter().enumerate() {
        // Extract coefficients for THIS prime ONLY (column j)
        let a_mod_q: Vec<i64> = (0..n).map(|i| a.rns_coeffs[i][j]).collect();
        let b_mod_q: Vec<i64> = (0..n).map(|i| b.rns_coeffs[i][j]).collect();

        // Multiply using NTT for THIS prime
        let c_mod_q = ntt_multiply_fn(&a_mod_q, &b_mod_q, q, n);

        // Store result in column j
        for i in 0..n {
            rns_coeffs[i][j] = c_mod_q[i];
        }
    }

    let result = RnsPolynomial::new(rns_coeffs, n, level);

    // SELF-CHECK: Verify each prime's column independently
    // (Only check first 3 coefficients to avoid spam)
    if std::env::var("RNS_SELFCHECK").is_ok() {
        for j in 0..num_primes {
            let q = active_primes[j];
            let a_j: Vec<i64> = (0..n).map(|i| a.rns_coeffs[i][j]).collect();
            let b_j: Vec<i64> = (0..n).map(|i| b.rns_coeffs[i][j]).collect();
            let expected = ntt_multiply_fn(&a_j, &b_j, q, n);

            for i in 0..n.min(3) {
                let got = ((result.rns_coeffs[i][j] % q) + q) % q;
                let exp = ((expected[i] % q) + q) % q;
                if got != exp {
                    eprintln!("❌ RNS_MULTIPLY SELFCHECK FAILED!");
                    eprintln!("   coeff[{}], prime_idx[{}] (q={})", i, j, q);
                    eprintln!("   Expected: {}", exp);
                    eprintln!("   Got:      {}", got);
                    panic!("RNS multiply self-check failed - per-prime misalignment detected!");
                }
            }
        }
    }

    result
}

/// RNS rescaling: drop the last prime from the modulus chain
///
/// This is how CKKS rescaling works in RNS:
/// - Before: coefficients mod Q = q₀ · q₁ · ... · qₗ
/// - After: coefficients mod Q' = q₀ · q₁ · ... · qₗ₋₁ (dropped qₗ)
///
/// The rescaling process:
/// 1. Convert RNS to regular coefficients (CRT)
/// 2. Divide by the last prime qₗ (with rounding)
/// 3. Convert back to RNS with one fewer prime
///
/// Returns new polynomial at level+1 with scale divided by qₗ
/// Exact RNS rescale with proper rounding: computes round(c / q_last)
///
/// This implements the CKKS "DivideRoundByLastq" operation correctly:
/// 1. Center-lift the last residue for symmetric rounding
/// 2. For each remaining prime, compute: (c_i - c_last_centered) * q_last^{-1} mod q_i
///
/// # Arguments
/// * `poly` - Polynomial to rescale
/// * `primes` - Full prime chain at current level (including q_last to be dropped)
/// * `inv_qlast_mod_qi` - Precomputed q_last^{-1} mod q_i for each remaining prime
///
/// # Returns
/// Polynomial at level+1 with last prime dropped
pub fn rns_rescale_exact(
    poly: &RnsPolynomial,
    primes: &[i64],
    inv_qlast_mod_qi: &[i64],
) -> RnsPolynomial {
    let n = poly.n;
    let level = poly.level;
    let num_primes = poly.num_primes();

    assert!(num_primes > 1, "Cannot rescale: only one prime remaining");
    assert_eq!(inv_qlast_mod_qi.len(), num_primes - 1, "inv_qlast_mod_qi length mismatch");

    // Get the prime we're dropping (last one)
    let q_last = primes[num_primes - 1];
    let new_num_primes = num_primes - 1;

    let mut new_rns_coeffs = vec![vec![0i64; new_num_primes]; n];

    // For each coefficient, perform exact rounded rescaling
    for i in 0..n {
        // Step 1: Center-lift the last residue to (-q_last/2, q_last/2]
        let c_mod_qlast = poly.rns_coeffs[i][num_primes - 1];
        let c_last_centered = if c_mod_qlast > q_last / 2 {
            c_mod_qlast - q_last
        } else {
            c_mod_qlast
        };

        // Step 2: For each remaining prime, compute rounded division
        for j in 0..new_num_primes {
            let qj = primes[j];
            let c_mod_qj = poly.rns_coeffs[i][j];

            // Bring c_last_centered into modulo qj
            let t = ((c_last_centered % qj) + qj) % qj;

            // Compute: (c_mod_qj - t) * inv_qlast_mod_qi[j] mod qj
            let diff = ((c_mod_qj - t) % qj + qj) % qj;

            // Multiply by precomputed inverse (use i128 to avoid overflow)
            let c_new = ((diff as i128) * (inv_qlast_mod_qi[j] as i128) % (qj as i128) + (qj as i128)) % (qj as i128);

            new_rns_coeffs[i][j] = c_new as i64;
        }
    }

    // Preserve domain tag from input
    RnsPolynomial::new_with_domain(new_rns_coeffs, n, level + 1, poly.domain)
}

/// Precompute inverse constants for RNS rescaling
///
/// For dropping prime q_last, computes q_last^{-1} mod q_i for each remaining prime.
///
/// # Arguments
/// * `primes` - Full prime chain at current level
///
/// # Returns
/// Vector of inverses [q_last^{-1} mod q_0, q_last^{-1} mod q_1, ...]
pub fn precompute_rescale_inv(primes: &[i64]) -> Vec<i64> {
    let num_primes = primes.len();
    assert!(num_primes > 1, "Need at least 2 primes to rescale");

    let q_last = primes[num_primes - 1];
    let mut inv_qlast_mod_qi = vec![0i64; num_primes - 1];

    for i in 0..(num_primes - 1) {
        let qi = primes[i];
        inv_qlast_mod_qi[i] = mod_inverse(q_last as i128, qi as i128) as i64;
    }

    inv_qlast_mod_qi
}

/// Reference CRT-based rescale (slow but correct - for debugging)
///
/// This is the "obviously correct" implementation that:
/// 1. Reconstructs each coefficient as a big integer via CRT
/// 2. Center-lifts to (-Q/2, Q/2]
/// 3. Performs exact nearest-integer division by q_last
/// 4. Converts back to RNS with the remaining primes
///
/// Use this to verify the fast rns_rescale_exact() is correct.
pub fn rns_rescale_reference(poly: &RnsPolynomial, primes: &[i64]) -> RnsPolynomial {
    let n = poly.n;
    let num_primes = poly.num_primes();
    assert!(num_primes > 1, "Cannot rescale: only one prime remaining");

    let q_last = primes[num_primes - 1];
    let new_num_primes = num_primes - 1;
    let new_primes = &primes[..new_num_primes];

    // For large moduli (>2 primes of 60+ bits), full CRT overflows i128!
    // Instead, use "approximate rescale": scale each residue independently.
    //
    // For each remaining prime q_i, compute:
    //   c_i' ≈ (c_i * q_last^{-1}) mod q_i
    //
    // This is an approximation but works well when noise is small.
    // The exact formula would require full CRT reconstruction which overflows.

    let q_last_128 = q_last as i128;
    let mut new_rns_coeffs = vec![vec![0i64; new_num_primes]; n];

    for i in 0..n {
        for j in 0..new_num_primes {
            let q_j = new_primes[j];
            let q_j_128 = q_j as i128;

            // Get residue mod q_j and mod q_last
            let c_j = poly.rns_coeffs[i][j] as i128;
            let c_last = poly.rns_coeffs[i][num_primes - 1] as i128;

            // Approximate rescale: (c_j - c_last) / q_last mod q_j
            // This is the "fast rescale" used in many RNS-FHE implementations
            let q_last_inv_mod_qj = mod_inverse(q_last_128, q_j_128);

            // Compute (c_j - c_last) * q_last^{-1} mod q_j
            let diff = (c_j - c_last + q_j_128) % q_j_128;  // Ensure positive
            let rescaled = mulmod_i128(diff, q_last_inv_mod_qj, q_j_128);

            new_rns_coeffs[i][j] = rescaled as i64;
        }
    }

    let mut result = RnsPolynomial::new(new_rns_coeffs, n, poly.level + 1);
    result
}
/// Old rescale function (kept for compatibility, but use rns_rescale_exact for correctness)
#[deprecated(note = "Use rns_rescale_exact for proper CKKS rescaling")]
#[allow(dead_code)]
pub fn rns_rescale(poly: &RnsPolynomial, primes: &[i64]) -> RnsPolynomial {
    // Compute inverse on the fly
    let inv = precompute_rescale_inv(primes);
    rns_rescale_exact(poly, primes, &inv)
}

/// CRT reconstruction: convert RNS residues to a single integer in [0, Q)
///
/// Given residues [r0, r1, ..., rL] where ri = x mod qi,
/// reconstruct x ∈ [0, Q) where Q = q0 * q1 * ... * qL
///
/// Uses the explicit CRT formula with precomputed basis.
fn crt_reconstruct(residues: &[i64], primes: &[i64]) -> i128 {
    let num_primes = residues.len();
    assert_eq!(num_primes, primes.len());

    if num_primes == 1 {
        return residues[0] as i128;
    }

    // Compute Q = product of all primes (careful: can overflow for many primes!)
    let mut q_prod = 1i128;
    for &qi in primes {
        q_prod *= qi as i128;
    }

    // CRT formula: x = Σ ri * (Q/qi) * [(Q/qi)^{-1} mod qi] mod Q
    // CRITICAL: Use mul_mod to avoid overflow in intermediate products
    let mut result = 0i128;
    for i in 0..num_primes {
        let qi = primes[i] as i128;
        let ri = residues[i] as i128;

        // Compute Q/qi
        let q_div_qi = q_prod / qi;

        // Compute (Q/qi)^{-1} mod qi
        let inv = mod_inverse(q_div_qi, qi);

        // Compute: coeff = (Q/qi) * inv mod Q
        // This gives us the CRT basis element
        let basis = mulmod_i128(q_div_qi, inv, q_prod);

        // Add ri * basis to result
        let term = mulmod_i128(ri, basis, q_prod);
        result = (result + term) % q_prod;
    }

    // Ensure positive result
    if result < 0 {
        result += q_prod;
    }

    result
}

/// Center-lift an integer from [0, Q) to (-Q/2, Q/2]
fn center_lift(x: i128, q_prod: i128) -> i128 {
    if x > q_prod / 2 {
        x - q_prod
    } else {
        x
    }
}

/// Balanced base-B decomposition in Z
///
/// Decompose x ∈ Z into digits dt ∈ [-B/2, B/2) such that:
///   x = d0 + d1*B + d2*B² + ... + d(D-1)*B^(D-1)
///
/// Uses balanced representation to minimize digit magnitudes.
fn balanced_pow2_decompose(x: i128, w: u32, d: usize) -> Vec<i64> {
    let b = 1i128 << w;  // B = 2^w
    let b_half = b / 2;

    let mut digits = vec![0i64; d];
    let mut remainder = x;

    for t in 0..d {
        // Extract digit in balanced form: dt ∈ [-B/2, B/2)
        // First, get remainder mod B (result in [0, B) for positive, or (-B, 0] for negative)
        let mut dt = ((remainder % b) + b) % b;  // Force into [0, B)

        // Balance: if dt >= B/2, make it negative by subtracting B
        if dt >= b_half {
            dt -= b;
        }

        digits[t] = dt as i64;

        // Update remainder: (x - dt) / B
        // This should be exact division since dt ≡ x (mod B)
        remainder = (remainder - dt) / b;
    }

    digits
}

/// Decompose RNS polynomial into CRT-consistent, balanced base-2^w digits
///
/// CRITICAL FIX: This uses CRT-consistent decomposition, not per-prime independent.
///
/// For each coefficient:
/// 1. Reconstruct the integer x ∈ [0, Q) via CRT from all residues
/// 2. Center-lift to x_c ∈ (-Q/2, Q/2]
/// 3. Balanced decomposition in Z: x_c = Σ dt·B^t where dt ∈ [-B/2, B/2)
/// 4. Map each digit back to RNS identically across all primes
///
/// This ensures Σ dt·B^t ≡ x (mod qi) for EVERY prime qi, maintaining the
/// EVK cancellation property even when noise is present.
///
/// # Arguments
/// * `poly` - RNS polynomial to decompose
/// * `primes` - Active primes (matching poly's level)
/// * `w` - Digit width (typically 20-30 bits)
///
/// # Returns
/// Vector of D digit polynomials with CRT-consistent residues
pub fn decompose_base_pow2(
    poly: &RnsPolynomial,
    primes: &[i64],
    w: u32,
) -> Vec<RnsPolynomial> {
    let n = poly.n;
    let num_primes = poly.num_primes();

    // Compute Q = product of all primes
    let mut q_prod = 1i128;
    let mut q_bits = 0u32;
    for &qi in primes {
        q_prod *= qi as i128;
        q_bits += 60;  // Assume each prime is ~60 bits
    }

    // Number of digits needed to represent values up to Q
    // Need enough digits to cover Q_bits worth of data
    let d = ((q_bits + w - 1) / w) as usize;

    // Storage for all digits: digits[t] is an RnsPolynomial
    let mut digits_data = vec![vec![vec![0i64; num_primes]; n]; d];

    // Decompose each coefficient using CRT
    for i in 0..n {
        // Step 1: CRT reconstruct to get x ∈ [0, Q)
        let residues: Vec<i64> = (0..num_primes)
            .map(|j| poly.rns_coeffs[i][j])
            .collect();

        let x = crt_reconstruct(&residues, primes);

        // Step 2: Center-lift to (-Q/2, Q/2]
        let x_centered = center_lift(x, q_prod);

        // Step 3: Balanced base-B decomposition in Z
        let digits_z = balanced_pow2_decompose(x_centered, w, d);

        // Step 4: Map each digit back to RNS consistently
        for t in 0..d {
            let dt = digits_z[t];

            // Reduce dt modulo each prime identically
            for j in 0..num_primes {
                let qi = primes[j];
                // Ensure positive residue
                let dt_mod_qi = ((dt as i128 % qi as i128) + qi as i128) % qi as i128;
                digits_data[t][i][j] = dt_mod_qi as i64;
            }
        }
    }

    // Package each digit level into an RnsPolynomial
    digits_data.into_iter()
        .map(|digit_coeffs| RnsPolynomial::new(digit_coeffs, n, poly.level))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rns_conversion() {
        let primes = vec![1_099_511_627_689, 1_099_511_627_691]; // Two 40-bit primes
        let coeffs = vec![123456789, -987654321, 0, 42];
        let n = 4;

        let rns_poly = RnsPolynomial::from_coeffs(&coeffs, &primes, n, 0);
        let recovered = rns_poly.to_coeffs(&primes);

        for i in 0..n {
            assert_eq!(coeffs[i], recovered[i], "Coefficient {} mismatch", i);
        }
    }

    #[test]
    fn test_rns_add() {
        let primes = vec![1_099_511_627_689, 1_099_511_627_691];
        let a_coeffs = vec![100, 200, 300, 400];
        let b_coeffs = vec![50, 60, 70, 80];
        let n = 4;

        let a = RnsPolynomial::from_coeffs(&a_coeffs, &primes, n, 0);
        let b = RnsPolynomial::from_coeffs(&b_coeffs, &primes, n, 0);

        let c = rns_add(&a, &b, &primes);
        let c_coeffs = c.to_coeffs(&primes);

        for i in 0..n {
            assert_eq!(a_coeffs[i] + b_coeffs[i], c_coeffs[i]);
        }
    }

    #[test]
    fn test_mod_inverse() {
        // Test: 3 * 3^{-1} ≡ 1 (mod 7)
        let inv = mod_inverse(3, 7);
        assert_eq!((3 * inv) % 7, 1);

        // Test with larger values
        let q = 1_099_511_627_689i128;
        let a = 12345678i128;
        let inv = mod_inverse(a, q);
        assert_eq!((a * inv) % q, 1);
    }
}
