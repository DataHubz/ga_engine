//! Number Theoretic Transform (NTT) for fast polynomial multiplication
//!
//! NTT is the integer arithmetic version of FFT, enabling O(N log N) polynomial
//! multiplication instead of O(N²) or O(N^1.585) with Karatsuba.
//!
//! **Parameters for Clifford-LWE with q=3329, N=32**:
//! - ω = 1996 (primitive 64th root of unity mod 3329)
//! - ω^(-1) = 1426
//! - N^(-1) = 3225 (for normalization in inverse NTT)

/// NTT context with precomputed parameters
#[derive(Clone, Debug)]
pub struct NTTContext {
    pub q: i64,              // Modulus
    pub n: usize,            // Polynomial degree
    pub omega: i64,          // Primitive 2N-th root of unity
    pub omega_inv: i64,      // ω^(-1) for inverse NTT
    pub n_inv: i64,          // N^(-1) for normalization
    pub psi: Vec<i64>,       // Twiddle factors: ψ[k] = ω^k mod q
    pub psi_inv: Vec<i64>,   // Inverse twiddle factors
}

impl NTTContext {
    /// Create NTT context for Clifford-LWE-256
    ///
    /// q = 3329, N = 32
    pub fn new_clifford_lwe() -> Self {
        Self::new(3329, 32, 1996, 1426, 3225)
    }

    /// Create NTT context with explicit parameters
    pub fn new(q: i64, n: usize, omega: i64, omega_inv: i64, n_inv: i64) -> Self {
        let mut psi = vec![0i64; 2 * n];
        let mut psi_inv = vec![0i64; 2 * n];

        // Precompute twiddle factors
        for k in 0..(2 * n) {
            psi[k] = mod_pow(omega, k as i64, q);
            psi_inv[k] = mod_pow(omega_inv, k as i64, q);
        }

        Self {
            q,
            n,
            omega,
            omega_inv,
            n_inv,
            psi,
            psi_inv,
        }
    }

    /// Forward NTT using Cooley-Tukey algorithm (in-place)
    ///
    /// Transforms polynomial coefficients a(x) to frequency domain â
    pub fn forward(&self, a: &mut [i64]) {
        assert_eq!(a.len(), self.n, "Input length must equal N");

        let n = self.n;
        let q = self.q;

        // Bit-reversal permutation
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j >= bit {
                j -= bit;
                bit >>= 1;
            }
            j += bit;

            if i < j {
                a.swap(i, j);
            }
        }

        // Cooley-Tukey butterfly
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = 2 * n / len;

            for start in (0..n).step_by(len) {
                let mut k = 0;
                for j in start..(start + half) {
                    let u = a[j];
                    let v = (a[j + half] * self.psi[k]) % q;

                    a[j] = (u + v) % q;
                    a[j + half] = (u - v + q) % q;

                    k += step;
                }
            }

            len *= 2;
        }
    }

    /// Inverse NTT (in-place)
    ///
    /// Transforms frequency domain â back to coefficients a(x)
    pub fn inverse(&self, a: &mut [i64]) {
        assert_eq!(a.len(), self.n, "Input length must equal N");

        let n = self.n;
        let q = self.q;

        // Bit-reversal permutation
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j >= bit {
                j -= bit;
                bit >>= 1;
            }
            j += bit;

            if i < j {
                a.swap(i, j);
            }
        }

        // Inverse butterfly (using ω^(-1))
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = 2 * n / len;

            for start in (0..n).step_by(len) {
                let mut k = 0;
                for j in start..(start + half) {
                    let u = a[j];
                    let v = (a[j + half] * self.psi_inv[k]) % q;

                    a[j] = (u + v) % q;
                    a[j + half] = (u - v + q) % q;

                    k += step;
                }
            }

            len *= 2;
        }

        // Normalize by N^(-1)
        for i in 0..n {
            a[i] = (a[i] * self.n_inv) % q;
        }
    }

    /// Polynomial multiplication using NTT
    ///
    /// Computes c(x) = a(x) * b(x) mod (x^N - 1)
    ///
    /// **Algorithm**:
    /// 1. â = NTT(a)
    /// 2. b̂ = NTT(b)
    /// 3. ĉ = â ⊙ b̂ (point-wise multiplication)
    /// 4. c = INTT(ĉ)
    pub fn multiply_scalar(&self, a: &[i64], b: &[i64]) -> Vec<i64> {
        assert_eq!(a.len(), self.n);
        assert_eq!(b.len(), self.n);

        // Forward NTT
        let mut a_ntt = a.to_vec();
        let mut b_ntt = b.to_vec();

        self.forward(&mut a_ntt);
        self.forward(&mut b_ntt);

        // Point-wise multiplication in frequency domain
        let mut c_ntt = vec![0i64; self.n];
        for i in 0..self.n {
            c_ntt[i] = (a_ntt[i] * b_ntt[i]) % self.q;
        }

        // Inverse NTT
        self.inverse(&mut c_ntt);

        c_ntt
    }
}

/// Fast modular exponentiation
#[inline]
fn mod_pow(mut base: i64, mut exp: i64, modulus: i64) -> i64 {
    let mut result = 1i64;
    base %= modulus;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntt_inverse() {
        let ntt = NTTContext::new_clifford_lwe();

        // Test with simple polynomial: [1, 2, 3, ..., 32]
        let mut a = (1..=32).map(|x| x as i64).collect::<Vec<_>>();
        let original = a.clone();

        // Forward + Inverse should reconstruct original
        ntt.forward(&mut a);
        ntt.inverse(&mut a);

        for i in 0..32 {
            assert_eq!(a[i], original[i], "NTT roundtrip failed at index {}", i);
        }
    }

    #[test]
    fn test_ntt_multiplication() {
        let ntt = NTTContext::new_clifford_lwe();
        let q = ntt.q;

        // Test: (1 + x) * (1 + x) = 1 + 2x + x²
        let a = {
            let mut v = vec![0i64; 32];
            v[0] = 1; // constant term
            v[1] = 1; // x term
            v
        };

        let result = ntt.multiply_scalar(&a, &a);

        // Result should be [1, 2, 1, 0, 0, ..., 0]
        assert_eq!(result[0], 1, "constant term");
        assert_eq!(result[1], 2, "x term");
        assert_eq!(result[2], 1, "x² term");

        for i in 3..32 {
            assert_eq!(result[i], 0, "higher terms should be zero");
        }
    }

    #[test]
    fn test_ntt_vs_naive_multiplication() {
        let ntt = NTTContext::new_clifford_lwe();
        let q = ntt.q;

        // Random test polynomials (small coefficients)
        let a = vec![1, 2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let b = vec![2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        // NTT multiplication
        let result_ntt = ntt.multiply_scalar(&a, &b);

        // Naive multiplication (for verification)
        let mut result_naive = vec![0i64; 32];
        for i in 0..32 {
            for j in 0..32 {
                if a[i] != 0 && b[j] != 0 {
                    // Modulo (x^32 - 1), so x^32 = 1
                    let idx = (i + j) % 32;
                    result_naive[idx] = (result_naive[idx] + a[i] * b[j]) % q;
                }
            }
        }

        // Results should match
        for i in 0..32 {
            assert_eq!(
                result_ntt[i], result_naive[i],
                "Mismatch at index {}: NTT={}, Naive={}",
                i, result_ntt[i], result_naive[i]
            );
        }
    }
}
