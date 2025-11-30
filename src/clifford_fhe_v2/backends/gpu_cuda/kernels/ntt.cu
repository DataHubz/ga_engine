/**
 * CUDA Kernels for Number Theoretic Transform (NTT)
 *
 * Implements Harvey Butterfly NTT algorithm on NVIDIA GPUs
 * Based on the Metal implementation but adapted for CUDA.
 */

extern "C" {

/**
 * Modular multiplication without 128-bit types
 * Returns (a * b) % q
 *
 * Simple but correct implementation using bit-by-bit reduction.
 * Optimized path for when product fits in 64 bits.
 */
__device__ unsigned long long mul_mod(unsigned long long a, unsigned long long b, unsigned long long q) {
    // Compute full 128-bit product using hi and lo parts
    unsigned long long lo = a * b;
    unsigned long long hi = __umul64hi(a, b);

    // Fast path: if hi == 0, simple modulo
    if (hi == 0) {
        // Product fits in 64 bits
        if (lo >= q) {
            lo -= q;
            if (lo >= q) {
                lo %= q;  // Only needed if lo >= 2*q
            }
        }
        return lo;
    }

    // Slow path: 128-bit modulo using bit-by-bit reduction
    // Build up the result from high bit to low bit
    unsigned long long result = 0;

    // Process high 64 bits
    for (int i = 63; i >= 0; i--) {
        // Double the current result: result = (result * 2) mod q
        // Since result < q, result * 2 < 2q, so at most one subtraction needed
        unsigned long long doubled = result + result;  // Safer than result * 2 or result << 1
        if (doubled >= q) doubled -= q;
        result = doubled;

        // Add current bit if set
        if ((hi >> i) & 1ULL) {
            result++;
            if (result >= q) result -= q;
        }
    }

    // Process low 64 bits
    for (int i = 63; i >= 0; i--) {
        // Double the current result
        unsigned long long doubled = result + result;
        if (doubled >= q) doubled -= q;
        result = doubled;

        // Add current bit if set
        if ((lo >> i) & 1ULL) {
            result++;
            if (result >= q) result -= q;
        }
    }

    return result;
}

/**
 * Modular addition: (a + b) % q
 */
__device__ unsigned long long add_mod(unsigned long long a, unsigned long long b, unsigned long long q) {
    unsigned long long sum = a + b;
    return (sum >= q) ? (sum - q) : sum;
}

/**
 * Modular subtraction: (a - b) % q
 */
__device__ unsigned long long sub_mod(unsigned long long a, unsigned long long b, unsigned long long q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

/**
 * Bit-reversal permutation
 * Required preprocessing step for NTT
 */
__global__ void bit_reverse_permutation(
    unsigned long long* coeffs,
    unsigned int n,
    unsigned int log_n
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= n / 2) return;

    // Compute bit-reversed index
    unsigned int reversed = 0;
    unsigned int temp = gid;
    for (unsigned int i = 0; i < log_n; i++) {
        reversed = (reversed << 1) | (temp & 1);
        temp >>= 1;
    }

    if (gid < reversed) {
        // Swap coeffs[gid] and coeffs[reversed]
        unsigned long long tmp = coeffs[gid];
        coeffs[gid] = coeffs[reversed];
        coeffs[reversed] = tmp;
    }
}

/**
 * Forward NTT (Cooley-Tukey butterfly)
 * Transforms polynomial from coefficient to evaluation representation
 */
__global__ void ntt_forward(
    unsigned long long* coeffs,
    const unsigned long long* twiddles,
    unsigned int n,
    unsigned long long q,
    unsigned int stage,
    unsigned int m
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_butterflies = n / 2;

    if (gid >= total_butterflies) return;

    // Butterfly indices (matches CPU Cooley-Tukey DIT)
    unsigned int m2 = m * 2;
    unsigned int k = gid / m;        // Which butterfly group
    unsigned int j = gid % m;        // Position within group

    unsigned int idx1 = k * m2 + j;
    unsigned int idx2 = idx1 + m;

    // FIXED: Correct twiddle indexing
    // w = omega^((n/m2) * j) = twiddles[(n/m2) * j]
    unsigned int twiddle_stride = n / m2;
    unsigned int twiddle_idx = (twiddle_stride * j) % n;
    unsigned long long w = twiddles[twiddle_idx];

    // Cooley-Tukey butterfly
    unsigned long long u = coeffs[idx1];
    unsigned long long t = mul_mod(w, coeffs[idx2], q);

    coeffs[idx1] = add_mod(u, t, q);
    coeffs[idx2] = sub_mod(u, t, q);
}

/**
 * Inverse NTT (Gentleman-Sande butterfly)
 * Transforms polynomial from evaluation to coefficient representation
 */
__global__ void ntt_inverse(
    unsigned long long* coeffs,
    const unsigned long long* twiddles_inv,
    unsigned int n,
    unsigned long long q,
    unsigned int stage,
    unsigned int m
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_butterflies = n / 2;

    if (gid >= total_butterflies) return;

    // Butterfly indices (same as forward - CPU uses same Cooley-Tukey for both)
    unsigned int m2 = m * 2;
    unsigned int k = gid / m;
    unsigned int j = gid % m;

    unsigned int idx1 = k * m2 + j;
    unsigned int idx2 = idx1 + m;

    // FIXED: Correct twiddle indexing (same formula as forward, but with omega_inv twiddles)
    unsigned int twiddle_stride = n / m2;
    unsigned int twiddle_idx = (twiddle_stride * j) % n;
    unsigned long long w = twiddles_inv[twiddle_idx];

    // Same Cooley-Tukey butterfly as forward (CPU does this for inverse too)
    unsigned long long u = coeffs[idx1];
    unsigned long long t = mul_mod(w, coeffs[idx2], q);

    coeffs[idx1] = add_mod(u, t, q);
    coeffs[idx2] = sub_mod(u, t, q);
}

/**
 * Pointwise multiplication in NTT domain
 * c[i] = (a[i] * b[i]) % q for all i
 */
__global__ void ntt_pointwise_multiply(
    const unsigned long long* a,
    const unsigned long long* b,
    unsigned long long* c,
    unsigned int n,
    unsigned long long q
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        c[gid] = mul_mod(a[gid], b[gid], q);
    }
}

/**
 * Scalar multiplication: a[i] = (a[i] * scalar) % q
 */
__global__ void ntt_scalar_multiply(
    unsigned long long* a,
    unsigned long long scalar,
    unsigned int n,
    unsigned long long q
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        a[gid] = mul_mod(a[gid], scalar, q);
    }
}

/**
 * BATCHED OPERATIONS - Process multiple primes in parallel
 *
 * These kernels use 2D grid: (butterfly_blocks, num_primes)
 * to process all RNS primes simultaneously, dramatically reducing
 * kernel launch overhead.
 */

/**
 * Batched Forward NTT - Process all primes in parallel
 *
 * Input layout: data[prime_idx * n + coeff_idx] (flat RNS)
 * Grid: (butterfly_blocks, num_primes, 1)
 *
 * This replaces num_primes sequential kernel launches with ONE launch.
 */
__global__ void ntt_forward_batched(
    unsigned long long* data,              // All primes' data [num_primes * n]
    const unsigned long long* twiddles,    // Twiddles for ALL primes [num_primes * n]
    const unsigned long long* moduli,      // RNS moduli [num_primes]
    unsigned int n,                        // Ring dimension
    unsigned int num_primes,               // Number of RNS primes
    unsigned int stage,                    // Current NTT stage
    unsigned int m                         // Butterfly group size
) {
    // 2D grid: x = butterfly index, y = prime index
    unsigned int butterfly_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int prime_idx = blockIdx.y;

    unsigned int total_butterflies = n / 2;

    if (butterfly_idx >= total_butterflies || prime_idx >= num_primes) return;

    // Get modulus for this prime
    unsigned long long q = moduli[prime_idx];

    // Offset to this prime's data
    unsigned int prime_offset = prime_idx * n;
    unsigned long long* coeffs = data + prime_offset;
    const unsigned long long* twid = twiddles + prime_offset;

    // Butterfly indices (same logic as single-prime version)
    unsigned int k = butterfly_idx / m;
    unsigned int j = butterfly_idx % m;
    unsigned int butterfly_span = m * 2;

    unsigned int idx1 = k * butterfly_span + j;
    unsigned int idx2 = idx1 + m;

    // Harvey butterfly: (a, b) -> (a + w*b, a - w*b)
    unsigned long long a = coeffs[idx1];
    unsigned long long b = coeffs[idx2];
    unsigned long long w = twid[m + j];

    unsigned long long wb = mul_mod(w, b, q);

    coeffs[idx1] = add_mod(a, wb, q);
    coeffs[idx2] = sub_mod(a, wb, q);
}

/**
 * Batched Inverse NTT - Process all primes in parallel
 *
 * Input layout: data[prime_idx * n + coeff_idx] (flat RNS)
 * Grid: (butterfly_blocks, num_primes, 1)
 */
__global__ void ntt_inverse_batched(
    unsigned long long* data,
    const unsigned long long* twiddles_inv,
    const unsigned long long* moduli,
    unsigned int n,
    unsigned int num_primes,
    unsigned int stage,
    unsigned int m
) {
    unsigned int butterfly_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int prime_idx = blockIdx.y;

    unsigned int total_butterflies = n / 2;

    if (butterfly_idx >= total_butterflies || prime_idx >= num_primes) return;

    unsigned long long q = moduli[prime_idx];

    unsigned int prime_offset = prime_idx * n;
    unsigned long long* coeffs = data + prime_offset;
    const unsigned long long* twid_inv = twiddles_inv + prime_offset;

    // Butterfly indices (inverse pattern)
    unsigned int k = butterfly_idx / m;
    unsigned int j = butterfly_idx % m;
    unsigned int butterfly_span = m * 2;

    unsigned int idx1 = k * butterfly_span + j;
    unsigned int idx2 = idx1 + m;

    // Inverse butterfly
    unsigned long long a = coeffs[idx1];
    unsigned long long b = coeffs[idx2];
    unsigned long long w_inv = twid_inv[m + j];

    unsigned long long sum = add_mod(a, b, q);
    unsigned long long diff = sub_mod(a, b, q);
    unsigned long long w_diff = mul_mod(w_inv, diff, q);

    coeffs[idx1] = sum;
    coeffs[idx2] = w_diff;
}

/**
 * Batched Pointwise Multiplication - Process all primes in parallel
 *
 * c[prime_idx * n + i] = (a[...] * b[...]) % q[prime_idx]
 * Grid: (coeff_blocks, num_primes, 1)
 */
__global__ void ntt_pointwise_multiply_batched(
    const unsigned long long* a,
    const unsigned long long* b,
    unsigned long long* c,
    const unsigned long long* moduli,
    unsigned int n,
    unsigned int num_primes
) {
    unsigned int coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int prime_idx = blockIdx.y;

    if (coeff_idx >= n || prime_idx >= num_primes) return;

    unsigned long long q = moduli[prime_idx];
    unsigned int idx = prime_idx * n + coeff_idx;

    c[idx] = mul_mod(a[idx], b[idx], q);
}

} // extern "C"
