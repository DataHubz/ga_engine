/**
 * CUDA Kernels for Number Theoretic Transform (NTT)
 *
 * Implements Harvey Butterfly NTT algorithm on NVIDIA GPUs
 * Based on the Metal implementation but adapted for CUDA.
 */

extern "C" {

/**
 * Modular multiplication using Barrett reduction
 * Returns (a * b) % q
 */
__device__ unsigned long long mul_mod(unsigned long long a, unsigned long long b, unsigned long long q) {
    // Use 128-bit arithmetic via __umul64hi for high bits
    unsigned long long lo = a * b;
    unsigned long long hi = __umul64hi(a, b);

    // Barrett reduction approximation
    // For FHE primes (~60 bits), simple modulo is acceptable
    // CUDA has efficient 64-bit division
    __uint128_t product = ((__uint128_t)hi << 64) | lo;
    return (unsigned long long)(product % q);
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

    // Butterfly indices
    unsigned int k = gid / m;
    unsigned int j = gid % m;
    unsigned int butterfly_span = m * 2;

    unsigned int idx1 = k * butterfly_span + j;
    unsigned int idx2 = idx1 + m;

    // Harvey butterfly: (a, b) -> (a + w*b, a - w*b)
    unsigned long long a = coeffs[idx1];
    unsigned long long b = coeffs[idx2];
    unsigned long long w = twiddles[m + j];

    unsigned long long wb = mul_mod(w, b, q);

    coeffs[idx1] = add_mod(a, wb, q);
    coeffs[idx2] = sub_mod(a, wb, q);
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

    // Butterfly indices (inverse pattern)
    unsigned int k = gid / m;
    unsigned int j = gid % m;
    unsigned int butterfly_span = m * 2;

    unsigned int idx1 = k * butterfly_span + j;
    unsigned int idx2 = idx1 + m;

    // Inverse butterfly: (a, b) -> ((a + b)/2, w*(a - b)/2)
    unsigned long long a = coeffs[idx1];
    unsigned long long b = coeffs[idx2];
    unsigned long long w_inv = twiddles_inv[m + j];

    unsigned long long sum = add_mod(a, b, q);
    unsigned long long diff = sub_mod(a, b, q);
    unsigned long long w_diff = mul_mod(w_inv, diff, q);

    coeffs[idx1] = sum;
    coeffs[idx2] = w_diff;
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

} // extern "C"
