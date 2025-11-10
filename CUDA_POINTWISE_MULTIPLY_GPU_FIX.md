# CUDA Pointwise Multiply GPU Fix

**Date**: 2025-11-09
**Status**: ✅ **IMPLEMENTED & COMPILED SUCCESSFULLY**

## Problem

The `cuda_multiply_plain()` function in V3 bootstrap had CPU loops doing 128-bit modular multiplication:

```rust
// BEFORE (CPU LOOPS - 30,720 iterations per call):
for coeff_idx in 0..n {
    for prime_idx in 0..num_primes {
        let idx = coeff_idx * num_primes + prime_idx;
        let q = ckks_ctx.params().moduli[prime_idx];

        c0_result[idx] = ((c0_val as u128 * pt_val as u128) % q as u128) as u64;
        c1_result[idx] = ((c1_val as u128 * pt_val as u128) % q as u128) as u64;
    }
}
```

This function is called **~20 times per bootstrap** during CoeffToSlot/SlotToCoeff diagonal matrix multiplications.

## Solution

### 1. Created New CUDA Kernel

Added `rns_pointwise_multiply_strided` kernel to `src/clifford_fhe_v2/backends/gpu_cuda/kernels/rns.cu`:

```cuda
__global__ void rns_pointwise_multiply_strided(
    const unsigned long long* a,         // First polynomial (strided)
    const unsigned long long* b,         // Second polynomial (strided)
    unsigned long long* c,               // Result (strided)
    const unsigned long long* moduli,    // RNS moduli
    unsigned int n,                      // Ring dimension
    unsigned int stride,                 // Stride (usually num_primes_total)
    unsigned int num_primes              // Number of active primes
) {
    unsigned int coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (coeff_idx >= n) return;

    // Process all primes for this coefficient
    for (unsigned int prime_idx = 0; prime_idx < num_primes; prime_idx++) {
        unsigned int idx = coeff_idx * stride + prime_idx;
        unsigned long long q = moduli[prime_idx];

        // Multiply with 128-bit safety using Russian peasant algorithm
        unsigned long long result = mul_mod_128(a[idx], b[idx], q);
        c[idx] = result;
    }
}
```

**Key Feature**: Uses `mul_mod_128()` device function (Russian peasant algorithm) to avoid overflow, same as rescaling kernel.

### 2. Added V2 Wrapper Function

Added to `src/clifford_fhe_v2/backends/gpu_cuda/ckks.rs` (line 1525):

```rust
pub fn pointwise_multiply_polynomials_gpu_strided(
    &self,
    a: &[u64],
    b: &[u64],
    stride: usize,
    num_primes: usize,
) -> Result<Vec<u64>, String>
```

This wraps the CUDA kernel with proper:
- Memory transfers (host → device → host)
- Launch configuration (one thread per coefficient)
- Error handling

### 3. Updated Kernel Loading

Modified `src/clifford_fhe_v2/backends/gpu_cuda/ckks.rs` (line 128):

```rust
device.device.load_ptx(ptx, "rns_module", &[
    "rns_exact_rescale",
    "rns_exact_rescale_strided",
    "rns_strided_to_flat",
    "rns_flat_to_strided",
    "rns_add",
    "rns_sub",
    "rns_negate",
    "rns_pointwise_multiply_strided",  // ← NEW
])
```

### 4. Fixed V3 cuda_multiply_plain()

Updated `src/clifford_fhe_v3/bootstrapping/cuda_coeff_to_slot.rs` (line 186):

```rust
// AFTER (GPU CALLS - 100% parallel):
pub fn cuda_multiply_plain(
    ct: &CudaCiphertext,
    pt: &[u64],
    ckks_ctx: &Arc<CudaCkksContext>,
    scale_for_diag: f64,
) -> Result<CudaCiphertext, String> {
    let n = ct.n;
    let num_primes = ct.num_primes;

    // Use V2's GPU pointwise multiplication with strided layout
    let c0_result = ckks_ctx.pointwise_multiply_polynomials_gpu_strided(
        &ct.c0, pt, num_primes, num_primes
    )?;
    let c1_result = ckks_ctx.pointwise_multiply_polynomials_gpu_strided(
        &ct.c1, pt, num_primes, num_primes
    )?;

    // Rescale using GPU
    let c0_rescaled = ckks_ctx.exact_rescale_gpu(&c0_result, num_primes - 1)?;
    let c1_rescaled = ckks_ctx.exact_rescale_gpu(&c1_result, num_primes - 1)?;

    // ... return new ciphertext
}
```

## Results

### Code Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Function length** | 48 lines | 33 lines | **31% shorter** |
| **CPU loops** | 30,720 iterations | **0** | **Eliminated** |
| **GPU operations** | 1 (rescale only) | **3** (multiply×2 + rescale×2) | **3× more GPU** |

### Files Changed

1. `src/clifford_fhe_v2/backends/gpu_cuda/kernels/rns.cu` (+30 lines)
   - Added `rns_pointwise_multiply_strided` kernel

2. `src/clifford_fhe_v2/backends/gpu_cuda/ckks.rs` (+73 lines)
   - Added `pointwise_multiply_polynomials_gpu_strided()` wrapper
   - Added kernel to PTX loading list

3. `src/clifford_fhe_v3/bootstrapping/cuda_coeff_to_slot.rs` (-15 lines)
   - Replaced CPU loops with GPU calls
   - Simplified function logic

### Build Status

✅ **Compiled successfully** in 14.77s

```bash
cargo build --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap
    Finished `release` profile [optimized] target(s) in 14.77s
```

## Expected Performance Impact

### Per Bootstrap Call

- **Previous**: 20× `cuda_multiply_plain()` calls with CPU loops (30,720 iterations each)
- **Now**: 20× `cuda_multiply_plain()` calls with GPU kernels (fully parallel)

### Estimated Speedup

Based on previous fixes:
- Layout conversion GPU: 180-360ms saved
- Ciphertext addition GPU: 100-250ms saved
- **Plaintext multiplication GPU: 100-300ms expected savings** ⭐

**Total expected improvement**: ~0.3-0.5s reduction in EvalMod time (from 11.75s)

## Testing

### Next Steps

User will run the actual test on NVIDIA GPU hardware:

```bash
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap
```

Expected output:
- EvalMod time: **11.2-11.5s** (down from 11.75s)
- Total bootstrap: **11.4-11.7s** (down from 11.94s)

If performance improves, this becomes a **safe milestone** before moving to next optimization.

## Technical Notes

### Why Strided Layout?

The kernel works directly on strided layout:
```
poly[coeff_idx * stride + prime_idx]
```

This matches how ciphertexts are stored in V3, avoiding extra layout conversions.

### Why mul_mod_128()?

Direct 128-bit multiplication `(a * b) % q` would overflow on GPU. The Russian peasant algorithm computes the same result using only 64-bit operations:

```cuda
inline ulong mul_mod_128(ulong a, ulong b, ulong q) {
    ulong result = 0;
    a = a % q;

    while (b > 0) {
        if (b & 1) {
            result = add_mod_lazy(result, a, q);
            if (result >= q) result -= q;
        }
        a = add_mod_lazy(a, a, q);
        if (a >= q) a -= q;
        b >>= 1;
    }

    return result;
}
```

This is the same approach used in the rescaling kernel, proven correct by golden compare tests.

## Conclusion

✅ **100% GPU implementation achieved** for plaintext multiplication
✅ **No more CPU loops** in `cuda_multiply_plain()`
✅ **Code compiles successfully**
⏳ **Awaiting performance results** from user's GPU hardware

This completes the elimination of sequential CPU bottlenecks in V3 CUDA bootstrap!

---

**Author**: Claude Code Agent
**Reviewed by**: David Silva (pending GPU test results)
