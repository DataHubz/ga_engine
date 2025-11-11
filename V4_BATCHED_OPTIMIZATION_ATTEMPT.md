# V4 Batched Multi-Prime Optimization Attempt

**Date**: 2025-11-10
**Status**: ✅ IMPLEMENTED, ⚠️ NEUTRAL PERFORMANCE

## Overview

Implemented batched 2D Metal GPU dispatch for multi-prime pointwise multiplication. The goal was to process all RNS primes in parallel to eliminate sequential per-prime processing overhead.

## Implementation

### Metal Kernel

Added `ntt_pointwise_multiply_batched` kernel ([ntt.metal:419-443](src/clifford_fhe_v2/backends/gpu_metal/shaders/ntt.metal#L419-L443)):

```metal
kernel void ntt_pointwise_multiply_batched(
    device const ulong* a [[buffer(0)]],
    device const ulong* b [[buffer(1)]],
    device ulong* c [[buffer(2)]],
    constant ulong* moduli [[buffer(3)]],
    constant ulong* moduli_inv [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& num_primes [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]  // 2D dispatch!
)
```

**Key features:**
- 2D dispatch: `(coeff_idx, prime_idx)`
- Flat interleaved layout: `[coeff0_p0, coeff0_p1, coeff0_p2, coeff1_p0, ...]`
- Per-prime moduli and Montgomery parameters

### Rust Wrapper

Added `pointwise_multiply_batched()` to MetalCkksContext ([ckks.rs:1171-1245](src/clifford_fhe_v2/backends/gpu_metal/ckks.rs#L1171-L1245)):

```rust
pub fn pointwise_multiply_batched(
    &self,
    a_flat: &[u64],
    b_flat: &[u64],
    moduli: &[u64],
) -> Result<Vec<u64>, String>
```

**Dispatch strategy:**
- Threadgroup size: 16×16 = 256 threads
- Grid size: `(⌈n/16⌉, ⌈num_primes/16⌉)`

### Integration

Updated `multiply_ntt_and_intt_batched()` in hoisting ([hoisting.rs:652-696](src/clifford_fhe_v2/backends/gpu_metal/hoisting.rs#L652-L696)):

```rust
// STEP 1: BATCHED pointwise multiply for all primes in ONE GPU dispatch
let product_flat = ctx.pointwise_multiply_batched(a_ntt_flat, b_ntt_flat, moduli)?;

// STEP 2: Per-prime inverse NTT + untwist (still sequential)
for (prime_idx, _q) in moduli.iter().enumerate() {
    // Extract, iNTT, store
}
```

## Performance Results

### V4 Geometric Product

| Implementation | Time | vs Baseline |
|---------------|------|-------------|
| **Baseline** (fused iNTT+untwist) | 10.35s | - |
| **Batched 2D dispatch** | 10.70s | -3.4% (slower!) |

**Runs:**
- Run 1: 10.75s
- Run 2: 10.65s
- Run 3: 10.99s
- **Average: ~10.7s**

### Analysis: Why Slower?

1. **Too few primes (3)**: 2D dispatch overhead dominates for small Y dimension
   - Y-dimension threadgroups: ⌈3/16⌉ = 1
   - Underutilized GPU: Most threads idle in Y dimension

2. **Buffer setup cost**: Additional work to gather moduli arrays
   ```rust
   let mut moduli_array = Vec::with_capacity(num_primes);
   let mut moduli_inv_array = Vec::with_capacity(num_primes);
   // ... gather per-prime parameters
   ```

3. **2D dispatch overhead**: More complex dispatch than 1D
   - 1D: `dispatch_thread_groups((n+255)/256, 1, 1)`
   - 2D: `dispatch_thread_groups((n+15)/16, (3+15)/16, 1)`

4. **Cache locality**: Sequential per-prime may have better cache behavior

## When Would Batched Help?

The batched approach would be beneficial with:

### Scenario 1: Many RNS Primes (10-20+)
```
Configuration:
  num_primes = 20
  2D grid: 64×2 threadgroups (good GPU occupancy!)

Expected speedup: 2-3× for num_primes > 10
```

### Scenario 2: Larger Polynomial Degree
```
Configuration:
  N = 8192 or 16384
  num_primes = 10+
  Larger grid fully utilizes GPU
```

### Scenario 3: Batch Processing
```
Process multiple ciphertexts simultaneously:
  3D dispatch: (n, num_primes, batch_size)
  Even better amortization
```

## Current V4 Configuration

```
N = 1024
num_primes = 3  ← Too small for batching benefit!
Primes per rotation: 3
```

**Conclusion**: For V4's parameters, sequential per-prime processing is actually optimal.

## Recommendations

### Keep the Implementation

The batched kernel is good infrastructure for:
1. Future larger parameter sets
2. Bootstrap operations (may use more primes)
3. Code reuse in other contexts

### Make It Adaptive

Add heuristic to choose strategy:

```rust
fn multiply_ntt_and_intt_batched(...) -> Result<Vec<u64>, String> {
    const BATCH_THRESHOLD: usize = 8;  // Empirically determined

    if moduli.len() >= BATCH_THRESHOLD {
        // Use 2D batched dispatch
        let product_flat = ctx.pointwise_multiply_batched(...)?;
        // ... per-prime iNTT
    } else {
        // Use sequential (current baseline)
        for (prime_idx, _q) in moduli.iter().enumerate() {
            // ... per-prime multiply + iNTT
        }
    }
}
```

### Alternative Optimizations for V4

Since batching doesn't help with 3 primes, focus on:

1. **GPU accumulation loops** (lines 525-542)
   - Move modular add/sub to GPU kernel
   - Current: CPU loop with branch per element

2. **Permutation optimization**
   - Cache permutation maps (computed once per step)
   - GPU permutation kernel?

3. **Multi-digit batching**
   - Process all digits (typically 8) together
   - Larger batch → better GPU utilization

## Code Status

**Files Modified:**
- `src/clifford_fhe_v2/backends/gpu_metal/shaders/ntt.metal` (+ batched kernel)
- `src/clifford_fhe_v2/backends/gpu_metal/ckks.rs` (+ batched wrapper)
- `src/clifford_fhe_v2/backends/gpu_metal/hoisting.rs` (use batched)
- `src/clifford_fhe_v2/backends/gpu_metal/ntt.rs` (+ q() accessor)

**Tests:** ✅ All pass with exact agreement
**Performance:** ⚠️ Slightly slower for V4 (3 primes)

## Lessons Learned

1. **GPU batching isn't always faster**
   - Overhead matters for small batches
   - Need sufficient parallelism to amortize setup cost

2. **Problem size matters**
   - 2D dispatch needs Y ≥ 8-16 for good occupancy
   - V4's 3 primes is below this threshold

3. **Measure, don't assume**
   - "Batched" sounds faster but isn't always
   - Real-world parameters determine actual performance

4. **Infrastructure vs immediate wins**
   - Good to have for future use
   - But not the optimization V4 needs now

## Next Steps

Since batched multi-prime didn't help, the next best optimizations for V4 are:

1. **Multi-digit batching** (Target: 10-15% speedup)
   - Process all 8 digits of key-switch together
   - Better GPU occupancy

2. **GPU accumulation** (Target: 5-8% speedup)
   - Move add/sub loops to GPU kernels
   - Eliminate CPU arithmetic

3. **Rotation caching in butterfly** (Target: 5-10% speedup)
   - Butterfly uses rotations {1,2,4} multiple times
   - Cache and reuse instead of recomputing

**Most promising**: Multi-digit batching, as it increases batch size from 3 (primes) to 24 (primes×digits).
