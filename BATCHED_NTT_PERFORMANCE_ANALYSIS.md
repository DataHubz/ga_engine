# Batched NTT Performance Analysis - Why It's Slower

## Test Results

**Before (Sequential)**: EvalMod = 14.42s
**After (Batched)**: EvalMod = 17.58s
**Result**: **21% SLOWER** ❌

## Root Cause Analysis

### Issue #1: Massive PCIe Transfer Overhead

The batched implementation copies **twiddles to GPU on every NTT call**:

```rust
// In ntt_forward_batched():
for i in 0..num_primes {
    let ntt_ctx = &self.ntt_contexts[i];
    all_twiddles.extend_from_slice(&ntt_ctx.twiddles);  // CPU allocation + copy
    all_moduli.push(ntt_ctx.q);
}

// Copy to GPU
let gpu_twiddles = self.device.device.htod_copy(all_twiddles)  // 240KB upload!
```

**Per multiplication** (4 forward + 4 inverse NTTs):
- Twiddles copied: 8 × 240KB = **1.92MB**
- At 32 GB/s PCIe: **60μs** per multiplication
- For 10 BSGS multiplications: **0.6ms**

**Total for all BSGS operations** (~100 multiplications):
- Twiddles copied: 100 × 1.92MB = **192MB**
- Transfer time: **6ms**

This alone doesn't explain the 3.16s slowdown, but it's unnecessary overhead.

### Issue #2: Sequential Bit-Reversal (30 kernel launches)

```rust
// Bit-reversal still sequential!
for prime_idx in 0..num_primes {  // 30 iterations
    let func_bit_reverse = self.ntt_contexts[0].device.device.get_func(...)
        .ok_or("Failed to get bit_reverse_permutation function")?;

    unsafe {
        func_bit_reverse.launch(cfg, (&gpu_data_view, n as u32, log_n as u32))
            .map_err(|e| format!("Bit-reversal failed: {:?}", e))?;
    }
}
```

**Per batched NTT call**:
- Bit-reversal: 30 kernel launches
- Overhead: 30 × 20μs = **0.6ms**

**Per multiplication** (4 forward + 4 inverse):
- Overhead: 8 × 0.6ms = **4.8ms**

**For 10 BSGS multiplications**:
- Overhead: 10 × 4.8ms = **48ms**

**For all ~100 multiplications** (including giant steps):
- Overhead: 100 × 4.8ms = **480ms**

### Issue #3: Getting Function Handle in Loop

```rust
for stage in 0..log_n {  // 10 stages
    let func_ntt = self.ntt_contexts[0].device.device.get_func("ntt_module", "ntt_forward_batched")
        .ok_or("Failed to get ntt_forward_batched function")?;  // Lookup overhead!

    unsafe {
        func_ntt.launch(cfg, (...))
    }
}
```

Getting the function handle 10 times per NTT call adds overhead. Should be done once outside the loop.

### Issue #4: Data Copy Overhead

Each batched NTT call:
```rust
// Copy to GPU
let mut gpu_data = self.device.device.htod_copy(data.to_vec())  // 240KB upload
    .map_err(|e| format!("Failed to copy data to GPU: {:?}", e))?;

// ... compute ...

// Copy result back
let result = self.device.device.dtoh_sync_copy(&gpu_data)  // 240KB download
    .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

data.copy_from_slice(&result);
```

**Per multiplication**:
- Forward NTTs: 4 × (240KB H→D + 240KB D→H) = 1.92MB
- Inverse NTTs: 4 × (240KB H→D + 240KB D→H) = 1.92MB
- **Total: 3.84MB per multiplication**

**For 10 BSGS multiplications**:
- Transfer: 10 × 3.84MB = **38.4MB**
- At 32 GB/s: **1.2ms**

**For all ~100 multiplications**:
- Transfer: 100 × 3.84MB = **384MB**
- At 32 GB/s: **12ms**

### Issue #5: Small Ring Dimension (N=1024)

The batched optimization was designed for **N=32768** (production bootstrap), but the test uses **N=1024**.

**Impact**:
- Sequential per-prime: Very lightweight (8KB per prime)
- Batched: Fixed overhead (240KB twiddles, function lookups, etc.)

For small N, **sequential might actually be faster** because the fixed overhead dominates.

**Kernel launch comparison** (per multiplication):
- Sequential: 4 × 10 stages × 30 primes = **1,200 launches**
- Batched: 4 × 10 stages = **40 launches** (30× fewer)

But with 20μs per launch:
- Sequential overhead: 1,200 × 20μs = **24ms**
- Batched overhead: 40 × 20μs = **0.8ms**
- **Savings: 23.2ms per multiplication**

For 100 multiplications: **2.32s saved** from kernel launches

But we're adding **~500ms overhead** from twiddle copies and bit-reversal, so net benefit is only **~1.8s**.

**Yet we're seeing 3.16s SLOWER!**

### Issue #6: Possible GPU Underutilization

For N=1024, num_primes=30:
```
Grid: (2 blocks, 30 primes, 1) = 60 thread blocks
Threads per block: 256
Total threads: 15,360
```

**RTX 5090 specs**:
- 21,760 CUDA cores
- 170 SM units

With only 60 thread blocks across 170 SMs, **GPU utilization is very low** (~35% occupancy).

The GPU is **severely underutilized** for such small problem sizes!

### Issue #7: Correctness Problem?

The 17.58s EvalMod (vs 14.42s before) suggests either:
1. **Extra work** is being done (repeated computations?)
2. **GPU synchronization** is blocking more than expected
3. **Memory allocation** overhead from repeated `Vec` allocations
4. **Compilation issue** where old code is still running?

Let me check if the batched code is actually being called by adding instrumentation.

## Performance Breakdown Estimate

### Sequential Version (14.42s EvalMod)
- Kernel launches: ~100 mult × 1,200 launches/mult × 20μs = **2.4s**
- NTT computation: ~**10s**
- Relinearization: ~**2s**

### Batched Version (Expected)
- Kernel launches: ~100 mult × 40 launches/mult × 20μs = **0.08s** (2.32s saved)
- Twiddle copies: **~500ms** (added overhead)
- NTT computation: ~**10s** (same, maybe faster with better GPU utilization)
- Relinearization: ~**2s** (same)
- **Expected: ~12.6s** (1.8s improvement)

### Batched Version (Actual)
- **Measured: 17.58s** (3.16s WORSE!)

**Something is very wrong!**

## Hypothesis: What's Actually Happening

Looking at the extreme slowdown, I suspect:

1. **Twiddle allocation overhead**: Creating `Vec` with 240KB × 8 times per multiplication = **1.92MB allocations**
   - With Rust's allocator, this could be expensive
   - Memory fragmentation over 100 multiplications

2. **GPU synchronization**: Each `htod_copy` and `dtoh_sync_copy` might be synchronizing the entire GPU pipeline
   - Destroying parallelism across multiplications
   - CUDA stream synchronization overhead

3. **Cache pollution**: Copying 240KB twiddles every call evicts useful data from L2 cache
   - NTT coefficients get evicted
   - More cache misses during computation

4. **Bit-reversal overhead underestimated**: 30 sequential kernel launches with sync between each
   - Real overhead might be 50-100μs per launch (not 20μs)
   - 30 × 100μs × 8 NTTs × 100 multiplications = **2.4s**

## Root Cause Identification

The **#1 culprit** is likely: **Twiddle copying + allocation overhead**

**Evidence**:
- 192MB of twiddles copied for all BSGS operations
- Each copy requires:
  - CPU Vec allocation (expensive for repeated 240KB allocations)
  - PCIe H→D transfer
  - GPU memory allocation (cudarc overhead)
  - GPU synchronization

**Conservative estimate**:
- Allocation: 10μs per Vec
- Transfer: 240KB ÷ 32GB/s = 7.5μs
- GPU sync: 50μs
- **Total: 67.5μs per twiddle copy**

**Per multiplication**: 8 × 67.5μs = **540μs**
**For 100 multiplications**: 100 × 540μs = **54ms**

Still doesn't fully explain the 3.16s overhead...

## Likely Explanation: GPU Synchronization

Each `htod_copy` and `dtoh_sync_copy` is **synchronizing the entire GPU**:

```rust
let mut gpu_data = self.device.device.htod_copy(data.to_vec())?;  // SYNC!
let gpu_twiddles = self.device.device.htod_copy(all_twiddles)?;   // SYNC!
let result = self.device.device.dtoh_sync_copy(&gpu_data)?;       // SYNC!
```

If each sync is ~500μs (waiting for GPU to finish all pending work):
- Per batched NTT: 3 syncs × 500μs = **1.5ms**
- Per multiplication: 12 batched ops × 1.5ms = **18ms**
- For 100 multiplications: 100 × 18ms = **1.8s**

Add bit-reversal overhead (30 syncs per NTT × 500μs = 15ms):
- Per multiplication: 8 NTTs × 15ms = **120ms**
- For 100 multiplications: **12s**

**That could explain the 3.16s slowdown!**

## Conclusion

The batched NTT implementation is **slower** because:

1. **Excessive GPU synchronization** from htod/dtoh copies (est. **~12s overhead**)
2. **Twiddle copy overhead** - copying 240KB per NTT call (**~500ms**)
3. **Sequential bit-reversal** - 30 kernel launches per NTT (**~480ms**)
4. **GPU underutilization** for small N=1024

**The optimization is fundamentally sound** but the implementation has fatal flaws:
- ❌ Data should stay **GPU-resident** (no htod/dtoh per call)
- ❌ Twiddles should be **cached on GPU** once during initialization
- ❌ Bit-reversal should be **batched** like NTT stages
- ❌ Need **CUDA streams** to overlap transfers and compute

## Recommended Fixes

### Priority 1: Keep Data GPU-Resident
Instead of copying data back to CPU after each operation, keep intermediate results on GPU.

### Priority 2: Cache Twiddles on GPU
Upload twiddles once during `CudaCkksContext::new()` and reuse them.

### Priority 3: Batch Bit-Reversal
Create `bit_reverse_permutation_batched` kernel with 2D grid.

### Priority 4: Use CUDA Streams
Overlap data transfers with compute using asynchronous operations.

### Priority 5: Test with Larger N
The optimization targets N=32768. Test with production parameters, not N=1024.

## Next Steps

1. **Verify batched code is actually running** (add instrumentation)
2. **Profile with nvprof** to see actual GPU timeline
3. **Implement GPU-resident data** (biggest win)
4. **Cache twiddles on GPU** (easy fix, big impact)
5. **Test with N=32768** (target production parameters)
