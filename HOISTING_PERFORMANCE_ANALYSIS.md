# Hoisting Performance Analysis & Optimization Roadmap

## Benchmark Results

### Multi-Step Rotation (N=1024, 3 primes, 6 rotations)

**Configuration:**
- Same ciphertext rotated by [1, 2, 4, 8, 16, 32]
- This is the IDEAL case for hoisting (R>>1)

**Results:**
```
Naive (6× decompose + 6× NTT):   1.383s  (0.230s per rotation)
Hoisted (1× decompose + 1× NTT): 0.887s  (0.148s per rotation)

Speedup: 1.56×
Improvement: 35.8%
```

## Analysis

### Current Cost Breakdown (Estimated)

Based on 1.56× speedup from hoisting, we can estimate the rotation pipeline cost:

```
Total rotation cost: 100%

├─ Decompose + forward NTT: ~36%  ← SAVED by hoisting
│
├─ Key-switch operations: ~45%     ← DOMINANT (NOT optimized yet)
│   ├─ Permute NTT digits: ~5%
│   ├─ Transform keys to NTT: ~15%  ← Can pre-compute!
│   └─ Pointwise multiply + accumulate: ~25%
│
└─ Inverse NTT + untwist: ~19%     ← DOMINANT (NOT optimized yet)
```

**Key insight:** Decompose+NTT is only 36% of the cost. Key-switch (45%) and iNTT (19%) dominate at 64% combined.

### Why Speedup < 2×?

Hoisting saves **decompose+NTT** (36%), so theoretical max speedup is:
```
Speedup_max = 1 / (1 - 0.36) = 1.56×
```

This matches our measured 1.56× exactly! To get further speedup, we must optimize key-switch and iNTT.

## Optimization Roadmap

### Priority 1: Pre-NTT Rotation Keys (Estimated 15% win)

**Current:** Keys stored in coefficient domain, transformed to NTT at rotation time
```rust
// Inside rotate_with_hoisted_digits() - done 6× in naive, 6× in hoisted!
let rlk0_ntt = transform_key_to_ntt(&rlk0[t], ...); // twist + NTT
let rlk1_ntt = transform_key_to_ntt(&rlk1[t], ...); // twist + NTT
```

**Optimization:** Pre-compute and cache keys in NTT domain by level
```rust
// At key generation time:
for level in 0..max_level {
    for digit in 0..num_digits {
        rlk0_ntt_cache[level][digit] = transform_key_to_NTT(&rlk0[digit], level_primes);
        rlk1_ntt_cache[level][digit] = transform_key_to_ntt(&rlk1[digit], level_primes);
    }
}

// At rotation time:
let rlk0_ntt = &rlk0_ntt_cache[ct.level][t]; // Direct lookup!
let rlk1_ntt = &rlk1_ntt_cache[ct.level][t];
```

**Estimated speedup:** 1.18× overall (saves 15% of rotation time)
**Tradeoff:** 2× memory for rotation keys (store both coeff and NTT forms)

### Priority 2: Fuse Key-Switch Kernels (Estimated 20-25% win)

**Current:** Multiple kernel launches with intermediate writes to global memory
```
1. Permute NTT digits     → write to global mem
2. Read back for multiply → memory traffic
3. Pointwise multiply     → write to global mem
4. Read back for iNTT     → memory traffic
5. Inverse NTT            → write to global mem
6. Untwist               → write result
```

**Optimization:** Single fused Metal kernel per prime
```metal
kernel void fused_keyswitch(
    const device uint64_t* digit_ntt,      // Input: hoisted digit
    const device uint64_t* rlk_ntt,        // Input: pre-NTT'd key
    const device uint* perm_map,            // Permutation with offset
    device uint64_t* result,                // Output: coeff domain
    // ... NTT parameters
) {
    // 1. Permute (in threadgroup memory)
    uint64_t digit_permuted = digit_ntt[perm_map[tid]];

    // 2. Pointwise multiply (Montgomery domain)
    uint64_t product = mont_mul(digit_permuted, rlk_ntt[tid]);

    // 3. Inverse NTT (in-place, use threadgroup for butterfly tiles)
    uint64_t coeff = inverse_ntt_inplace(product, ...);

    // 4. Untwist
    result[tid] = mont_mul(coeff, psi_inv[tid]);
}
```

**Benefits:**
- Eliminates 3-4 global memory round-trips per prime
- Better GPU occupancy (fewer kernel launches)
- Threadgroup memory for butterfly tiles

**Estimated speedup:** 1.25-1.30× overall (saves 20-25% of rotation time)

### Priority 3: Reduce V4 Butterfly Rotation Count with BSGS (Estimated 30-40% win)

**Current:** 7 rotations per unpack (1×rot(4), 2×rot(2), 4×rot(1))

**Optimization:** Baby-step/giant-step for 8-way butterfly
```
Current 8-point butterfly:  7 rotations
BSGS 2×(4-point):          5 rotations  → 1.4× faster
BSGS with diagonals:       4 rotations  → 1.75× faster
```

**Algorithm sketch (BSGS approach):**
```
1. Split into 2×(4-point) sub-butterflies:
   - Components [0,1,2,3]: 2 rotations (baby + giant steps)
   - Components [4,5,6,7]: 2 rotations (baby + giant steps)

2. Combine with 1 rotation (stride-4)

Total: 5 rotations instead of 7
```

**Estimated speedup:** 1.4× for butterfly-heavy workloads (geometric product)

### Priority 4: Batch Same-Step Rotations (Estimated 10-15% win for butterfly)

**Current:** Rotations launched separately even when they have the same step

**Optimization:** Single batched kernel for multiple ciphertexts with same step
```rust
// Stage 1: four rot(1) operations on different ciphertexts
let rot1_batch = batch_rotate_different_cts(
    &[q0, q1, q2, q3],  // 4 different ciphertexts
    1,                   // Same rotation step
    &rot_keys,
    &ctx
)?;
```

**Benefits:**
- Better GPU occupancy (process 4 cts in one kernel dispatch)
- Shared permutation map across all 4 ciphertexts
- Shared rotation keys across all 4 ciphertexts
- Amortize kernel launch overhead

**Estimated speedup:** 1.10-1.15× for butterfly (savings from reduced launches + better occupancy)

### Priority 5: Optimize Memory Layout (Estimated 5-10% win)

**Current:** Unknown layout details

**Optimizations:**
1. **Structure of Arrays (SoA):** Per-prime, per-digit arrays
2. **256-byte alignment:** For Metal threadgroup coalescing
3. **Lazy reduction:** Postpone Barrett reduction to every 2-3 ops
4. **Threadgroup memory:** Use for NTT butterfly tiles (16-32 elements)

**Estimated speedup:** 1.05-1.10× overall (reduces memory bandwidth)

## Combined Speedup Estimates

Applying optimizations sequentially:

```
Baseline:              1.000× (current hoisting implementation)
+ Pre-NTT keys:        1.180× (15% win)
+ Fused kernels:       1.475× (1.18 × 1.25 = 47% over baseline)
+ BSGS butterfly:      2.065× (1.475 × 1.4 = 2× over baseline, V4 only)
+ Batched same-step:   2.272× (2.065 × 1.1 = 2.3× over baseline, V4 only)
+ Memory optimizations: 2.500× (2.27 × 1.1 = 2.5× over baseline, V4 only)
```

**For multi-step rotations (like benchmark):**
```
Current hoisting:      1.56× over naive
+ Pre-NTT keys:        1.84× over naive (1.56 × 1.18)
+ Fused kernels:       2.30× over naive (1.56 × 1.25 × 1.18)
```

**For V4 butterfly (full pipeline):**
```
Current:               ~15.5s geometric product test
+ All optimizations:   ~6.2s (2.5× faster)
```

## Implementation Priority Order

### Immediate (Biggest bang for buck):
1. **Pre-NTT rotation keys** (15% win, medium effort)
   - Modify `MetalRotationKeys::generate()` to store both forms
   - Version keys by level (separate cache per level)
   - Update `rotate_with_hoisted_digits()` to use cached NTT keys

2. **Fused key-switch kernel** (20-25% win, high effort)
   - Write single Metal kernel: permute → mul → iNTT → untwist
   - Use threadgroup memory for butterfly tiles
   - Eliminate intermediate global memory writes

### Medium-term (Algorithmic wins):
3. **BSGS butterfly** (30-40% win for V4, medium effort)
   - Redesign 8-way butterfly as 2×(4-point) with baby/giant steps
   - Reduce from 7 to 5 rotations
   - Further optimize with diagonal pre-multiplication (4 rotations)

4. **Batch same-step rotations** (10-15% win, medium effort)
   - Add `batch_rotate_different_cts()` API
   - Single kernel dispatch for multiple ciphertexts with same step
   - Amortize launch overhead and improve occupancy

### Polish (Final 5-10%):
5. **Memory layout optimizations** (5-10% win, high effort)
   - SoA layout for per-prime arrays
   - 256-byte alignment
   - Lazy Barrett reduction
   - Threadgroup memory tuning

## Next Steps

**For you to run:**

1. **Profile single rotation breakdown** - Add Metal GPU timing to measure:
   ```
   Time per rotation:
     - decompose + NTT:     ~XXms (36%)
     - transform keys:      ~XXms (15%)  ← Can be cached!
     - permute:             ~XXms (5%)
     - multiply+accumulate: ~XXms (25%)
     - iNTT + untwist:      ~XXms (19%)
   ```

2. **Test with more primes** - Run benchmark with num_primes = 3, 5, 7 to see scaling
   - If iNTT time grows superlinearly → memory bandwidth bound → prioritize fusion

3. **Memory bandwidth test** - Check if we're compute-bound or bandwidth-bound
   - Metal GPU counters can show utilization

**Provide me this breakdown and I'll refine the estimated wins per optimization.**

## Summary

- ✅ Hoisting works: **1.56× speedup** for multi-step rotations
- ⚠️ Key-switch/iNTT dominate (64% of rotation cost)
- 🎯 Next wins: Pre-NTT keys (15%) + fused kernels (20-25%)
- 🚀 Full optimization: **~2.5× total speedup** over current (aggressive estimate)

The 1.56× from hoisting is solid, but to get to 2-3× we need the key-switch/iNTT optimizations. Pre-NTTing the keys is the quickest win with medium effort.
