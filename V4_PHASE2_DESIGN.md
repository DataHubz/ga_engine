# V4 Phase 2: Automorphism Hoisting - Implementation Design

## Goal

Reduce key-switching overhead from **~2.7s → ~1.0s** by hoisting gadget decomposition and forward NTT operations outside the rotation loop.

## Current Bottleneck Analysis

### Per-Rotation Cost (current implementation)

```rust
pub fn rotate_by_steps(&self, step: i32, ...) -> Result<Self, String> {
    // 1. Extract rotation keys for this step (~negligible)
    let (rlk0, rlk1) = rot_keys.get_key_for_step(step)?;

    // 2. Apply Galois automorphism (cheap: just permutation + sign flip)
    let c0_rotated = apply_galois_gpu(&c0, galois_map, ...)?;  // ~0.01s
    let c1_rotated = apply_galois_gpu(&c1, galois_map, ...)?;  // ~0.01s

    // 3. Key-switch using gadget decomposition (EXPENSIVE!)
    let (c0_final, c1_final) = key_switch_gpu_gadget(
        c0_rotated, c1_rotated, rlk0, rlk1, ...
    )?;  // ~0.28s
}
```

###  `key_switch_gpu_gadget` Breakdown (per rotation)

```rust
fn key_switch_gpu_gadget(...) -> Result<(Vec<u64>, Vec<u64>), String> {
    // Decompose c1 into base-w digits
    let c1_digits = gadget_decompose_flat(c1_rotated, base_w, ...)?;  // ~0.01s (CPU)

    // For each digit (typically 8 digits with base_w=32):
    for t in 0..num_digits {  // 8 iterations
        // Forward NTT + pointwise multiply + inverse NTT
        let term0 = multiply_digit_by_ntt_key(&c1_digits[t], &rlk0[t], ...)?;  // ~0.03s
        let term1 = multiply_digit_by_ntt_key(&c1_digits[t], &rlk1[t], ...)?;  // ~0.03s

        // Accumulate results (~negligible)
        c0_final -= term0;
        c1_final += term1;
    }
}
```

### `multiply_digit_by_ntt_key` Cost (per digit)

```rust
fn multiply_digit_by_ntt_key(digit_coeff, key_coeff, ...) -> Result<Vec<u64>, String> {
    for each prime:
        // Forward NTT on digit
        digit_ntt = forward_ntt(digit_coeff)?;        // ~0.015s (GPU)

        // Pointwise multiply (key already in NTT domain)
        result_ntt = pointwise_mult(digit_ntt, key_coeff)?;  // ~0.005s (GPU)

        // Inverse NTT
        result = inverse_ntt(result_ntt)?;            // ~0.015s (GPU)
}
```

### Total Cost (9 rotations in butterfly transform)

```
Per rotation: 0.01 (decompose) + 8 × 0.03 (NTT ops per digit) = ~0.25s
9 rotations: 9 × 0.25s = ~2.25s
```

This matches our observed ~2-3s overhead for rotations!

## Optimization Strategy: Automorphism Hoisting

### Key Insight

All 9 rotations in the butterfly transform operate on the SAME `c1` component. We're decomposing the same `c1` nine times and forward-NTT'ing the same digits 9 × 8 = 72 times!

### Hoisted Algorithm

```rust
// NEW: Batch rotation API
pub fn rotate_batch_with_hoisting(
    &self,
    steps: &[i32],
    rot_keys: &MetalRotationKeys,
    ctx: &MetalCkksContext,
) -> Result<Vec<Self>, String> {

    // Step 1: Extract active primes ONCE
    let num_primes_active = self.level + 1;
    let moduli = &ctx.params.moduli[..num_primes_active];
    let (c0_active, c1_active) = extract_active_primes(&self)?;

    // Step 2: Decompose c1 ONCE for all rotations
    let c1_digits = gadget_decompose_flat(&c1_active, base_w, moduli, n)?;  // 1×

    // Step 3: Forward-NTT all digits ONCE
    let c1_digits_ntt = forward_ntt_all_digits(&c1_digits, moduli, ctx)?;  // 8× forward NTT

    // Step 4: For EACH rotation (now much cheaper!)
    let mut results = Vec::with_capacity(steps.len());
    for &step in steps {
        // Galois automorphism on c0 and c1
        let c0_rotated = apply_galois_gpu(&c0_active, step, ...)?;  // ~0.01s
        let c1_rotated = apply_galois_gpu(&c1_active, step, ...)?;  // ~0.01s

        // Key-switch using PRE-COMPUTED NTT digits (FAST!)
        let (c0_final, c1_final) = key_switch_with_hoisted_digits(
            &c0_rotated,
            &c1_rotated,  // We need c1_rotated in frequency domain
            &c1_digits_ntt,  // Pre-computed!
            step,
            rot_keys,
            moduli,
            ctx
        )?;  // ~0.08s (just inverse NTT + accumulate)

        results.push(Self { c0: c0_final, c1: c1_final, ... });
    }

    Ok(results)
}
```

### Wait - There's a Problem!

The hoisting works for `c1` decomposition and forward NTT, but **each rotation applies a different Galois automorphism** to `c1`. So we can't directly reuse the NTT digits - we need `c1_rotated` in NTT domain, not `c1`.

### Refined Strategy: Hoist What We Can

After analysis, here's what we can actually hoist:

**❌ Cannot hoist:**
- Galois automorphism (different for each rotation)
- Forward NTT of c1_digits (they change after Galois)

**✅ Can hoist:**
- Rotation key extraction (amortize across batch)
- GPU buffer allocations
- Decomposition base computation
- Metal kernel setup

Actually, let me reconsider the Galois operation...

### Key Realization: Galois + Decompose Ordering

Current:
```
c1 → apply_galois(k) → c1_rotated → decompose → forward_NTT
```

Could we do:
```
c1 → decompose → c1_digits → apply_galois(k) to each digit → forward_NTT
```

**NO!** Gadget decomposition is base-w digit extraction, which doesn't commute with Galois automorphism. The Galois operation permutes coefficients and flips signs, which breaks the decomposition structure.

### Realistic Optimization: Batch Processing

Since we can't mathematically hoist the NTT operations, we focus on **GPU kernel fusion** and **batch dispatch**:

1. **Fused Galois + Decompose + NTT kernel** - Do all three in one GPU dispatch
2. **Batch key-switching** - Process all 9 rotations in parallel on GPU
3. **Persistent GPU buffers** - Avoid repeated CPU↔GPU copies

## Revised Phase 2 Approach

Given the mathematical constraints, Phase 2 should focus on:

### Option A: Fused GPU Kernels (High Impact)
Combine Galois + Decompose + NTT into single Metal shader:
- **Benefit:** Eliminate kernel launch overhead, better cache locality
- **Expected speedup:** 1.5-2× (2.7s → 1.4-1.8s)
- **Complexity:** Medium (new Metal shader)

### Option B: Parallel Rotation Batch (Medium Impact)
Process multiple rotations in parallel on GPU:
- **Benefit:** Better GPU utilization
- **Expected speedup:** 1.3-1.5× (2.7s → 1.8-2.1s)
- **Complexity:** Low (use existing kernels)

### Option C: Mixed Strategy (Recommended)
1. Start with **Option B** (parallel batching) - quick win
2. Then implement **Option A** (fused kernels) - bigger gain
3. Combine both for maximum performance

## Implementation Plan

### Step 1: Batch Rotation API (Quick Win)

```rust
impl MetalCiphertext {
    /// Rotate by multiple steps simultaneously (GPU parallel processing)
    pub fn rotate_batch(
        &self,
        steps: &[i32],
        rot_keys: &MetalRotationKeys,
        ctx: &MetalCkksContext,
    ) -> Result<Vec<Self>, String> {
        // Process all rotations in parallel on GPU
        // Use Metal command buffer to dispatch all at once
    }
}
```

**Expected:** 2.7s → 2.0s (1.35× speedup)

### Step 2: Fused Rotation Kernel (Bigger Gain)

Create `rotation_fused.metal`:
```metal
kernel void fused_galois_decompose_ntt(
    device const uint64_t* c1_input,
    device const uint64_t* rotation_keys,
    constant GaloisParams* galois_params,
    device uint64_t* c1_output_ntt,
    uint gid [[thread_position_in_grid]]
) {
    // 1. Apply Galois automorphism
    // 2. Extract base-w digit
    // 3. Forward NTT
    // All in one kernel!
}
```

**Expected:** 2.0s → 1.2s (combined: 2.25× overall speedup)

## Success Metrics

- **Phase 2 Target:** Geometric product 12.97s → 10-11s
- **Rotation overhead:** 2.7s → 1.2s (55% reduction)
- **Test:** All V4 tests still pass
- **Code:** Maintainable, well-documented

## Next Steps

1. Implement `rotate_batch()` for parallel processing
2. Benchmark improvement
3. If successful, proceed with fused kernel
4. Update V4 butterfly to use batch API

---

**Note:** The original "automorphism hoisting" isn't mathematically possible due to non-commutativity. Phase 2 focuses on **batch processing** and **kernel fusion** instead, which are more realistic optimizations.
