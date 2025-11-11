# V4 Optimization Roadmap: From 13.5s to ~2-3s

## Current Bottleneck Analysis

**Current pack/unpack pattern (per GP):**
```
Pack A:   7 rotations (components 1-7)
Pack B:   7 rotations (components 1-7)
Unpack R: 7 rotations + 8 mask multiplies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:    21 rotations + 8 masks = ~7.2s
```

**Rotation pattern:**
- Pack: rotate component i by +i steps (i=0..7)
- Unpack: rotate packed by -i, then mask multiply

This is **maximally inefficient** - we're doing individual rotations sequentially with no reuse.

## Optimization Strategy (Ranked by Impact)

### Phase 1: Butterfly Transform (HIGHEST IMPACT)
**Target: 21 rotations → 3-4 rotations**
**Expected speedup: 5-7×**

#### Current Slot Layout (Naive)
```
Slots: [s₀, e1₀, e2₀, e3₀, e12₀, e23₀, e31₀, I₀, s₁, e1₁, ...]
       └─────────── batch 0 ──────────┘ └───── batch 1 ─────┘
```

Component i is at positions: i, i+8, i+16, ...

#### Optimized Slot Layout (Bit-Reversal)
Map component index to low 3 bits of slot index:
```
Slot index:  [batch_id * 8 + component_id]
Component:   slot & 0x7

Example (N=1024, batch_size=64):
- Component 0 (scalar): slots 0, 8, 16, ... 504
- Component 1 (e₁):     slots 1, 9, 17, ... 505
- Component 2 (e₂):     slots 2, 10, 18, ... 506
...
```

#### Butterfly Unpack (3 stages, log₂(8)=3)
```rust
// Stage 1: Separate odd/even by component bit 0
let rot_4 = packed.rotate(4);           // Rotate by N/2 in component space
let even = packed.add(&rot_4);          // Components with bit0=0
let odd  = packed.sub(&rot_4);          // Components with bit0=1

// Stage 2: Separate by component bit 1
let even_rot2 = even.rotate(2);
let q0 = even.add(&even_rot2);          // Components 0,1
let q1 = even.sub(&even_rot2);          // Components 2,3

let odd_rot2 = odd.rotate(2);
let q2 = odd.add(&odd_rot2);            // Components 4,5
let q3 = odd.sub(&odd_rot2);            // Components 6,7

// Stage 3: Final separation by bit 2
let c0 = q0.add(&q0.rotate(1));         // Component 0
let c1 = q0.sub(&q0.rotate(1));         // Component 1
let c2 = q1.add(&q1.rotate(1));         // Component 2
let c3 = q1.sub(&q1.rotate(1));         // Component 3
let c4 = q2.add(&q2.rotate(1));         // Component 4
let c5 = q2.sub(&q2.rotate(1));         // Component 5
let c6 = q3.add(&q3.rotate(1));         // Component 6
let c7 = q3.sub(&q3.rotate(1));         // Component 7
```

**Total rotations: 3 unique (by 4, 2, 1)**
**Reused across all stages: Yes!**

#### Butterfly Pack (Inverse transform)
```rust
// Inverse butterfly: combine components → packed
// Stage 1: Combine pairs
let q0 = (c0.add(&c1.rotate(-1))).mult_plain(0.5);
let q1 = (c2.add(&c3.rotate(-1))).mult_plain(0.5);
let q2 = (c4.add(&c5.rotate(-1))).mult_plain(0.5);
let q3 = (c6.add(&c7.rotate(-1))).mult_plain(0.5);

// Stage 2: Combine quads
let even = (q0.add(&q1.rotate(-2))).mult_plain(0.5);
let odd  = (q2.add(&q3.rotate(-2))).mult_plain(0.5);

// Stage 3: Final combine
let packed = (even.add(&odd.rotate(-4))).mult_plain(0.5);
```

**Total rotations: 3 unique (by 1, 2, 4)**

**Savings per GP:**
- Before: 21 rotations
- After: 3 (unpack) + 3 (pack a) + 3 (pack b) = 9 rotations
- **Reduction: 21 → 9 = 2.3×** fewer rotations
- Time: 7.2s → 3.1s

### Phase 2: Automorphism Hoisting (HIGH IMPACT)
**Target: Amortize NTT costs across rotation batch**
**Expected speedup: 2-3×**

#### Problem
Current rotation (per key-switch):
```rust
for each rotation k:
    digits = decompose(c1, base_w)      // CPU work
    for each digit:
        digit_ntt = forward_ntt(digit)  // GPU NTT
        prod = mult(digit_ntt, rlk_k)   // GPU pointwise
    c1_rot = inverse_ntt(sum(prod))     // GPU inverse NTT
```

**Cost per rotation: decompose + 8 forward NTTs + 8 inverse NTTs**

#### Solution: Hoist Decomposition
```rust
// Do ONCE per rotation batch:
digits = decompose(c1, base_w)                  // 1×
digits_ntt = [forward_ntt(d) for d in digits]  // 8× forward NTT

// For EACH rotation k (cheap):
for each rotation k:
    prod_k = [mult(digits_ntt[i], rlk_k[i]) for i in 0..8]  // Just multiply
    c1_rot_k = inverse_ntt(sum(prod_k))                      // 1× inverse NTT
```

**Before:** 9 rotations × (1 decompose + 8 forward NTT + 8 inv NTT) = 9 + 72 + 72 = 153 NTT ops
**After:** 1 decompose + 8 forward NTT + 9 inverse NTT = 1 + 8 + 9 = 18 NTT ops

**Reduction: 153 → 18 NTT ops = 8.5× fewer NTTs**
**Time: 3.1s → 1.0s**

### Phase 3: Fused Multi-Rotation Kernel (MEDIUM IMPACT)
**Target: Eliminate command buffer overhead**
**Expected speedup: 1.5-2×**

Instead of 9 separate kernel launches, launch one fused kernel:
```metal
kernel void fused_multi_rotation(
    device uint64_t* hoisted_digits [[buffer(0)]],     // Pre-NTT digits
    constant GaloisMap* galois_maps [[buffer(1)]],     // 9 rotation maps
    device uint64_t* rotation_keys [[buffer(2)]],      // All RLKs
    device uint64_t* output [[buffer(3)]],             // 9 outputs
    constant RotationParams* params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // Load hoisted digit once
    uint64_t digit = hoisted_digits[gid];

    // Apply all 9 rotations in one kernel
    for (int k = 0; k < 9; k++) {
        uint galois_idx = galois_maps[k].map[gid % N];
        uint64_t rlk_val = rotation_keys[k * N * num_digits + ...];
        uint64_t prod = montgomery_mul(digit, rlk_val, q);
        output[k * N + galois_idx] = prod;
    }
}
```

**Benefit:** No sync between rotations, better cache locality, reduced PCIe overhead
**Time: 1.0s → 0.6s**

### Phase 4: Baby-Step/Giant-Step (BSGS) for Rotations (LOW IMPACT)
**Target: Reduce 9 rotations → 6 rotations**
**Expected speedup: 1.5×**

For rotation set {1, 2, 4}, BSGS decomposition:
- Baby steps: {1, 2}
- Giant steps: {0, 2}
- Represent: rot(4) = rot(2) ∘ rot(2)

**Net: 3 unique rotations (1, 2, 4) with reuse**
Already covered by butterfly approach - this is mainly for larger rotation sets.

**For V4's tiny set (1,2,4), benefit is minimal.**

### Phase 5: Device-Resident Buffers (LOW IMPACT)
**Target: Eliminate CPU↔GPU copies**
**Expected speedup: 1.2×**

Current:
```rust
let extracted = extract_prime_from_flat(&ct.c0, ...);  // Copy to CPU
let result = metal_gp.compute(&extracted)?;             // Upload to GPU
insert_prime_into_flat(&result, ...);                  // Copy to CPU
```

Optimized:
```rust
// Keep everything on GPU
let gpu_view = ct.get_gpu_slice(prime_idx)?;           // Device pointer
let result_gpu = metal_gp.compute_inplace(gpu_view)?;  // Stay on device
```

**Time: 0.6s → 0.5s**

## Implementation Plan

### Week 1: Butterfly Transform (Core)
**Files to create/modify:**
- `src/clifford_fhe_v4/packing_butterfly.rs` (NEW)
- `src/clifford_fhe_v4/packing.rs` (add feature flag)

**Steps:**
1. Implement 3-stage butterfly unpack
2. Implement 3-stage butterfly pack
3. Add `cfg(feature = "butterfly-packing")` flag
4. Test correctness against naive implementation
5. Benchmark improvement

**Expected outcome:** 7.2s → 3.1s

### Week 2: Automorphism Hoisting
**Files to modify:**
- `src/clifford_fhe_v2/backends/gpu_metal/rotation_keys.rs`
- Add `MetalCiphertext::rotate_batch_with_hoisting()`

**Steps:**
1. Add `hoist_c1_digits()` method
2. Add `apply_rotation_batch()` method
3. Modify butterfly to use batch API
4. Benchmark improvement

**Expected outcome:** 3.1s → 1.0s

### Week 3: Fused Kernel
**Files to create/modify:**
- `src/clifford_fhe_v2/backends/gpu_metal/shaders/multi_rotation.metal` (NEW)
- `src/clifford_fhe_v2/backends/gpu_metal/rotation_keys.rs`

**Steps:**
1. Write fused Metal shader
2. Integrate with hoisting API
3. Benchmark improvement

**Expected outcome:** 1.0s → 0.6s

### Week 4: Device-Resident Buffers
**Files to modify:**
- `src/clifford_fhe_v4/geometric_ops.rs`
- Add GPU memory management

**Expected outcome:** 0.6s → 0.5s

## Final Performance Target

```
Component Breakdown (optimized):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pack/Unpack: ~0.5s (butterfly + hoisting + fused kernel)
Metal GP:    ~0.5s (3 primes with device reuse)
Overhead:    ~0.2s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:       ~1.2s per geometric product
```

**Improvement: 13.5s → 1.2s = 11× speedup**

Compare to V2: 33ms (unpacked, direct)
- V4 still slower, but **much more reasonable** for 8× memory savings
- 36× slower instead of 400× slower

## Concrete Next Steps

### Immediate (This Week):
1. ✅ Document current rotation pattern (DONE)
2. Create `packing_butterfly.rs` with 3-stage transform
3. Test correctness with small example
4. Measure rotation count (should be 9 instead of 21)

### Short-term (Next 2 Weeks):
1. Implement hoisting API
2. Integrate hoisting with butterfly
3. Profile GPU kernel times
4. Write fused multi-rotation shader

### Medium-term (Next Month):
1. Device-resident buffer management
2. Consider "compute while packed" for simple GA ops
3. Document trade-offs in user guide
4. Add performance benchmarks to CI

## Questions for Implementation

1. **Slot layout migration:** Do we need backward compatibility with old packed format?
2. **Testing strategy:** How to verify butterfly correctness? Use V2 as ground truth?
3. **Feature flags:** Enable butterfly by default, or keep both implementations?
4. **Metal shader complexity:** Is fused kernel maintainable, or too complex?

---

**Bottom line:** With butterfly + hoisting + fused kernels, V4 can achieve **~1.2s per GP** (11× improvement) while keeping 8× memory savings. This makes V4 viable for practical use, not just a theoretical exercise.
