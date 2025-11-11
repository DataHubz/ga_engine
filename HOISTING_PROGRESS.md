# Hoisting Implementation Progress

## Current Status

The hoisting optimization has been implemented and structurally debugged. The core issue - **negacyclic sign cancellation** - has been identified and understood mathematically, but the test validation still needs refinement.

## Key Mathematical Insight (from Expert Review)

For negacyclic NTT with Galois automorphism σ_g, the negacyclic reduction signs **CANCEL** with the twist index change:

```
(-1)^⌊gi/N⌋ · ψ^{i'} = ψ^{gi}
```

This happens because:
- ψ^{i'} = ψ^{gi - N⌊gi/N⌋} = ψ^{gi} · ψ^{-N⌊gi/N⌋} = ψ^{gi} · (ψ^N)^{-⌊gi/N⌋}
- Since ψ^N = -1 (primitive 2N-th root), we get: ψ^{gi} · (-1)^{-⌊gi/N⌋}
- The two (-1) factors cancel!

**Result:** No per-coefficient sign tables needed in hoisting! The diagonal twist D_g[j] = ψ^{(g-1)j} handles everything.

## Implementation Status

### ✅ Completed

1. **Core Hoisting Functions**
   - `hoist_decompose_ntt()` - Decompose + NTT once (line 238-297 in hoisting.rs)
   - `compute_diagonal_twist()` - Compute D_g[j] = ψ^{(g-1)j} in Montgomery (line 315-360)
   - `rotate_with_hoisted_digits()` - Fast rotation via permute + diagonal (line 440-509)
   - `permute_in_place_ntt()` - PULL semantics permutation (line 186-204)

2. **Batch API**
   - `rotate_batch_with_hoisting()` - Fixed to hoist ONCE before loop (line 1650-1652 in ckks.rs)
   - Previously hoisted inside loop (defeating the purpose) - now fixed ✅

3. **Formula Implementation**
   - Permutation: `map[j] = (j * g) mod N` with PULL semantics ✅
   - Diagonal: `D_g[j] = ψ^{(g-1)j}` via rolling Montgomery power ✅
   - Montgomery domain: All multiplications in Montgomery ✅

### 🔍 In Progress

**Negacyclic Sanity Test** - Test created but still failing

The test `test_negacyclic_hoisting_sanity` compares:
- **Path 1:** `Galois(coeff with signs) → twist by ψ → NTT`
- **Path 2:** `twist by ψ → NTT → permute → multiply D_g`

**Issue:** Values still don't match (diff ~half of q).

**Hypothesis:** Path 1's Galois application may not correctly model the sign cancellation. The signs are applied in coefficient domain, but the twist is applied AFTER, and the interaction between signs and twist needs careful handling.

**Next Steps:**
1. Create minimal N=4 trace-through to verify the formula manually
2. Possibly remove explicit sign application in Path 1's Galois since signs cancel with twist
3. Or implement Path 1 using a different approach that matches the mathematical derivation

## Files Status

- ✅ `src/clifford_fhe_v2/backends/gpu_metal/hoisting.rs` - Core implementation complete
- ✅ `src/clifford_fhe_v2/backends/gpu_metal/ckks.rs` - Batch API fixed
- ✅ `tests/test_hoisting_sanity_check.rs` - Cyclic NTT test PASSES
- ❌ `tests/test_negacyclic_hoisting_sanity.rs` - Negacyclic test FAILS (validation issue)
- ❌ `tests/test_hoisted_rotation.rs` - Integration test FAILS (depends on sanity test)

## Mathematical Formula (Correct)

```
NTT_neg(σ_g a)[j] = D_g[j] · NTT_neg(a)[j·g mod N]
```

where:
- D_g[j] = ψ^{(g-1)j} in Montgomery domain
- Permutation: `out[j] = in[j·g mod N]` (PULL semantics)
- No sign tables needed (signs cancel via ψ^N = -1)

## Implementation Checklist (from Expert)

- [x] Hoisted digits are negacyclic NTT output (twist by ψ^i, then forward NTT)
- [x] Permutation uses PULL semantics: `out[j] = in[(j*g) & (N-1)]`
- [x] Diagonal D_g[j] built with rolling Montgomery power
- [x] Multiply diagonal in Montgomery domain
- [ ] **Negacyclic sanity test passes** ← Current blocker
- [ ] Plug into rotation and verify vs naive
- [ ] Benchmark speedup

## Performance Target

**Without hoisting:**
- 9 rotations × 0.25s = 2.25s

**With hoisting:**
- 1 × decompose+NTT = 0.13s
- 9 × (permute + diagonal + key-switch) = 9 × 0.08s = 0.72s
- **Total: 0.85s → 2.6× speedup** 🎯

## Next Session Tasks

1. Debug negacyclic sanity test - understand why Path 1 vs Path 2 differ
2. Consider simplified Path 1: Apply Galois as index permutation in the summation (no explicit coeff-domain Galois)
3. Once sanity test passes, run integration test
4. Benchmark and verify 2.6× speedup
5. Update V4 butterfly to use hoisted rotations

## References

- Halevi & Shoup 2014: "Algorithms in HElib" (original hoisting paper)
- Kim et al. 2018: "Bootstrapping for Approximate Homomorphic Encryption" (CKKS bootstrapping with hoisting)
- Expert explanation: Sign cancellation via ψ^N = -1 identity
