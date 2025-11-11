# Hoisting Bug Analysis

## Summary

The hoisting optimization for batch rotations has been partially implemented, but the results are incorrect. The hoisted rotations produce values that are millions off from the correct values.

## What Works ✅

1. **Cyclic NTT Sanity Check** - The test `test_hoisting_sanity_check` PASSES
   - Tests pure cyclic NTT (no negacyclic signs)
   - Verifies: `NTT(σ_g(a)) == Π_g(NTT(a))` for cyclic rings
   - Confirms the permutation formula is correct: `Π_g(j) = (j · g) mod N`
   - Confirms PULL semantics work: `out[j] = in[map[j]]`

2. **Fixed Structural Bug** - Moved hoisting outside the loop
   - **Before:** Hoisted c1 INSIDE the loop for each rotation step (defeating the purpose!)
   - **After:** Hoist c1 ONCE before the loop, then use permutation for each step
   - This is now structurally correct for batch hoisting

3. **Naive Rotation Works** - The non-hoisted `rotate_by_steps()` works correctly
   - Produces correct rotation results (e.g., rotate left by 1: `[10, 20, 30, ...]` → `[20, 30, 40, ...]`)
   - This confirms the Galois automorphism logic is correct

## What's Broken ❌

**Hoisted Rotation Results Are Garbage:**

Test output for step=1:
```
Naive result:   [10.0, 20.0, 30.0, 40.0, ...]  ✅ Correct
Hoisted result: [979427.4, 3390622.5, 11964095.6, ...]  ❌ Wrong by millions!
```

The hoisted results differ by **~10-20 million** from the correct values.

## Root Cause Analysis

### The Negacyclic Problem

CKKS uses **negacyclic** polynomial rings: `Z[X]/(X^N + 1)`, not cyclic rings `Z[X]/(X^N - 1)`.

**Key difference:** When applying Galois automorphism σ_g: X^i → X^{g·i}:
- In **cyclic** rings: Coefficients just permute
- In **negacyclic** rings: Coefficients permute **and some get negated**

Example for N=4, g=3 (which gives i·g mod 8):
- i=0: 0·3=0 < N → coeff[0] → coeff[0], sign = +1
- i=1: 1·3=3 < N → coeff[1] → coeff[3], sign = +1
- i=2: 2·3=6 ≥ N → coeff[2] → coeff[6-4=2], sign = -1 (because X^6 = X^2 · X^4 = -X^2)
- i=3: 3·3=9 ≥ N → coeff[3] → coeff[9-4=5-4=1], sign = -1

The `galois_signs` array encodes these sign flips: `[1, 1, -1, -1]`

### Current Implementation Gap

The current hoisting implementation (`rotate_with_hoisted_digits`):
1. ✅ Permutes NTT digits: `permute_in_place_ntt()`
2. ✅ Applies diagonal twist: `D_g[j] = ψ^{(g-1)·j}`
3. ❌ **MISSING:** Sign corrections from `galois_signs`

The diagonal twist D_g[j] = ψ^{(g-1)·j} accounts for the NTT structure, but **does NOT account for the negacyclic sign flips** that occur in the coefficient domain.

### Why Cyclic NTT Sanity Check Passes

The test `test_hoisting_sanity_check` uses `apply_galois_coefficient_cyclic()` which explicitly does **NOT** apply negacyclic signs:

```rust
fn apply_galois_coefficient_cyclic(poly: &[u64], ...) {
    // Pure cyclic: X^i -> X^{(g*i) mod N}, NO SIGNS
    let target_idx = (i * g) % n;
    result[target_idx * num_primes + prime_idx] = val;  // No sign flip!
}
```

This tests the **pure cyclic case**, which is why it passes. But CKKS needs the **negacyclic case**.

## Potential Solutions

### Option 1: Incorporate Signs into Diagonal Twist (Complex)

Modify `compute_diagonal_twist()` to incorporate both:
- The standard diagonal D_g[j] = ψ^{(g-1)·j}
- Additional factors for negacyclic sign flips

**Challenge:** The signs are per-coefficient in coefficient domain, but the diagonal is per-evaluation-point in NTT domain. The mapping is non-trivial.

### Option 2: Apply Signs in NTT Domain (Research Needed)

Research the correct formula for negacyclic hoisted automorphisms. The literature (Halevi & Shoup, Kim et al.) may have a different formula that inherently handles negacyclic.

**Key question:** Does the diagonal twist formula D_g[j] = ψ^{(g-1)·j} need modification for negacyclic rings?

### Option 3: Pre-apply Signs Before Hoisting (Incorrect)

Apply Galois automorphism to c1 before hoisting.

**Problem:** This defeats the purpose! We'd have different c1_rotated for each step, so we can't hoist once and reuse.

### Option 4: Post-apply Signs After Permutation (Investigate)

After permuting the NTT digits, apply sign corrections in the NTT domain.

**Challenge:** How do we apply coefficient-domain signs in the NTT (evaluation) domain?

## Next Steps

1. **Research negacyclic hoisted automorphisms** - Find the correct mathematical formula
2. **Create negacyclic sanity check** - Test that includes proper sign handling
3. **Fix diagonal twist or add sign correction** - Based on findings
4. **Verify with integration test** - Ensure hoisted == naive results

## Test Status

- ✅ `test_hoisting_sanity_check` - Cyclic NTT permutation (PASSES)
- ❌ `test_hoisted_rotation_correctness` - Full negacyclic hoisting (FAILS by 10M+)
- ✅ `test_geometric_operations_v4` - V4 operations without hoisting (PASSES)

## Files Involved

- `src/clifford_fhe_v2/backends/gpu_metal/hoisting.rs` - Core hoisting implementation
- `src/clifford_fhe_v2/backends/gpu_metal/ckks.rs` - `rotate_batch_with_hoisting()` method
- `src/clifford_fhe_v2/backends/gpu_metal/rotation.rs` - `compute_galois_map()` with signs
- `tests/test_hoisting_sanity_check.rs` - Cyclic NTT test (passes)
- `tests/test_hoisted_rotation.rs` - Negacyclic integration test (fails)
