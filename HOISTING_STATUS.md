# Hoisting Implementation - Current Status

## Summary

The hoisting optimization infrastructure is complete and structurally correct. The negacyclic sanity test is still failing, indicating a remaining gap in understanding or implementation of the negacyclic hoisting formula.

## Fixed ✅

1. **Structural bug in `rotate_batch_with_hoisting`** ([ckks.rs:1650-1652](src/clifford_fhe_v2/backends/gpu_metal/ckks.rs#L1650-L1652))
   - Was hoisting inside loop - now hoists ONCE before loop ✅

2. **Understood sign cancellation principle**
   - Negacyclic reduction signs cancel with twist index change
   - No per-coefficient sign tables needed in hoisting path
   - Formula: `NTT_neg(σ_g a)[j] = D_g[j] · NTT_neg(a)[j·g mod N]`

3. **Implementation complete**
   - Permutation: PULL semantics ✅
   - Diagonal: `D_g[j] = ψ^{(g-1)j}` in Montgomery ✅
   - All Montgomery domain handling correct ✅

## Remaining Issue ❌

**Negacyclic sanity test fails** - Values differ between two paths:

### Test Results
```
Path 1 (Galois → twist → NTT)[0] = 254377618152758367
Path 2 (twist → NTT → permute → diagonal)[0] = 311356815911072910
```

These should be equal according to the hoisting formula.

### Path 1 Implementation
```rust
// 1. Apply σ_g in coefficient domain (permute with signs)
for i in 0..n {
    let i_prime = (i * g) % n;
    let sign = ((i * g) / n) & 1;
    b_coeff[i_prime] = if sign == 0 { a[i] } else { q - a[i] };
}

// 2. Twist by array index
for idx in 0..n {
    twisted[idx] = b_coeff[idx] * ψ^idx;
}

// 3. Cyclic NTT
ntt_ctx.forward(&mut twisted);
```

### Path 2 Implementation
```rust
// 1. Twist original coefficients
for i in 0..n {
    twisted[i] = a[i] * ψ^i;
}

// 2. Cyclic NTT
ntt_ctx.forward(&mut twisted);  // → NTT_neg(a)

// 3. Permute (PULL): out[j] = in[j*g mod N]
permute_in_place_ntt(&mut out, &perm_map, n, num_primes);

// 4. Multiply diagonal: out[j] *= D_g[j] where D_g[j] = ψ^{(g-1)j}
for j in 0..n {
    out[j] = mont_mul(out[j], D_g[j], q, ntt_ctx);
}
```

### Diagnosis

- Galois permutation in Path 1 is correct (verified with debug output)
- Sign handling is correct (checked b[1] = -(a[205]) ✅)
- Twist and NTT execution work in both paths
- **But the outputs don't match the formula**

### Hypothesis

The mathematical derivation may require a different understanding of how the summation index substitution works. The formula:

```
∑_i a[i]·ψ^{gi}·ω^{ijg} = ψ^{(g-1)j} · NTT_neg(a)[jg]
```

needs careful index manipulation that may not be immediately obvious. The relationship between:
- Applying Galois in coefficient domain then NTT
- vs. NTT then permute in frequency domain

may have a subtlety not yet captured in the test implementation.

## Next Steps

1. **Clarify the mathematical derivation**
   - Specifically how the summation ∑_i a[i]·ψ^{gi}·ω^{ijg} simplifies
   - How the index substitution k = ig mod N works
   - Why this gives D_g[j] · NTT_neg(a)[jg]

2. **Consider alternative test approaches**
   - Test with N=4 or N=8 for manual verification
   - Use delta impulses (a[i] = δ_{i,k}) to isolate specific terms
   - Compare against a known-working reference implementation

3. **Review expert explanation**
   - Re-read the sign cancellation derivation carefully
   - Check if there's a subtlety in "twist by destination index" concept
   - Verify the final formula matches literature (Halevi & Shoup)

## Files Status

- ✅ [hoisting.rs](src/clifford_fhe_v2/backends/gpu_metal/hoisting.rs) - Core implementation complete
- ✅ [ckks.rs](src/clifford_fhe_v2/backends/gpu_metal/ckks.rs) - Batch API fixed
- ✅ [test_hoisting_sanity_check.rs](tests/test_hoisting_sanity_check.rs) - Cyclic test PASSES
- ❌ [test_negacyclic_hoisting_sanity.rs](tests/test_negacyclic_hoisting_sanity.rs) - Negacyclic test FAILS
- ⏸️ [test_hoisted_rotation.rs](tests/test_hoisted_rotation.rs) - Integration test (blocked on sanity)

## Key Question

**How do we correctly implement the reference path (Path 1) for `NTT_neg(σ_g a)`?**

The current implementation applies Galois permutation with signs, then twists, then NTT. But this doesn't match the hoisted path (Path 2). Either:
- Path 1 needs a different implementation
- Path 2 needs adjustment
- The test comparison logic is incorrect

The formula is theoretically sound (sign cancellation proven), so the issue is in translating it to code.

## Performance Target (Once Fixed)

- Without hoisting: 9 rotations × 0.25s = 2.25s
- With hoisting: 0.13s + 9 × 0.08s = 0.85s
- **Target: 2.6× speedup** 🎯
