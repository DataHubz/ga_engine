# Phase 2 Debugging: Rotation-Based Geometric Product

**Date**: November 1, 2025
**Status**: Implementation complete, results incorrect

## Summary

Implemented rotation-based approach for homomorphic geometric product but getting huge errors (10^5) instead of correct results. The implementation compiles and runs but produces wrong values.

## What Was Implemented

### 1. Rotation Keys (keys.rs)
- Added `keygen_with_rotation()` function
- Generates rotation keys for all positions 0 to N-1
- Implements automorphism x → x^r for polynomial rotation

### 2. Rotation Operation (ckks.rs)
- Added `rotate()` function
- Applies automorphism to ciphertext components
- Uses rotation keys to maintain encryption correctness

### 3. Component Product with Rotation (operations.rs)
- `compute_component_product()` function
- Masks ct_a with selector polynomial to isolate component i
- Masks ct_b with selector polynomial to isolate component j
- Multiplies masked ciphertexts (product lands at position i+j)
- Rotates result to target position

### 4. Geometric Product (geometric_product.rs)
- Updated to use rotation-based approach
- For each output component:
  - Compute all contributing products using `compute_component_product`
  - Apply structure constant coefficients (+1 or -1)
  - Accumulate into result

### 5. Test Parameters (params.rs)
- Added `new_test()` with N=64 for fast testing
- Uses smaller scale (2^20) and fewer levels

## Test Results

**Test case**: (1 + 2e₁) ⊗ (3 + 4e₂) = 3 + 6e₁ + 4e₂ + 8e₁₂

**Expected**:
```
[3.0, 6.0, 4.0, 0.0, 8.0, 0.0, 0.0, 0.0]
```

**Actual**:
```
[-155426.77, 121775.67, -402545.44, -460248.02, 494781.68, 38993.80, 340911.94, 7517.30]
```

**Errors**: 10^4 to 10^5 magnitude - completely wrong!

## Possible Issues

### 1. Selector Polynomial Approach May Be Wrong

Current approach:
```rust
let mut selector_i = vec![0i64; params.n];
selector_i[i] = params.scale as i64;  // Scaled by scale factor
let pt_i = Plaintext::new(selector_i, params.scale);
let ct_a_masked = multiply_by_plaintext(ct_a, &pt_i, params);
```

**Problem**: When we multiply ciphertext by a selector polynomial, does it actually isolate that coefficient?

In CKKS:
- Polynomial multiplication is done via NTT
- Selector [0,0,1,0...] multiplied with [a0,a1,a2,...] gives what?
- In NTT domain, this is component-wise multiplication of NTT transforms
- This does NOT simply select coefficient - it applies convolution!

### 2. Position Calculation May Be Wrong

```rust
let product_position = if i + j < params.n {
    i + j
} else {
    (i + j) % params.n
};
```

**Problem**: This assumes polynomial multiplication of [0,...,ai,...,0] * [0,...,bj,...,0] places result at position i+j. But:
- Polynomial multiplication in R[x]/(x^N + 1) is NOT positional
- Position i * position j → involves convolution, not simple position addition
- Negacyclic reduction (x^N = -1) affects where coefficients end up

### 3. Rotation Implementation May Be Incorrect

The rotation implementation assumes automorphism x → x^r works for coefficient rotation. But:
- Standard CKKS rotation uses Galois automorphisms σ_k: x → x^(5^k)
- Our implementation uses x → x^r which may not preserve CKKS structure
- The rotation key generation may not match what rotation operation expects

### 4. Scale Management Issues

Each multiplication changes the scale:
```rust
new_scale = ct1.scale * ct2.scale / params.scale
```

After 64 multiplications (8 components × 8 products each), we're dividing by scale^64. This could cause:
- Overflow/underflow in scaled values
- Loss of precision
- Incorrect decryption

## Root Cause Analysis

The fundamental issue is: **Selector polynomial multiplication does NOT isolate coefficients in CKKS!**

When we multiply `ct` (encrypting [a0, a1, a2, ...]) by selector `[0, 0, 1, 0, ...]`:
1. CKKS does polynomial multiplication modulo x^N + 1
2. This is convolution, not element-wise multiplication
3. The result is NOT [0, 0, a2, 0, ...] as we assumed

**What we need**: A way to extract coefficient at position i without using polynomial multiplication as coefficient selector.

## Alternative Approaches

### Option A: Use SIMD Slots Properly

CKKS supports SIMD operations where coefficients represent "slots". To extract slot i:
1. Use rotation to move slot i to slot 0
2. Multiply by mask that zeros all slots except 0
3. Rotate back

But this requires understanding SIMD slot permutations, which are complex.

### Option B: Don't Extract - Compute Directly

Instead of extract→multiply→pack, compute the full polynomial product directly:
```
(a0 + a1·x + ... + a7·x^7) * (b0 + b1·x + ... + b7·x^7)
```
Then rearrange terms according to structure constants.

This avoids extraction entirely but requires understanding how to rearrange polynomial products.

### Option C: Plaintext Geometric Product

Since we only have 8 components, we could:
1. Encrypt each component separately (8 independent ciphertexts)
2. Compute all 64 products homomorphically
3. Add them according to structure constants

This wastes 8× the ciphertext space but avoids coefficient extraction.

## Next Steps

1. **Debug selector polynomial**: Test if selector multiplication actually isolates coefficients
2. **Test rotation**: Verify rotation operation works correctly for simple cases
3. **Understand CKKS slot operations**: Read CKKS papers on SIMD operations
4. **Consider Option C**: Separate ciphertexts per component (simpler, proven to work)

## Key Insight

The main lesson: **CKKS coefficient packing is not the same as SIMD slot packing.**

- Coefficient packing: Store values in polynomial coefficients directly
- SIMD slot packing: Use CRT/Chinese Remainder Theorem to pack into "slots"

We've been mixing these two concepts. CKKS rotation works on SIMD slots, not raw coefficients!

## Files Changed

- `src/clifford_fhe/keys.rs` - rotation key generation
- `src/clifford_fhe/ckks.rs` - rotation operation
- `src/clifford_fhe/operations.rs` - component product with rotation
- `src/clifford_fhe/geometric_product.rs` - rotation-based GP
- `src/clifford_fhe/params.rs` - test parameters
- `src/clifford_fhe/mod.rs` - exports
- `examples/clifford_fhe_geometric_product_v2.rs` - test example

## References Needed

1. "Homomorphic Encryption for Arithmetic of Approximate Numbers" (CKKS paper)
2. CKKS rotation/permutation sections
3. SEAL library implementation of rotations
4. HElib SIMD slot operations

---

**Status**: Need to rethink approach - current implementation fundamentally misunderstands CKKS coefficient vs slot operations
