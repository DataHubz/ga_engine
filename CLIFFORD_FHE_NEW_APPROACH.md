# New Approach: Polynomial-Level Geometric Product Encoding

**Date**: November 1, 2025
**Status**: Design Phase

## The Problem with Component Extraction

### What Went Wrong

**Attempt 1**: Polynomial multiplication with selector
```rust
// selector = [0, 0, 1, 0, 0, 0, 0, 0, ...]
ct_component = multiply_by_plaintext(ct, selector)
```
**Result**: ❌ Rotates coefficients instead of selecting them (negacyclic NTT behavior)

**Attempt 2**: Direct coefficient masking
```rust
c0_masked[i] = ct.c0[i]  // Keep only position i
c1_masked[i] = ct.c1[i]  // Keep only position i
```
**Result**: ❌ Breaks CKKS decryption structure (decrypt = c0 + c1·s requires relationship between all coefficients)

### Root Cause

CKKS encrypts into polynomial ring R = Z[x]/(x^N + 1). The ciphertext (c0, c1) satisfies:
```
decrypt(ct) = c0(x) + c1(x) · s(x)  (mod x^N + 1)
```

This means:
1. Each coefficient position depends on polynomial evaluation, not independent storage
2. Masking individual coefficients destroys the polynomial structure
3. Component extraction requires rotation keys and SIMD slot permutations (not yet implemented)

## The New Approach: Structure-Constant Polynomials

### Core Idea

Instead of:
1. ❌ Extract component i from ct_a
2. ❌ Extract component j from ct_b
3. ❌ Multiply
4. ❌ Accumulate into result component k

Do:
1. ✅ Create a "projection polynomial" P_{ij→k}(x) that encodes the GP rule
2. ✅ Apply this polynomial operation to full ciphertexts
3. ✅ Accumulate results using homomorphic addition

### Mathematical Foundation

For Clifford algebra Cl(3,0), the geometric product is:
```
(a₀ + a₁e₁ + ... + a₇e₁₂₃) ⊗ (b₀ + b₁e₁ + ... + b₇e₁₂₃)
= Σᵢ Σⱼ aᵢbⱼ (eᵢ ⊗ eⱼ)
```

Each basis blade product eᵢ ⊗ eⱼ = ±eₖ is encoded in structure constants.

### Encoding in CKKS

**Current encoding**: Multivector → polynomial
```
[a₀, a₁, a₂, ..., a₇] → polynomial with coefficients at positions 0-7
```

**Key insight**: We need to implement geometric product at the polynomial level!

### Strategy 1: Polynomial Projection (Initial Attempt)

For each structure constant rule (i, j) → k with coefficient c:

```rust
// Create a polynomial that "projects out" positions i and j
// and places result at position k

// Example: e₁ ⊗ e₂ = e₁₂ (positions 1, 2 → position 4)
// We need: ct_a[1] * ct_b[2] → result[4]
```

**Challenge**: How to express "multiply coefficient i of ct_a with coefficient j of ct_b" as a polynomial operation?

## Strategy 2: Explicit Polynomial Multiplication with Rearrangement

### Approach

1. **Full polynomial multiplication**:
   ```
   ct_result = ct_a ⊗ ct_b  (full ciphertext-ciphertext multiplication)
   ```
   This gives us a degree-2 ciphertext with ALL cross-terms aᵢ·bⱼ

2. **Rearrangement using structure constants**:
   Create a linear transformation that rearranges the cross-terms according to GP rules:
   ```
   result[k] = Σ_{(i,j)→k} c_{ij} · (aᵢ·bⱼ)
   ```

### Implementation Sketch

```rust
pub fn geometric_product_homomorphic(
    ct_a: &Ciphertext,
    ct_b: &Ciphertext,
    evk: &EvaluationKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    // Step 1: Full polynomial multiplication (gives all aᵢ·bⱼ terms)
    // This uses standard CKKS multiplication
    let ct_product = multiply(ct_a, ct_b, evk, params);

    // Step 2: Rearrange terms according to structure constants
    // This is the KEY innovation!
    let ct_rearranged = rearrange_by_structure_constants(&ct_product, params);

    ct_rearranged
}
```

### The Rearrangement Challenge

After `ct_a ⊗ ct_b`, we have a polynomial where position `i*8 + j` contains `aᵢ·bⱼ`.

We need to:
1. Map these positions to the correct output positions according to GP rules
2. Apply signs (+1 or -1) from structure constants
3. Accumulate terms that contribute to the same output component

**Problem**: This requires coefficient rearrangement, which brings us back to the extraction problem!

## Strategy 3: Duplicate Encoding (SIMD-like)

### Idea

Instead of encoding 8 components in 8 coefficient positions, replicate each component across multiple positions to enable polynomial operations.

**Example encoding**:
```
a₀ at positions: 0, 8, 16, 24, ...
a₁ at positions: 1, 9, 17, 25, ...
...
```

Then polynomial multiplication with appropriate masks can select the right components.

**Challenge**: Requires N ≥ 64 positions just for one multivector (wasteful)

## Strategy 4: Batched Computation (Most Promising!)

### Core Insight

What if we DON'T try to compute the full GP in one operation?

Instead, compute each OUTPUT component separately:

```rust
pub fn geometric_product_homomorphic(
    ct_a: &Ciphertext,
    ct_b: &Ciphertext,
    evk: &EvaluationKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    let structure = StructureConstants::new_cl30();

    // For each output component k:
    let mut result_polys: Vec<Vec<i64>> = vec![vec![0; params.n]; 8];

    for k in 0..8 {
        // Get all (i,j) pairs that contribute to component k
        let products = structure.get_products_for(k);

        for &(coeff, _k, i, j) in products {
            // Compute ct_a[position i] * ct_b[position j]
            // This is the KEY operation we need to implement!
            let term = compute_component_product(ct_a, i, ct_b, j, evk, params);

            // Apply sign
            let scaled = multiply_by_scalar(&term, coeff as i64, params);

            // Accumulate into result[k]
            result_polys[k] = add_polynomials(&result_polys[k], &scaled.c0, params.modulus_at_level(0));
        }
    }

    // Pack the 8 result polynomials into one ciphertext
    pack_result_components(&result_polys, params)
}
```

**The Key Function**: `compute_component_product(ct_a, i, ct_b, j, ...)`

This needs to compute the product of coefficient i from ct_a with coefficient j from ct_b.

### Possible Implementation of `compute_component_product`

**Observation**: While we can't EXTRACT a single component, we CAN use polynomial multiplication to ISOLATE the cross-term we want!

```rust
fn compute_component_product(
    ct_a: &Ciphertext,
    i: usize,
    ct_b: &Ciphertext,
    j: usize,
    evk: &EvaluationKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    // Create selection polynomials
    let mut select_i = vec![0i64; params.n];
    select_i[i] = 1;  // Polynomial that is 1 at position i, 0 elsewhere

    let mut select_j = vec![0i64; params.n];
    select_j[j] = 1;  // Polynomial that is 1 at position j, 0 elsewhere

    // Multiply ct_a by select_i polynomial (this is plaintext-ciphertext mult)
    let pt_i = Plaintext::new(select_i, params.scale);
    let ct_a_selected = multiply_by_plaintext(ct_a, &pt_i, params);

    // Multiply ct_b by select_j polynomial
    let pt_j = Plaintext::new(select_j, params.scale);
    let ct_b_selected = multiply_by_plaintext(ct_b, &pt_j, params);

    // Now multiply the two selected ciphertexts
    let result = multiply(&ct_a_selected, &ct_b_selected, evk, params);

    // The result will have the product at position (i+j) mod N
    // due to polynomial multiplication in R[x]/(x^N + 1)

    result
}
```

**Wait!** This brings us back to the rotation problem. When we multiply polynomials with single coefficients, we get rotation, not the product at a specific position.

## Strategy 5: Accept Rotation, Then Rotate Back

### The Breakthrough Idea

Polynomial multiplication DOES preserve information - it just moves it to a different position!

When we multiply:
- Polynomial with coefficient at position i
- Polynomial with coefficient at position j

We get a result with the product at position (i+j) mod N (with sign from negacyclic reduction if i+j ≥ N).

**Solution**: Use rotation keys to move it back!

```rust
fn compute_component_product_with_rotation(
    ct_a: &Ciphertext,
    i: usize,
    ct_b: &Ciphertext,
    j: usize,
    evk: &EvaluationKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    // Select and multiply (product ends up at position i+j)
    let ct_product = compute_component_product(ct_a, i, ct_b, j, evk, params);

    // Rotate to move product from position (i+j) to position k
    // where k is the target component from structure constants
    let target_position = structure.get_target(i, j);
    let rotation_amount = target_position - (i + j);

    rotate_ciphertext(&ct_product, rotation_amount, evk, params)
}
```

**Problem**: We haven't implemented rotation keys yet! (That's Phase 3)

## Decision: Path Forward

### Immediate Implementation (Phase 2 completion)

**Option A**: Implement rotation keys early (move Phase 3 → Phase 2.5)
- Required for proper GP implementation
- Well-understood in CKKS literature
- ~200 lines of code

**Option B**: Simplified GP without full correctness
- Accept that positions rotate
- Document as "proof of concept"
- Return to fix with rotation keys later

### Recommendation: **Option A** (Implement Rotation Keys)

**Rationale**:
1. Rotation is fundamental to CKKS SIMD operations
2. We'll need it anyway for GA rotations (Phase 3)
3. Doing it now unblocks geometric product
4. Better to do it right than ship broken code

**Implementation plan**:
1. Add rotation key generation to `keys.rs`
2. Implement `rotate_left` and `rotate_right` operations in `ckks.rs`
3. Use rotation to implement `compute_component_product` correctly
4. Complete geometric product using Strategy 4 + rotation

## Summary

**Failed approaches**:
- ❌ Component extraction via polynomial multiplication (rotates, doesn't select)
- ❌ Component extraction via coefficient masking (breaks CKKS structure)

**Promising approach**:
- ✅ Strategy 4 (batched computation) + Strategy 5 (rotation)
- Compute each output component separately
- Use rotation keys to handle position offsets from polynomial multiplication
- Requires implementing rotation keys (move Phase 3 early)

**Next steps**:
1. Implement rotation key generation
2. Implement rotation operations
3. Implement `compute_component_product` with rotation
4. Complete geometric product
5. Test with (1 + 2e₁) ⊗ (3 + 4e₂) = 3 + 6e₁ + 4e₂ + 8e₁₂

**Timeline**:
- Rotation keys: 2-3 hours
- GP implementation: 1-2 hours
- Testing: 1 hour

**Total**: ~4-6 hours to complete Phase 2 correctly
