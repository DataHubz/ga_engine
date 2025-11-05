# Help Request: Rotor Sandwich Product Implementation in Clifford Algebra Cl(n,0)

## Problem Summary

I'm implementing n-dimensional rotors in Clifford algebra Cl(n,0) with Euclidean signature for a lattice reduction library. The rotor sandwich product R·v·R† is not preserving vector norms as it should for unit rotors. I need help deriving or verifying the correct formula.

## Background

**Rotor Definition**: A rotor in Cl(n,0) is an element of the even subalgebra:
```
R = s + B
```
where:
- `s` is a scalar (grade 0)
- `B` is a bivector (grade 2): `B = Σ b_ij e_i∧e_j` for i < j

**Unit Rotor**: `||R||² = s² + ||B||² = 1` where `||B||² = Σ b_ij²`

**Reversal**: `R† = s - B`

**Goal**: Compute the sandwich product `v' = R·v·R†` which should be a rotation (norm-preserving transformation).

## Current Implementation

### Data Structure

```rust
pub struct RotorND {
    dimension: usize,
    coefficients: Vec<f64>, // [s, b_01, b_02, b_03, ..., b_12, b_13, ..., b_(n-1)(n)]
}
```

For dimension n:
- `coefficients[0]` = scalar part (s)
- `coefficients[1..]` = bivector components in lexicographic order

### Current Sandwich Product Code

```rust
fn apply_sandwich_direct(&self, v: &[f64]) -> Vec<f64] {
    let n = self.dimension;
    let s = self.coefficients[0];

    // Step 1: Compute R·v = (s + B)·v
    let mut rv_vector = vec![0.0; n];

    // Scalar times vector
    for i in 0..n {
        rv_vector[i] = s * v[i];
    }

    // Bivector times vector: B·v (keeping only vector part)
    let mut idx = 1;
    for i in 0..n {
        for j in (i + 1)..n {
            let b_ij = self.coefficients[idx];
            rv_vector[j] += b_ij * v[i];  // e_ij·e_i → e_j component
            rv_vector[i] -= b_ij * v[j];  // e_ij·e_j → e_i component
            idx += 1;
        }
    }

    // Step 2: Compute (R·v)·R† = (R·v)·(s - B)
    let mut result = vec![0.0; n];

    // Vector times scalar
    for i in 0..n {
        result[i] = s * rv_vector[i];
    }

    // Vector times bivector: rv_vector · (-B)
    idx = 1;
    for i in 0..n {
        for j in (i + 1)..n {
            let b_ij = self.coefficients[idx];
            result[j] -= b_ij * rv_vector[i];
            result[i] += b_ij * rv_vector[j];
            idx += 1;
        }
    }

    // ATTEMPTED FIX: Add ||B||² * v term
    let b_norm_sq: f64 = self.coefficients[1..].iter().map(|x| x * x).sum();
    for i in 0..n {
        result[i] += b_norm_sq * v[i];
    }

    result
}
```

## Test Cases

### Test Case 1: Simple 2D Rotation (PASSES)
```rust
// Rotate [1,0,0,0] by 90° in xy-plane
// R = cos(45°) + sin(45°) e_01 = 0.707 + 0.707 e_01

let s = 0.7071067811865476;
let b_01 = 0.7071067811865476;
let v = [1.0, 0.0, 0.0, 0.0];

// Expected: [0.0, 1.0, 0.0, 0.0]
// Actual: [0.0, 1.0, 0.0, 0.0] ✓
// Norm preserved: 1.0 → 1.0 ✓
```

### Test Case 2: 4D Vector in 2D Rotation Plane (FAILS)
```rust
// Same rotor, but vector has components in all dimensions
let s = 0.7071067811865476;
let b_01 = -0.7071067811865476; // From from_vectors([1,0,0,0], [0,1,0,0])
let v = [1.0, 2.0, 3.0, 4.0];

// Expected: Components in xy-plane rotate, z and w stay unchanged
// Expected norm: 5.477 (preserved)

// Actual WITHOUT ||B||² term:
// Output: [1.0, 2.0, 1.5, 2.0]
// z changed: 3.0 → 1.5 (scaled by s² = 0.5) ✗
// w changed: 4.0 → 2.0 (scaled by s² = 0.5) ✗
// Norm: 3.354 ✗ NOT PRESERVED

// Actual WITH ||B||² term:
// Output: [1.5, 3.0, 3.0, 4.0]
// z: 3.0 → 3.0 ✓ (correct!)
// w: 4.0 → 4.0 ✓ (correct!)
// But x,y are wrong: [1.0, 2.0] → [1.5, 3.0] (scaled, not rotated) ✗
// Norm: 6.021 ✗ NOT PRESERVED
```

## The Problem

Components of the vector that are **orthogonal to the rotation plane** (e.g., z and w when rotating in xy-plane) are getting scaled by `s²` instead of being preserved.

- Without the `||B||²·v` correction: orthogonal components scaled by `s² = 0.5`
- With the `||B||²·v` correction: orthogonal components preserved, but components IN the rotation plane are also affected, breaking the rotation

## Questions for Expert

1. **What is the correct formula** for computing `R·v·R†` where `R = s + B` in Cl(n,0)?

2. **Should I be tracking trivector terms?** When computing `(s + B)·v`, I get trivector terms (grade 3) which I've been discarding. Do these feed back into the final result when multiplied by `R† = s - B`?

3. **Is there a simpler formula** that avoids computing the full geometric product? For example:
   - `v' = (s² - ||B||²)v + 2s(B·v) + 2(B·(B·v))` (some sources)
   - Or Rodrigues-like formula using only s, B, and v?

4. **Sign conventions**: I empirically determined that for `e_ij·e_k`:
   ```
   e_ij·e_i → +e_j (contributes to j-component)
   e_ij·e_j → -e_i (contributes to i-component)
   ```
   Is this correct for Cl(n,0) with geometric product?

5. **Reference implementation**: Is there a standard reference for rotor sandwich products in arbitrary dimensions that I can follow?

## What I've Tried

1. ✗ Formula: `v' = (s² - ||B||²)v + 2s(B·v) + 2(B·(B·v))` - doesn't preserve norm
2. ✗ Formula: `v' = (s² + ||B||²)v + 2s(B·v) - 2(B·(B·v))` - doesn't preserve norm
3. ✗ Step-by-step sandwich product keeping only vector parts - scales orthogonal components by s²
4. ✗ Adding `||B||²·v` correction term - fixes orthogonal components but breaks in-plane rotation
5. ✓ Tested exhaustive sign combinations on simple case - found working signs for 2D rotation in 4D space
6. ✗ Same signs fail when vector has components in all dimensions

## Expected Behavior

For a unit rotor `R` (with `||R||² = 1`), the sandwich product `R·v·R†` should:
- **Preserve norm**: `||R·v·R†|| = ||v||` for all vectors v
- **Rotate** components in the plane spanned by the bivector B
- **Leave unchanged** components orthogonal to all bivectors in B

## Code Context

- Language: Rust
- Precision: f64
- Application: Lattice reduction for cryptanalysis (GA-accelerated projections)
- Full code: See `src/lattice_reduction/rotor_nd.rs` lines 169-230

## Additional Information

### How the Rotor is Created

```rust
pub fn from_vectors(a: &[f64], b: &[f64]) -> Self {
    // Creates rotor that rotates vector a toward vector b
    // Uses formula: R = (1 + b·a) + (b ∧ a)
    // Normalized to unit rotor
}
```

For `from_vectors([1,0,0,0], [0,1,0,0])`:
- Creates rotor for 90° rotation in xy-plane
- Coefficients: `[0.707, -0.707, 0, 0, 0, 0, 0]`
- Note the NEGATIVE b_01 coefficient

### Metric

Using Euclidean metric: `e_i·e_j = δ_ij` (1 if i=j, 0 otherwise)

Vectors anticommute: `e_i·e_j = -e_j·e_i` for i ≠ j

### Test Command

```bash
cargo test --lib lattice_reduction::rotor_nd::tests::test_apply_preserves_norm -- --nocapture
```

## What I Need

The **correct, complete formula** for computing `R·v·R†` in Cl(n,0) that:
1. Works for any dimension n
2. Works for any vector v (not just vectors in the rotation plane)
3. Preserves norms for unit rotors
4. Can be implemented with explicit loops over bivector components

OR

A clear explanation of **which terms I'm missing** in my current implementation and how to compute them.

## References I've Consulted

- Doran & Lasenby, "Geometric Algebra for Physicists" (2003)
- Dorst, Fontijne, Mann, "Geometric Algebra for Computer Science" (2007)
- Various online GA resources

But I'm struggling to translate the general formulas to explicit code for the n-dimensional case with multiple bivector components.

---

**Thank you for any help!** This is blocking a critical component of our lattice reduction implementation for an FSE 2026 submission.
