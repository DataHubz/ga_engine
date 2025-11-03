# NTT Bug Report: 4+ Prime Multiplication Failure in RNS-CKKS

## Executive Summary

Multiplication in RNS-CKKS works correctly with 3 primes but **fails completely with 4+ primes**. Through systematic debugging, we isolated the root cause to a bug in the NTT (Number Theoretic Transform) implementation that affects specific primes, specifically **q = 1099511693313** (the 4th prime in our modulus chain).

## Problem Description

### Symptoms
- ✅ **3 primes**: Multiplication works perfectly (error < 0.001)
- ❌ **4 primes**: Multiplication produces garbage (error ~50,000 for 2×3=6)
- ❌ **5 primes**: Multiplication produces garbage (error ~190,000 for 2×3=6)

### Impact
Cannot use depth-2+ circuits (wedge product, inner product, rotations, etc.) because they require 4+ primes.

## Root Cause Investigation

### Step 1: Eliminated False Leads

1. **EVK Digit Count Mismatch** ✅ FIXED
   - EVK was generated with wrong number of digits
   - Fixed by using `Q.bits()` instead of sum of individual prime bits
   - This reduced error but didn't fix the core issue

2. **Scale Parameter** ✅ FIXED
   - Scale cannot equal any moduli prime (causes zero residues)
   - Changed scale from q₁ back to 2^40

3. **CRT Reconstruction Overflow** ✅ NOT THE ISSUE
   - `to_coeffs()` overflows with large Q (Q > i64_MAX)
   - But this is in decoding, not the cause of multiplication failure

4. **Gadget Decomposition** ✅ WORKS CORRECTLY
   - Verified all digits are CRT-consistent

5. **Secret Key Generation** ✅ WORKS CORRECTLY
   - Secret key is CRT-consistent

### Step 2: Identified CRT Inconsistency

Testing revealed that **s² (secret key squared) has CRT-inconsistent residues**:

**With 4 primes:**
```
coeff[0]: [1141392289560813561, 1099511678969, 1099511683065, 870019386691]
                                                                  ^^^^^^^^^^^
                                                                  WRONG!

coeff[2]: [23, 23, 23, 262598929341]
                       ^^^^^^^^^^^^
                       WRONG!
```

**Pattern:** Primes 0, 1, 2 agree perfectly, but **prime[3] gives completely different values**.

**Result:** ALL 1024 coefficients of s² are CRT-inconsistent (0% consistency).

**Removing prime[3]:** CRT consistency improves from 0% to 51.2%.

### Step 3: Confirmed Prime[3] NTT is Broken

Direct NTT test for q = 1099511693313:

```rust
// Test: 1 × 1 should give 1
let a = [1, 0, 0, ...];  // polynomial "1"
let b = [1, 0, 0, ...];  // polynomial "1"
let result = polynomial_multiply_ntt(&a, &b, q, n);

// Expected: [1, 0, 0, ...]
// Actual:   [454788377539, ?, ?, ...]  ❌ COMPLETELY WRONG
```

**Even the simplest possible NTT test fails for this prime!**

### Step 4: Investigated Primitive Root Finding

The primitive root algorithm was initially flawed (only checked phi/2 and phi/odd_part).

**Fixed** by implementing proper primitive root test:
```rust
// g is primitive root iff g^(phi/p) ≠ 1 for ALL prime factors p of phi
```

**Prime factorization of phi = q-1 = 1099511693312:**
```
phi = 2^18 × 97 × 257 × 673
```

**Verified g=2 IS a valid primitive root:**
```
g^(phi/2)   = 34800621601   ≠ 1 ✅
g^(phi/97)  = 834416120866  ≠ 1 ✅
g^(phi/257) = 699622560580  ≠ 1 ✅
g^(phi/673) = 225324903721  ≠ 1 ✅
```

So the primitive root finding is correct!

## Current Status: NTT Algorithm Bug

**The bug is NOT in:**
- ✅ Primitive root finding (fixed and verified)
- ✅ Prime selection (all primes are NTT-friendly: q ≡ 1 mod 2N)
- ✅ Overall RNS-CKKS structure

**The bug IS in:**
- ❌ NTT computation for large primes (specifically q = 1099511693313)

## Technical Details

### Primes Used
```rust
q₀ = 1141392289560813569  // 60-bit, works ✅
q₁ = 1099511678977        // 41-bit, works ✅
q₂ = 1099511683073        // 41-bit, works ✅
q₃ = 1099511693313        // 41-bit, BROKEN ❌
q₄ = 1099511697409        // 41-bit, not tested yet
```

All satisfy q ≡ 1 (mod 2048) for N=1024.

### NTT Implementation (Simplified)

```rust
pub fn polynomial_multiply_ntt(a: &[i64], b: &[i64], q_i64: i64, n: usize) -> Vec<i64> {
    let q = q_i64 as u64;

    // Convert to u64 in [0, q)
    let au: Vec<u64> = a.iter().map(|&x| ((x % q_i64) + q_i64) as u64).collect();
    let bu: Vec<u64> = b.iter().map(|&x| ((x % q_i64) + q_i64) as u64).collect();

    // Get twisted NTT roots
    let (psi, omega) = negacyclic_roots(q, n);

    // Forward NTT
    let a_ntt = negacyclic_ntt(au, q, psi, omega);
    let b_ntt = negacyclic_ntt(bu, q, psi, omega);

    // Pointwise multiply
    let c_ntt: Vec<u64> = (0..n).map(|i| mod_mul_u64(a_ntt[i], b_ntt[i], q)).collect();

    // Inverse NTT
    let c = negacyclic_intt(c_ntt, q, psi, omega);

    // Convert back to i64
    c.into_iter().map(|v| v as i64).collect()
}

fn negacyclic_roots(q: u64, n: usize) -> (u64, u64) {
    let g = primitive_root(q);  // Now correctly finds g=2
    let two_n = 2u64 * (n as u64);
    let exp = (q - 1) / two_n;
    let psi = mod_pow_u64(g, exp, q);      // 2N-th root of unity
    let omega = mod_mul_u64(psi, psi, q);  // N-th root: omega = psi^2
    (psi, omega)
}
```

### Arithmetic Operations

```rust
fn mod_mul_u64(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

fn mod_pow_u64(mut base: u64, mut exp: u64, q: u64) -> u64 {
    let mut acc = 1u64;
    while exp > 0 {
        if (exp & 1) == 1 {
            acc = mod_mul_u64(acc, base, q);
        }
        base = mod_mul_u64(base, base, q);
        exp >>= 1;
    }
    acc
}
```

**Note:** All operations use u128 for intermediate results to avoid overflow, so arithmetic should be safe for q ≈ 10^12.

## Questions for Expert Review

### 1. Is there a known issue with NTT for specific prime sizes?

The broken prime q₃ = 1099511693313 is approximately 2^40. Could there be:
- Issues with root computation for primes of this size?
- Numeric stability problems in the butterfly operations?
- Off-by-one errors in bit-reversal indexing?

### 2. Are the NTT roots correct?

For q = 1099511693313 with N = 1024:
- g = 2 (primitive root) ✅ Verified
- psi should be a primitive 2N-th root: psi^(2N) = 1, psi^N = -1
- omega = psi^2 should be a primitive N-th root: omega^N = 1

Could you verify the computation:
```
phi = 1099511693312 = 2^18 × 97 × 257 × 673
exp = phi / (2N) = 1099511693312 / 2048 = 536870944
psi = 2^536870944 mod 1099511693313 = ?
```

### 3. Could this be a Cooley-Tukey implementation bug?

The NTT uses standard Cooley-Tukey FFT with bit-reversal. Could there be:
- Incorrect twiddle factor computation?
- Issues with the negacyclic twisting (multiply by psi^i before NTT)?
- Problems in the inverse NTT (especially the n^{-1} scaling)?

### 4. Why does it work for q₀, q₁, q₂ but not q₃?

All primes satisfy:
- q ≡ 1 (mod 2048) ✅
- q is prime ✅
- Have primitive roots ✅

**Size comparison:**
- q₀ = 1.14 × 10^18 (60-bit) - works
- q₁ = 1.10 × 10^12 (41-bit) - works
- q₂ = 1.10 × 10^12 (41-bit) - works
- q₃ = 1.10 × 10^12 (41-bit) - BROKEN

All the 41-bit primes are essentially the same size (~51k different), so why does q₃ fail while q₁ and q₂ work?

**Hypothesis:** Could there be something special about q₃'s specific value or its factorization that breaks the NTT?

### 5. Test suggestion?

Could you suggest a minimal test case that would isolate the issue? For example:
- Specific coefficients that should work but don't?
- Manual computation of one butterfly operation to compare?
- Verification of psi/omega properties?

## Code Location

**Main NTT implementation:**
- File: `src/clifford_fhe/ckks_rns.rs`
- Function: `polynomial_multiply_ntt` (line 171)
- Helper functions: `negacyclic_ntt`, `negacyclic_intt`, `negacyclic_roots`
- Primitive root: `primitive_root` (line 48, recently fixed)

**Test demonstrating the bug:**
- File: `tests/test_prime3_ntt.rs`
- Test: `test_prime3_ntt_simple`
- Minimal reproduction: 1 × 1 = 454788377539 (should be 1)

## Impact

This bug blocks all depth-2+ operations in our Clifford FHE implementation:
- ❌ Wedge product: (a⊗b - b⊗a) / 2
- ❌ Inner product: (a⊗b + b⊗a) / 2
- ❌ Rotations: R ⊗ v ⊗ R̃
- ❌ Projections, rejections (depth-3)

Currently limited to "toy example" status with only depth-1 (geometric product).

## Request

We need help identifying:
1. **What is wrong with the NTT for q = 1099511693313?**
2. **How to fix it?**
3. **Are there other primes in our chain that might have similar issues?**

Any insights into NTT implementation for large primes in the context of RNS-CKKS would be greatly appreciated.

---

**Generated:** 2025-11-02
**Context:** Implementing Clifford FHE (geometric algebra over CKKS) in Rust
**Repository:** ga_engine
