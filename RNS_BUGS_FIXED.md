# RNS-CKKS Bugs Fixed - Session Summary

## Date
2025-11-02

## Bugs Found and Fixed

### Bug #1: RNS-Inconsistent Public Key Generation (CRITICAL)
**Location**: `src/clifford_fhe/keys_rns.rs`, lines 94-102

**Problem**:
The uniform polynomial `a` in the public key was being sampled with independent random values for each prime:

```rust
// WRONG:
for i in 0..n {
    for (j, &qj) in primes.iter().enumerate() {
        a_rns_coeffs[i][j] = rng.gen_range(0..qj);  // Independent per prime!
    }
}
```

This created RNS-inconsistent residues where `a[i][0]` and `a[i][1]` represented DIFFERENT underlying values, violating the fundamental RNS invariant that all residues represent the same value modulo Q = q₀ · q₁ · ....

**Impact**:
- Ciphertexts had inconsistent residues across primes
- Decryption produced garbage values (errors ~10^11 instead of correct values)
- All subsequent operations were corrupted

**Fix**:
Sample coefficients from a consistent range [0, q₀), then reduce modulo all primes:

```rust
// CORRECT:
let q0 = primes[0]; // Largest prime
let a_coeffs: Vec<i64> = (0..n).map(|_| rng.gen_range(0..q0)).collect();
let a_rns = RnsPolynomial::from_coeffs(&a_coeffs, primes, n, 0);
```

This ensures all residues represent the same underlying polynomial.

**Expert's Guidance**:
> "Lock down per-prime multiply... Make sure your RNS multiply builds each prime's product independently... The most likely fix is ensuring every coefficient's residues are a valid CRT tuple"

The expert correctly identified this as a "per-prime misalignment" issue where residues from different primes were inconsistent.

---

### Bug #2: Wrong Decryption Formula (CRITICAL)
**Location**: `src/clifford_fhe/ckks_rns.rs`, line 290

**Problem**:
The decryption formula used subtraction when it should use addition:

```rust
// WRONG:
let m_prime = rns_sub(&ct.c0, &c1s, active_primes);  // c0 - c1*s
```

**Correct Formula**:
For CKKS with public key b = -a·s + e, the decryption formula is:

```
m' = c0 + c1·s
```

**Proof**:
```
c0 + c1·s = (b·r + e0 + m) + (a·r + e1)·s
          = ((-a·s + e)·r + e0 + m) + (a·r)·s + e1·s
          = -a·s·r + e·r + e0 + m + a·r·s + e1·s
          = m + e·r + e0 + e1·s     [a·s·r terms cancel!]
```

With zero noise (e = e0 = e1 = 0), we get m' = m exactly.

**Fix**:
```rust
// CORRECT:
let m_prime = rns_add(&ct.c0, &c1s, active_primes);  // c0 + c1*s
```

---

## Previous Bugs Fixed (from earlier sessions)

### Bug #3: Premature Modular Reduction in Polynomial Multiply
**Location**: `src/clifford_fhe/keys_rns.rs`, lines 110-137, 180-205

**Problem**: Polynomial multiplication was reducing modulo after each product instead of accumulating first.

**Fix**: Accumulate all products in i128, then reduce once at the end.

### Bug #4: Wrong Public Key Formula
**Location**: `src/clifford_fhe/keys_rns.rs`, lines 141-161

**Problem**: Public key was computed as b = a·s + e instead of b = -a·s + e.

**Fix**: Added negation step to compute b = -a·s + e (CKKS standard).

---

## Testing Results

### Before Fixes:
```
Decrypted:
  coeff[0] residues: [855696110723587709, 411715506363091052]  ← Inconsistent!
  Expected residues for 2Δ: [2097152, 2097152]

Recovered value: -272461108052.46912
Expected:        2.0

❌ Basic encryption/decryption FAILED!
   Error: 272461108054.46912
```

### After Fixes:
```
Decrypted:
  coeff[0] residues: [2097152, 2097152]  ← Perfect!
  Expected residues for 2Δ: [2097152, 2097152]

Recovered value: 2
Expected:        2.0

✅ Basic encryption/decryption WORKS!
```

---

## Key Learnings

1. **RNS Consistency is Critical**: All residues must represent the same underlying value. Sampling must be done in a consistent range.

2. **CKKS Decryption Formula**: For b = -a·s + e, use c0 + c1·s (not subtraction).

3. **Self-Check Approach**: Adding runtime verification (RNS_SELFCHECK) helped isolate that the polynomial multiply itself was correct, narrowing down the bug to key generation.

4. **Expert Diagnosis Was Correct**: The expert identified "per-prime misalignment" as the root cause, which led directly to finding the inconsistent sampling bug.

---

## Files Modified

1. `src/clifford_fhe/keys_rns.rs` - Fixed key generation sampling
2. `src/clifford_fhe/ckks_rns.rs` - Fixed decryption formula
3. `src/clifford_fhe/rns.rs` - Added self-check and assertions
4. `examples/test_basic_enc_dec.rs` - Added CRT consistency checks

---

## Next Steps

1. ✅ Basic encryption/decryption works (N=64, 2 primes, noise=0)
2. ⏭️ Test with noise > 0
3. ⏭️ Test with full parameters (N=1024, 10 primes)
4. ⏭️ Test homomorphic multiplication
5. ⏭️ Test geometric product

All foundational RNS-CKKS bugs have been fixed!
