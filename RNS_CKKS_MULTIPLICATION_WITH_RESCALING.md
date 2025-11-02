# RNS-CKKS Multiplication with Rescaling - Step-by-Step Reproduction

## Date
2025-11-02

## Purpose
This document reproduces the full RNS-CKKS multiplication pipeline including rescaling, to identify where the remaining bug lies.

## Status Summary

### ✅ Working Components
1. **Basic Encryption/Decryption** - Perfect (N=64, 2 primes, noise=0)
2. **Public Key Relation** - Perfect (b + a·s = 0)
3. **Tensor Product** - Perfect (d0 + d1·s + d2·s² = 6·Δ²)
4. **Relinearization** - 99.998% accurate (error 2.37e-5)

### ❌ Failing Component
5. **Rescaling** - After rescaling, errors are ~25-95 (down from ~165 before fixes)

## Mathematical Background

### CKKS Multiplication Pipeline

Given two ciphertexts encrypting m₁ and m₂ at scale Δ:
- ct₁ encrypts m₁: ct₁ = (c0₁, c1₁) where c0₁ + c1₁·s ≈ m₁·Δ
- ct₂ encrypts m₂: ct₂ = (c0₂, c1₂) where c0₂ + c1₂·s ≈ m₂·Δ

**Step 1: Tensor Product**
Compute element-wise polynomial multiplication:
```
(c0₁, c1₁) ⊗ (c0₂, c1₂) = (d0, d1, d2)

where:
  d0 = c0₁ · c0₂
  d1 = c0₁ · c1₂ + c1₁ · c0₂
  d2 = c1₁ · c1₂
```

**Decryption formula for degree-2 ciphertext:**
```
d0 + d1·s + d2·s² = (c0₁ + c1₁·s) · (c0₂ + c1₂·s)
                  ≈ (m₁·Δ) · (m₂·Δ)
                  = m₁·m₂·Δ²
```

**Step 2: Relinearization**
Use evaluation key EVK to convert degree-2 back to degree-1:

With EVK satisfying: `evk0[t] - evk1[t]·s = -B^t·s² + e_t`

Decompose d2 into digits: `d2 = Σ d_t·B^t` where B = 2^w

Compute:
```
c0' = d0 - Σ d_t·evk0[t]
c1' = d1 + Σ d_t·evk1[t]
```

**Verification:**
```
c0' + c1'·s = (d0 - Σ d_t·evk0[t]) + (d1 + Σ d_t·evk1[t])·s
            = d0 + d1·s + Σ d_t·(-evk0[t] + evk1[t]·s)
            = d0 + d1·s + Σ d_t·(B^t·s² - e_t)
            = d0 + d1·s + d2·s² - noise
            ≈ m₁·m₂·Δ²
```

**Step 3: Rescaling**
The result is at scale Δ². To bring it back to scale Δ, divide by q_last (the last prime in the modulus chain):

For RNS representation with primes Q = q₀ · q₁ · ... · q_L:

```
c_rescaled = round(c / q_L) mod Q'

where Q' = q₀ · q₁ · ... · q_{L-1}
```

**RNS Rescaling Formula (per prime):**

For each coefficient c with residues [c mod q₀, c mod q₁, ..., c mod q_L]:

1. Center-lift the last residue:
   ```
   c_L' = c mod q_L  (lifted to (-q_L/2, q_L/2])
   ```

2. For each remaining prime q_i (i = 0..L-1):
   ```
   c'_i = (c_i - c_L') · q_L^{-1} mod q_i
   ```

This computes `round(c / q_L)` exactly in RNS form.

**New scale:**
```
Δ_new = Δ² / q_L
```

If we choose q_L ≈ Δ, then Δ_new ≈ Δ, maintaining consistent scale.

## Test Case: [2] × [3] = [6]

### Parameters
- N = 64 (ring dimension)
- Number of primes = 2
- q₀ = 1141392289560813569 (≈ 2^60)
- q₁ = 1141392289560840193 (≈ 2^60)
- Δ = 2^20 = 1048576
- w = 10 (digit width, B = 2^10 = 1024)
- D = 6 (number of digits for 60-bit primes)

### Expected Result
- m₁ = 2, m₂ = 3
- After tensor product: m₁·m₂·Δ² = 6 · (2^20)² = 6 · 2^40 = 6597069766656
- After rescaling: m₁·m₂·Δ²/q₁ ≈ 6 · Δ = 6291456 (exact value depends on rounding)

### Actual Execution Trace

#### Step 1: Tensor Product ✅
```
d0[0] residues: [174619855359483838, 634430955580491654]
d1[0] residues: [991657569553000741, 314568848755863851]
d2[0] residues: [548173707513442054, 658672117879067243]

d0 + d1·s + d2·s²:
  result[0] residues: [6597069766656, 6597069766656]
  Expected: 6597069766656
  ✅ PERFECT MATCH
```

#### Step 2: Relinearization ✅
```
Decomposition:
  d2_digits[0][0] residues: [543, 575]
  d2_digits[1][0] residues: [764, 521]
  d2_digits[2][0] residues: [710, 812]
  d2_digits[3][0] residues: [438, 252]
  d2_digits[4][0] residues: [94, 520]
  d2_digits[5][0] residues: [640, 888]

After relinearization:
  c0'[0] residues: [544018115545252468, 534549521798736170]
  c1'[0] residues: [714565562049919750, 903307477712063292]
  c1'·s[0] residues: [597380770929115820, 606849364675659039]

  result[0] residues: [6596913554719, 6596913555016]
  Expected: 6597069766656
  Error: 156211937
  Relative error: 2.37e-5 (0.00237%)
  ✅ EXCELLENT (within noise tolerance)
```

#### Step 3: Rescaling ❌
```
Before rescaling (c0', c1') at level 0 (2 primes):
  c0'[0] = [544018115545252468, 534549521798736170]
  c1'[0] = [714565562049919750, 903307477712063292]

After rescaling to level 1 (1 prime):
  c0''[0] = [?]
  c1''[0] = [?]

After decryption (c0'' + c1''·s):
  result = ???
  Expected: ≈ 6 · Δ = 6291456

  Actual from test: ~30.694 (after dividing by Δ)
  This corresponds to: 30.694 · 1048576 ≈ 32181299
  Error: HUGE (~5× off)
```

## Current Rescaling Implementation

**File:** `src/clifford_fhe/rns.rs:488-535`

```rust
pub fn rns_rescale_exact(
    poly: &RnsPolynomial,
    primes: &[i64],
    inv_qlast_mod_qi: &[i64],
) -> RnsPolynomial {
    let n = poly.n;
    let level = poly.level;
    let num_primes = poly.num_primes();

    assert!(num_primes > 1, "Cannot rescale: only one prime remaining");
    assert_eq!(inv_qlast_mod_qi.len(), num_primes - 1, "inv_qlast_mod_qi length mismatch");

    // Get the prime we're dropping (last one)
    let q_last = primes[num_primes - 1];
    let new_num_primes = num_primes - 1;

    let mut new_rns_coeffs = vec![vec![0i64; new_num_primes]; n];

    // For each coefficient, perform exact rounded rescaling
    for i in 0..n {
        // Step 1: Center-lift the last residue to (-q_last/2, q_last/2]
        let c_mod_qlast = poly.rns_coeffs[i][num_primes - 1];
        let c_last_centered = if c_mod_qlast > q_last / 2 {
            c_mod_qlast - q_last
        } else {
            c_mod_qlast
        };

        // Step 2: For each remaining prime, compute rounded division
        for j in 0..new_num_primes {
            let qj = primes[j];
            let c_mod_qj = poly.rns_coeffs[i][j];

            // Bring c_last_centered into modulo qj
            let t = ((c_last_centered % qj) + qj) % qj;

            // Compute: (c_mod_qj - t) * inv_qlast_mod_qi[j] mod qj
            let diff = ((c_mod_qj - t) % qj + qj) % qj;

            // Multiply by precomputed inverse (use i128 to avoid overflow)
            let c_new = ((diff as i128) * (inv_qlast_mod_qi[j] as i128) % (qj as i128) + (qj as i128)) % (qj as i128);

            new_rns_coeffs[i][j] = c_new as i64;
        }
    }

    RnsPolynomial::new(new_rns_coeffs, n, level + 1)
}
```

## Potential Issues in Rescaling

### Issue 1: Prime Indexing
The code uses `primes[num_primes - 1]` as `q_last`, but this assumes:
- Input `primes` array has exactly `num_primes` elements
- The last prime in the array is the one to drop

**Question:** When called from `rns_multiply_ciphertexts`, is the `primes` array correctly sliced to `active_primes`?

**Code in `ckks_rns.rs:386`:**
```rust
let new_primes = &active_primes[..active_primes.len()-1];
```

This looks correct - it takes all but the last prime.

### Issue 2: Center-Lifting Sign
```rust
let c_last_centered = if c_mod_qlast > q_last / 2 {
    c_mod_qlast - q_last
} else {
    c_mod_qlast
};
```

This lifts to (-q_last/2, q_last/2], which is correct for signed arithmetic.

### Issue 3: Modular Arithmetic
```rust
let t = ((c_last_centered % qj) + qj) % qj;
```

When `c_last_centered` is negative (which it often is after center-lifting), this should correctly compute the positive residue.

**BUT:** If `c_last_centered` is a large negative i64, the `%` operation in Rust returns a negative result!

Example:
```rust
let x: i64 = -1000;
let q: i64 = 7;
let r = x % q;  // r = -6 (not 1!)
let r_correct = ((x % q) + q) % q;  // r_correct = 1
```

So the code `((c_last_centered % qj) + qj) % qj` should be correct.

### Issue 4: Domain Tag Not Updated
```rust
RnsPolynomial::new(new_rns_coeffs, n, level + 1)
```

This creates a new polynomial with default domain `Coef`, which should be correct. But it doesn't preserve the domain tag from the input! If the input was in NTT domain, this would create an inconsistency.

**Fix:** Use `new_with_domain`:
```rust
RnsPolynomial::new_with_domain(new_rns_coeffs, n, level + 1, poly.domain)
```

### Issue 5: Potential Overflow in Precomputed Inverse
The precomputed inverse is computed as:
```rust
inv_qlast_mod_qi[i] = mod_inverse(q_last as i128, qi as i128) as i64;
```

This should be safe for 60-bit primes since `mod_inverse` returns a value in [0, qi).

## Debugging Strategy

### Test 1: Verify Rescale Identity
After rescaling c to c', verify:
```
c'[i] · q_last + c_last_centered ≡ c[i] (mod q_i)
```

This is the definition of correct rescaling.

### Test 2: Check Scale Tracking
After rescaling, verify:
```
new_scale = old_scale² / q_last
```

If scale tracking is wrong, decryption will be off by a constant factor.

### Test 3: Deterministic Rescale Test
Create a test with known values:
```
c = [6597069766656, 6597069766656]  (both primes)
q_last = 1141392289560840193

Expected after rescale:
c' = round(6597069766656 / 1141392289560840193)
   = round(5.778e-6)
   = 0

Wait, this doesn't make sense!
```

**AH! THE PROBLEM!**

The value `6597069766656` is tiny compared to `q_last ≈ 1.14e18`!

So `6597069766656 / 1141392289560840193 ≈ 5.78e-6`, which rounds to 0!

**This is the bug!**

### The Core Issue: Wrong Prime for Rescaling

In CKKS, we should rescale by dropping the **smallest** prime, not the **largest**!

Or alternatively, the primes should be ordered with the **rescaling prime last** (conventionally the smallest).

**Current prime ordering:**
```
primes[0] = 1141392289560813569  (largest)
primes[1] = 1141392289560840193  (also large, ≈ 2^60)
```

Both primes are huge (≈ 2^60), so rescaling by either one will make the value ≈ 0!

**What should happen:**

For CKKS multiplication:
1. Fresh ciphertexts are at level 0 with ALL primes active
2. After multiplication, scale becomes Δ²
3. Rescale by dropping ONE prime (the last one)
4. Choose this prime to be ≈ Δ so that new_scale = Δ²/Δ = Δ

**Typical CKKS prime chain:**
```
Q₀ = q₀ · q₁ · q₂ · ... · q_L

where:
  q₀ ≈ 2^60 (large, for initial ciphertext)
  q₁ ≈ Δ ≈ 2^40 (rescaling prime, dropped after first mult)
  q₂ ≈ Δ ≈ 2^40 (rescaling prime, dropped after second mult)
  ...
```

**Current implementation:**
Both primes are ≈ 2^60, which is way too large for rescaling!

## The Fix

### Option 1: Use Smaller Rescaling Primes
Modify the prime generation to include primes ≈ Δ:

```rust
params.moduli = vec![
    1141392289560813569,  // q₀ ≈ 2^60 (base prime)
    1099511627776,        // q₁ ≈ 2^40 = Δ (rescaling prime)
];
```

### Option 2: Don't Actually Rescale (Skip Division)
For testing, just track that scale is now Δ² and DON'T divide:

```rust
// Skip rescaling, just update scale metadata
let new_scale = (ct1.scale * ct2.scale);  // Δ²
// Keep same level (don't drop prime)
RnsCiphertext::new(new_c0, new_c1, ct1.level, new_scale)
```

Then decode with scale Δ² instead of Δ.

### Option 3: Proper Multi-Prime CKKS (Correct Long-Term)
Use a full modulus chain with L primes:
```
Q₀ = q₀ · q₁ · q₂ · q₃ · q₄  (5 primes for depth-4 circuits)

where:
  q₀ ≈ 2^60  (base)
  q₁ ≈ 2^40  (rescale after mult 1)
  q₂ ≈ 2^40  (rescale after mult 2)
  q₃ ≈ 2^40  (rescale after mult 3)
  q₄ ≈ 2^40  (rescale after mult 4)
```

## Recommended Immediate Fix

For the test to pass, use **Option 1** with a smaller second prime:

```rust
// In test_canonical_slot_multiplication.rs or params
params.moduli = vec![
    1152921504606584833,  // 61-bit prime (base)
    1099511627776,        // 40-bit prime ≈ 2^40 = Δ (rescaling)
];
```

This way:
```
Before rescale: c ≈ 6597069766656 at scale Δ²
After rescale:  c' = round(6597069766656 / 1099511627776)
                   = round(6.0)
                   = 6 (at scale Δ)
Decode:         6 / Δ = 6 / 1048576 ≈ 6e-6...

Wait, that's still wrong!
```

**Actually, the decoded value should be:**
```
After rescale: c' ≈ 6 · Δ (at scale Δ)
Decode: (6 · Δ) / Δ = 6 ✓
```

So the rescaled value should be `6 · Δ = 6291456`, not just `6`.

Let me recalculate:
```
Before rescale: c ≈ 6 · Δ² = 6597069766656
After rescale:  c' = round((6 · Δ²) / q_L)

If q_L ≈ Δ:
  c' = round((6 · Δ²) / Δ) = round(6 · Δ) = 6 · Δ = 6291456

Decode: c' / Δ = (6 · Δ) / Δ = 6 ✓
```

Perfect! So the fix is to use a rescaling prime ≈ Δ.

## Conclusion

**Root cause:** Both primes in the test are ≈ 2^60, which is WAY larger than Δ = 2^20. When we rescale by dividing by q_last ≈ 2^60, the value `6·Δ² ≈ 6.6e12` becomes `6.6e12 / 1.14e18 ≈ 5.78e-6 ≈ 0`.

**Fix:** Use a rescaling prime ≈ Δ = 2^20 so that:
```
new_value = (6·Δ²) / Δ = 6·Δ  (preserves the message scaled by Δ)
```

**Implementation:** Modify prime selection in params or test to include a prime ≈ 2^40.
