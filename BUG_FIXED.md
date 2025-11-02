# Bug Fixed: RNS-CKKS Multiplication Now Works With Noise! 🎉

## Summary
**FIXED!** RNS-CKKS homomorphic multiplication now works correctly even with noise (error_std=3.2).

## The Root Cause
The bug was in the **gadget decomposition** function used during relinearization. The original implementation decomposed each prime's residues **independently**, which violated CRT consistency. This caused catastrophic errors (~10^5x too large) when noise was present, even though the zero-noise case worked.

## The Fix
Implemented **CRT-consistent, balanced base-2^w decomposition**:

1. **CRT Reconstruct**: Convert RNS residues to a single integer x ∈ [0, Q) via Chinese Remainder Theorem
2. **Center-Lift**: Map to x_c ∈ (-Q/2, Q/2] for signed representation
3. **Balanced Decomposition**: Decompose x_c into digits d_t ∈ [-B/2, B/2) such that x_c = Σ d_t·B^t
4. **Map to RNS**: Reduce each digit modulo all primes **identically**

This ensures that Σ d_t·B^t ≡ original_value (mod q_i) for **EVERY** prime q_i, maintaining the EVK cancellation property.

### Key Changes

#### 1. Fixed Decomposition ([src/clifford_fhe/rns.rs](src/clifford_fhe/rns.rs#L798))
```rust
pub fn decompose_base_pow2(
    poly: &RnsPolynomial,
    primes: &[i64],
    w: u32,
) -> Vec<RnsPolynomial> {
    // For each coefficient:
    // 1. CRT reconstruct to get x ∈ [0, Q)
    let x = crt_reconstruct(&residues, primes);

    // 2. Center-lift to (-Q/2, Q/2]
    let x_centered = center_lift(x, q_prod);

    // 3. Balanced base-B decomposition in Z
    let digits_z = balanced_pow2_decompose(x_centered, w, d);

    // 4. Map each digit back to RNS consistently
    for t in 0..d {
        let dt = digits_z[t];
        for j in 0..num_primes {
            let qi = primes[j];
            let dt_mod_qi = ((dt as i128 % qi as i128) + qi as i128) % qi as i128;
            digits_data[t][i][j] = dt_mod_qi as i64;
        }
    }
}
```

#### 2. Fixed CRT Reconstruction to Avoid Overflow
Used modular multiplication (`mulmod_i128`) instead of naive `(a * b) % m` to prevent i128 overflow in intermediate products.

#### 3. Fixed Number of Digits
Changed from hardcoded `d = 3` to `d = (num_primes * 60 + w - 1) / w` to handle Q = product of all primes (which can be ~2^100 for 2 primes).

#### 4. Updated EVK Generation
EVK now creates the correct number of digit keys matching the decomposition.

## Test Results

### Before Fix ❌
```
With Δ = 2^40, error_std = 3.2:
  Decoded: 473290.810540
  Expected: 6.0
  Error: 473284.810540  (79000x too large!)
```

### After Fix ✅
```
With Δ = 2^40, error_std = 3.2:
  Decoded: 6.000000
  Expected: 6.0
  Error: 0.000000
  Relative error: 3.79e-12  (PERFECT!)
```

## Verification Tests

### 1. Decomposition Correctness
[test_decomp_verify.rs](examples/test_decomp_verify.rs) - Verifies Σ d_t·B^t ≡ original (mod each prime)
```
Prime j=0: ✅ MATCH
Prime j=1: ✅ MATCH
```

### 2. EVK Identity
[test_evk_identity.rs](examples/test_evk_identity.rs) - Verifies evk0 - evk1·s = -B^t·s² + e (small)
```
Max centered error: 6
Expected ~6σ = 19.2
✅ EVK identity holds! Errors are small.
```

### 3. Multiplication Tests
- **Zero noise** ([test_mult_zero_noise.rs](examples/test_mult_zero_noise.rs)): ✅ PASS (error 0.0)
- **With noise** ([test_mult_proper_primes.rs](examples/test_mult_proper_primes.rs)): ✅ PASS (error 3.79e-12)
- **Relinearization** ([test_relin_no_rescale.rs](examples/test_relin_no_rescale.rs)): ✅ PASS

## Key Insights

1. **Per-prime decomposition breaks CRT**: Each prime's residues must represent the SAME underlying integer
2. **Noise amplifies inconsistencies**: With zero noise, the bug was hidden; noise exposed the CRT violation
3. **Digit count matters**: Must have enough digits to cover Q = ∏ q_i, not just individual primes
4. **Balanced digits reduce noise**: Using d_t ∈ [-B/2, B/2) instead of [0, B) roughly halves noise growth

## Files Modified

- [src/clifford_fhe/rns.rs](src/clifford_fhe/rns.rs) - Added CRT-consistent decomposition
- [src/clifford_fhe/keys_rns.rs](src/clifford_fhe/keys_rns.rs) - Fixed EVK digit count
- [src/clifford_fhe/ckks_rns.rs](src/clifford_fhe/ckks_rns.rs) - Fixed debug tracing

## Credit

This fix was implemented based on expert guidance that identified the exact issue:
> "Do not decompose per-prime independently. That makes the digit tuples not represent the same integer across primes, so the EVK relation doesn't cancel properly when noise is present."

The solution uses the standard CKKS approach: CRT-consistent, balanced gadget decomposition as used in SEAL, HElib, and other production FHE libraries.

## Next Steps

- ✅ Basic multiplication works
- ⏭️ Test with larger parameters (N=1024, 10+ primes)
- ⏭️ Implement NTT for faster polynomial multiplication
- ⏭️ Add bootstrapping for deeper computation
