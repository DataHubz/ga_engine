# Current Bug: RNS-CKKS Multiplication Fails With Noise

## Summary
Homomorphic multiplication in RNS-CKKS works **perfectly with zero noise** but fails catastrophically when even small noise (std=3.2) is added. The decrypted value is off by a factor of ~10^5 to ~10^7.

## Reproduction

### Working Case (Zero Noise)
```bash
cargo run --release --example test_mult_zero_noise
```

**Result**: ✅ PASS
```
Expected: 6.0
Decoded: 6.000000
Error: 0.0000000000
```

### Failing Case (With Noise)
```bash
cargo run --release --example test_mult_proper_primes
```

**Result**: ❌ FAIL
```
Expected: 6.0
Decoded: 473290.810540
Error: 473284.810540
```

## Test Setup

Both tests use identical parameters except for `error_std`:

```rust
params.n = 64;
params.scale = 2f64.powi(40);  // Δ = 2^40 = 1099511627776
params.moduli = vec![
    1152921504606851201,  // q0 ≈ 2^60
    1099511628161,        // q1 ≈ 2^40 ≈ Δ
];

// Only difference:
params.error_std = 0.0;  // Zero noise test - WORKS ✅
params.error_std = 3.2;  // With noise test - FAILS ❌
```

Both tests encrypt [2] × [3] and expect result [6].

## Key Observations

### 1. Zero Noise Works Perfectly
With `error_std = 0.0`:
- Tensor product: d0 + d1·s + d2·s² = exactly 6·Δ² ✅
- Relinearization: works perfectly ✅
- Rescaling: produces exactly 6·Δ ✅
- Final decryption: 6.000000 ✅

### 2. With Noise, Value is Corrupted
With `error_std = 3.2`:
- Individual encryptions decrypt correctly (2.0 and 3.0) ✅
- After multiplication + rescaling: ~473290 instead of 6 ❌
- Error factor: ~79000x too large
- This is FAR beyond normal CKKS noise amplification

### 3. Relinearization Alone Works
```bash
cargo run --release --example test_relin_no_rescale
```

This test does tensor product + relinearization WITHOUT rescaling:
- Uses Δ = 2^20 (smaller scale)
- With error_std = 3.2
- Result: error only 1.87e-5 ✅

**Conclusion**: Relinearization works correctly even with noise. The bug is specific to the interaction between relinearization and rescaling.

### 4. Pattern Across Different Scales

| Δ     | error_std | Rescale | Result        |
|-------|-----------|---------|---------------|
| 2^20  | 3.2       | No      | ✅ Works      |
| 2^40  | 0.0       | Yes     | ✅ Works      |
| 2^40  | 3.2       | Yes     | ❌ ~473k err  |
| 2^30  | 3.2       | Yes     | ❌ ~79M err   |

## Diagnostic Output

### Before Rescaling (With Noise)
```
[BEFORE RESCALE] new_c0[0] residues:
  j=0 r=871322087425746527 centered=-281599417181104674
  j=1 r=56088646085 centered=56088646085
```

Using CRT to reconstruct the actual value before rescaling:
```python
# Python CRT reconstruction
q0 = 1152921504606851201
q1 = 1099511628161
r0 = 871322087425746527
r1 = 56088646085

value_centered = -574930502140634234923496445033
value / Δ² = -475571.36  # Should be ~6!
```

**The value BEFORE rescaling is already wrong by a factor of ~80000!**

This proves the bug is NOT in rescaling itself, but in how values are computed during relinearization when noise is present.

### After Rescaling
```
c0_after_rescale coeff[0] residues:
  j=0  r=630025259205390163  centered=-522896245401461038
```

Decoded: `630025259205390163 / Δ ≈ 473290` (still wrong)

## Code Involved

### 1. Multiplication Function
**File**: `src/clifford_fhe/ckks_rns.rs:354`

```rust
pub fn rns_multiply_ciphertexts(
    ct1: &RnsCiphertext,
    ct2: &RnsCiphertext,
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> RnsCiphertext {
    // Step 1: Tensor product
    let c0d0 = rns_poly_multiply(&ct1.c0, &ct2.c0, ...);
    let c0d1 = rns_poly_multiply(&ct1.c0, &ct2.c1, ...);
    let c1d0 = rns_poly_multiply(&ct1.c1, &ct2.c0, ...);
    let c1d1 = rns_poly_multiply(&ct1.c1, &ct2.c1, ...);

    let d_mid = rns_add(&c0d1, &c1d0, active_primes);

    // Step 2: Relinearization
    let (new_c0, new_c1) = rns_relinearize_degree2(&c0d0, &d_mid, &c1d1, evk, ...);

    // Step 3: Rescaling
    let inv = precompute_rescale_inv(active_primes);
    let rescaled_c0 = rns_rescale_exact(&new_c0, active_primes, &inv);
    let rescaled_c1 = rns_rescale_exact(&new_c1, active_primes, &inv);

    RnsCiphertext::new(rescaled_c0, rescaled_c1, new_level, new_scale)
}
```

### 2. Relinearization Function
**File**: `src/clifford_fhe/ckks_rns.rs:481`

```rust
fn rns_relinearize_degree2(
    d0: &RnsPolynomial,
    d1: &RnsPolynomial,
    d2: &RnsPolynomial,
    evk: &RnsEvaluationKey,
    primes: &[i64],
    _n: usize,
) -> (RnsPolynomial, RnsPolynomial) {
    // Decompose d2 into digits in base B = 2^w
    let d2_digits = decompose_base_pow2(d2, primes, evk.base_w);

    let mut c0 = d0.clone();
    let mut c1 = d1.clone();

    for t in 0..d2_digits.len() {
        let u0 = rns_poly_multiply(&d2_digits[t], &evk.evk0[t], primes, ...);
        let u1 = rns_poly_multiply(&d2_digits[t], &evk.evk1[t], primes, ...);

        c0 = rns_sub(&c0, &u0, primes);  // c0 - Σ d_t·evk0[t]
        c1 = rns_add(&c1, &u1, primes);  // c1 + Σ d_t·evk1[t]
    }

    (c0, c1)
}
```

Current digit base: `w = 20` (B = 2^20 = 1048576)

### 3. Rescaling Function
**File**: `src/clifford_fhe/rns.rs:556`

```rust
pub fn rns_rescale_exact(
    poly: &RnsPolynomial,
    primes: &[i64],
    inv_qlast_mod_qi: &[i64],
) -> RnsPolynomial {
    let q_last = primes[num_primes - 1];

    for i in 0..n {
        // Center-lift last residue
        let c_last_centered = if c_mod_qlast > q_last / 2 {
            c_mod_qlast - q_last
        } else {
            c_mod_qlast
        };

        for j in 0..new_num_primes {
            let qj = primes[j];
            let c_mod_qj = poly.rns_coeffs[i][j];

            let t = ((c_last_centered % qj) + qj) % qj;
            let diff = ((c_mod_qj - t) % qj + qj) % qj;
            let c_new = ((diff as i128) * (inv_qlast_mod_qi[j] as i128) % (qj as i128)
                        + (qj as i128)) % (qj as i128);

            new_rns_coeffs[i][j] = c_new as i64;
        }
    }

    RnsPolynomial::new_with_domain(new_rns_coeffs, n, level + 1, poly.domain)
}
```

## Hypothesis

The issue appears to be in the **relinearization** step when noise is present:

1. With zero noise, the EVK errors are zero, so the relinearization is exact
2. With noise, the EVK contains error terms that get amplified during gadget decomposition
3. The amplified error manifests as a massive corruption of the actual value

Possible causes:
- **Gadget decomposition** creating wrong digit values when noise is present
- **EVK noise sampling** might be incorrect (wrong scale or distribution)
- **Polynomial multiplication** might have overflow issues with noisy values
- **Sign convention** in EVK formula might be subtly wrong for noisy case

## What We've Tried

1. ✅ Fixed RNS consistency bugs in key generation
2. ✅ Fixed decryption formula (c0 + c1·s instead of c0 - c1·s)
3. ✅ Added domain tags to prevent COEF/NTT mixing
4. ✅ Verified rescaling formula is mathematically correct
5. ✅ Increased digit base from w=10 to w=20
6. ✅ Tested with different scales (2^20, 2^30, 2^40)
7. ✅ Verified tensor product works correctly
8. ✅ Verified relinearization works without rescaling

## Next Steps to Investigate

1. **Compare EVK values**: Check if EVK with noise has unexpectedly large values
2. **Trace gadget decomposition**: Verify digit values are correct when noise is present
3. **Check i128 overflow**: Large intermediate products might overflow during multiplication
4. **Verify EVK formula**: Double-check the sign convention in EVK generation
5. **Test with larger N**: Current N=64 is small, might expose edge cases

## Files

- Test (failing): `examples/test_mult_proper_primes.rs`
- Test (working): `examples/test_mult_zero_noise.rs`
- Test (relin only): `examples/test_relin_no_rescale.rs`
- Multiplication: `src/clifford_fhe/ckks_rns.rs`
- RNS operations: `src/clifford_fhe/rns.rs`
- Key generation: `src/clifford_fhe/keys_rns.rs`
