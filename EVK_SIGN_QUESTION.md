# EVK Sign Convention - Need Clarification

## Current Status

✅ **Basic encrypt/decrypt works perfectly**
✅ **Exact rescaling implemented** (center-lift + proper rounding)
❌ **Relinearization produces wrong results**

## Implementation Details

### Decrypt Formula
```rust
m' = c0 - c1*s
```

### Public Key Generation
```rust
b = a*s + e  (not -a*s + e)
```

### Relinearization Application
```rust
c0_new = d0 + d2*evk0
c1_new = d1 + d2*evk1
```

Where (d0, d1, d2) is the degree-2 ciphertext from multiplication.

## The Math

For decrypt to work:
```
c0_new - c1_new*s = d0 - d1*s + d2*s²
(d0 + d2*evk0) - (d1 + d2*evk1)*s = d0 - d1*s + d2*s²
d2*(evk0 - evk1*s) = d2*s²
evk0 - evk1*s = s²
```

Therefore:
```
evk0 = s² + evk1*s
```

With `evk1 = a_k` (fresh uniform):
```
evk0 = s² + a_k*s
```

## What I Tried

### Attempt 1: Negative sign (from expert's formula)
```rust
evk0 = -a_k*s + e_k + s²  // Expert said this
evk1 = a_k
```

**Result**: Gives `evk0 - evk1*s = -2*a_k*s + e_k + s²` ❌

### Attempt 2: Positive sign (my derivation)
```rust
evk0 = a_k*s + e_k + s²   // My derivation
evk1 = a_k
```

**Result**: Gives `evk0 - evk1*s = e_k + s²` ✓ (almost, just +e_k noise)

But test still fails!

## Test Results (Attempt 2)

**Test**: Multiply [2] × [3], relinearize (NO rescaling), decrypt at scale²

**Expected coefficient**: `6 × scale² ≈ 7.3×10²⁴`
**Got coefficient**: `-6.8×10¹⁸`

Way too small and negative!

**Decrypted RNS**: [525673017102, 782978988476, 927405550720]
- Third residue is in upper half → negative after center-lift
- Inconsistent residues → CRT gives wrong value

## Questions

1. **Is the EVK formula `evk0 = a_k*s + e_k + s²` correct for decrypt `m' = c0 - c1*s`?**

2. **Or should I change the decrypt formula to `m' = c0 + c1*s` and use `evk0 = -a_k*s + e_k + s²`?**

3. **Is there a sign issue in how I'm computing the degree-2 ciphertext?**
   Current: `d0 = c0_a * c0_b`, `d1 = c0_a*c1_b + c1_a*c0_b`, `d2 = c1_a * c1_b`

4. **Should the tensor product multiplication handle signs differently?**

## Code References

- **EVK generation**: [keys_rns.rs:180-220](src/clifford_fhe/keys_rns.rs#L180-L220)
- **Decrypt**: [ckks_rns.rs:223](src/clifford_fhe/ckks_rns.rs#L223) - uses `rns_sub`
- **Multiplication**: [ckks_rns.rs:283-289](src/clifford_fhe/ckks_rns.rs#L283-L289)
- **Relinearization**: [ckks_rns.rs:316-338](src/clifford_fhe/ckks_rns.rs#L316-L338)

## Specific Test

Run: `cargo run --example test_mult_no_rescale`

This tests multiplication + relinearization WITHOUT rescaling to isolate the issue.

---

**Summary**: I'm confused about the correct sign convention for EVK when decrypt uses `c0 - c1*s`. Please clarify!
