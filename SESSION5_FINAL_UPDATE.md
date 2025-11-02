# Session 5 Final Update - RNS-CKKS Debugging

## Summary

✅ **Implemented** all expert recommendations:
1. Exact RNS rescaling with center-lift
2. Correct EVK formula: `evk0 = a_k*s + e_k + s²`, `evk1 = a_k`
3. Precomputed inverse constants

✅ **Verified** EVK correctness:
- Test `test_evk_correctness` shows EVK correctly encrypts s²
- Differences are just noise (±1)

❌ **Still failing**: Degree-2 decryption gives wrong result

## Test Results

### Probe B: EVK Correctness ✅
```
cargo run --example test_evk_correctness

✓ PASS: EVK correctly encrypts s² (differences are just noise)
```

**EVK is correct!**

### Probe A: Degree-2 Decryption ❌
```
cargo run --example test_degree2_decrypt

Messages: 2 × 3 = 6
Expected coefficient at scale²: 7.3×10²⁴
Got RNS: [362195, 361603, 361059]
CRT reconstruction: -809240557662323
Decoded: 0.000000
Expected: 6.000000

✗ FAIL
```

## The Mystery

The degree-2 manual decryption formula is:
```
m = d0 - d1*s + d2*s²
```

All components computed correctly:
- d0, d1, d2 from tensor product multiplication
- s² verified correct (matches EVK test)
- All operations use time-domain (no NTT mismatch)
- All modular reductions applied

Result: RNS values [362195, 361603, 361059]
- All less than q/2, so positive
- All consistent across primes
- But value is ~3.6×10⁵ instead of expected ~7.3×10²⁴

**Factor off**: ~10¹⁹ (which is suspiciously close to scale² / small_number)

## Hypothesis

Something is wrong with how the multiplication accumulates values, or there's a subtle bug in the polynomial multiplication that only shows up in the product of ciphertexts.

Possible issues:
1. **Missing carry/reduction** somewhere in nested multiplications
2. **Sign error** in negacyclic reduction during d2*s² computation
3. **Accumulation bug** in the tensor product
4. **Scale mismatch** - but we're not applying any scaling, just ring operations

## What We Know Works

✅ Basic encrypt/decrypt (level 0)
✅ RNS polynomial operations (tested in isolation)
✅ EVK generation and structure
✅ CRT reconstruction (when values are correct)
✅ Exact rescaling formula (implemented but not yet tested due to earlier failure)

## Code References

All implemented changes:
- **Exact rescale**: [rns.rs:335-428](src/clifford_fhe/rns.rs#L335-L428)
- **EVK generation**: [keys_rns.rs:180-226](src/clifford_fhe/keys_rns.rs#L180-L226)
- **Multiplication**: [ckks_rns.rs:267-309](src/clifford_fhe/ckks_rns.rs#L267-L309)

## Tests to Run

```bash
# Verify EVK encrypts s²
cargo run --example test_evk_correctness

# Test degree-2 decryption (currently failing)
cargo run --example test_degree2_decrypt

# Test with relinearization (will fail until degree-2 works)
cargo run --example test_mult_no_rescale
```

## Request for Expert

The EVK is provably correct (Probe B passes). But degree-2 decryption (Probe A) fails with values that are 10¹⁹ times too small.

**Question**: Is there a known gotcha in computing the degree-2 product or in handling the s² term that could cause this massive magnitude error?

The formula `m = d0 - d1*s + d2*s²` seems straightforward, but the result suggests something fundamental is wrong with how these polynomial products combine.

## Next Steps

If degree-2 decryption can be fixed, then:
1. Probe C (relin only) should pass automatically (EVK is correct)
2. Full multiply with rescale should work
3. Clifford-FHE geometric product pipeline can proceed

**Blocker**: Understanding why degree-2 decryption gives values 10¹⁹ times too small.
