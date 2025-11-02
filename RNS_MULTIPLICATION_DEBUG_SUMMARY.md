# RNS-CKKS Homomorphic Multiplication Debug Request

## Context

I've successfully implemented RNS-CKKS encrypt/decrypt which works perfectly:
- ✅ Encrypt `[5]` → Decrypt → Get `[5]` (exact)
- ✅ RNS polynomial operations working
- ✅ CRT reconstruction working with i128 overflow protection

## The Problem

Homomorphic multiplication `[2] × [3]` produces wrong results after rescaling:
- **Expected**: ≈ 6
- **Got**: Large negative value (coefficient ≈ -4×10^11 at scale ≈ 1.1×10^12)

## Test Output

```
Test: test_rns_mult_debug.rs

Plaintext A RNS (coeff[0]): [174, 170, 166]  // 2 * scale mod each prime
Plaintext B RNS (coeff[0]): [261, 255, 249]  // 3 * scale mod each prime

Before multiplication:
  ct_a.level = 0, scale = 1.10e12
  ct_b.level = 0, scale = 1.10e12

After multiplication:
  ct_result.level = 1, scale = 1.10e12
  Active primes: 2 (dropped last prime in rescaling)

Decrypted plaintext (level 1):
  RNS (coeff[0]): [685153833332, 361291302902]

These residues are in the UPPER HALF of their moduli:
  Center-lifted: [-414357794357, -358266537434]
  → Represents large NEGATIVE value, not +6!
```

## Relevant Code Files

### 1. Multiplication Function
**File**: `src/clifford_fhe/ckks_rns.rs:267-308`

```rust
pub fn rns_multiply_ciphertexts(
    ct1: &RnsCiphertext,
    ct2: &RnsCiphertext,
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> RnsCiphertext {
    // Step 1: Multiply ciphertexts (tensored product) → degree-2
    let c0d0 = rns_poly_multiply(&ct1.c0, &ct2.c0, active_primes, polynomial_multiply_ntt);
    let c0d1 = rns_poly_multiply(&ct1.c0, &ct2.c1, active_primes, polynomial_multiply_ntt);
    let c1d0 = rns_poly_multiply(&ct1.c1, &ct2.c0, active_primes, polynomial_multiply_ntt);
    let c1d1 = rns_poly_multiply(&ct1.c1, &ct2.c1, active_primes, polynomial_multiply_ntt);

    let d_mid = rns_add(&c0d1, &c1d0, active_primes);

    // Step 2: Relinearization (degree-2 → degree-1)
    let (new_c0, new_c1) = rns_relinearize_degree2(&c0d0, &d_mid, &c1d1, evk, active_primes, n);

    // Step 3: Rescaling - drop last prime
    let rescaled_c0 = rns_rescale(&new_c0, active_primes);
    let rescaled_c1 = rns_rescale(&new_c1, active_primes);

    let q_last = active_primes[num_primes - 1];
    let new_scale = (ct1.scale * ct2.scale) / (q_last as f64);

    RnsCiphertext::new(rescaled_c0, rescaled_c1, ct1.level + 1, new_scale)
}
```

### 2. Relinearization Function
**File**: `src/clifford_fhe/ckks_rns.rs:316-338`

```rust
fn rns_relinearize_degree2(
    d0: &RnsPolynomial,
    d1: &RnsPolynomial,
    d2: &RnsPolynomial,
    evk: &RnsEvaluationKey,
    primes: &[i64],
    _n: usize,
) -> (RnsPolynomial, RnsPolynomial) {
    // Use RNS evaluation key (encrypts s²)
    let (evk0, evk1) = &evk.relin_keys[0];

    // Multiply d2 (coefficient of s²) by evaluation key
    let d2_evk0 = rns_poly_multiply(d2, evk0, primes, polynomial_multiply_ntt);
    let d2_evk1 = rns_poly_multiply(d2, evk1, primes, polynomial_multiply_ntt);

    // c0 = d0 + d2*evk0
    // c1 = d1 + d2*evk1
    let c0 = rns_add(d0, &d2_evk0, primes);
    let c1 = rns_add(d1, &d2_evk1, primes);

    (c0, c1)
}
```

### 3. Rescaling Function
**File**: `src/clifford_fhe/rns.rs:335-384`

```rust
pub fn rns_rescale(poly: &RnsPolynomial, primes: &[i64]) -> RnsPolynomial {
    let n = poly.n;
    let level = poly.level;
    let num_primes = poly.num_primes();

    let q_last = primes[num_primes - 1];  // Prime to drop
    let new_num_primes = num_primes - 1;

    let mut new_rns_coeffs = vec![vec![0i64; new_num_primes]; n];

    for i in 0..n {
        let c_mod_qlast = poly.rns_coeffs[i][num_primes - 1];

        for j in 0..new_num_primes {
            let qj = primes[j];
            let c_mod_qj = poly.rns_coeffs[i][j];

            // Fast basis conversion formula:
            // c_new mod qj = (c_mod_qj - c_mod_qlast) * q_last^{-1} mod qj

            let qlast_inv = mod_inverse(q_last as i128, qj as i128) as i64;
            let diff = ((c_mod_qj - c_mod_qlast % qj) % qj + qj) % qj;
            let c_new = ((diff as i128) * (qlast_inv as i128) % (qj as i128) + (qj as i128)) % (qj as i128);

            new_rns_coeffs[i][j] = c_new as i64;
        }
    }

    RnsPolynomial::new(new_rns_coeffs, n, level + 1)
}
```

## Questions for Expert

1. **Is the rescaling formula correct?**
   - Using: `(c_mod_qj - c_mod_qlast) * q_last^{-1} mod qj`
   - This is "fast basis conversion" approximation
   - Should we use exact rounding instead: `round(c / q_last) mod qj`?

2. **Is relinearization correct?**
   - Are we applying the evaluation key correctly?
   - Should there be additional noise management?

3. **Why are we getting large negative residues?**
   - Input: small positive values (2, 3)
   - After multiply + rescale: large negative residues
   - Sign error somewhere?

4. **Scale management:**
   - After rescale: `new_scale = (scale1 * scale2) / q_last`
   - With scale ≈ q_last, this gives `new_scale ≈ scale` ✓
   - Is this correct?

## What Works

1. **Encrypt/Decrypt at level 0** (all 3 primes): ✅ Perfect
2. **Polynomial multiplication** (in plaintext): ✅ Works
3. **RNS operations** (add, multiply): ✅ Work
4. **CRT reconstruction**: ✅ Works (with i128)

## What Doesn't Work

1. **Homomorphic multiplication**: ❌ Wrong result after rescaling
2. **Decrypt at level 1** (2 primes after rescale): ⚠️ Large errors

## Test Files

- `examples/test_rns_simple.rs` - Basic encrypt/decrypt (PASSING)
- `examples/test_rns_homomorphic_mult.rs` - Multiplication test (FAILING)
- `examples/test_rns_mult_debug.rs` - Detailed debug output (FAILING)
- `examples/test_rns_level1_decrypt.rs` - Level-1 decrypt test (PARTIAL)

## Parameters

```rust
N = 1024
Primes = [1099511627689, 1099511627691, 1099511627693]  // Three 40-bit primes
Scale = 2^40 ≈ 1.1×10^12 (approximately equal to each prime)
Error std = 3.2
```

## Hypothesis

The rescaling formula `(c_i - c_last) * q_last^{-1} mod q_i` is an **approximation** that works when the coefficient `c` is "well-distributed" across the RNS representation. However, after relinearization, the coefficient values might not satisfy this assumption, leading to large errors.

**Possible fix**: Use exact rounding by reconstructing `c` via CRT, computing `round(c / q_last)`, then converting back to RNS. But this defeats the purpose of RNS (staying in RNS form throughout).

## Reference Implementations

- **Microsoft SEAL**: Has working RNS-CKKS with rescaling
- **HElib**: Also has RNS implementation
- **PALISADE**: Another option

Could you review the rescaling formula against these implementations?

## Priority

This is blocking the entire Clifford-FHE homomorphic geometric product implementation. Once multiplication works, we can proceed to the full geometric algebra pipeline.

**Impact**: HIGH - Core functionality blocker
**Urgency**: MEDIUM - Making good progress, but stuck on this specific issue
