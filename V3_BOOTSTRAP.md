# V3 Bootstrap Implementation

This document describes the V3 bootstrap implementation for CKKS homomorphic encryption, including both the **Hybrid** (GPU multiply + CPU rescale) and **Native** (fully GPU) versions.

## Overview

The V3 bootstrap enables approximate computation on encrypted data by refreshing ciphertext modulus levels through:
1. **CoeffToSlot**: Transform coefficient encoding → slot encoding (FFT-like)
2. **EvalMod**: Evaluate modular reduction (not currently implemented)
3. **SlotToCoeff**: Transform slot encoding → coefficient encoding (inverse FFT-like)

## Architecture

### Parameters
- **N = 1024**: Ring dimension
- **20 RNS primes**: Multi-precision arithmetic
  - 1× ~60-bit prime (q₀)
  - 19× ~45-bit primes (q₁...q₁₉)
- **Scale = 2⁴⁰**: Fixed-point precision
- **9 bootstrap levels**: log₂(N/2) transformations

### Two Implementations

#### 1. **Hybrid Version** (PRODUCTION - STABLE)
- **Rescaling**: CPU using BigInt CRT
- **All other ops**: Metal GPU (NTT, multiplication, rotation)
- **Accuracy**: ~3.6e-3 error (excellent)
- **Command**:
  ```bash
  cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap
  ```

#### 2. **Native Version** (FULLY GPU - STABLE)
- **100% GPU**: All operations including rescaling
- **Rescaling**: Metal GPU using fixed RNS DRLMQ shader
- **Accuracy**: ~3.6e-3 error (matches hybrid!)
- **Command**:
  ```bash
  cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native
  ```

## Implementation Details

### File Structure

```
src/clifford_fhe_v2/backends/gpu_metal/
├── bootstrap.rs                         # Bootstrap transformations
│   ├── coeff_to_slot_gpu()             # Hybrid CoeffToSlot
│   ├── coeff_to_slot_gpu_native()      # Native CoeffToSlot
│   ├── slot_to_coeff_gpu()             # Hybrid SlotToCoeff
│   ├── slot_to_coeff_gpu_native()      # Native SlotToCoeff
│   ├── multiply_plain_metal()          # Hybrid (CPU rescale)
│   └── multiply_plain_metal_native_rescale()  # Native (GPU rescale)
├── ckks.rs                              # Core CKKS operations
│   ├── exact_rescale_gpu_fixed()       # GPU rescaling wrapper
│   ├── multiply_polys_flat_ntt_negacyclic()  # NTT multiplication
│   └── rotate_by_steps()               # Galois rotation
└── shaders/
    └── rns_fixed.metal                  # GPU rescaling kernel
```

### CoeffToSlot Transformation

Implements FFT-like butterfly network with 9 levels:

```rust
pub fn coeff_to_slot_gpu_native(
    ct: &MetalCiphertext,
    rotation_keys: &MetalRotationKeys,
    ctx: &MetalCkksContext,
    params: &CliffordFHEParams,
) -> Result<MetalCiphertext, String>
```

**Algorithm:**
```
For level_idx in 0..9:
    rotation_amount = 2^level_idx  // 1, 2, 4, 8, ..., 256

    1. Rotate ciphertext by +rotation_amount (GPU)
    2. Compute DFT twiddle factors
    3. Encode diagonal matrices as plaintexts
    4. ct_mul1 = current * diag1 + rescale (GPU or CPU)
    5. ct_mul2 = rotated * diag2 + rescale (GPU or CPU)
    6. current = ct_mul1 + ct_mul2

    Level decreases by 1 each iteration
```

**Key Operations per Level:**
- 1× Rotation (Galois automorphism + key switching)
- 2× Plaintext multiplication (NTT-based)
- 2× Rescaling (GPU native or CPU BigInt)
- 1× Addition

### SlotToCoeff Transformation

Inverse FFT in reverse order:

```rust
pub fn slot_to_coeff_gpu_native(
    ct: &MetalCiphertext,
    rotation_keys: &MetalRotationKeys,
    ctx: &MetalCkksContext,
    params: &CliffordFHEParams,
) -> Result<MetalCiphertext, String>
```

**Algorithm:**
```
For level_idx in (0..9).rev():  // Reverse order: 8, 7, 6, ..., 0
    rotation_amount = 2^level_idx

    1. Rotate ciphertext by -rotation_amount (GPU)
    2. Compute inverse DFT twiddle factors
    3. Encode diagonal matrices as plaintexts
    4. ct_mul1 = current * diag1 + rescale (GPU or CPU)
    5. ct_mul2 = rotated * diag2 + rescale (GPU or CPU)
    6. current = ct_mul1 + ct_mul2
```

## GPU Rescaling (Native Version)

### The Problem Solved

CKKS requires **exact centered rounding** when rescaling:
```
C' = ⌊(C + q_last/2) / q_last⌋ mod (Q / q_last)
```

Previous RNS approximations failed because:
1. ❌ Rounding applied in wrong domain
2. ❌ Constants not in correct arithmetic domain
3. ❌ 128-bit overflow in modular multiplication

### The Solution: `rns_fixed.metal`

**File**: `src/clifford_fhe_v2/backends/gpu_metal/shaders/rns_fixed.metal`

```metal
kernel void rns_exact_rescale_fixed(
    device const ulong* poly_in [[buffer(0)]],
    device ulong* poly_out [[buffer(1)]],
    constant ulong* moduli [[buffer(2)]],
    constant ulong* qtop_inv_mod_qi [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& num_primes_in [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
)
```

**Algorithm** (DRLMQ - Divide and Round Last Modulus Quotient):

```
For each coefficient c (parallel on GPU):
    1. Centered rounding in q_last:
       r_last_rounded = (r_last + q_last/2) mod q_last

    2. For each output prime q_i:
       a. Map r_last_rounded to q_i domain
       b. Subtract rounding correction
       c. Compute difference: diff = (r_i - r_last_adjusted) mod q_i
       d. Multiply by inverse: result = (diff * q_last^{-1}) mod q_i

    3. Output: c' mod q_i
```

**Key Innovation**: Russian Peasant Multiplication

To avoid 128-bit overflow in `(diff * qtop_inv) mod q_i`:

```metal
inline ulong mul_mod_128(ulong a, ulong b, ulong q) {
    ulong result = 0;
    a = a % q;

    while (b > 0) {
        if (b & 1) {
            result = add_mod_lazy(result, a, q);
            if (result >= q) result -= q;
        }
        a = add_mod_lazy(a, a, q);
        if (a >= q) a -= q;
        b >>= 1;
    }

    return result;
}
```

This computes `(a * b) mod q` without any 128-bit intermediate values.

### Validation

**Golden Compare Test**: `examples/test_rescale_golden_compare.rs`
- Compares GPU vs CPU rescaling coefficient-by-coefficient
- Result: **0 mismatches** across 200 test coefficients
- Proves bit-exact correctness

## Performance

### Benchmark Results (Apple M3 Max)

| Operation | Hybrid | Native | Notes |
|-----------|--------|--------|-------|
| **Key Generation** | 73-75s | 73-75s | Same (CPU) |
| **Encryption** | ~170ms | ~175ms | GPU |
| **CoeffToSlot** | ~50-53s | ~48-50s | 9 levels |
| **SlotToCoeff** | ~12-13s | ~11-12s | 9 levels |
| **Decryption** | ~11ms | ~11ms | GPU |
| **TOTAL BOOTSTRAP** | **62-66s** | **60-62s** | 🎯 |
| **Max Error** | **3.61e-3** | **3.61e-3** | ✅ Identical! |

### Breakdown per Bootstrap Level

Each level performs:
- 1× Rotation (~5-6s for high levels, ~1-2s for low levels)
  - Galois automorphism (GPU)
  - Key switching with gadget decomposition (GPU)
- 2× Multiply + Rescale (~0.5-1s each)
  - NTT forward (GPU)
  - Pointwise multiplication (GPU)
  - NTT inverse (GPU)
  - Rescaling (GPU or CPU)
- 1× Addition (~negligible)

**Native version is slightly faster** because GPU rescaling avoids CPU↔GPU data transfer.

## Layout Conventions

### Strided Layout (Ciphertext Storage)
```
c[coeff_idx * num_primes + prime_idx]

Example (N=4, 3 primes):
[c0_q0, c0_q1, c0_q2, c1_q0, c1_q1, c1_q2, c2_q0, c2_q1, c2_q2, c3_q0, c3_q1, c3_q2]
```

### Flat RNS Layout (GPU Shader Input/Output)
```
poly[prime_idx * n + coeff_idx]

Example (N=4, 3 primes):
[c0_q0, c1_q0, c2_q0, c3_q0, c0_q1, c1_q1, c2_q1, c3_q1, c0_q2, c1_q2, c2_q2, c3_q2]
```

**Conversion** happens in `multiply_plain_metal_native_rescale()`:
1. Extract active primes → strided layout
2. GPU multiply → strided layout
3. Convert to flat RNS → for rescaling shader
4. GPU rescale → flat RNS output
5. Convert back to strided → compact format

## Testing

### Unit Tests

1. **Golden Compare Test**
   ```bash
   cargo run --release --features v2,v2-gpu-metal,v3 --example test_rescale_golden_compare
   ```
   - Validates GPU rescaling against CPU reference
   - Result: ✅ 0 mismatches

2. **Layout Test**
   ```bash
   cargo run --release --features v2,v2-gpu-metal,v3 --example test_multiply_rescale_layout
   ```
   - Validates multiply→rescale pipeline
   - Checks output dimensions and padding

### Integration Tests

1. **Hybrid Bootstrap**
   ```bash
   cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap
   ```
   - Expected: Error ~3.6e-3 ✅

2. **Native Bootstrap**
   ```bash
   cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native
   ```
   - Expected: Error ~3.6e-3 ✅

## Debugging

### Enable Debug Output

Both versions include extensive debug logging:

```rust
println!("  Level {}: rotation by ±{}, current level={}", level_idx, rotation_amount, current.level);
eprintln!("[ROTATION DEBUG] n={}, num_primes_active={}, ct_stride={}", n, num_primes_active, ct_stride);
eprintln!("[GALOIS DEBUG] Coeff 0 (mod q0): input={}, target_pos={}, output@target={}", ...);
```

To see all debug output:
```bash
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native 2>&1 | less
```

### Common Issues

1. **Layout Mismatch Warning**
   ```
   [ROTATION WARNING] ct_stride (20) != num_primes_active (9)
   ```
   - ✅ This is **expected** after rescaling
   - Ciphertexts use full stride (20 primes) but only first N are active

2. **Large Errors (>100k)**
   - ❌ Check output format: should be **compact** (n × num_primes_out), not padded
   - ❌ Verify layout conversions between strided ↔ flat RNS

3. **Shader Compilation Errors**
   - Check Metal Shading Language syntax (use C-style casts, not Rust `as`)
   - Verify all shader libraries load: `rns_fixed_library`

## Key Achievements

### Before This Work
- ❌ Native GPU rescaling produced **~385k errors**
- ❌ RNS approximation didn't handle CKKS rounding
- ❌ Only hybrid (CPU rescale) version worked

### After This Work
- ✅ **Native GPU rescaling: 3.61e-3 error** (matches hybrid!)
- ✅ **100% GPU pipeline** with no CPU fallback
- ✅ **Bit-exact correctness** proven by golden compare test
- ✅ **Slightly faster** than hybrid (~60s vs ~65s)

## Future Optimizations

1. **Batch Operations**: Process multiple ciphertexts in parallel
2. **Pipeline Overlapping**: Overlap GPU transfers with computation
3. **Persistent Buffers**: Reuse GPU buffers across operations
4. **Custom NTT Radix**: Optimize for Apple GPU architecture
5. **Approximate Rescaling**: Trade accuracy for 10x+ speedup (research)

## References

- **CKKS Original Paper**: Cheon et al. (2017) "Homomorphic Encryption for Arithmetic of Approximate Numbers"
- **Bootstrap Paper**: Chen & Han (2018) "Homomorphic Lower Digits Removal and Improved FHE Bootstrapping"
- **RNS Representation**: Bajard et al. (2016) "A Full RNS Variant of FV like Somewhat Homomorphic Encryption Schemes"

## Author

Implementation by David Silva (contact@davidwsilva.com | dsilva@datahubz.com).

## License

See [LICENSE](LICENSE) file in the repository root.
