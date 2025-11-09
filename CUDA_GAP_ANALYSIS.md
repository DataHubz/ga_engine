# CUDA Implementation - Gap Analysis

## Current Status (What Exists)

### ✅ Core Infrastructure
- **device.rs** - CUDA device management with cudarc
  - Device initialization
  - Buffer management (htod/dtoh)
  - Launch configuration

- **ntt.rs** - Complete NTT implementation
  - Forward/Inverse NTT
  - Pointwise multiplication
  - Precomputed twiddle factors
  - Runtime PTX compilation via cudarc

- **kernels/ntt.cu** - NTT CUDA kernels
  - Bit-reversal permutation
  - Forward NTT (Cooley-Tukey butterfly)
  - Inverse NTT (Gentleman-Sande butterfly)
  - Pointwise multiply
  - Scalar multiply
  - Modular arithmetic helpers

- **geometric.rs** - Geometric product (V1-related, not needed for V3)

- **mod.rs** - Module structure and feature flags

## Missing Components for V3 CUDA Bootstrap

### ❌ CKKS Operations (`ckks.rs`)
**Priority: CRITICAL**

Needed:
```rust
pub struct CudaCkksContext {
    device: Arc<CudaDeviceContext>,
    params: CliffordFHEParams,
    ntt_contexts: Vec<CudaNttContext>,
    rescale_inv_table: Vec<Vec<u64>>,
}

impl CudaCkksContext {
    // Core operations
    fn encode(&self, values: &[f64]) -> Vec<u64>;
    fn decode(&self, poly: &[u64]) -> Vec<f64>;
    fn encrypt(&self, pt: &[u64], pk: &PublicKey) -> Ciphertext;
    fn decrypt(&self, ct: &Ciphertext, sk: &SecretKey) -> Vec<u64>;

    // Arithmetic
    fn add(&self, ct1: &Ciphertext, ct2: &Ciphertext) -> Ciphertext;
    fn multiply(&self, ct1: &Ciphertext, ct2: &Ciphertext) -> Ciphertext;
    fn multiply_plain(&self, ct: &Ciphertext, pt: &[u64]) -> Ciphertext;

    // GPU rescaling (CRITICAL!)
    fn exact_rescale_gpu(&self, poly: &[u64], level: usize) -> Vec<u64>;
}
```

**Reference**: `src/clifford_fhe_v2/backends/gpu_metal/ckks.rs` (2183 lines)

### ❌ RNS Rescaling Kernel (`kernels/rns.cu`)
**Priority: CRITICAL**

Needed:
```cuda
// Exact CKKS rescaling with centered rounding
// C' = ⌊(C + q_last/2) / q_last⌋ mod Q'
__global__ void rns_exact_rescale(
    const unsigned long long* poly_in,   // [n × num_primes_in] flat layout
    unsigned long long* poly_out,        // [n × num_primes_out] flat layout
    const unsigned long long* moduli,    // [num_primes_in]
    const unsigned long long* qtop_inv,  // [num_primes_out]
    unsigned int n,
    unsigned int num_primes_in,
    unsigned int num_primes_out
);

// Helper: 128-bit modular multiplication (Russian peasant algorithm)
__device__ unsigned long long mul_mod_128(
    unsigned long long a,
    unsigned long long b,
    unsigned long long q
);
```

**Reference**: `src/clifford_fhe_v2/backends/gpu_metal/shaders/rns_fixed.metal` (324 lines)

### ❌ Rotation Operations (`rotation.rs`)
**Priority: HIGH**

Needed:
```rust
// Galois element computation
pub fn rotation_step_to_galois_element(step: i32, n: usize) -> usize;

// Compute Galois permutation map
pub fn compute_galois_map(galois_element: usize, n: usize) -> Vec<usize>;

// Apply rotation on GPU
pub fn apply_galois_automorphism_gpu(
    poly: &[u64],
    galois_element: usize,
    n: usize,
    num_primes: usize,
    device: &CudaDeviceContext
) -> Result<Vec<u64>, String>;
```

**Reference**: `src/clifford_fhe_v2/backends/gpu_metal/rotation.rs` (388 lines)

### ❌ Galois Automorphism Kernel (`kernels/galois.cu`)
**Priority: HIGH**

Needed:
```cuda
// Apply Galois automorphism σ_k: x → x^k
// Permutes polynomial coefficients according to Galois map
__global__ void galois_automorphism(
    const unsigned long long* poly_in,   // [n × num_primes] flat layout
    unsigned long long* poly_out,        // [n × num_primes] flat layout
    const unsigned int* galois_map,      // [n] permutation indices
    unsigned int n,
    unsigned int num_primes
);
```

**Reference**: `src/clifford_fhe_v2/backends/gpu_metal/shaders/galois.metal` (78 lines)

### ❌ Rotation Keys (`rotation_keys.rs`)
**Priority: HIGH**

Needed:
```rust
pub struct CudaRotationKeys {
    device: Arc<CudaDeviceContext>,
    keys: HashMap<usize, (Vec<Vec<u64>>, Vec<Vec<u64>>)>,  // Galois element → (rlk0, rlk1)
    base_w: u32,
    num_digits: usize,
    n: usize,
    num_primes: usize,
    level: usize,
}

impl CudaRotationKeys {
    fn generate(
        device: Arc<CudaDeviceContext>,
        sk: &SecretKey,
        rotation_steps: &[i32],
        params: &CliffordFHEParams,
        ntt_contexts: &[CudaNttContext],
        base_w: u32,
    ) -> Result<Self, String>;

    fn get_key_for_step(&self, step: i32) -> Option<&(Vec<Vec<u64>>, Vec<Vec<u64>>)>;
}
```

**Reference**: `src/clifford_fhe_v2/backends/gpu_metal/rotation_keys.rs` (662 lines)

### ❌ Bootstrap Implementation (`bootstrap.rs`)
**Priority: CRITICAL**

Needed:
```rust
// Hybrid: GPU multiply + CPU rescale
pub fn coeff_to_slot_cuda(
    ct: &CudaCiphertext,
    rot_keys: &CudaRotationKeys,
    ctx: &CudaCkksContext,
) -> Result<CudaCiphertext, String>;

pub fn slot_to_coeff_cuda(
    ct: &CudaCiphertext,
    rot_keys: &CudaRotationKeys,
    ctx: &CudaCkksContext,
) -> Result<CudaCiphertext, String>;

// Native: 100% GPU
pub fn coeff_to_slot_cuda_native(
    ct: &CudaCiphertext,
    rot_keys: &CudaRotationKeys,
    ctx: &CudaCkksContext,
) -> Result<CudaCiphertext, String>;

pub fn slot_to_coeff_cuda_native(
    ct: &CudaCiphertext,
    rot_keys: &CudaRotationKeys,
    ctx: &CudaCkksContext,
) -> Result<CudaCiphertext, String>;
```

**Reference**: `src/clifford_fhe_v2/backends/gpu_metal/bootstrap.rs` (1011 lines)

## Implementation Strategy

### Phase 1: RNS Rescaling (Foundation) 🔥
1. Create `kernels/rns.cu` with exact rescaling kernel
2. Add `exact_rescale_gpu()` to a minimal `ckks.rs`
3. Test with golden compare (bit-exact validation)

**Why first?** Rescaling is the critical GPU operation. If this works correctly, everything else follows.

### Phase 2: Basic CKKS Operations
1. Complete `ckks.rs` with encode/decode/encrypt/decrypt
2. Add multiply operations (uses NTT which already exists)
3. Test basic FHE pipeline

### Phase 3: Rotation Infrastructure
1. Create `rotation.rs` with Galois element computation
2. Create `kernels/galois.cu` for GPU rotation
3. Test rotation operations

### Phase 4: Rotation Keys
1. Create `rotation_keys.rs` with gadget decomposition
2. Generate rotation keys using CUDA NTT
3. Test key generation

### Phase 5: Bootstrap
1. Create `bootstrap.rs` with CoeffToSlot/SlotToCoeff
2. Implement hybrid version first (easier to debug)
3. Implement native version (100% GPU)
4. Test full bootstrap pipeline

## Lines of Code Estimate

| Component | Metal LOC | CUDA LOC (Est.) | Complexity |
|-----------|-----------|-----------------|------------|
| ckks.rs | 2183 | ~2000 | High |
| rotation.rs | 388 | ~350 | Medium |
| rotation_keys.rs | 662 | ~600 | Medium |
| bootstrap.rs | 1011 | ~950 | High |
| kernels/rns.cu | 324 | ~300 | High |
| kernels/galois.cu | 78 | ~80 | Low |
| **TOTAL** | **4646** | **~4280** | **High** |

## Testing Requirements

### Unit Tests
- [ ] CUDA NTT (forward/inverse) ✅ (already exists)
- [ ] RNS rescaling (golden compare)
- [ ] Rotation operations
- [ ] Key generation
- [ ] Bootstrap (hybrid)
- [ ] Bootstrap (native)

### Integration Tests
- [ ] Full FHE pipeline (encrypt → bootstrap → decrypt)
- [ ] Error validation (< 1e-2)
- [ ] Performance benchmarks (vs Metal)

### Examples Needed
- [ ] `test_cuda_ckks.rs` - Basic CKKS ops
- [ ] `test_cuda_rescale_golden_compare.rs` - Validate rescaling
- [ ] `test_cuda_rotation.rs` - Rotation test
- [ ] `test_cuda_bootstrap.rs` - Hybrid bootstrap
- [ ] `test_cuda_bootstrap_native.rs` - Native GPU bootstrap

## Feature Flags

Add to `Cargo.toml`:
```toml
v3-cuda = ["v3", "v2-gpu-cuda"]  # V3 bootstrap with CUDA GPU
```

## Success Metrics

✅ **Correctness**:
- GPU rescaling: 0 mismatches vs CPU
- Hybrid bootstrap error: < 1e-2
- Native bootstrap error: < 1e-2

✅ **Performance** (RTX 4090 target):
- CoeffToSlot: < 2s (vs ~4s on Metal M3 Max)
- Full Bootstrap: < 30s (vs ~60s on Metal)

✅ **Completeness**:
- All operations GPU-accelerated
- No CPU fallbacks in native mode
- Comprehensive test coverage

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 128-bit arithmetic in CUDA | High | Use Russian peasant multiplication |
| Memory layout differences | Medium | Match Metal's flat RNS layout |
| Kernel compilation issues | Low | Use runtime PTX compilation (cudarc) |
| No local NVIDIA GPU for testing | High | Test on RunPod.io with RTX 4090 |

## Next Steps

1. **Start with RNS rescaling** - This is the critical component
2. **Create minimal CKKS wrapper** - Just enough to test rescaling
3. **Test on RunPod** - Validate kernel compilation and execution
4. **Iterate based on results** - Fix issues, optimize, expand

---

**Estimated Effort**: 4-6 hours for Phase 1 (rescaling), 12-16 hours total for complete implementation.

**Recommendation**: Start with Phase 1 (RNS rescaling) to validate the CUDA setup on RunPod before committing to the full implementation.
