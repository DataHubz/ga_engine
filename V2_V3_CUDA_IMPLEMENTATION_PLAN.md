# V2/V3 CUDA GPU Implementation Plan

## Overview

Implement V2 CUDA GPU backend and V3 CUDA bootstrap, mirroring the Metal GPU implementation for NVIDIA GPUs on RunPod.io.

## Architecture

```
src/clifford_fhe_v2/backends/gpu_cuda/
├── mod.rs              # Module exports ✅ (exists)
├── device.rs           # CUDA device management ✅ (exists)
├── ntt.rs              # NTT operations ✅ (exists)
├── ckks.rs             # CKKS operations (TO CREATE)
├── rotation.rs         # Galois automorphisms (TO CREATE)
├── rotation_keys.rs    # Key switching keys (TO CREATE)
├── bootstrap.rs        # CoeffToSlot/SlotToCoeff (TO CREATE)
└── kernels/            # CUDA kernels (TO CREATE)
    ├── ntt.cu          # NTT kernels
    ├── rns.cu          # RNS rescaling
    ├── galois.cu       # Rotation kernels
    └── gadget.cu       # Key switching
```

## Feature Flags

Add to `Cargo.toml`:

```toml
v2-gpu-cuda = ["v2", "cudarc"]     # Already exists
v3-cuda = ["v3", "v2-gpu-cuda"]    # TO ADD - V3 bootstrap with CUDA
```

## Implementation Phases

### Phase 1: Core CUDA Infrastructure ✅
- [x] Basic device management (`device.rs`)
- [x] NTT operations (`ntt.rs`)
- [ ] Compile CUDA kernels to PTX
- [ ] Load PTX kernels in Rust

### Phase 2: CKKS Operations
File: `src/clifford_fhe_v2/backends/gpu_cuda/ckks.rs`

Mirror Metal GPU implementation:
- `CudaCkksContext` struct
- Encode/Decode
- Encrypt/Decrypt
- Add/Multiply
- **GPU Rescaling** (RNS DRLMQ with centered rounding)

### Phase 3: Rotation Operations
File: `src/clifford_fhe_v2/backends/gpu_cuda/rotation.rs`

- Galois element computation
- Rotation permutation on GPU
- `rotate_by_steps()` operation

### Phase 4: Rotation Keys
File: `src/clifford_fhe_v2/backends/gpu_cuda/rotation_keys.rs`

- Gadget decomposition key generation
- Key switching with GPU NTT
- Similar to `MetalRotationKeys` but for CUDA

### Phase 5: Bootstrap Implementation
File: `src/clifford_fhe_v2/backends/gpu_cuda/bootstrap.rs`

Functions:
- `coeff_to_slot_cuda()` - Hybrid (GPU + CPU rescale)
- `slot_to_coeff_cuda()` - Hybrid
- `coeff_to_slot_cuda_native()` - 100% GPU
- `slot_to_coeff_cuda_native()` - 100% GPU

### Phase 6: CUDA Kernels

#### `kernels/ntt.cu`
```cuda
// Forward NTT kernel
__global__ void ntt_forward(
    uint64_t* poly,
    const uint64_t* omega_powers,
    uint64_t q,
    uint32_t n,
    uint32_t log_n
);

// Inverse NTT kernel
__global__ void ntt_inverse(
    uint64_t* poly,
    const uint64_t* omega_inv_powers,
    uint64_t n_inv,
    uint64_t q,
    uint32_t n,
    uint32_t log_n
);
```

#### `kernels/rns.cu`
```cuda
// Exact rescaling with centered rounding
// Uses 128-bit modular arithmetic (Russian peasant multiplication)
__global__ void rns_exact_rescale(
    const uint64_t* poly_in,
    uint64_t* poly_out,
    const uint64_t* moduli,
    const uint64_t* qtop_inv,
    uint32_t n,
    uint32_t num_primes_in,
    uint32_t num_primes_out
);

// Helper: 128-bit modular multiplication
__device__ uint64_t mul_mod_128(uint64_t a, uint64_t b, uint64_t q);
```

#### `kernels/galois.cu`
```cuda
// Apply Galois automorphism (rotation)
__global__ void galois_automorphism(
    const uint64_t* poly_in,
    uint64_t* poly_out,
    uint32_t galois_element,
    uint32_t n,
    uint32_t num_primes
);
```

#### `kernels/gadget.cu`
```cuda
// Gadget decomposition for key switching
__global__ void gadget_decompose(
    const uint64_t* poly_in,
    uint64_t* digits_out,
    uint32_t base_w,
    uint32_t num_digits,
    uint32_t n,
    uint32_t num_primes
);
```

### Phase 7: Testing

Create examples:
- `examples/test_cuda_ckks.rs` - Basic CKKS operations
- `examples/test_cuda_rotation.rs` - Rotation test
- `examples/test_cuda_bootstrap.rs` - Hybrid bootstrap
- `examples/test_cuda_bootstrap_native.rs` - Native GPU bootstrap
- `examples/test_cuda_golden_compare.rs` - Rescaling validation

### Phase 8: Documentation

Create:
- `CUDA_SETUP.md` - RunPod setup instructions
- `V3_CUDA_BOOTSTRAP.md` - CUDA bootstrap guide
- Update `ARCHITECTURE.md`
- Update `COMMANDS.md`

## Key Differences: Metal vs CUDA

| Aspect | Metal (Apple) | CUDA (NVIDIA) |
|--------|---------------|---------------|
| Language | Metal Shading Language | CUDA C++ |
| Compilation | Runtime (source → binary) | Ahead-of-time (CUDA → PTX → binary) |
| Memory | Unified (shared CPU/GPU) | Discrete (explicit transfers) |
| API | metal-rs crate | cudarc crate |
| Thread Model | Threads in threadgroups | Threads in blocks |
| Max Threads/Block | Device-specific (~1024) | 1024 (Compute 7.5+) |

## CUDA Kernel Compilation Strategy

**Option 1: PTX (Portable)**
```bash
# Compile CUDA to PTX intermediate representation
nvcc -ptx kernels/ntt.cu -o kernels/ntt.ptx
nvcc -ptx kernels/rns.cu -o kernels/rns.ptx
```

Embed PTX in Rust:
```rust
const NTT_PTX: &str = include_str!("kernels/ntt.ptx");
let module = device.load_ptx(NTT_PTX.into(), "ntt", &[])?;
```

**Option 2: Runtime Compilation (cudarc)**
```rust
// Compile CUDA source at runtime
let ptx = compile_ptx("
    __global__ void kernel(...) { ... }
")?;
let module = device.load_ptx(ptx, "module", &[])?;
```

We'll use **Option 1** for better error messages and compile-time verification.

## Performance Targets (RTX 4090 on RunPod)

Based on Metal GPU benchmarks (M3 Max):

| Operation | Metal M3 Max | CUDA RTX 4090 (Target) |
|-----------|--------------|------------------------|
| NTT (1024) | ~0.5ms | ~0.2ms (2.5× faster) |
| Rotation | ~50ms | ~20ms (2.5× faster) |
| CoeffToSlot (9 levels) | ~4s | ~1.6s (2.5× faster) |
| Full Bootstrap | ~60s | ~24s (2.5× faster) |

RTX 4090 has ~3× more CUDA cores than M3 Max GPU cores, so we expect 2-3× speedup.

## RunPod Setup Requirements

### Docker Image
Use NVIDIA CUDA image:
```dockerfile
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y curl build-essential
# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
```

### RunPod Configuration
- **GPU**: RTX 4090 (48GB VRAM recommended)
- **CUDA**: 12.3+
- **Storage**: 50GB+ for compilation
- **Region**: Any with RTX 4090 availability

### Build Commands
```bash
# Clone repo
git clone <repo> && cd ga_engine

# Compile CUDA kernels
cd src/clifford_fhe_v2/backends/gpu_cuda/kernels
nvcc -ptx *.cu

# Build with CUDA features
cargo build --release --features v2,v2-gpu-cuda,v3

# Test
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap_native
```

## Implementation Checklist

- [ ] Phase 1: CUDA kernel compilation setup
- [ ] Phase 2: CudaCkksContext implementation
- [ ] Phase 3: Rotation operations
- [ ] Phase 4: Rotation keys with gadget decomposition
- [ ] Phase 5: Bootstrap (hybrid + native)
- [ ] Phase 6: All CUDA kernels written and compiled
- [ ] Phase 7: Test suite
- [ ] Phase 8: Documentation
- [ ] Phase 9: RunPod testing and validation

## Success Criteria

✅ Hybrid bootstrap: Error < 1e-2
✅ Native bootstrap: Error < 1e-2 (same as Metal)
✅ GPU rescaling: Bit-exact with CPU (0 mismatches)
✅ Performance: 2-3× faster than Metal on RTX 4090

## Author

Implementation by David Silva (contact@davidwsilva.com | dsilva@datahubz.com).

## References

- Metal GPU implementation: `src/clifford_fhe_v2/backends/gpu_metal/`
- V3 Bootstrap guide: `V3_BOOTSTRAP.md`
- cudarc documentation: https://docs.rs/cudarc/
