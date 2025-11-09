# GA Engine Architecture

## Overview

The GA Engine implements Clifford Homomorphic Encryption with **three parallel implementations**:

- **V1:** Research prototype, stable (for Paper 1)
- **V2:** Metal GPU-accelerated backend (active development)
- **V3:** Bootstrap-optimized implementation (latest)

## Directory Structure

```
src/
├── clifford_fhe/              # V1 - STABLE (Paper 1)
│   ├── ckks_rns.rs            # Basic RNS-CKKS
│   ├── geometric_product_rns.rs
│   └── ...
│
├── clifford_fhe_v2/           # V2 - Metal GPU Backend
│   ├── backends/
│   │   ├── cpu_optimized/     # CPU with NTT optimization
│   │   │   ├── ckks.rs
│   │   │   ├── ntt.rs
│   │   │   └── keys.rs
│   │   │
│   │   └── gpu_metal/         # Apple Metal GPU backend
│   │       ├── ckks.rs        # Core CKKS operations
│   │       ├── ntt.rs         # GPU NTT transforms
│   │       ├── keys.rs        # Key generation (CPU)
│   │       ├── bootstrap.rs   # Bootstrap transformations ⭐
│   │       ├── device.rs      # Metal device management
│   │       └── shaders/
│   │           ├── ntt.metal         # NTT kernels
│   │           ├── pointwise.metal   # Element-wise ops
│   │           ├── galois.metal      # Galois automorphism
│   │           ├── gadget.metal      # Key switching
│   │           └── rns_fixed.metal   # GPU rescaling ⭐
│   │
│   └── params.rs              # Parameter sets
│
└── clifford_fhe_v3/           # V3 - Bootstrap Optimized
    ├── bootstrapping/         # Bootstrap operations
    │   ├── coeff_to_slot.rs   # CoeffToSlot transform
    │   ├── slot_to_coeff.rs   # SlotToCoeff transform
    │   ├── eval_mod.rs        # Modular reduction
    │   └── keys.rs            # Rotation key generation
    │
    ├── params.rs              # V3-optimized parameters
    └── prime_gen.rs           # Dynamic NTT-friendly prime generation
```

## Feature Flags

Defined in `Cargo.toml`:

```toml
[features]
default = ["v1"]

# Version selection
v1 = []                        # Stable research implementation
v2 = []                        # Metal GPU backend
v3 = []                        # Bootstrap optimized

# V2 backends (requires v2)
v2-cpu-optimized = ["v2"]      # CPU with NTT
v2-gpu-metal = ["v2", "metal"] # Apple Metal GPU
```

## Bootstrap Implementation

### V2 Metal GPU Bootstrap

Located in `src/clifford_fhe_v2/backends/gpu_metal/bootstrap.rs`

**Two versions:**

1. **Hybrid** (GPU multiply + CPU rescale)
   - Functions: `coeff_to_slot_gpu()`, `slot_to_coeff_gpu()`
   - Rescaling: CPU BigInt CRT
   - Status: ✅ Production stable (~3.6e-3 error)

2. **Native** (100% GPU)
   - Functions: `coeff_to_slot_gpu_native()`, `slot_to_coeff_gpu_native()`
   - Rescaling: GPU via `rns_fixed.metal` shader
   - Status: ✅ Production stable (~3.6e-3 error)

### V3 Bootstrap

Located in `src/clifford_fhe_v3/bootstrapping/`

- **CPU-only implementation** with correct scale management
- Used as reference for V2 Metal GPU development
- Status: ✅ Working

See [V3_BOOTSTRAP.md](V3_BOOTSTRAP.md) for detailed implementation guide.

## Key Components

### RNS (Residue Number System)

All versions use multi-prime RNS representation for efficiency:
- **V1/V2:** Fixed prime sets
- **V3:** Dynamic NTT-friendly prime generation

Example: 20 primes for N=1024 bootstrap
- 1× ~60-bit prime
- 19× ~45-bit primes

### NTT (Number Theoretic Transform)

Fast polynomial multiplication:
- **V1:** Schoolbook multiplication (O(n²))
- **V2 CPU:** Harvey butterfly NTT (O(n log n))
- **V2 GPU:** Metal shader NTT with optimized memory access
- **V3:** Uses V2's NTT implementation

### Metal GPU Shaders

Located in `src/clifford_fhe_v2/backends/gpu_metal/shaders/`:

| Shader | Purpose | Key Feature |
|--------|---------|-------------|
| `ntt.metal` | Forward/inverse NTT | Optimized for Apple GPU |
| `pointwise.metal` | Element-wise multiplication | Fused ops |
| `galois.metal` | Galois automorphism | Rotation permutation |
| `gadget.metal` | Gadget decomposition | Key switching |
| `rns_fixed.metal` | **Exact rescaling** | Russian peasant mul_mod_128 ⭐ |

### GPU Rescaling (`rns_fixed.metal`)

**The Innovation:**

Previous GPU rescaling attempts failed (~385k errors) due to:
- ❌ Rounding in wrong domain
- ❌ 128-bit overflow in modular multiplication

**Solution:**

```metal
// Russian peasant multiplication (no 128-bit overflow)
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

**Validation:**
- Golden compare test: ✅ 0 mismatches (bit-exact with CPU)
- Bootstrap test: ✅ 3.6e-3 error (same as hybrid)

## Testing

### Unit Tests
```bash
# V1 tests
cargo test --features v1

# V2 CPU tests
cargo test --features v2,v2-cpu-optimized

# V2 Metal GPU tests
cargo test --features v2,v2-gpu-metal

# V3 tests
cargo test --features v3
```

### Bootstrap Tests
```bash
# Hybrid (GPU + CPU rescale)
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap

# Native (100% GPU)
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native

# V3 CPU reference
cargo run --release --features v2,v2-gpu-metal,v3 --example test_v3_metal_bootstrap_correct
```

### Validation Tests
```bash
# GPU rescaling golden compare
cargo run --release --features v2,v2-gpu-metal,v3 --example test_rescale_golden_compare

# Layout conversion test
cargo run --release --features v2,v2-gpu-metal,v3 --example test_multiply_rescale_layout
```

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing instructions.

## Performance (Apple M3 Max)

### Bootstrap Comparison

| Version | Total Time | Error | Notes |
|---------|-----------|-------|-------|
| **V3 CPU** | ~70s | 3.6e-3 | Reference implementation |
| **V2 Hybrid** | ~65s | 3.6e-3 | GPU multiply + CPU rescale |
| **V2 Native** | ~60s | 3.6e-3 | **100% GPU** ⭐ |

### Per-Operation Breakdown

| Operation | V2 Native | Notes |
|-----------|-----------|-------|
| Key Generation | ~73s | CPU only |
| Encryption | ~175ms | GPU |
| CoeffToSlot (9 levels) | ~50s | GPU |
| SlotToCoeff (9 levels) | ~12s | GPU |
| Decryption | ~11ms | GPU |

See [BENCHMARKS.md](BENCHMARKS.md) for detailed performance analysis.

## Development Guidelines

### V1 - DO NOT MODIFY
- ❌ Frozen for Paper 1 publication
- ✅ Only critical bug fixes allowed
- ✅ Must maintain exact reproducibility

### V2 - Active Development
- ✅ Metal GPU optimization
- ✅ Bootstrap implementation
- ✅ Performance tuning
- ✅ New shader development

### V3 - Bootstrap Research
- ✅ Parameter optimization
- ✅ Algorithm improvements
- ✅ Reference implementations

## Documentation

Core docs (kept):
- [README.md](README.md) - Project overview
- [V3_BOOTSTRAP.md](V3_BOOTSTRAP.md) - Bootstrap implementation guide
- [BENCHMARKS.md](BENCHMARKS.md) - Performance measurements
- [COMMANDS.md](COMMANDS.md) - Command reference
- [INSTALLATION.md](INSTALLATION.md) - Setup instructions
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing procedures

## Key Achievements

### V1 (2024)
- ✅ First working Clifford FHE implementation
- ✅ All 7 geometric operations
- ✅ Paper 1 published

### V2 (2024)
- ✅ Metal GPU backend operational
- ✅ NTT optimization (10x+ speedup)
- ✅ Rotation operations working
- ✅ **100% GPU bootstrap** (November 2024) ⭐

### V3 (2024)
- ✅ Bootstrap-optimized parameters
- ✅ Dynamic prime generation
- ✅ Reference implementation for V2 validation

## Future Work

1. **Batching**: Process multiple ciphertexts in parallel
2. **EvalMod**: Complete modular reduction for full bootstrap
3. **Optimization**: Pipeline overlapping, persistent GPU buffers
4. **Portability**: Vulkan backend for cross-platform GPU support

## Author

Implementation by David Silva (contact@davidwsilva.com | dsilva@datahubz.com).

## License

See [LICENSE](LICENSE) file in the repository root.
