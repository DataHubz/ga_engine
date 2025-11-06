# GA Engine Feature Flags

Complete reference for feature flags and build configurations.

## Core Feature Flags

### FHE Version Selection

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `v1` | V1 baseline reference implementation | None |
| `v2` | V2 CPU-optimized backend (NTT + Rayon) | None |
| `v3` | V3 bootstrapping and SIMD batching | Requires `v2` |

### GPU Backend Selection

| Feature | Description | Platform | Dependencies |
|---------|-------------|----------|--------------|
| `v2-gpu-metal` | Metal GPU acceleration (V2/V3) | macOS (Apple Silicon) | `metal`, `objc` |
| `v2-gpu-cuda` | CUDA GPU acceleration (V2/V3) | Linux/Windows (NVIDIA) | `cudarc` |

### Optional Modules

| Feature | Description | Use Case | Dependencies |
|---------|-------------|----------|--------------|
| `lattice-reduction` | Lattice reduction for security analysis | CPU-only cryptanalysis | `nalgebra`, `nalgebra-lapack`, `blas-src` |

## Default Features

By default, the following features are enabled:

```toml
default = ["f64", "nd", "v1", "lattice-reduction"]
```

This provides:
- V1 FHE implementation (stable baseline)
- Lattice reduction for security analysis
- Full functionality on local development machines

## Common Build Patterns

### Local Development (Full Features)

```bash
# Use default features (includes lattice-reduction)
cargo build --release

# Or explicitly:
cargo build --release --features v1,v2,v3,lattice-reduction

# Run all tests
cargo test --release --features v1,v2,v3,lattice-reduction
```

### Cloud GPU Instances (CUDA, No Lattice)

```bash
# Build without lattice-reduction to avoid netlib-src/simba issues
cargo build --release --features v2-gpu-cuda --no-default-features

# Run CUDA benchmarks
cargo test --release --features v2-gpu-cuda --no-default-features \
  --test test_geometric_operations_cuda -- --nocapture
```

### Local macOS (Metal GPU + Lattice)

```bash
# Build with Metal GPU and lattice reduction
cargo build --release --features v2,v2-gpu-metal,v3,lattice-reduction

# Run Metal GPU tests
cargo test --release --features v2-gpu-metal,lattice-reduction \
  --test test_geometric_operations_metal -- --nocapture
```

### V2 CPU Only (No GPU, No Lattice)

```bash
# Build V2 CPU backend only
cargo build --release --features v2 --no-default-features

# Run V2 CPU tests
cargo test --release --features v2 --no-default-features
```

## Feature Flag Dependencies

### Automatic Dependencies

When you enable certain features, they automatically enable their dependencies:

- `v3` → automatically enables `v2`
- `v2-gpu-metal` → automatically enables `v2`, `metal`, `objc`
- `v2-gpu-cuda` → automatically enables `v2`, `cudarc`
- `lattice-reduction` → automatically enables `nalgebra`, `nalgebra-lapack`, `blas-src`

### Optional Dependencies

These dependencies are only compiled when their corresponding features are enabled:

| Dependency | Feature Required | Purpose |
|------------|------------------|---------|
| `nalgebra` | `lattice-reduction` | Linear algebra for lattice reduction |
| `nalgebra-lapack` | `lattice-reduction` | LAPACK bindings for QR decomposition |
| `blas-src` | `lattice-reduction` | BLAS backend (Accelerate on macOS, netlib elsewhere) |
| `metal` | `v2-gpu-metal` | Metal GPU compute API |
| `objc` | `v2-gpu-metal` | Objective-C runtime for Metal |
| `cudarc` | `v2-gpu-cuda` | CUDA GPU compute API |

## Why Feature Flags?

### Problem: Build Issues on Cloud GPU Instances

When building on cloud GPU instances (RunPod, Lambda Labs, etc.), lattice reduction dependencies cause issues:

1. **netlib-src**: Compiles BLAS/LAPACK from Fortran source (very slow, 1+ hours)
2. **simba**: Floods build output with thousands of nvptx cfg warnings
3. **Not needed**: Lattice reduction is CPU-only security analysis, not used for FHE GPU operations

### Solution: Optional lattice-reduction Feature

By making lattice reduction optional:

- **Local development**: Include lattice-reduction (default behavior)
- **Cloud GPU builds**: Omit lattice-reduction (fast builds, clean output)
- **FHE operations**: Unaffected (V1/V2/V3 work with or without lattice-reduction)

## Test Coverage

| Component | Test Count | Command |
|-----------|------------|---------|
| V1 Unit Tests | 31 | `cargo test --lib --features v1` |
| V2 Unit Tests | 127 | `cargo test --lib --features v2 --no-default-features` |
| V3 Unit Tests | 100 | `cargo test --lib --features v2,v3 --no-default-features` |
| Lattice Reduction | 95 | `cargo test --lib lattice_reduction --features lattice-reduction` |
| **Total** | **353** | `cargo test --features v1,v2,v3,lattice-reduction` |

## Troubleshooting

### Build stuck at netlib-src compilation

**Problem**: Accidentally included `lattice-reduction` feature on cloud GPU instance

**Solution**:
```bash
cargo clean
cargo build --release --features v2-gpu-cuda --no-default-features
```

### Lattice reduction tests not found

**Problem**: Built without `lattice-reduction` feature

**Solution**:
```bash
cargo test --lib lattice_reduction --features lattice-reduction
```

### undefined reference to BLAS symbols

**Problem**: Enabled `lattice-reduction` but BLAS backend not available

**Solution** (Linux):
```bash
# Install system BLAS (optional, netlib-src will compile from source if not found)
sudo apt-get install libblas-dev liblapack-dev
cargo clean && cargo build --release --features lattice-reduction
```

**Solution** (macOS):
```bash
# Xcode Command Line Tools includes Accelerate framework
xcode-select --install
cargo clean && cargo build --release --features lattice-reduction
```

## Performance Impact

### Compile Time

| Configuration | Compile Time (from clean) | Reason |
|---------------|--------------------------|--------|
| `--features v2 --no-default-features` | ~2 minutes | No Fortran compilation |
| `--features v2,lattice-reduction` | ~5-10 minutes | netlib-src Fortran compilation (if no system BLAS) |
| `--features v2 --no-default-features` (incremental) | ~10 seconds | Cached dependencies |

### Runtime Performance

**No impact**: The `lattice-reduction` feature only affects what modules are compiled, not runtime performance of FHE operations. V2/V3 FHE operations have identical performance with or without lattice-reduction.

## See Also

- [COMMANDS.md](COMMANDS.md) - Complete command reference
- [CUDA_BUILD_NOTES.md](CUDA_BUILD_NOTES.md) - CUDA build configuration details
- [RUNPOD_QUICKSTART.md](RUNPOD_QUICKSTART.md) - RunPod GPU benchmark guide
- [INSTALLATION.md](INSTALLATION.md) - Setup guide and system requirements
