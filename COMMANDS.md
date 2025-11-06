# Command Reference

Complete reference of all build, test, example, and benchmark commands for GA Engine.

## Important: Feature Flags

GA Engine uses feature flags to control which components are compiled:

**Default features** (local development):
```bash
cargo build --release  # Includes: v1, lattice-reduction
```

**Cloud GPU instances** (works with lattice-reduction, CMake 4.0 fix applied):
```bash
cargo build --release --features v2-gpu-cuda  # lattice-reduction works now
```

**Key points**:
- `lattice-reduction` is included by default and now works with CMake 4.0+
- CMake 4.0 compatibility fixed via `.cargo/config.toml` (see [CMAKE_FIX.md](CMAKE_FIX.md))
- Lattice reduction is CPU-only security analysis, not needed for FHE operations
- See [FEATURE_FLAGS.md](FEATURE_FLAGS.md) for detailed explanation

## Table of Contents

- [Important: Feature Flags](#important-feature-flags)
- [Installation](#installation)
- [V1: Baseline Reference](#v1-baseline-reference)
- [V2: CPU Optimized](#v2-cpu-optimized)
- [V2: Metal GPU](#v2-metal-gpu)
- [V2: CUDA GPU](#v2-cuda-gpu)
- [V3: Bootstrapping](#v3-bootstrapping)
- [Lattice Reduction](#lattice-reduction)
- [All Versions Combined](#all-versions-combined)
- [Quick Reference Tables](#quick-reference-tables)
- [Troubleshooting](#troubleshooting)

## Installation

### Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version  # Verify version 1.75+
```

### Clone Repository
```bash
git clone https://github.com/davidwilliamsilva/ga_engine.git
cd ga_engine
```

## V1: Baseline Reference

### Build V1
```bash
# Development build (without lattice-reduction, RECOMMENDED)
cargo build --features f64,nd,v1 --no-default-features

# Release build (optimized, RECOMMENDED)
cargo build --release --features f64,nd,v1 --no-default-features
```

### Test V1
```bash
# Run all V1 unit tests (31 tests)
cargo test --lib --features f64,nd,v1 --no-default-features

# Run V1 integration tests
cargo test --test clifford_fhe_integration_tests --features f64,nd,v1 --no-default-features -- --nocapture

# Run comprehensive geometric operations test suite (all 7 operations)
cargo test --test test_geometric_operations --features f64,nd,v1 --no-default-features -- --nocapture

# Run isolated operation tests (individual tests for clean output)
cargo test --test test_clifford_operations_isolated test_key_generation --features f64,nd,v1 --no-default-features -- --nocapture
cargo test --test test_clifford_operations_isolated test_encryption_decryption --features f64,nd,v1 --no-default-features -- --nocapture
cargo test --test test_clifford_operations_isolated test_reverse --features f64,nd,v1 --no-default-features -- --nocapture
cargo test --test test_clifford_operations_isolated test_geometric_product --features f64,nd,v1 --no-default-features -- --nocapture
cargo test --test test_clifford_operations_isolated test_wedge_product --features f64,nd,v1 --no-default-features -- --nocapture
cargo test --test test_clifford_operations_isolated test_inner_product --features f64,nd,v1 --no-default-features -- --nocapture
cargo test --test test_clifford_operations_isolated test_rotation --features f64,nd,v1 --no-default-features -- --nocapture
cargo test --test test_clifford_operations_isolated test_projection --features f64,nd,v1 --no-default-features -- --nocapture
cargo test --test test_clifford_operations_isolated test_rejection --features f64,nd,v1 --no-default-features -- --nocapture

# Run all V1 tests
cargo test --features f64,nd,v1 --no-default-features
```

### Examples V1
```bash
# Encrypted 3D classification demo (main application)
cargo run --release --features f64,nd,v1 --no-default-features --example encrypted_3d_classification

# Basic FHE encryption/decryption
cargo run --release --features f64,nd,v1 --no-default-features --example clifford_fhe_basic
```

### Performance V1
```bash
# Actual performance (Apple M3 Max, 14-core):
# - Geometric product: 11.42 seconds
# - Full network inference: 308.3 seconds (27 operations)
# - Error: <1.30e-10
```

## V2: CPU Optimized

### Build V2 CPU
```bash
# Development build (without lattice-reduction, RECOMMENDED)
cargo build --features f64,nd,v2 --no-default-features

# Release build (optimized, RECOMMENDED)
cargo build --release --features f64,nd,v2 --no-default-features
```

### Test V2 CPU
```bash
# Run all V2 unit tests (132 tests, <1 second)
cargo test --lib --features f64,nd,v2 --no-default-features

# Run specific module tests
cargo test --lib clifford_fhe_v2::backends::cpu_optimized::ntt --features f64,nd,v2 --no-default-features -- --nocapture
cargo test --lib clifford_fhe_v2::backends::cpu_optimized::rns --features f64,nd,v2 --no-default-features -- --nocapture
cargo test --lib clifford_fhe_v2::backends::cpu_optimized::ckks --features f64,nd,v2 --no-default-features -- --nocapture
cargo test --lib clifford_fhe_v2::backends::cpu_optimized::keys --features f64,nd,v2 --no-default-features -- --nocapture
cargo test --lib clifford_fhe_v2::backends::cpu_optimized::multiplication --features f64,nd,v2 --no-default-features -- --nocapture
cargo test --lib clifford_fhe_v2::backends::cpu_optimized::geometric --features f64,nd,v2 --no-default-features -- --nocapture

# Run V2 geometric operations integration test
cargo test --test test_geometric_operations_v2 --features f64,nd,v2 --no-default-features -- --nocapture
```

### Examples V2 CPU
```bash
# Encrypted 3D classification (38x faster than V1)
cargo run --release --features f64,nd,v2 --no-default-features --example encrypted_3d_classification
```

### Performance V2 CPU
```bash
# Actual performance (Apple M3 Max, 14-core):
# - Geometric product: 0.30s (38x faster than V1)
# - Rotation (depth-2): 0.43s
# - Wedge product: 0.54s
# - Inner product: 0.53s
# - Projection (depth-3): 0.67s
# - Rejection (depth-3): 0.67s
# - Keygen: 0.01s
# - Error: <9.67e-8 (all operations)
```

## V2: Metal GPU

### Build V2 Metal
```bash
# Install Xcode Command Line Tools (macOS only)
xcode-select --install

# Build with Metal support (RECOMMENDED: without lattice-reduction)
cargo build --release --features f64,nd,v2-gpu-metal --no-default-features
```

### Test V2 Metal
```bash
# Run Metal GPU geometric operations test (includes benchmarking)
cargo test --release --features f64,nd,v2-gpu-metal --no-default-features --test test_geometric_operations_metal -- --nocapture
```

### Performance V2 Metal
```bash
# Actual performance (Apple M3 Max GPU):
# - Geometric product: 33ms mean (30ms min, 37ms max)
# - Speedup: 346× vs V1 (11.42s → 0.033s)
# - Speedup: 9.1× vs V2 CPU (0.30s → 0.033s)
# - Throughput: 30.3 operations/second
# - Standard deviation: 2.6ms (7.8% CV)
# - Statistical confidence: High (n=10 iterations)
```

## V2: CUDA GPU

### Build V2 CUDA
```bash
# Verify CUDA installation
nvcc --version

# Set CUDA path if needed
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Build with CUDA support (RECOMMENDED for cloud GPU instances)
cargo build --release --features f64,nd,v2-gpu-cuda --no-default-features
```

### Test V2 CUDA
```bash
# Run CUDA GPU geometric operations test (RECOMMENDED for cloud instances)
cargo test --release --features f64,nd,v2-gpu-cuda --no-default-features --test test_geometric_operations_cuda -- --nocapture
```

### Performance V2 CUDA
```bash
# Actual performance (NVIDIA RTX 5090):
# - Geometric product: 5.7ms mean (5ms min, 5ms max)
# - Speedup: 2,002× vs V1 (11.42s → 5.7ms)
# - Speedup: 77× vs V2 CPU (441ms → 5.7ms)
# - Ratio: 0.17× vs Metal GPU (5.7ms vs 34ms, CUDA is 6× faster)
# - Throughput: 174.8 operations/second
# - Standard deviation: 0.08ms (1.4% CV)
# - Statistical confidence: High (n=10 iterations)
# - Hardware: NVIDIA GeForce RTX 5090
```

## V3: Bootstrapping

### Build V3
```bash
# Build V3 with V2 backend
cargo build --release --features v2,v3
```

### Test V3
```bash
# Run all V3 unit tests (52 tests, 100% passing)
cargo test --lib --features v2,v3 clifford_fhe_v3

# Run full test suite (V1 + V2 + V3 + lattice-reduction = 249 tests)
cargo test --lib --features v2,v3

# Run V3 bootstrapping tests
cargo test --lib clifford_fhe_v3::bootstrapping --features v2,v3 -- --nocapture

# Run SIMD batching tests
cargo test --lib clifford_fhe_v3::batched --features v2,v3 -- --nocapture

# Run rotation tests
cargo test --lib clifford_fhe_v3::bootstrapping::rotation --features v2,v3 -- --nocapture

# Run CoeffToSlot/SlotToCoeff tests
cargo test --lib clifford_fhe_v3::bootstrapping::coeff_to_slot --features v2,v3 -- --nocapture
cargo test --lib clifford_fhe_v3::bootstrapping::slot_to_coeff --features v2,v3 -- --nocapture

# Run extraction tests (Pattern A mask-only approach)
cargo test --lib clifford_fhe_v3::batched::extraction --features v2,v3 -- --nocapture
```

### Examples V3
```bash
# V3 bootstrap simple test
cargo run --release --features v2,v3 --example test_v3_bootstrap_simple

# SIMD batching demonstration (512x throughput)
cargo run --release --features v2,v3 --example test_batching

# Medical imaging: encrypted deep GNN
cargo run --release --features v2,v3 --example medical_imaging_encrypted
```

### Performance V3
```bash
# Expected performance:
# - Bootstrap (CPU): ~1 second per ciphertext
# - Bootstrap (GPU, projected): ~500ms per ciphertext
# - SIMD batch (512 samples): 0.656ms per sample (19,817x faster than V1)
# - Deep GNN throughput: 1,524 operations/second
```

## Lattice Reduction

Lattice reduction is used for **security analysis** (cryptanalysis) of the FHE scheme. It is **not required** for FHE operations.

**Note**: CMake 4.0 compatibility has been fixed via `.cargo/config.toml`. The lattice-reduction feature now builds successfully on all platforms. See [CMAKE_FIX.md](CMAKE_FIX.md) for details.

### Build Lattice Reduction
```bash
# With lattice reduction (default, now works with CMake 4.0+)
cargo build --release --features lattice-reduction

# Or build with GPU backends (lattice-reduction included by default)
cargo build --release --features v2-gpu-cuda
cargo build --release --features v2-gpu-metal
```

### Test Lattice Reduction
```bash
# Run all lattice reduction tests (included in full suite)
cargo test --lib lattice_reduction

# Run specific module tests
cargo test --lib lattice_reduction::stable_gso --features lattice-reduction
cargo test --lib lattice_reduction::bkz_stable --features lattice-reduction
cargo test --lib lattice_reduction::ga_lll --features lattice-reduction
cargo test --lib lattice_reduction::enumeration --features lattice-reduction
```

### Examples Lattice Reduction
```bash
# Lattice reduction examples (all require lattice-reduction feature)
cargo run --release --features lattice-reduction --example test_stable_bkz
cargo run --release --features lattice-reduction --example test_lll
cargo run --release --features lattice-reduction --example benchmark_lll_comparison
```

## All Versions Combined

### Build All
```bash
# Build all versions (V1, V2) without lattice reduction (RECOMMENDED)
cargo build --release --features f64,nd,v1,v2 --no-default-features

# Build with GPU support
cargo build --release --features f64,nd,v1,v2,v2-gpu-metal --no-default-features  # macOS
cargo build --release --features f64,nd,v1,v2,v2-gpu-cuda --no-default-features   # Linux/Cloud GPU
```

### Test All
```bash
# Run all tests across all versions (without lattice reduction, RECOMMENDED)
cargo test --features f64,nd,v1,v2 --no-default-features

# Run all tests including GPU backends
cargo test --features f64,nd,v1,v2,v2-gpu-metal --no-default-features  # macOS
cargo test --features f64,nd,v1,v2,v2-gpu-cuda --no-default-features   # Linux/Cloud GPU

# Run with lattice reduction (requires working CMake)
cargo test --features v1,v2,lattice-reduction
```

### Benchmarks
```bash
# V1 vs V2 comparison benchmark
cargo bench --bench v1_vs_v2_benchmark --features v1,v2

# Run with specific backend
cargo bench --features v2-gpu-cuda
```

### Documentation
```bash
# Generate and open documentation
cargo doc --open --features v2,v3

# Generate documentation for specific version
cargo doc --open --features v1
cargo doc --open --features v2
cargo doc --open --features v2,v3
```

## Quick Reference Tables

### Feature Flags

| Feature | Description | Required For |
|---------|-------------|--------------|
| `v1` | V1 baseline reference implementation | V1 examples and tests |
| `v2` | V2 CPU-optimized backend (Rayon parallel) | V2 CPU examples and tests |
| `v2-gpu-metal` | V2/V3 Metal GPU backend (Apple Silicon) | Metal GPU tests (V2/V3) |
| `v2-gpu-cuda` | V2/V3 CUDA GPU backend (NVIDIA) | CUDA GPU tests (V2/V3) |
| `v3` | V3 bootstrapping and SIMD batching | V3 examples and tests |
| `lattice-reduction` | Lattice reduction for security analysis | Lattice reduction tests and examples |

### Test Counts

| Component | Test Count | Command |
|-----------|------------|---------|
| V1 Unit Tests | 31 | `cargo test --lib --features v1` |
| V2 Unit Tests | 127 | `cargo test --lib --features v2` |
| V3 Unit Tests | 52 | `cargo test --lib --features v2,v3 clifford_fhe_v3` |
| Lattice Reduction | ~60 | `cargo test --lib lattice_reduction` |
| Medical Imaging | ~25 | `cargo test --lib medical_imaging --features v2,v3` |
| **Total** | **249** | `cargo test --lib --features v2,v3` |

### Performance Summary

| Backend | Hardware | Time | Speedup vs V1 |
|---------|----------|------|---------------|
| V1 CPU | Apple M3 Max (14-core) | 11.42s | 1× |
| V2 CPU | Apple M3 Max (14-core) | 0.30s | 38× |
| V2 Metal | Apple M3 Max GPU | 33ms | 346× |
| V2 CUDA | NVIDIA RTX 5090 | 5.7ms | 2,002× |
| V3 SIMD | (projected) | 0.656ms/sample | 17,408× |

## Troubleshooting

### Compilation Issues

**Problem**: Feature flag not recognized
```bash
# Solution: Ensure correct feature syntax
cargo build --features v2,v3  # Correct
cargo build --features v2 v3  # Incorrect
```

**Problem**: Metal not found
```bash
# Solution: Install Xcode Command Line Tools
xcode-select --install
```

**Problem**: CUDA not found
```bash
# Solution: Install CUDA Toolkit and set environment variables
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

**Problem**: CMake 4.0 errors with netlib-src
```bash
# Error: "Compatibility with CMake < 3.5 has been removed from CMake"

# Solution: This is FIXED in the repository via .cargo/config.toml
# The fix is automatic - just rebuild:
cargo clean
cargo build --release --features v2,v3

# If you still see issues, verify .cargo/config.toml contains:
# [env]
# CMAKE_POLICY_VERSION_MINIMUM = "3.5"

# See CMAKE_FIX.md for detailed explanation
```

### Test Failures

**Problem**: Tests timing out
```bash
# Solution: Use release mode for performance-intensive tests
cargo test --release --features v1,v2,v3
```

**Problem**: Specific test failing
```bash
# Solution: Run test in isolation with nocapture
cargo test --lib specific_test_name --features v2 -- --nocapture
```

**Problem**: Lattice reduction tests not found
```bash
# Root cause: Built without lattice-reduction feature

# Solution: Add lattice-reduction feature
cargo test --lib lattice_reduction --features lattice-reduction

# Or use default features (which include lattice-reduction)
cargo test --lib lattice_reduction
```

### Performance Issues

**Problem**: Slower than expected performance
```bash
# Solution: Always use --release flag for benchmarking
cargo run --release --features v2 --example encrypted_3d_classification

# Solution: Set optimization level in Cargo.toml (already configured)
# [profile.release]
# opt-level = 3
# lto = true
```

## Additional Resources

- **CMake 4.0 Fix**: See [CMAKE_FIX.md](CMAKE_FIX.md) - netlib-src compatibility solution
- **V3 Bootstrapping**: See [V3_BOOTSTRAP.md](V3_BOOTSTRAP.md) - Complete V3 documentation (52/52 tests passing)
- **Feature Flags Guide**: See [FEATURE_FLAGS.md](FEATURE_FLAGS.md)
- **Installation Guide**: See [INSTALLATION.md](INSTALLATION.md)
- **CUDA Build Notes**: See [CUDA_BUILD_NOTES.md](CUDA_BUILD_NOTES.md)
- **Architecture Details**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Performance Benchmarks**: See [BENCHMARKS.md](BENCHMARKS.md)

## Support

For issues or questions:
- **GitHub Issues**: https://github.com/davidwilliamsilva/ga_engine/issues
- **Email**: dsilva@datahubz.com
