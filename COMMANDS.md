# Command Reference

Complete reference of all build, test, example, and benchmark commands for GA Engine.

## Table of Contents

- [Installation](#installation)
- [V1: Baseline Reference](#v1-baseline-reference)
- [V2: CPU Optimized](#v2-cpu-optimized)
- [V2: Metal GPU](#v2-metal-gpu)
- [V2: CUDA GPU](#v2-cuda-gpu)
- [V3: Bootstrapping](#v3-bootstrapping)
- [Lattice Reduction](#lattice-reduction)
- [All Versions Combined](#all-versions-combined)

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
# Development build
cargo build --features v1

# Release build (optimized)
cargo build --release --features v1
```

### Test V1
```bash
# Run all V1 unit tests (31 tests)
cargo test --lib --features v1

# Run V1 integration tests
cargo test --test clifford_fhe_integration_tests --features v1 -- --nocapture

# Run comprehensive geometric operations test suite (all 7 operations)
cargo test --test test_geometric_operations --features v1 -- --nocapture

# Run isolated operation tests (individual tests for clean output)
cargo test --test test_clifford_operations_isolated test_key_generation --features v1 -- --nocapture
cargo test --test test_clifford_operations_isolated test_encryption_decryption --features v1 -- --nocapture
cargo test --test test_clifford_operations_isolated test_reverse --features v1 -- --nocapture
cargo test --test test_clifford_operations_isolated test_geometric_product --features v1 -- --nocapture
cargo test --test test_clifford_operations_isolated test_wedge_product --features v1 -- --nocapture
cargo test --test test_clifford_operations_isolated test_inner_product --features v1 -- --nocapture
cargo test --test test_clifford_operations_isolated test_rotation --features v1 -- --nocapture
cargo test --test test_clifford_operations_isolated test_projection --features v1 -- --nocapture
cargo test --test test_clifford_operations_isolated test_rejection --features v1 -- --nocapture

# Run all V1 tests
cargo test --features v1
```

### Examples V1
```bash
# Encrypted 3D classification demo (main application)
cargo run --release --features v1 --example encrypted_3d_classification

# Basic FHE encryption/decryption
cargo run --release --features v1 --example clifford_fhe_basic
```

### Performance V1
```bash
# Expected performance:
# - Geometric product: 13 seconds
# - Rotation (depth-2): 26 seconds
# - Projection (depth-3): 115 seconds
# - Full network inference: ~361 seconds
```

## V2: CPU Optimized

### Build V2 CPU
```bash
# Development build
cargo build --features v2

# Release build (optimized, recommended)
cargo build --release --features v2
```

### Test V2 CPU
```bash
# Run all V2 unit tests (127 tests, <1 second)
cargo test --lib --features v2

# Run specific module tests
cargo test --lib clifford_fhe_v2::backends::cpu_optimized::ntt --features v2 -- --nocapture
cargo test --lib clifford_fhe_v2::backends::cpu_optimized::rns --features v2 -- --nocapture
cargo test --lib clifford_fhe_v2::backends::cpu_optimized::ckks --features v2 -- --nocapture
cargo test --lib clifford_fhe_v2::backends::cpu_optimized::keys --features v2 -- --nocapture
cargo test --lib clifford_fhe_v2::backends::cpu_optimized::multiplication --features v2 -- --nocapture
cargo test --lib clifford_fhe_v2::backends::cpu_optimized::geometric --features v2 -- --nocapture

# Run V2 geometric operations integration test
cargo test --test test_geometric_operations_v2 --features v2 -- --nocapture
```

### Examples V2 CPU
```bash
# Encrypted 3D classification (30x faster than V1)
cargo run --release --features v2 --example encrypted_3d_classification
```

### Performance V2 CPU
```bash
# Expected performance (14-core Apple M3 Max):
# - Geometric product: 441ms (30x faster than V1)
# - Keygen: 16ms (3.2x faster)
# - Encryption: 2.7ms (4.2x faster)
# - Decryption: 1.3ms (4.4x faster)
```

## V2: Metal GPU

### Build V2 Metal
```bash
# Install Xcode Command Line Tools (macOS only)
xcode-select --install

# Build with Metal support
cargo build --release --features v2-gpu-metal
```

### Test V2 Metal
```bash
# Run Metal GPU geometric operations test (includes benchmarking)
cargo test --release --features v2-gpu-metal --test test_geometric_operations_metal -- --nocapture
```

### Performance V2 Metal
```bash
# Expected performance (Apple M3 Max GPU):
# - Geometric product: 34ms (387x faster than V1, 13x faster than V2 CPU)
# - Throughput: 25 operations/second
# - Statistical analysis: 10 iterations with mean/min/max/std dev
```

## V2: CUDA GPU

### Build V2 CUDA
```bash
# Verify CUDA installation
nvcc --version

# Set CUDA path if needed
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Build with CUDA support
cargo build --release --features v2-gpu-cuda
```

### Test V2 CUDA
```bash
# Run CUDA GPU geometric operations test (includes benchmarking)
cargo test --release --features v2-gpu-cuda --test test_geometric_operations_cuda -- --nocapture
```

### Performance V2 CUDA
```bash
# Expected performance (NVIDIA RTX 4090):
# - Geometric product: 5.4ms (2,407x faster than V1, 82x faster than V2 CPU, 6.3x faster than Metal)
# - Throughput: 184 operations/second
# - Statistical analysis: 10 iterations with mean/min/max/std dev
# - Hardware: 16,384 CUDA cores, CUDA 12.9
```

## V3: Bootstrapping

### Build V3
```bash
# Build V3 with V2 backend
cargo build --release --features v2,v3
```

### Test V3
```bash
# Run all V3 unit tests (100 tests)
cargo test --lib --features v2,v3

# Run V3 bootstrapping tests
cargo test --lib clifford_fhe_v3::bootstrapping --features v2,v3 -- --nocapture

# Run SIMD batching tests
cargo test --lib batching --features v2,v3 -- --nocapture

# Run rotation tests
cargo test --lib rotation --features v2,v3 -- --nocapture

# Run CoeffToSlot/SlotToCoeff tests
cargo test --lib coeff_to_slot --features v2,v3 -- --nocapture
cargo test --lib slot_to_coeff --features v2,v3 -- --nocapture
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

### Build Lattice Reduction
```bash
# No special features required
cargo build --release
```

### Test Lattice Reduction
```bash
# Run all lattice reduction tests (95 tests)
cargo test --lib lattice_reduction

# Run specific module tests
cargo test --lib lattice_reduction::stable_gso
cargo test --lib lattice_reduction::bkz_stable
cargo test --lib lattice_reduction::ga_lll
cargo test --lib lattice_reduction::enumeration
```

### Examples Lattice Reduction
```bash
# Lattice reduction demonstration with GA-accelerated BKZ
cargo run --release --example lattice_reduction_demo
```

## All Versions Combined

### Build All
```bash
# Build all versions (V1, V2, V3)
cargo build --release --features v1,v2,v3

# Build with GPU support
cargo build --release --features v1,v2,v2-gpu-metal,v3  # macOS
cargo build --release --features v1,v2,v2-gpu-cuda,v3   # Linux
```

### Test All
```bash
# Run all tests across all versions
cargo test --features v1,v2,v3

# Run all tests including GPU backends
cargo test --features v1,v2,v2-gpu-metal,v3  # macOS
cargo test --features v1,v2,v2-gpu-cuda,v3   # Linux
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
| `v2-gpu-metal` | V2 Metal GPU backend (Apple Silicon) | Metal GPU tests |
| `v2-gpu-cuda` | V2 CUDA GPU backend (NVIDIA) | CUDA GPU tests |
| `v3` | V3 bootstrapping and SIMD batching | V3 examples and tests |

### Test Counts

| Component | Test Count | Command |
|-----------|------------|---------|
| V1 Unit Tests | 31 | `cargo test --lib --features v1` |
| V2 Unit Tests | 127 | `cargo test --lib --features v2` |
| V3 Unit Tests | 100 | `cargo test --lib --features v2,v3` |
| Lattice Reduction | 95 | `cargo test --lib lattice_reduction` |
| **Total** | **353** | `cargo test --features v1,v2,v3` |

### Performance Summary

| Backend | Command | Time | Speedup |
|---------|---------|------|---------|
| V1 CPU | `cargo run --release --features v1 --example encrypted_3d_classification` | 13s | 1x |
| V2 CPU | `cargo run --release --features v2 --example encrypted_3d_classification` | 441ms | 30x |
| V2 Metal | `cargo test --release --features v2-gpu-metal --test test_geometric_operations_metal -- --nocapture` | 34ms | 387x |
| V2 CUDA | `cargo test --release --features v2-gpu-cuda --test test_geometric_operations_cuda -- --nocapture` | 5.4ms | 2,407x |
| V3 SIMD | `cargo run --release --features v2,v3 --example test_batching` | 0.656ms/sample | 19,817x |

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

- **Installation Guide**: See [INSTALLATION.md](INSTALLATION.md)
- **Architecture Details**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Performance Benchmarks**: See [BENCHMARKS.md](BENCHMARKS.md)
- **V3 Bootstrapping**: See [V3_BOOTSTRAP.md](V3_BOOTSTRAP.md)

## Support

For issues or questions:
- **GitHub Issues**: https://github.com/davidwilliamsilva/ga_engine/issues
- **Email**: dsilva@datahubz.com
