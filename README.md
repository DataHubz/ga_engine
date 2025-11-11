# GA Engine: High-Performance Geometric Algebra for Homomorphic Encryption

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-100%25%20passing-brightgreen.svg)](TESTING_GUIDE.md)

A production-grade Rust framework implementing the first fully homomorphic encryption scheme with native geometric algebra support, enabling privacy-preserving computation on encrypted 3D spatial data.

## Overview

**GA Engine** implements **Clifford FHE**, a novel cryptographic scheme combining Ring Learning With Errors (RLWE) based fully homomorphic encryption with Clifford geometric algebra operations. This enables practical privacy-preserving machine learning on encrypted geometric data, a capability critical for medical imaging, autonomous systems, and secure spatial computing applications.

The framework achieves **production-candidate performance** through systematic optimization: from baseline reference implementation (V1) to hardware-accelerated backends featuring Metal and CUDA GPU support achieving **2,002× speedup**, delivering **5.7ms** homomorphic geometric products on NVIDIA RTX 5090 architecture.

## Technical Achievements

### Cryptographic Innovation
- **First RLWE-based FHE with native Clifford algebra**: Complete implementation of all 7 fundamental geometric operations under encryption
- **≥118-bit post-quantum security**: Verified using Lattice Estimator against primal, dual, and hybrid attacks
- **Production-grade RNS-CKKS foundation**: Multi-prime modulus chain enabling deep computation circuits
- **Full bootstrapping implementation** (V3): Unlimited multiplication depth through homomorphic noise refresh
  - **CUDA GPU**: 11.95s bootstrap (5.86× faster than CPU, 100% GPU execution with relinearization)
  - **Metal GPU**: 60s bootstrap (100% GPU execution)
  - **CPU**: 70s bootstrap (reference implementation)

### Performance Engineering

#### Single Operation Performance (Geometric Product)

| Backend | Hardware | Performance | Speedup | Throughput |
|---------|----------|-------------|---------|------------|
| V1 Baseline | Apple M3 Max CPU | 11,420 ms | 1× | 0.09 ops/sec |
| V2 CPU (Rayon) | Apple M3 Max (14-core) | 300 ms | 38× | 3.3 ops/sec |
| V2 Metal GPU | Apple M3 Max GPU | 33 ms | 346× | 30.3 ops/sec |
| **V2 CUDA GPU** | **NVIDIA RTX 5090** | **5.7 ms** | **2,002×** | **175 ops/sec** |

#### Bootstrap Performance (V3 Full Bootstrap)

| Backend | Hardware | Total Time | Speedup vs CPU | Status |
|---------|----------|------------|----------------|--------|
| V3 CPU | Apple M3 Max | ~70s | 1× | Reference |
| V3 Metal GPU | Apple M3 Max | ~60s | 1.17× | ✅ Production Stable |
| **V3 CUDA GPU** | **NVIDIA GPU** | **~11.95s** | **5.86×** | **✅ Production Stable** |

**V3 CUDA GPU Bootstrap Breakdown**:
- EvalMod: ~11.76s (98% of total time)
- CoeffToSlot: ~0.15s
- SlotToCoeff: ~0.04s
- Error: ~1e-3 (excellent accuracy)
- Full relinearization support
- 100% GPU execution (no CPU fallback)

### Algorithmic Optimizations
1. **Harvey Butterfly NTT**: O(n log n) polynomial multiplication replacing O(n²) schoolbook method
2. **RNS Modular Arithmetic**: Chinese Remainder Theorem decomposition for 60-bit prime moduli
3. **Barrett Reduction**: Fast approximate modular reduction eliminating division operations
4. **Galois Automorphism Optimization**: Native slot rotation via key-switching for SIMD batching
5. **Metal/CUDA GPU Acceleration**: Unified memory architecture (Metal) and massively parallel execution (CUDA)
6. **Russian Peasant Multiplication**: GPU-safe 128-bit modular multiplication avoiding overflow

### Machine Learning Capabilities
- **99% accuracy** on encrypted 3D point cloud classification (sphere/cube/pyramid discrimination)
- **Rotational equivariance by construction**: Geometric algebra encoding eliminates need for data augmentation
- **Deep neural network support**: V3 bootstrapping enables unlimited circuit depth for complex models
- **SIMD batching** (in development): Slot packing for throughput multiplication

## System Architecture

### Four-Tier Implementation Strategy

#### **V1: Reference Baseline**
- **Purpose**: Correctness verification, academic reproducibility, performance baseline
- **Status**: Complete, stable, 31 unit tests passing
- **Performance**: 11.42s per homomorphic geometric product
- **Characteristics**: Straightforward implementation, comprehensive documentation, O(n²) algorithms

#### **V2: Production Optimization**
- **Purpose**: Practical deployment, multiple hardware backends, maximum single-operation performance
- **Status**: Complete with CPU/Metal/CUDA backends, 127 unit tests passing
- **Performance**: 5.7ms per operation (CUDA), 33ms (Metal), 300ms (CPU)
- **Backends**:
  - CPU with Rayon parallelization (14-core utilization)
  - Apple Metal GPU (unified memory, runtime shader compilation)
  - NVIDIA CUDA GPU (massively parallel execution, kernel caching)
- **Optimizations**: O(n log n) NTT, Barrett reduction, SIMD-ready Montgomery infrastructure

#### **V3: Unlimited Depth Computing**
- **Purpose**: Deep neural networks, complex circuits, production ML deployment
- **Status**: Complete and validated, 52/52 tests passing (100%)
- **Performance**:
  - CUDA GPU: **11.95s bootstrap** (5.86× faster than CPU)
  - Metal GPU: 60s bootstrap
  - CPU: 70s bootstrap (reference)
- **Architecture**: **V3 uses V2 backend** (not backend-agnostic)
  - V3 provides bootstrap algorithms (CoeffToSlot, SlotToCoeff, EvalMod)
  - V2 provides low-level operations (NTT, rescaling, rotation, key switching)
  - GPU backends work with both V2 and V3
- **Components**:
  - Rotation keys (Galois automorphism key-switching)
  - Homomorphic rotation (verified correctness)
  - CoeffToSlot/SlotToCoeff (FFT-like transformations)
  - EvalMod (homomorphic modular reduction via BSGS polynomial evaluation)
  - Full bootstrap pipeline (ModRaise → CoeffToSlot → EvalMod → SlotToCoeff)
  - Relinearization keys (CUDA)

#### **V4: Packed Multivector Layout**
- **Purpose**: Memory-efficient geometric operations, SIMD slot packing for multivectors
- **Status**: Complete with Metal GPU backend, 3 integration tests passing
- **Performance**: ~5.0s per packed geometric product (Metal GPU)
- **Architecture**: **V4 uses V2 backend** (builds on V2 GPU infrastructure)
  - V4 provides packing/unpacking operations (slot-interleaved layout)
  - V2 Metal GPU provides low-level operations (NTT, rotation, multiplication)
  - Single packed ciphertext holds all 8 Clifford algebra components
- **Memory Efficiency**: 8× reduction (1 packed ciphertext instead of 8 separate)
- **Components**:
  - Slot-interleaved packing (8 components → 1 ciphertext)
  - Unpacking via homomorphic rotations
  - Geometric product on packed multivectors
  - Compatible with V3 bootstrapping
- **Trade-off**: Slower than unpacked V2 (~5s vs 33ms) but uses 8× less memory

## Core Capabilities

### Homomorphic Geometric Operations

All operations preserve mathematical structure under encryption with error <10⁻⁶:

| Operation | Depth | Description | V1 Time | V2 CUDA Time | Speedup |
|-----------|-------|-------------|---------|--------------|---------|
| **Geometric Product** | 1 | Fundamental Clifford product: a⊗b | 11.42s | 5.7ms | 2,002× |
| **Reverse** | 0 | Grade involution: ~a | <1ms | <1ms | - |
| **Rotation** | 2 | Rotor-based rotation: R⊗v⊗~R | 22.8s | 11.4ms | 2,000× |
| **Wedge Product** | 2 | Exterior product: a∧b = (a⊗b - b⊗a)/2 | 22.8s | 11.4ms | 2,000× |
| **Inner Product** | 2 | Contraction: a·b = (a⊗b + b⊗a)/2 | 22.8s | 11.4ms | 2,000× |
| **Projection** | 3 | Parallel component: proj_a(b) | 34.3s | 17.1ms | 2,006× |
| **Rejection** | 3 | Orthogonal component: b - proj_a(b) | 34.3s | 17.1ms | 2,006× |

**Mathematical Foundation**: Operations preserve Clifford algebra Cl(3,0) structure:
- Basis: {1, e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃}
- Multiplication table: 64 structure constants encoding geometric product
- Graded structure: scalar ⊕ vector ⊕ bivector ⊕ trivector

### Cryptographic Security

**Post-Quantum Security Level**: ≥118 bits (NIST Level 1 equivalent)

**Security Analysis** (Lattice Estimator verification):
```
Parameters: N=1024, log₂(Q)=100-180, σ=3.2
Attack Complexity:
  • Primal uSVP: 2¹²⁰ operations
  • Dual attack: 2¹¹⁸ operations
  • Hybrid attack: 2¹¹⁹ operations
Conservative estimate: λ ≥ 118 bits
```

**Cryptographic Basis**:
- Ring-LWE hardness assumption over polynomial ring Z[x]/(x^1024 + 1)
- RNS-CKKS approximate homomorphic encryption
- IND-CPA security via game-hopping reduction
- Modulus chain: 3-30 primes (45-60 bits each) for depth management

**Important**: Research prototype—not constant-time, requires security audit for production deployment.

## Documentation

### Quick Navigation

| Document | Description |
|----------|-------------|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design, V1/V2/V3 comparison, backend architecture |
| **[COMMANDS.md](COMMANDS.md)** | Complete command reference for all builds, tests, and examples |
| **[FEATURE_FLAGS.md](FEATURE_FLAGS.md)** | Feature flag reference and build configuration patterns |
| **[INSTALLATION.md](INSTALLATION.md)** | Setup guide, system requirements, build instructions |
| **[BENCHMARKS.md](BENCHMARKS.md)** | Performance benchmarks and optimization techniques |
| **[TESTING_GUIDE.md](TESTING_GUIDE.md)** | Comprehensive testing procedures |

## Quick Start

### Installation

```bash
# Install Rust 1.75+
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone repository
git clone https://github.com/davidwilliamsilva/ga_engine.git
cd ga_engine

# Build with optimizations
cargo build --release --features v2
```

See [INSTALLATION.md](INSTALLATION.md) for GPU backend setup (Metal/CUDA).

### Running Examples

```bash
# V2 CPU: Encrypted 3D classification (38× faster than V1)
cargo run --release --features v2 --example encrypted_3d_classification

# V2 CUDA GPU: Maximum performance (2,002× faster than V1)
cargo test --release --features v2,v2-gpu-cuda --test test_geometric_operations_cuda -- --nocapture

# V3 CUDA GPU: Full bootstrap (100% GPU, 11.95s)
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap

# V3 Metal GPU: Full bootstrap (100% GPU, 60s)
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native

# V4 Metal GPU: Packed geometric product (8× memory reduction)
cargo test --release --features v4,v2-gpu-metal --test test_geometric_operations_v4 -- --nocapture
```

### Running Tests

```bash
# V2: Complete test suite (127 tests, <1 second)
cargo test --lib --features v2

# V3: Bootstrap tests (52 tests, 100% passing)
cargo test --lib --features v2,v3 clifford_fhe_v3

# V4: Packed multivector tests (3 integration tests)
cargo test --test test_geometric_operations_v4 --features v4,v2-gpu-metal --no-default-features -- --nocapture

# All versions (V1 + V2 + V3 = ~210 tests)
cargo test --lib --features v1,v2,v3
```

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing instructions.

## Applications

### Privacy-Preserving 3D Classification

**Problem**: Medical imaging, autonomous vehicles, and secure CAD require classification of 3D spatial data without compromising privacy.

**Solution**: Clifford FHE enables encrypted neural network inference on geometric data.

**Results**:
- **Accuracy**: 99% on encrypted 3D point clouds (100 points/sample)
- **Latency**:
  - Single sample: 5.7ms per operation (V2 CUDA)
  - With bootstrap: ~11.95s refresh (V3 CUDA GPU)
- **Error**: <10⁻⁶ relative precision maintained throughout computation
- **Privacy**: Zero-knowledge inference—server never observes plaintext data or model weights

**Dataset**: Synthetic geometric shapes (spheres, cubes, pyramids) with rotational invariance.

**Architecture**: 3-layer geometric neural network (1→16→8→3 multivectors), 27 homomorphic geometric products.

### Lattice Cryptanalysis

**GA-Accelerated BKZ**: Geometric algebra rotors for n-dimensional lattice reduction.

**Achievements**:
- Stable numerically-accurate Gram-Schmidt orthogonalization
- 100% test pass rate on challenging lattice problems
- Novel μ-coefficient computation using geometric algebra

## Citation

If you use this work, please cite:

```bibtex
@software{silva2025gaengine,
  title={GA Engine: High-Performance Geometric Algebra for Homomorphic Encryption},
  author={Silva, David William},
  year={2025},
  url={https://github.com/davidwilliamsilva/ga_engine},
  note={Rust framework implementing Clifford FHE with GPU acceleration (Metal, CUDA)}
}
```

## Contact & Support

- **Author**: David William Silva
- **Email**: dsilva@datahubz.com
- **GitHub**: https://github.com/davidwilliamsilva/ga_engine
- **Issues**: https://github.com/davidwilliamsilva/ga_engine/issues

## Acknowledgments

- **Eckhard Hitzer** and **Dietmar Hildenbrand**: Guidance on geometric algebra applications
- **Leo Dorst**: Foundational discussions on conformal geometric algebra
- **Vinod Vaikuntanathan**: Public work on lattice-based cryptography and bootstrapping
- **DataHubz**: Research sponsorship and computational resources
- **Rust Community**: Exceptional tooling, documentation, and ecosystem support

## License

MIT License - See [LICENSE](LICENSE) file

**Open Source Philosophy**: All code released under permissive license to enable:
- Verification of academic claims and reproducibility
- Extension and improvement by research community
- Advancement of privacy-preserving machine learning

## Project Status

| Component | Status | Tests | Documentation |
|-----------|--------|-------|---------------|
| V1 Baseline | Complete | 31/31 passing | Full |
| V2 CPU Backend | Complete | 127/127 passing | Full |
| V2 Metal GPU | Complete | Verified | Full |
| V2 CUDA GPU | Complete | Verified | Full |
| V3 Bootstrap (CPU) | Complete | 52/52 passing | Full |
| V3 Bootstrap (Metal GPU) | **Production Stable** | Verified | Full |
| V3 Bootstrap (CUDA GPU) | **Production Stable** | Verified | Full |
| V4 Packed Layout (Metal GPU) | Complete | 3/3 passing | Full |
| Lattice Reduction | Complete | ~60/60 passing | Full |

**Overall**: Production-ready framework with comprehensive testing and documentation.

---

**GA Engine** - Privacy-preserving geometric computing at scale.
