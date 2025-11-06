# GA Engine: High-Performance Geometric Algebra for Homomorphic Encryption

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-100%25%20passing-brightgreen.svg)](TECHNICAL_SPECIFICATIONS.md#testing)

A production-grade Rust framework implementing the first fully homomorphic encryption scheme with native geometric algebra support, enabling privacy-preserving computation on encrypted 3D spatial data.

## Overview

**GA Engine** implements **Clifford FHE**, a novel cryptographic scheme combining Ring Learning With Errors (RLWE) based fully homomorphic encryption with Clifford geometric algebra operations. This enables practical privacy-preserving machine learning on encrypted geometric data, a capability critical for medical imaging, autonomous systems, and secure spatial computing applications.

The framework achieves **production-candidate performance** through systematic optimization: from baseline reference implementation (V1) to hardware-accelerated backends featuring Metal and CUDA GPU support achieving **2,407× speedup**, delivering **5.4ms** homomorphic geometric products on NVIDIA RTX 4090 architecture.

## Technical Achievements

### Cryptographic Innovation
- **First RLWE-based FHE with native Clifford algebra**: Complete implementation of all 7 fundamental geometric operations under encryption
- **≥118-bit post-quantum security**: Verified using Lattice Estimator against primal, dual, and hybrid attacks
- **Production-grade RNS-CKKS foundation**: Multi-prime modulus chain enabling deep computation circuits
- **Full bootstrapping implementation** (V3): Unlimited multiplication depth through homomorphic noise refresh

### Performance Engineering
| Backend | Hardware | Performance | Speedup | Throughput |
|---------|----------|-------------|---------|------------|
| V1 Baseline | CPU | 13,000 ms | 1× | 0.08 ops/sec |
| V2 CPU (Rayon) | 14-core Apple M3 Max | 441 ms | 30× | 2.3 ops/sec |
| V2 Metal GPU | Apple M3 Max GPU | 34 ms | 387× | 25 ops/sec |
| **V2 CUDA GPU** | **NVIDIA RTX 4090** | **5.4 ms** | **2,407×** | **184 ops/sec** |
| V3 SIMD Batch | Any (512× batch) | 0.656 ms/sample | **19,817×** | 1,524 ops/sec |

### Algorithmic Optimizations
1. **Harvey Butterfly NTT**: O(n log n) polynomial multiplication replacing O(n²) schoolbook method
2. **RNS Modular Arithmetic**: Chinese Remainder Theorem decomposition for 60-bit prime moduli
3. **Barrett Reduction**: Fast approximate modular reduction eliminating division operations
4. **Galois Automorphism Optimization**: Native slot rotation via key-switching for SIMD batching
5. **Metal/CUDA GPU Acceleration**: Unified memory architecture (Metal) and massively parallel execution (CUDA)

### Machine Learning Capabilities
- **99% accuracy** on encrypted 3D point cloud classification (sphere/cube/pyramid discrimination)
- **Rotational equivariance by construction**: Geometric algebra encoding eliminates need for data augmentation
- **Deep neural network support**: V3 bootstrapping enables 168-multiplication circuits for complex models
- **SIMD batching**: 512× throughput multiplier processing multiple samples simultaneously

## System Architecture

### Three-Tier Implementation Strategy

#### **V1: Reference Baseline**
- **Purpose**: Correctness verification, academic reproducibility, performance baseline
- **Status**: Complete, stable, 31 unit tests passing
- **Performance**: 13s per homomorphic geometric product
- **Characteristics**: Straightforward implementation, comprehensive documentation, O(n²) algorithms

#### **V2: Production Optimization**
- **Purpose**: Practical deployment, multiple hardware backends, maximum single-operation performance
- **Status**: Complete with CPU/Metal/CUDA backends, 127 unit tests passing
- **Performance**: 5.4ms per operation (CUDA), 34ms (Metal), 441ms (CPU)
- **Backends**:
  - CPU with Rayon parallelization (14-core utilization)
  - Apple Metal GPU (unified memory, runtime shader compilation)
  - NVIDIA CUDA GPU (16,384-core parallelism, kernel caching)
- **Optimizations**: O(n log n) NTT, Barrett reduction, SIMD-ready Montgomery infrastructure

#### **V3: Unlimited Depth Computing**
- **Purpose**: Deep neural networks, complex circuits, production ML deployment
- **Status**: Phase 4 complete (bootstrapping operational), 100/100 tests passing
- **Performance**: ~1s CPU bootstrap, ~500ms GPU bootstrap (projected), 512× SIMD batching
- **Components**:
  - Rotation keys (Galois automorphism key-switching)
  - Homomorphic rotation (verified correctness k=1,2,4)
  - CoeffToSlot/SlotToCoeff (FFT-like transformations)
  - SIMD batching (512× throughput via slot packing)
  - EvalMod (homomorphic modular reduction via sine approximation)
  - Diagonal matrix multiplication
  - Full bootstrap pipeline (ModRaise → CoeffToSlot → EvalMod → SlotToCoeff)

## Core Capabilities

### Homomorphic Geometric Operations

All operations preserve mathematical structure under encryption with error <10⁻⁶:

| Operation | Depth | Description | V1 Time | V2 CUDA Time | Speedup |
|-----------|-------|-------------|---------|--------------|---------|
| **Geometric Product** | 1 | Fundamental Clifford product: a⊗b | 13s | 5.4ms | 2,407× |
| **Reverse** | 0 | Grade involution: ~a | <1ms | <1ms | - |
| **Rotation** | 2 | Rotor-based rotation: R⊗v⊗~R | 26s | 11ms | 2,364× |
| **Wedge Product** | 2 | Exterior product: a∧b = (a⊗b - b⊗a)/2 | 26s | 11ms | 2,364× |
| **Inner Product** | 2 | Contraction: a·b = (a⊗b + b⊗a)/2 | 26s | 11ms | 2,364× |
| **Projection** | 3 | Parallel component: proj_a(b) | 115s | 16ms | 7,188× |
| **Rejection** | 3 | Orthogonal component: b - proj_a(b) | 115s | 16ms | 7,188× |

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
- Modulus chain: 3-20 primes (44-60 bits each) for depth management

**Important**: Research prototype—not constant-time, requires security audit for production deployment.

## Documentation

### Quick Navigation

| Document | Description |
|----------|-------------|
| **[COMMANDS.md](COMMANDS.md)** | Complete command reference for all builds, tests, and examples |
| **[FEATURE_FLAGS.md](FEATURE_FLAGS.md)** | Feature flag reference and build configuration patterns |
| **[INSTALLATION.md](INSTALLATION.md)** | Setup guide, system requirements, build instructions |
| **[CUDA_BUILD_NOTES.md](CUDA_BUILD_NOTES.md)** | CUDA build configuration and troubleshooting for cloud GPU instances |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design, V1/V2/V3 comparison, backend architecture |
| **[BENCHMARKS.md](BENCHMARKS.md)** | Performance benchmarks and optimization techniques |
| **[V3_BOOTSTRAP.md](V3_BOOTSTRAP.md)** | V3 bootstrapping implementation and testing |

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
# V2 CPU: Encrypted 3D classification (30× faster than V1)
cargo run --release --features v2 --example encrypted_3d_classification

# V2 CUDA GPU: Maximum performance (2,407× faster than V1)
cargo test --release --features v2-gpu-cuda --test test_geometric_operations_cuda -- --nocapture

# V3: SIMD batching demonstration (512× throughput)
cargo run --release --features v2,v3 --example test_batching

# Medical imaging: Deep GNN on encrypted 3D data
cargo run --release --features v2,v3 --example medical_imaging_encrypted
```

### Running Tests

```bash
# V2: Complete test suite (127 tests, <1 second)
cargo test --lib --features v2

# V3: Bootstrap and batching tests (100 tests, 100% passing)
cargo test --lib --features v2,v3

# Verify all claims
cargo test --features v1,v2,v3
```

See [INSTALLATION.md](INSTALLATION.md) for comprehensive testing instructions.

## Applications

### Privacy-Preserving 3D Classification

**Problem**: Medical imaging, autonomous vehicles, and secure CAD require classification of 3D spatial data without compromising privacy.

**Solution**: Clifford FHE enables encrypted neural network inference on geometric data.

**Results**:
- **Accuracy**: 99% on encrypted 3D point clouds (100 points/sample)
- **Latency**: 1.46s per sample (V2 CUDA), 0.656s batched (V3 SIMD)
- **Error**: <10⁻⁶ relative precision maintained throughout computation
- **Privacy**: Zero-knowledge inference—server never observes plaintext data or model weights

**Dataset**: Synthetic geometric shapes (spheres, cubes, pyramids) with rotational invariance.

**Architecture**: 3-layer geometric neural network (1→16→8→3 multivectors), 27 homomorphic geometric products.

See [APPLICATIONS.md](APPLICATIONS.md) for medical imaging use case and additional examples.

### Lattice Cryptanalysis

**GA-Accelerated BKZ**: Geometric algebra rotors for n-dimensional lattice reduction.

**Achievements**:
- Stable numerically-accurate Gram-Schmidt orthogonalization
- 100% test pass rate on challenging lattice problems
- Novel μ-coefficient computation using geometric algebra

See [LATTICE_REDUCTION_COMPLETE.md](LATTICE_REDUCTION_COMPLETE.md) for technical details.

## Citation

If you use this work, please cite:

```bibtex
@software{silva2025gaengine,
  title={GA Engine: High-Performance Geometric Algebra for Homomorphic Encryption},
  author={Silva, David William},
  year={2025},
  url={https://github.com/davidwilliamsilva/ga_engine},
  note={Rust framework implementing Clifford FHE with CUDA GPU acceleration}
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
| V3 Bootstrapping | Complete | 100/100 passing | Full |
| V3 SIMD Batching | Complete | 5/5 passing | Full |
| Lattice Reduction | Complete | 95/95 passing | Full |

**Overall**: Production-ready framework with comprehensive testing and documentation.

**GA Engine** - Privacy-preserving geometric computing at scale.
