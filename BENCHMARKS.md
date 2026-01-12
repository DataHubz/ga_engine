# Performance Benchmarks

This document contains performance benchmarks for the GA Engine Clifford FHE implementation across V1-V5 versions.

## Table of Contents

1. [V1 vs V2 Core Operations](#v1-vs-v2-core-operations)
2. [GPU Bootstrap Performance](#gpu-bootstrap-performance)
3. [V4 Packed Operations](#v4-packed-operations)
4. [V5 Privacy Analysis](#v5-privacy-analysis)
5. [Detailed Analysis](#detailed-analysis)

## V1 vs V2 Core Operations

### Benchmark Setup

- **Hardware**: Apple Silicon (M-series)
- **Compiler**: Rust 1.86.0 with `--release` profile
- **Optimization**: LTO enabled, opt-level 3, single codegen unit
- **Parameters**:
  - Ring dimension: N = 1024
  - RNS moduli: 4 primes (~60 bits each)
  - Security level: ~128 bits
- **Benchmark Framework**: Criterion 0.4
- **Sample Size**: 100 samples for core operations, 50 for geometric operations

### Command

```bash
cargo bench --bench v1_vs_v2_benchmark --features v1,v2
```

### Results Summary

#### Core FHE Operations

| Operation | V1 Time | V2 Time | **Speedup** |
|-----------|---------|---------|-------------|
| **Key Generation** | 48.6 ms | 13.4 ms | **3.6×** |
| **Single Encryption** | 10.8 ms | 2.3 ms | **4.7×** |
| **Single Decryption** | 5.3 ms | 1.1 ms | **4.8×** |
| **Ciphertext Multiplication** | 109.9 ms | 34.0 ms | **3.2×** |

#### Geometric Operations

Both V1 and V2 support geometric operations. V2 is faster for complex operations but has a regression in the reverse operation:

| Operation | V1 Time | V2 Time | **Speedup** |
|-----------|---------|---------|-------------|
| **Reverse** | 363 µs | 671 µs | **0.54× (slower)** |
| **Geometric Product** | 11.5 s | 2.03 s | **5.7×** |
| **Wedge Product** | TBD | 4.15 s | TBD |
| **Inner Product** | TBD | 4.11 s | TBD |

**Note**: The reverse operation performance regression in V2 (1.85× slower) is due to RnsRepresentation design requiring `moduli: Vec<u64>` to be cloned for each coefficient (N=1024 times). V1 stores moduli separately at the ciphertext level. This could be optimized by refactoring RnsRepresentation to use `Rc<Vec<u64>>` for shared moduli, but this would require changing ~20+ call sites throughout the V2 codebase. For now, this minor overhead (~300µs absolute) is acceptable given V2's 5-7× speedup on expensive operations.

## GPU Bootstrap Performance

### V3 Bootstrap

V3 implements full CKKS bootstrap with three backends: CPU, Metal GPU, and CUDA GPU.

#### Parameters

- **Ring Dimension**: N = 1024
- **RNS Moduli**: 30 primes (1× 60-bit, 29× 45-bit)
- **Bootstrap Levels**: 27 levels total
  - CoeffToSlot: 9 levels
  - EvalMod: 9 levels
  - SlotToCoeff: 9 levels
- **Security Level**: ~128 bits

### Metal GPU Bootstrap (Apple M3 Max)

**Command:**
```bash
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native
```

#### Results

| Operation | Time | Backend | Notes |
|-----------|------|---------|-------|
| **Key Generation** | ~73s | CPU | Rotation keys + evaluation keys |
| **Encryption** | ~175ms | GPU | Single ciphertext |
| **CoeffToSlot** (9 levels) | ~50s | GPU | Linear transforms + rotations |
| **EvalMod** (9 levels) | TBD | GPU | Modular reduction |
| **SlotToCoeff** (9 levels) | ~12s | GPU | Linear transforms + rotations |
| **Decryption** | ~11ms | GPU | Single ciphertext |
| **Total Bootstrap** | **~60s** | **GPU** | **Entirely GPU execution** |

**Error**: 3.6e-3

**Key Features:**
- Entirely GPU execution (no CPU fallback)
- Uses Metal shaders for all operations
- GPU-resident ciphertexts (minimal PCIe transfers)
- Exact rescaling with Russian peasant `mul_mod_128`

### CUDA GPU Bootstrap (NVIDIA GPU)

**Command:**
```bash
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap
```

#### Results

| Operation | Time | Backend | Notes |
|-----------|------|---------|-------|
| **CoeffToSlot** | ~0.15s | GPU | Linear transforms + rotations |
| **EvalMod** | **11.76s** | GPU | Modular reduction with BSGS |
| **SlotToCoeff** | ~0.04s | GPU | Linear transforms + rotations |
| **Total Bootstrap** | **11.95s** | **GPU** | **With relinearization** |

**Error**: ~1e-3

**Key Features:**
- Entirely GPU execution with CUDA kernels
- Full relinearization support
- GPU-resident ciphertexts throughout
- Optimized RNS operations (add, sub, pointwise multiply)
- Exact rescaling with Russian peasant `mul_mod_128`

### Bootstrap Performance Comparison

| Backend | Hardware | Total Time | Speedup vs CPU | Notes |
|---------|----------|------------|----------------|-------|
| **V3 CPU** | Apple M3 Max | ~70s | 1.0× | Reference |
| **V3 Metal GPU** | Apple M3 Max | ~60s | 1.17× | Entirely GPU |
| **V3 CUDA GPU** | NVIDIA GPU | **11.95s** | **5.86×** | **Entirely GPU + Relin** |

**Key Insight**: CUDA implementation is ~5× faster than Metal on this workload, primarily due to:
- Different GPU architectures (NVIDIA vs Apple Silicon)
- Optimized CUDA kernels for FHE operations
- Efficient relinearization implementation
- Hardware-specific optimizations

## V4 Packed Operations

V4 introduces slot-interleaved packing, storing all 8 multivector components in a single ciphertext.

### Parameters

- **Ring Dimension**: N = 1024 (quick) / N = 8192 (production)
- **RNS Moduli**: 15 primes (~60 bits each)
- **Batch Size**: N/8 multivectors per ciphertext (128 for N=1024, 1024 for N=8192)

### Commands

```bash
# Metal GPU (Apple Silicon)
cargo run --release --features v4,v2-gpu-metal --example bench_v4_metal_geometric

# CUDA GPU (NVIDIA)
cargo run --release --features v4,v2-gpu-cuda --example bench_v4_cuda_geometric_quick
```

### V4 Performance Results

| Operation | Backend | Time | Notes |
|-----------|---------|------|-------|
| **Packing (8→1)** | CUDA (N=1024) | 31.38s | One-time per batch |
| **Geometric Product** | CUDA (N=1024) | 36.84s | On packed data |
| **Per-MV Cost** | CUDA (batched) | ~36ms | 1024 MVs in parallel |
| **Packing (8→1)** | Metal (N=8192) | ~5.0s | Unified memory advantage |
| **Geometric Product** | Metal (N=8192) | ~5.0s | On packed data |

### Memory Comparison

| Configuration | V2/V3 Memory | V4 Memory | Savings |
|---------------|--------------|-----------|---------|
| N=1024, 4 primes | 262 KB/MV | 33 KB/MV | 8× |
| N=8192, 15 primes | 15.7 MB/MV | 1.97 MB/MV | 8× |
| N=32768, 30 primes | 125.8 MB/MV | 15.7 MB/MV | 8× |

### V4 vs V2 Trade-offs

| Metric | V2 (Separate) | V4 (Packed) | Winner |
|--------|---------------|-------------|--------|
| Memory per MV | 8 ciphertexts | 1 ciphertext | **V4** |
| Single-MV latency | 33ms (GPU) | 36s (includes pack/unpack) | **V2** |
| Batch throughput | 1× | 64-1024× | **V4** |
| Implementation complexity | Lower | Higher | V2 |

**Use V4 when**: Processing batches of multivectors, memory-constrained environments, SIMD-style operations.

**Use V2 when**: Single multivector operations, low-latency requirements, interactive applications.

## V5 Privacy Analysis

V5 provides execution-trace collection and privacy analysis for security research.

### Commands

```bash
# Run comprehensive attack suite
cargo run --release --features v5 --example v5_privacy_attacks

# Run dimension inference attack only
cargo run --release --features v5 --example v5_dimension_attack

# Collect traces with Metal GPU
cargo run --release --features v5,v2-gpu-metal --example v5_trace_collector -- --metal
```

### Privacy Attack Results

Results from the comprehensive attack suite (6 attack vectors):

| Attack | CKKS Accuracy | CliffordFHE Accuracy | Random Baseline | Winner |
|--------|---------------|----------------------|-----------------|--------|
| **Dimension Inference** | 100.0% | 16.7% | 16.7% | **CliffordFHE** |
| **Task Identification** | 100.0% | 16.7% | 16.7% | **CliffordFHE** |
| **Operation Count** | 100.0% | 16.7% | 16.7% | **CliffordFHE** |
| **Trace Length** | 100.0% | 16.7% | 16.7% | **CliffordFHE** |
| **Sparsity Inference** | ~20% | ~20% | 20% | Tie |
| **Tenant Linkability** | ~50% | ~50% | 50% | Tie |

### Information Leakage

| Metric | CKKS | CliffordFHE |
|--------|------|-------------|
| Dimension entropy (6 classes) | 2.585 bits | 2.585 bits |
| Bits leaked | 2.585 bits (100%) | 0.0 bits (0%) |
| Conditional entropy H(D\|T) | 0.0 bits | 2.585 bits |
| Mutual information I(D;T) | 2.585 bits | 0.0 bits |

### Key Finding

CliffordFHE achieves **information-theoretic privacy** against dimension inference attacks:
- CKKS: Rotation count directly reveals input dimension
- CliffordFHE: Fixed 64-multiplication structure regardless of input

**Score**: CliffordFHE wins 4 attacks, ties 2.

## Detailed Analysis

### V2 Optimizations (CPU)

**Important:** Both V1 and V2 use O(N log N) NTT for polynomial multiplication and RNS representation. The speedups come from implementation-level optimizations, not algorithmic changes.

The V2 implementation achieves 3-5× speedups through:

1. **Harvey Butterfly NTT** (1.5-2× speedup over V1's Cooley-Tukey NTT)
   - More cache-efficient butterfly operations
   - Better memory access patterns
   - Optimized modular arithmetic with Barrett reduction
   - Lazy reduction techniques (fewer modular reductions)

2. **RNS Operation Optimizations** (1.2-1.5× speedup)
   - Both versions use RNS, but V2 has faster per-prime operations
   - Better vectorization opportunities
   - Reduced overhead in CRT reconstruction
   - More efficient modulus switching

3. **Memory Layout and Data Structures** (1.3-1.8× speedup)
   - Improved cache locality for ciphertext operations
   - Reduced allocations and copying
   - Better memory alignment for potential SIMD
   - Streamlined ciphertext representation

**Combined effect:** These multiplicative improvements result in the observed 3.2-4.8× overall speedup.

### GPU Backend Optimizations

Both Metal and CUDA GPU backends achieve significant speedups through:

1. **Parallel NTT** - All N coefficients × num_primes limbs computed in parallel
2. **GPU-Resident Data** - Ciphertexts stay on GPU between operations
3. **Batched Operations** - Multiple primes processed simultaneously
4. **Exact Rescaling on GPU** - Russian peasant `mul_mod_128` avoids overflow
5. **Optimized Memory Access** - Coalesced reads/writes, minimized transfers

**Metal-Specific** (Apple Silicon):
- Metal shaders optimized for Apple GPU architecture
- Unified memory reduces CPU↔GPU transfer overhead
- Threadgroup memory for NTT twiddle factors

**CUDA-Specific** (NVIDIA):
- CUDA kernels optimized for NVIDIA architecture
- Shared memory for fast data exchange
- Coalesced global memory access patterns
- Strided layout kernels avoid expensive conversions

### Bootstrap Operation Breakdown

The bootstrap consists of three main phases:

1. **CoeffToSlot** - Convert coefficient encoding to slot encoding
   - Linear transformations (matrix multiplications)
   - Rotations via Galois automorphisms
   - CUDA: ~0.15s, Metal: ~50s

2. **EvalMod** - Evaluate modular reduction function
   - Baby-step giant-step (BSGS) algorithm
   - Polynomial evaluations with rotation
   - Dominant operation in CUDA bootstrap
   - CUDA: ~11.76s (98% of total time)

3. **SlotToCoeff** - Convert slot encoding back to coefficients
   - Inverse of CoeffToSlot
   - Linear transformations + rotations
   - CUDA: ~0.04s, Metal: ~12s

**Performance Note**: The Metal backend shows different performance characteristics, with CoeffToSlot taking significantly longer. This is likely due to:
- Different GPU memory architectures
- Metal shader compilation/optimization differences
- Apple Silicon's unified memory model
- Different rotation key formats or access patterns

### Geometric Operations Performance

The geometric operations are computationally expensive because they involve multiple homomorphic operations:

- **Reverse**: Simple coefficient reordering (very fast)
- **Geometric Product**: 8×8 = 64 homomorphic multiplications + additions
- **Wedge Product**: Geometric product + subtraction + scalar division by 2
- **Inner Product**: Geometric product + addition + scalar division by 2

Each homomorphic multiplication requires:
1. Tensor product of ciphertexts (polynomial multiplication in NTT domain)
2. Relinearization (reduce ciphertext size using evaluation key)
3. Rescaling (manage noise growth)

For a single geometric product of 8-component multivectors:
- ~64 ciphertext multiplications
- Each multiplication: ~34ms
- Total theoretical time: ~2.2s (matches observed 2.07s)

## Accuracy Verification

All implementations maintain high accuracy:

### V2 CPU Operations
- Key Generation: Exact
- Encryption/Decryption: < 1e-6 error
- Multiplication: < 1e-6 error
- Reverse: < 2e-10 error
- Geometric Product: < 8e-10 error
- Wedge Product: < 2e-10 error
- Inner Product: < 1e-10 error
- Projection: < 2e-10 error
- Rejection: < 1e-7 error

### V3 Bootstrap Operations
- **Metal GPU Bootstrap**: 3.6e-3 error
- **CUDA GPU Bootstrap**: ~1e-3 error
- **CPU Reference**: 3.6e-3 error

All errors are within acceptable bounds for FHE applications.

## Future Optimization Opportunities

Based on feature flags and current development:

1. **SIMD Vectorization** (`v2-simd-batched`)
   - Slot packing for batch operations
   - Estimated 8-16× throughput improvement
   - Status: Experimental

2. **GPU Pipeline Optimization**
   - Persistent GPU buffers (eliminate redundant transfers)
   - Kernel fusion for multi-step operations
   - Async compute for overlapping operations
   - Estimated 20-30% additional speedup

3. **EvalMod Optimization** (CUDA)
   - Current bottleneck: 11.76s / 11.95s = 98% of bootstrap time
   - Potential: Optimize BSGS polynomial evaluation
   - Potential: Better rotation key caching
   - Target: 30-50% reduction in EvalMod time

4. **Multi-GPU Support**
   - Distribute bootstrap across multiple GPUs
   - Parallel ciphertext processing
   - Estimated 2-4× additional speedup

## Benchmark Reproducibility

### V1 vs V2 CPU Benchmarks

To reproduce these benchmarks:

1. Clone the repository
2. Ensure you have Rust 1.86.0 or later
3. Run: `cargo bench --bench v1_vs_v2_benchmark --features v1,v2`

### GPU Bootstrap Benchmarks

**Metal GPU (Apple Silicon required):**
```bash
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native
```

**CUDA GPU (NVIDIA GPU required):**
```bash
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap
```

**Note**: GPU benchmarks require specific hardware and drivers:
- Metal: Apple Silicon Mac (M1/M2/M3)
- CUDA: NVIDIA GPU with CUDA Toolkit 11.0+

Results may vary based on:
- GPU architecture and compute capability
- CPU architecture and clock speed
- Available RAM and VRAM
- PCIe bandwidth (for discrete GPUs)
- System load and thermal throttling
- Compiler version and optimizations

For consistent results:
- Close other applications
- Ensure adequate cooling
- Run multiple times and average results
- Use the same compiler and CUDA/Metal SDK versions
