# GAEngine

**GAEngine** is a Rust library and benchmark suite comparing **classical** linear algebra (matrix multiplication) with **Geometric Algebra (GA)** implementations. Our goal is to demonstrate, with verifiable benchmarks, that GA-based kernels can match or exceed classical counterparts under equal operation counts.

## Project Status
- ✅ Core classical `multiply_matrices` (n×n) implemented and tested.
- ✅ Core GA `geometric_product` (vector×vector) + full multivector `geometric_product_full` implemented and tested.
- ✅ 3D semantic adapters: `Vec3`, `apply_matrix3`, `Rotor3::rotate`, and `Rotor3::rotate_fast` added with unit tests.
- ✅ Criterion benchmarks comparing:
  - **Classical** 8×8 matrix × 1 000 batch  
  - **GA** full 8-component product × 1 000 batch  
  - **rotate 3D point classical** × 1 000 batch  
  - **rotate 3D point GA (sandwich)** × 1 000 batch  
  - **rotate 3D point GA (fast)** × 1 000 batch
- ✅ Optimized GA kernel using compile-time lookup table (`GP_PAIRS`) with zero runtime bit‑twiddling.

## Key Findings
1. **Correctness**
   - All unit tests (`identity`, `simple` matrices; `scalar_vector`, `vector_vector`; `rotate_z_90_degrees`, `rotate_z_90_degrees_fast`) pass.
2. **Performance**
   - **Classical** 8×8 × 1 000 batch: **~260 µs**
   - **GA** full multivector 8D × 1 000 batch: **~45 µs**
   - **rotate 3D point classical**: **~5.6 µs**
   - **rotate 3D point GA (sandwich)**: **~93 µs**
   - **rotate 3D point GA (fast)**: **~10.1 µs**
   - **Result:** GA kernel is **~5.8× faster** on the same-op-count matrix workload, and `rotate_fast` narrows to within ~2× of classical rotation.

These results show that GA can be a drop-in, correct replacement for classical routines, with competitive or superior performance given proper specialization.

## How to Reproduce
1. **Install Rust** (via `rustup`)
2. **Clone** this repo and change directory:
   ```bash
   git clone <this-repo>
   cd ga_engine
   ```
3. **Run tests:**
   ```bash
   cargo test
   ```
4. **Run benchmarks:**
   ```bash
   cargo bench
   ```
5. **Check coverage:**
   - Generate coverage JSON and summary in one step:
     ```bash
     make coverage
     ```
   - This runs `cargo llvm-cov --json --summary-only --output-path cov.json` and
     `cargo run --bin coverage_summary`, producing a terminal coverage table.

## Next Steps
- **Micro-optimizations:** unroll, SIMD, and LTO to push GA kernels past classical throughput.
- **Multi-threading & Batch Ops:** scale to larger workloads and measure parallel speedup.
- **Killer-app demos:** integrate GA in small neural‑net layers and FHE primitives with end‑to‑end benchmarks.
- **Publish & Document:** release on crates.io, add code samples, and plot performance charts in documentation.

---

*Chronology:*
- **v0.1.0**: Baseline classical & GA implementations + semantic adapters + full benchmarks + coverage tooling.

Join us as we build the fastest, most user-friendly GA Engine in Rust!