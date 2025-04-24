# GAEngine

**GAEngine** is a Rust library and benchmark suite comparing **classical** linear algebra (matrix multiplication) with **Geometric Algebra (GA)** implementations. Our goal is to demonstrate, with verifiable benchmarks, that GA-based kernels can match or exceed classical counterparts under equal operation counts—and to provide both 3-D and general N-dimensional APIs.

## Project Status

- ✅ **Classical**  
  - `multiply_matrices(n×n)` implemented and tested.  
- ✅ **3-D GA Core**  
  - `geometric_product` (vector×vector) + `geometric_product_full` implemented and tested.  
- ✅ **3-D Semantic Adapters**  
  - `Vec3`, `apply_matrix3`, `Rotor3::rotate`, `Rotor3::rotate_fast` + unit tests.  
- ✅ **SIMD Rotors**  
  - `Rotor3::rotate_simd` (4×) & `rotate_simd8` (8×) + tests.  
- ✅ **Criterion Benchmarks**  
  - **Classical** 8×8 matrix × 1 000 batch  
  - **GA** full 8-component product × 1 000 batch  
  - **rotate 3-D point**: classical, GA (sandwich), GA fast, GA SIMD-4×, GA SIMD-8×  
- ✅ **High-Level Ops**  
  - Projection, rejection, reflection, motors, slerp interpolation  
- ✅ **N-Dimensional GA** (feature-gated `"nd"`)  
  - `VecN<const N>`, `Multivector<const N>`, runtime `make_gp_table(n)`  

## Key Findings

1. **Correctness**  
   - All unit tests pass: linear algebra, GA products, rotors, SIMD, high-level ops.  
2. **Performance**  
   - **Classical** 8×8 × 1 000: **~260 µs**  
   - **GA** full 8D × 1 000: **~45 µs** (~5.8× faster)  
   - **rotate 3D** classical: **~5.6 µs**  
   - **rotate GA (sandwich)**: **~96 µs**  
   - **rotate GA (fast)**: **~7.9 µs** (~1.4× slower than classical)  
   - **rotate GA SIMD 4×**: **~10 µs** (~2.3× per vector)  
   - **rotate GA SIMD 8×**: **~12.3 µs** (~3.7× per vector)  

## How to Reproduce

```bash
# 1. Install Rust
rustup toolchain install stable

# 2. Clone & enter
git clone <repo-url>
cd ga_engine

# 3. Run all tests
cargo test

# 4. Lint with Clippy
cargo clippy --all-targets --all-features -- -D warnings

# 5. Run benchmarks
cargo bench

# 6. Generate coverage
make coverage
```

The `make coverage` target does:
```bash
cargo llvm-cov --json --summary-only --output-path cov.json
cargo run --bin coverage_summary
```

## 3D Example

```rust
use ga_engine::prelude::*;
use std::f64::consts::FRAC_PI_2;

const EPS: f64 = 1e-12;

// classical rotation 90° about Z
let p = Vec3::new(1.0, 0.0, 0.0);
let m = [
    0.0, -1.0, 0.0,
    1.0,  0.0, 0.0,
    0.0,  0.0, 1.0,
];
let p1 = apply_matrix3(&m, p);

// GA rotor
let r = Rotor3::from_axis_angle(Vec3::new(0.0,0.0,1.0), FRAC_PI_2);
let p2 = r.rotate_fast(p);

assert!((p1.x - p2.x).abs() < EPS);
assert!((p1.y - p2.y).abs() < EPS);
assert!((p1.z - p2.z).abs() < EPS);
```

## N-Dimensional Example (requires default `"nd"` feature)

```rust
use ga_engine::nd::vecn::VecN;
use ga_engine::nd::multivector::Multivector;
use ga_engine::nd::gp::make_gp_table;

// A 5D vector
let v5: VecN<5> = VecN::new([1.0,2.0,3.0,4.0,5.0]);
assert_eq!(v5.dot(&v5), 55.0);
assert!((v5.norm().powi(2) - 55.0).abs() < 1e-12);

// Build the 2D GA table at runtime (4 blades → 16 entries)
let gp2 = make_gp_table(2);
assert_eq!(gp2.len(), 4 * 4);

// A 2D multivector example
let mv0 = Multivector::<2>::zero();
let mv1 = Multivector::<2>::new(vec![0.0, 1.0, 2.0, 0.0]);
let mv2 = mv1.clone().gp(&mv1);  // geometric product
```

## Next Steps

- **Micro-optimizations:** loop-unroll, f32-precision builds (`--features f32`), wider SIMD  
- **Batch & parallel:** integrate Rayon, batch-APIs for GA products  
- **Applications:** neural-net layers, FHE primitives, physics demos  
- **Release & docs:** publish on crates.io, expand tutorials, add performance charts  

**v0.1.0**: baseline classical & GA cores + semantic adapters + N-D API + full benchmarks + coverage + Clippy.