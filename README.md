# GAEngine

**GAEngine** is a Rust library and benchmark suite comparing **classical** linear algebra (matrix multiplication) with **Geometric Algebra (GA)** implementations. Our goal is to demonstrate, with verifiable benchmarks, that GA-based kernels can match or exceed classical counterparts under equal operation counts and to provide both 3D-specific and fully generic N-dimensional APIs.

## Project Status

- ✅ **Classical**  
  - `multiply_matrices(n×n)` implemented & tested  
- ✅ **3D GA core**  
  - `geometric_product` (vector×vector)  
  - `geometric_product_full` (8-component multivector)  
- ✅ **3D semantic adapters**  
  - `Vec3` + `apply_matrix3`  
  - `Rotor3::rotate`, `rotate_fast`, `rotate_simd` (4×) & `rotate_simd8` (8×)  
- ✅ **N-dimensional GA (const-generic)**  
  - `VecN<N>` with `dot`/`norm`/ops  
  - `Multivector<N>` + compile-time `make_gp_table::<N>()`  
- ✅ **Benchmark suite** (Criterion)  
  - Classical 8×8 matrix × 1 000  
  - GA full 8D product × 1 000  
  - 3D point rotation (classical vs. GA sandwich, fast, SIMD 4×, SIMD 8×)  
- ✅ **Linting & coverage**  
  - `cargo clippy --all-targets --all-features -D warnings`  
  - Coverage via `cargo llvm-cov` + `coverage_summary`

## Key Findings

1. **Correctness**  
   All unit tests (classical kernels, GA products, 3D transforms, SIMD code, N-D ops) pass.  
2. **Performance**  
   | Operation                              | Time (μs)      | Relative        |
   |----------------------------------------|---------------:|----------------:|
   | Classical 8×8 × 1 000                  | ~260 μs        | —               |
   | GA full 8D × 1 000                     | ~45 μs         | ~5.8× faster    |
   | Rotate 3D point (classical) × 1 000    | ~5.6 μs        | —               |
   | Rotate 3D point GA (sandwich) × 1 000  | ~96 μs         | ~17× slower     |
   | Rotate 3D point GA (fast) × 1 000      | ~7.9 μs        | ~1.4× slower    |
   | Rotate 3D point GA (SIMD 4×) × 1 000   | ~10 μs         | ~2.3× faster    |
   | Rotate 3D point GA (SIMD 8×) × 1 000   | ~12.3 μs       | ~3.7× faster    |

---

## How to Reproduce

```bash
# 1. Install Rust (rustup)
rustup update stable

# 2. Clone & enter
git clone <this-repo>
cd ga_engine

# 3. Run all unit tests
cargo test

# 4. Run Clippy (linter)
cargo clippy --all-targets --all-features -- -D warnings

# 5. Run benchmarks
cargo bench

# 6. Generate coverage & summary
make coverage
```

`make coverage` invokes:
```bash
cargo llvm-cov --json --summary-only --output-path cov.json
cargo run --bin coverage_summary
```

## Quick Examples

### 3D: Classical vs. GA rotor

```rust
use ga_engine::{Vec3, Rotor3};
use ga_engine::transform::apply_matrix3;

const EPS: f64 = 1e-12;

// classical rotation matrix (90° about Z)
let p = Vec3::new(1.0, 0.0, 0.0);
let m = [ 0.0, -1.0, 0.0,
          1.0,  0.0, 0.0,
          0.0,  0.0, 1.0 ];
let p1 = apply_matrix3(&m, p);

// GA rotor
let r = Rotor3::from_axis_angle(Vec3::new(0.0,0.0,1.0), std::f64::consts::FRAC_PI_2);
let p2 = r.rotate(p);

// should match
assert!((p1.x-p2.x).abs()<EPS);
assert!((p1.y-p2.y).abs()<EPS);
assert!((p1.z-p2.z).abs()<EPS);
```

### N-D: VecN and Multivector

```rust
use ga_engine::nd::{VecN, Multivector};
use ga_engine::nd::gp::{make_gp_table, gp_table_3};

// 5-D vectors
let a: VecN<5> = VecN::new([1., 2., 3., 4., 5.]);
let b = VecN::new([5., 4., 3., 2., 1.]);
let dot = a.dot(&b);      // 35.0
let norm_sq = a.norm().powi(2); // 55.0

// 2-D multivector (4 components: 1, e1, e2, e12)
let m1 = Multivector::<2>::new(vec![1.0, 2.0, 3.0, 4.0]);
let m2 = Multivector::<2>::new(vec![5.0, 6.0, 7.0, 8.0]);
let gp_table = make_gp_table::<2>(); // 16 entries
let m3 = m1.gp(&m2);                  // geometric product
```

## Next Steps

- **Micro-optimizations**: unroll critical loops, add f32 kernels, widen SIMD to `std::simd::Simd` lanes  
- **Parallel & batch APIs**: integrate Rayon for large-scale workloads  
- **Applications & demos**: small neural-net layers, FHE primitives, physics engines  
- **Crate release**: publish on crates.io, expand documentation, add performance charts  

*Chronology:*  
- **v0.1.0**: Baseline classical & GA, 3D adapters, full benchmarks, coverage tooling, SIMD-4× & SIMD-8× rotors, N-dimensional support.