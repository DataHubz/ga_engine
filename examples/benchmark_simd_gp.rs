//! Micro-benchmark: SIMD-optimized geometric product
//!
//! Direct comparison between:
//! 1. Regular lazy geometric product
//! 2. SIMD-optimized (wrapping arithmetic + inline hints)
//!
//! This will tell us if the optimization helps before integrating into full Clifford-LWE

use ga_engine::clifford_ring_int::CliffordRingElementInt;
use ga_engine::clifford_ring_simd::geometric_product_lazy_optimized;
use ga_engine::lazy_reduction::LazyReductionContext;
use std::time::Instant;

fn main() {
    println!("=== Geometric Product SIMD Micro-Benchmark ===\n");

    let lazy = LazyReductionContext::new(3329);

    // Create test multivectors
    let a = CliffordRingElementInt::from_multivector([100, 200, 300, 400, 500, 600, 700, 800]);
    let b = CliffordRingElementInt::from_multivector([800, 700, 600, 500, 400, 300, 200, 100]);

    const NUM_OPS: usize = 10_000_000;

    // Warmup
    for _ in 0..1000 {
        let _ = a.geometric_product_lazy(&b, &lazy);
    }

    println!("--- Benchmark: Regular Lazy GP ({} ops) ---", NUM_OPS);
    let start = Instant::now();
    let mut result = a.clone();
    for _ in 0..NUM_OPS {
        result = a.geometric_product_lazy(&b, &lazy);
        std::hint::black_box(&result);
    }
    let lazy_time = start.elapsed();
    let lazy_ns = lazy_time.as_nanos() as f64 / NUM_OPS as f64;
    println!("Average per operation: {:.2} ns", lazy_ns);
    println!();

    // Warmup SIMD
    for _ in 0..1000 {
        let _ = geometric_product_lazy_optimized(&a, &b, &lazy);
    }

    println!("--- Benchmark: SIMD-Optimized GP ({} ops) ---", NUM_OPS);
    let start = Instant::now();
    let mut result2 = a.clone();
    for _ in 0..NUM_OPS {
        result2 = geometric_product_lazy_optimized(&a, &b, &lazy);
        std::hint::black_box(&result2);
    }
    let simd_time = start.elapsed();
    let simd_ns = simd_time.as_nanos() as f64 / NUM_OPS as f64;
    println!("Average per operation: {:.2} ns", simd_ns);
    println!();

    // Correctness check
    let result_lazy = a.geometric_product_lazy(&b, &lazy);
    let result_simd = geometric_product_lazy_optimized(&a, &b, &lazy);

    let mut correct = true;
    for i in 0..8 {
        if result_lazy.coeffs[i] != result_simd.coeffs[i] {
            correct = false;
            println!("❌ Mismatch at component {}: lazy={}, simd={}",
                    i, result_lazy.coeffs[i], result_simd.coeffs[i]);
        }
    }

    if correct {
        println!("✅ Correctness: PASS (results match)");
    }
    println!();

    println!("=== Performance Comparison ===\n");
    println!("| Version | Time (ns) | Speedup |");
    println!("|---------|-----------|---------|");
    println!("| Regular lazy | {:.2} | 1.00× |", lazy_ns);
    println!("| **SIMD-optimized** | **{:.2}** | **{:.2}×** |", simd_ns, lazy_ns / simd_ns);
    println!();

    if simd_ns < lazy_ns {
        let improvement = 100.0 * (lazy_ns - simd_ns) / lazy_ns;
        println!("🎉 SIMD is {:.1}% faster!", improvement);
        println!();
        println!("Expected impact on Clifford-LWE:");
        println!("- Current: 44.61 µs standard encryption");
        println!("- GP contributes ~20 µs (45%)");
        println!("- With {:.1}% GP speedup: ~{:.1} µs total", improvement, 44.61 - (20.0 * improvement / 100.0));
    } else if simd_ns > lazy_ns {
        let slowdown = 100.0 * (simd_ns - lazy_ns) / lazy_ns;
        println!("⚠️  SIMD is {:.1}% slower", slowdown);
        println!("The wrapping arithmetic or inline hints may not help on this platform.");
    } else {
        println!("≈ No significant difference");
        println!("The compiler is already doing a great job with the lazy version!");
    }
}
