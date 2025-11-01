//! Benchmark: SIMD vs Scalar Geometric Product
//!
//! Compares performance of AVX2-accelerated vs scalar geometric product

use ga_engine::ga::{geometric_product_full, geometric_product_full_simd};
use std::time::Instant;

fn random_multivector() -> [f64; 8] {
    let mut mv = [0.0; 8];
    for i in 0..8 {
        mv[i] = if rand::random::<bool>() { 1.0 } else { -1.0 };
    }
    mv
}

fn max_error(a: &[f64; 8], b: &[f64; 8]) -> f64 {
    let mut max = 0.0;
    for i in 0..8 {
        let err = (a[i] - b[i]).abs();
        if err > max {
            max = err;
        }
    }
    max
}

fn main() {
    println!("=== SIMD vs Scalar Geometric Product Benchmark ===\n");

    // Warm up
    let a = random_multivector();
    let b = random_multivector();
    let mut out = [0.0; 8];
    for _ in 0..1000 {
        geometric_product_full(&a, &b, &mut out);
    }

    const ITERS: usize = 1_000_000;

    println!("Running {} iterations...\n", ITERS);

    // Benchmark SCALAR version
    println!("--- Scalar Version ---");
    let a = random_multivector();
    let b = random_multivector();
    let mut out_scalar = [0.0; 8];

    let start = Instant::now();
    for _ in 0..ITERS {
        geometric_product_full(&a, &b, &mut out_scalar);
    }
    let scalar_time = start.elapsed();
    let scalar_ns = scalar_time.as_nanos() / ITERS as u128;

    println!("Total time: {:?}", scalar_time);
    println!("Average per operation: {} ns", scalar_ns);
    println!("Throughput: {:.2} M ops/sec", 1000.0 / scalar_ns as f64);
    println!();

    // Benchmark SIMD version
    println!("--- SIMD Version ---");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            println!("Platform: x86_64 with AVX2 ✓");
            println!("SIMD width: 256-bit (4 × f64)");
        } else if is_x86_feature_detected!("sse2") {
            println!("Platform: x86_64 with SSE2 ✓");
            println!("SIMD width: 128-bit (2 × f64)");
        } else {
            println!("Platform: x86_64 (no SIMD support)");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        println!("Platform: ARM64 (Apple Silicon) with NEON ✓");
        println!("SIMD width: 128-bit (2 × f64)");
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        println!("Platform: {} (no SIMD support)", std::env::consts::ARCH);
    }

    let mut out_simd = [0.0; 8];

    let start = Instant::now();
    for _ in 0..ITERS {
        geometric_product_full_simd(&a, &b, &mut out_simd);
    }
    let simd_time = start.elapsed();
    let simd_ns = simd_time.as_nanos() / ITERS as u128;

    println!("Total time: {:?}", simd_time);
    println!("Average per operation: {} ns", simd_ns);
    println!("Throughput: {:.2} M ops/sec", 1000.0 / simd_ns as f64);
    println!();

    // Verify correctness
    let error = max_error(&out_scalar, &out_simd);
    let correct = error < 1e-10;

    println!("--- Correctness Check ---");
    println!("Max error: {:.2e} {}", error, if correct { "✓" } else { "✗" });
    if !correct {
        println!("⚠ SIMD implementation produces different results!");
        println!("Scalar result: {:?}", out_scalar);
        println!("SIMD result:   {:?}", out_simd);
    }
    println!();

    // Performance summary
    println!("--- Performance Summary ---");
    println!("Scalar: {} ns", scalar_ns);
    println!("SIMD:   {} ns", simd_ns);

    if simd_ns < scalar_ns {
        let speedup = scalar_ns as f64 / simd_ns as f64;
        println!("Speedup: {:.2}× faster ✓", speedup);

        println!();
        println!("--- Impact on Clifford-LWE-256 ---");
        println!("Current encryption time: ~102 µs");
        println!("Geometric products per encryption: ~2048 (32 poly coeffs × 64 products each)");
        println!("Time in geometric product: ~{:.1} µs", (2048.0 * scalar_ns as f64) / 1000.0);
        println!("With SIMD: ~{:.1} µs", (2048.0 * simd_ns as f64) / 1000.0);
        println!("Potential savings: ~{:.1} µs ({:.1}% reduction)",
            (2048.0 * (scalar_ns - simd_ns) as f64) / 1000.0,
            100.0 * (scalar_ns - simd_ns) as f64 / scalar_ns as f64
        );
    } else if simd_ns > scalar_ns {
        let slowdown = simd_ns as f64 / scalar_ns as f64;
        println!("⚠ Slowdown: {:.2}× slower", slowdown);
        println!("(This can happen if AVX2 is not available or overhead dominates)");
    } else {
        println!("Performance: Same (no SIMD acceleration)");
    }

    println!();
    println!("--- Notes ---");
    println!("• AVX2 processes 4 f64 values in parallel (256-bit vectors)");
    println!("• Theoretical maximum speedup: 4× (limited by memory bandwidth)");
    println!("• Actual speedup depends on CPU, memory, and compiler optimizations");
    println!("• For best results, compile with: RUSTFLAGS='-C target-cpu=native'");
}
