//! Benchmark: All Geometric Product Variants
//!
//! Compares three implementations:
//! 1. Scalar (lookup table) - baseline
//! 2. SIMD (vectorized lookup table) - manual SIMD
//! 3. Optimized (explicit unrolled) - auto-vectorization friendly

use ga_engine::ga::{geometric_product_full, geometric_product_full_simd};
use ga_engine::ga_simd_optimized::geometric_product_full_optimized;
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
    println!("=== Geometric Product Implementation Comparison ===\n");

    // Warm up
    let a = random_multivector();
    let b = random_multivector();
    let mut out = [0.0; 8];
    for _ in 0..1000 {
        geometric_product_full(&a, &b, &mut out);
    }

    const ITERS: usize = 1_000_000;

    println!("Running {} iterations per method...\n", ITERS);

    // Generate multiple random inputs to prevent dead code elimination
    let mut inputs: Vec<([f64; 8], [f64; 8])> = Vec::with_capacity(100);
    for _ in 0..100 {
        inputs.push((random_multivector(), random_multivector()));
    }

    // Benchmark 1: SCALAR (lookup table)
    println!("--- Method 1: Scalar (Lookup Table) ---");
    let mut out_scalar = [0.0; 8];
    let mut sum_scalar = 0.0;

    let start = Instant::now();
    for i in 0..ITERS {
        let (a, b) = &inputs[i % 100];
        geometric_product_full(a, b, &mut out_scalar);
        sum_scalar += out_scalar[0]; // Prevent optimization
    }
    let scalar_time = start.elapsed();
    let scalar_ns = scalar_time.as_nanos() / ITERS as u128;

    // Use the sum to prevent dead code elimination
    if sum_scalar == f64::NAN { println!("impossible"); }

    println!("Implementation: 64-entry lookup table (GP_PAIRS)");
    println!("Total time: {:?}", scalar_time);
    println!("Average: {} ns", scalar_ns);
    println!("Throughput: {:.2} M ops/sec", 1000.0 / scalar_ns as f64);
    println!();

    // Benchmark 2: OPTIMIZED (explicit unrolled)
    println!("--- Method 2: Optimized (Explicit Unrolled) ---");
    let mut out_optimized = [0.0; 8];
    let mut sum_opt = 0.0;

    let start = Instant::now();
    for i in 0..ITERS {
        let (a, b) = &inputs[i % 100];
        geometric_product_full_optimized(a, b, &mut out_optimized);
        sum_opt += out_optimized[0];
    }
    let opt_time = start.elapsed();
    let opt_ns = opt_time.as_nanos() / ITERS as u128;

    if sum_opt == f64::NAN { println!("impossible"); }

    println!("Implementation: 8 explicit formulas (64 multiply-adds total)");
    println!("Total time: {:?}", opt_time);
    println!("Average: {} ns", opt_ns);
    println!("Throughput: {:.2} M ops/sec", 1000.0 / opt_ns as f64);

    let opt_error = max_error(&out_scalar, &out_optimized);
    println!("Correctness: {:.2e} {}", opt_error, if opt_error < 1e-10 { "✓" } else { "✗" });
    println!();

    // Benchmark 3: SIMD (manual vectorization)
    println!("--- Method 3: SIMD (Manual Vectorization) ---");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            println!("Platform: x86_64 with AVX2");
            println!("SIMD width: 256-bit (4 × f64)");
        } else if is_x86_feature_detected!("sse2") {
            println!("Platform: x86_64 with SSE2");
            println!("SIMD width: 128-bit (2 × f64)");
        } else {
            println!("Platform: x86_64 (no SIMD, fallback to scalar)");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        println!("Platform: ARM64 (Apple Silicon) with NEON");
        println!("SIMD width: 128-bit (2 × f64)");
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        println!("Platform: {} (no SIMD, fallback to scalar)", std::env::consts::ARCH);
    }

    let mut out_simd = [0.0; 8];
    let mut sum_simd = 0.0;

    let start = Instant::now();
    for i in 0..ITERS {
        let (a, b) = &inputs[i % 100];
        geometric_product_full_simd(a, b, &mut out_simd);
        sum_simd += out_simd[0];
    }
    let simd_time = start.elapsed();
    let simd_ns = simd_time.as_nanos() / ITERS as u128;

    if sum_simd == f64::NAN { println!("impossible"); }

    println!("Total time: {:?}", simd_time);
    println!("Average: {} ns", simd_ns);
    println!("Throughput: {:.2} M ops/sec", 1000.0 / simd_ns as f64);

    let simd_error = max_error(&out_scalar, &out_simd);
    println!("Correctness: {:.2e} {}", simd_error, if simd_error < 1e-10 { "✓" } else { "✗" });
    println!();

    // Performance summary
    println!("=== Performance Summary ===\n");
    println!("| Method | Time (ns) | Speedup | Status |");
    println!("|--------|-----------|---------|--------|");
    println!("| Scalar | {} | 1.00× | baseline |", scalar_ns);

    let opt_speedup = scalar_ns as f64 / opt_ns as f64;
    let opt_status = if opt_speedup >= 1.5 { "✓✓ excellent" }
                     else if opt_speedup >= 1.2 { "✓ good" }
                     else if opt_speedup >= 1.0 { "~ ok" }
                     else { "✗ slower" };
    println!("| Optimized | {} | {:.2}× | {} |", opt_ns, opt_speedup, opt_status);

    let simd_speedup = scalar_ns as f64 / simd_ns as f64;
    let simd_status = if simd_speedup >= 1.5 { "✓✓ excellent" }
                      else if simd_speedup >= 1.2 { "✓ good" }
                      else if simd_speedup >= 1.0 { "~ ok" }
                      else { "✗ slower" };
    println!("| SIMD | {} | {:.2}× | {} |", simd_ns, simd_speedup, simd_status);

    println!();
    println!("--- Impact on Clifford-LWE-256 ---");

    // Clifford-LWE-256 uses ~2048 geometric products per encryption
    // (32 polynomial coeffs × 64 products each)
    let current_gp_time = 2048.0 * scalar_ns as f64 / 1000.0;
    let optimized_gp_time = 2048.0 * opt_ns.min(simd_ns) as f64 / 1000.0;
    let savings = current_gp_time - optimized_gp_time;

    println!("Geometric products per encryption: ~2048");
    println!("Current GP time: {:.1} µs (baseline)", current_gp_time);
    println!("With best method: {:.1} µs", optimized_gp_time);
    println!("Potential savings: {:.1} µs ({:.1}% reduction)",
        savings, 100.0 * savings / current_gp_time);

    if savings > 10.0 {
        println!("\n✓ This would significantly improve Clifford-LWE-256 performance!");
    }

    println!();
    println!("--- Notes ---");
    println!("• Optimized version uses explicit formulas for better auto-vectorization");
    println!("• LLVM can auto-vectorize the optimized version without manual SIMD");
    println!("• Manual SIMD has overhead from loading/storing vector registers");
    println!("• For best results, compile with: RUSTFLAGS='-C target-cpu=native'");
}
