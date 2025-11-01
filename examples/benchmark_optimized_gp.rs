//! Benchmark: Optimized Geometric Product
//!
//! Uses black_box to prevent dead code elimination

use ga_engine::ga::geometric_product_full;
use ga_engine::ga_simd_optimized::geometric_product_full_optimized;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    println!("=== Optimized vs Scalar Geometric Product ===\n");

    const ITERS: usize = 10_000_000;

    // Test data
    let a = black_box([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = black_box([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

    // Benchmark SCALAR
    println!("--- Scalar (Lookup Table) ---");
    let mut out_scalar = [0.0; 8];

    let start = Instant::now();
    for _ in 0..ITERS {
        geometric_product_full(
            black_box(&a),
            black_box(&b),
            black_box(&mut out_scalar)
        );
    }
    let scalar_time = start.elapsed();
    let scalar_ns = scalar_time.as_nanos() / ITERS as u128;

    println!("Total time: {:?}", scalar_time);
    println!("Average: {} ns", scalar_ns);
    println!("Result: {:?}", out_scalar);
    println!();

    // Benchmark OPTIMIZED
    println!("--- Optimized (Explicit Formulas) ---");
    let mut out_opt = [0.0; 8];

    let start = Instant::now();
    for _ in 0..ITERS {
        geometric_product_full_optimized(
            black_box(&a),
            black_box(&b),
            black_box(&mut out_opt)
        );
    }
    let opt_time = start.elapsed();
    let opt_ns = opt_time.as_nanos() / ITERS as u128;

    println!("Total time: {:?}", opt_time);
    println!("Average: {} ns", opt_ns);
    println!("Result: {:?}", out_opt);
    println!();

    // Verify correctness
    let mut max_error = 0.0;
    for i in 0..8 {
        let err = (out_scalar[i] - out_opt[i]).abs();
        if err > max_error {
            max_error = err;
        }
    }

    println!("--- Results ---");
    println!("Correctness: {:.2e} {}", max_error, if max_error < 1e-10 { "✓" } else { "✗" });

    if scalar_ns > 0 && opt_ns > 0 {
        let speedup = scalar_ns as f64 / opt_ns as f64;
        println!("Speedup: {:.2}×", speedup);

        if speedup > 1.2 {
            println!("Status: ✓✓ Significant improvement!");
        } else if speedup > 1.0 {
            println!("Status: ✓ Modest improvement");
        } else {
            println!("Status: ~ No improvement (compiler already optimized scalar version)");
        }

        println!();
        println!("--- Impact on Clifford-LWE-256 ---");
        let products_per_enc = 2048;
        let current = products_per_enc as f64 * scalar_ns as f64 / 1000.0;
        let optimized = products_per_enc as f64 * opt_ns as f64 / 1000.0;
        let savings = current - optimized;

        println!("Geometric products per encryption: {}", products_per_enc);
        println!("Current: {:.1} µs", current);
        println!("Optimized: {:.1} µs", optimized);
        println!("Savings: {:.1} µs ({:.1}%)",
            savings, 100.0 * savings / current);
    }
}
