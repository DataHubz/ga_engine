//! Benchmark sparse polynomial multiplication vs dense

use ga_engine::clifford_ring_int::{CliffordPolynomialInt, CliffordRingElementInt};
use ga_engine::sparse_poly::SparseCliffordPoly;
use ga_engine::lazy_reduction::LazyReductionContext;
use std::time::Instant;
use rand::Rng;

fn discrete_poly(n: usize) -> CliffordPolynomialInt {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut mv = [0i64; 8];
        for j in 0..8 {
            mv[j] = rng.gen_range(-1..=1);  // {-1, 0, 1}
        }
        coeffs.push(CliffordRingElementInt::from_multivector(mv));
    }
    CliffordPolynomialInt::new(coeffs)
}

fn main() {
    println!("=== Sparse Polynomial Multiplication Benchmark ===\n");

    let q = 3329;
    let lazy = LazyReductionContext::new(q);

    const N: usize = 32;  // Clifford-LWE-256 polynomial degree
    const NUM_OPS: usize = 100;

    // Generate test polynomials
    let mut a = discrete_poly(N);
    a.reduce_modulo_xn_minus_1(N, q);

    let mut b = discrete_poly(N);
    b.reduce_modulo_xn_minus_1(N, q);

    // Convert to sparse
    let a_sparse = SparseCliffordPoly::from_dense(&a);
    let b_sparse = SparseCliffordPoly::from_dense(&b);

    println!("Polynomial degree: N = {}", N);
    println!("Sparsity of a: {:.1}%", a_sparse.sparsity());
    println!("Sparsity of b: {:.1}%", b_sparse.sparsity());
    println!("Non-zero coeffs in a: {} / {}", a_sparse.non_zero.len(), N);
    println!("Non-zero coeffs in b: {} / {}", b_sparse.non_zero.len(), N);
    println!();

    // Theoretical operation count
    let dense_ops = N * N;
    let sparse_ops = a_sparse.non_zero.len() * b_sparse.non_zero.len();
    let theoretical_speedup = dense_ops as f64 / sparse_ops as f64;

    println!("Operation count:");
    println!("  Dense:  {} operations", dense_ops);
    println!("  Sparse: {} operations", sparse_ops);
    println!("  Theoretical speedup: {:.2}×\n", theoretical_speedup);

    // Benchmark dense Karatsuba
    println!("--- Benchmark: Dense Karatsuba ({} ops) ---", NUM_OPS);
    let start = Instant::now();
    let mut result_dense = a.multiply_karatsuba_lazy(&b, &lazy);
    for _ in 1..NUM_OPS {
        result_dense = a.multiply_karatsuba_lazy(&b, &lazy);
    }
    let dense_time = start.elapsed();
    let dense_us = dense_time.as_micros() as f64 / NUM_OPS as f64;
    println!("Average time: {:.2} µs\n", dense_us);

    // Benchmark sparse naive
    println!("--- Benchmark: Sparse Naive ({} ops) ---", NUM_OPS);
    let start = Instant::now();
    let mut result_sparse = a_sparse.multiply_sparse_lazy(&b_sparse, &lazy);
    for _ in 1..NUM_OPS {
        result_sparse = a_sparse.multiply_sparse_lazy(&b_sparse, &lazy);
    }
    let sparse_time = start.elapsed();
    let sparse_us = sparse_time.as_micros() as f64 / NUM_OPS as f64;
    println!("Average time: {:.2} µs\n", sparse_us);

    // Verify correctness
    result_dense.reduce_modulo_xn_minus_1_lazy(N, &lazy);
    result_sparse.reduce_modulo_xn_minus_1_lazy(N, &lazy);

    let mut correct = true;
    for i in 0..result_dense.coeffs.len().min(result_sparse.coeffs.len()) {
        if result_dense.coeffs[i].coeffs != result_sparse.coeffs[i].coeffs {
            correct = false;
            println!("MISMATCH at index {}", i);
            println!("  Dense:  {:?}", result_dense.coeffs[i].coeffs);
            println!("  Sparse: {:?}", result_sparse.coeffs[i].coeffs);
            break;
        }
    }

    if correct {
        println!("✓ Correctness: PASS (results match!)\n");
    } else {
        println!("✗ Correctness: FAIL\n");
        return;
    }

    // Results
    let actual_speedup = dense_us / sparse_us;
    println!("=== Results ===\n");
    println!("| Method | Time (µs) | Speedup |");
    println!("|--------|-----------|---------|");
    println!("| Dense Karatsuba | {:.2} | 1.00× |", dense_us);
    println!("| **Sparse Naive** | **{:.2}** | **{:.2}×** |", sparse_us, actual_speedup);
    println!();

    if actual_speedup > 1.0 {
        println!("🎉 Sparse is {:.1}% faster!", (actual_speedup - 1.0) * 100.0);
    } else {
        println!("⚠️  Sparse is {:.1}% slower (overhead too high)", (1.0 - actual_speedup) * 100.0);
    }

    println!();
    println!("Theoretical speedup: {:.2}×", theoretical_speedup);
    println!("Actual speedup: {:.2}×", actual_speedup);
    println!("Efficiency: {:.1}% of theoretical", 100.0 * actual_speedup / theoretical_speedup);
}
