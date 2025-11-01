//! Benchmark NTT vs Karatsuba for Clifford polynomial multiplication

use ga_engine::clifford_ring_int::{CliffordPolynomialInt, CliffordRingElementInt};
use ga_engine::ntt::NTTContext;
use ga_engine::ntt_clifford::multiply_ntt;
use ga_engine::lazy_reduction::LazyReductionContext;
use std::time::Instant;
use rand::Rng;

fn discrete_poly(n: usize) -> CliffordPolynomialInt {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut mv = [0i64; 8];
        for j in 0..8 {
            mv[j] = rng.gen_range(-1..=1);
        }
        coeffs.push(CliffordRingElementInt::from_multivector(mv));
    }
    CliffordPolynomialInt::new(coeffs)
}

fn main() {
    println!("=== NTT vs Karatsuba Benchmark ===\n");

    let q = 3329;
    let n = 32;
    let lazy = LazyReductionContext::new(q);
    let ntt = NTTContext::new_clifford_lwe();

    const NUM_OPS: usize = 100;

    // Generate test polynomials
    let mut a = discrete_poly(n);
    a.reduce_modulo_xn_minus_1(n, q);

    let mut b = discrete_poly(n);
    b.reduce_modulo_xn_minus_1(n, q);

    println!("Polynomial degree: N = {}", n);
    println!("Modulus: q = {}", q);
    println!("Operations: {}", NUM_OPS);
    println!();

    // Benchmark Karatsuba
    println!("--- Benchmark: Karatsuba ({} ops) ---", NUM_OPS);
    let start = Instant::now();
    for _ in 0..NUM_OPS {
        let _ = a.multiply_karatsuba_lazy(&b, &lazy);
    }
    let karatsuba_time = start.elapsed();
    let karatsuba_us = karatsuba_time.as_micros() as f64 / NUM_OPS as f64;
    println!("Average time: {:.2} µs\n", karatsuba_us);

    // Benchmark NTT
    println!("--- Benchmark: NTT ({} ops) ---", NUM_OPS);
    let start = Instant::now();
    for _ in 0..NUM_OPS {
        let _ = multiply_ntt(&a, &b, &ntt, &lazy);
    }
    let ntt_time = start.elapsed();
    let ntt_us = ntt_time.as_micros() as f64 / NUM_OPS as f64;
    println!("Average time: {:.2} µs\n", ntt_us);

    // Verify correctness
    let result_karatsuba = a.multiply_karatsuba_lazy(&b, &lazy);
    let mut result_ntt = multiply_ntt(&a, &b, &ntt, &lazy);

    // Reduce both
    let mut result_karatsuba_reduced = result_karatsuba;
    result_karatsuba_reduced.reduce_modulo_xn_minus_1_lazy(n, &lazy);
    result_ntt.reduce_modulo_xn_minus_1_lazy(n, &lazy);

    let mut correct = true;
    for i in 0..n {
        for j in 0..8 {
            if result_karatsuba_reduced.coeffs[i].coeffs[j] != result_ntt.coeffs[i].coeffs[j] {
                correct = false;
                println!("MISMATCH at coeff[{}].component[{}]", i, j);
                println!("  Karatsuba: {}", result_karatsuba_reduced.coeffs[i].coeffs[j]);
                println!("  NTT: {}", result_ntt.coeffs[i].coeffs[j]);
                break;
            }
        }
        if !correct {
            break;
        }
    }

    if correct {
        println!("✓ Correctness: PASS\n");
    } else {
        println!("✗ Correctness: FAIL\n");
        return;
    }

    // Results
    let speedup = karatsuba_us / ntt_us;
    println!("=== Results ===\n");
    println!("| Method | Time (µs) | Speedup |");
    println!("|--------|-----------|---------|");
    println!("| Karatsuba | {:.2} | 1.00× |", karatsuba_us);
    println!("| **NTT** | **{:.2}** | **{:.2}×** |", ntt_us, speedup);
    println!();

    if speedup > 1.0 {
        println!("🎉 NTT is {:.1}% faster!", (speedup - 1.0) * 100.0);
    } else {
        println!("⚠️  NTT is {:.1}% slower", (1.0 - speedup) * 100.0);
    }

    // Theoretical analysis
    println!("\n=== Theoretical Analysis ===");
    println!("Karatsuba: O(N^1.585 × 64) ≈ {} operations", (32f64.powf(1.585) * 64.0) as usize);
    println!("NTT: O(8 × N log N + 8 × N × 64) ≈ {} operations",
             (8.0 * 32.0 * 5.0 + 8.0 * 32.0 * 64.0) as usize);
    println!("Theoretical speedup: {:.2}×", (32f64.powf(1.585) * 64.0) / (8.0 * 32.0 * 5.0 + 8.0 * 32.0 * 64.0));
}
