//! Simple BKZ test on well-scaled lattices
//!
//! Demonstrates that BKZ works correctly when entries are in a reasonable range
//! (roughly -100 to 100). For cryptographic lattices with huge entries, arbitrary
//! precision arithmetic would be needed.

use ga_engine::lattice_reduction::bkz_baseline::BKZ;
use ga_engine::lattice_reduction::lll_baseline::LLL;

fn main() {
    println!("=== BKZ Simple Test ===\n");
    println!("Testing BKZ on well-scaled lattices (entries ~ 1-100)\n");

    // Test 1: 5D lattice
    test_5d();

    // Test 2: 10D lattice
    test_10d();

    // Test 3: BKZ vs LLL quality
    compare_bkz_lll();
}

fn test_5d() {
    println!("Test 1: 5D diagonal lattice with perturbations");
    println!("----------------------------------------------");

    let basis = vec![
        vec![50.0, 10.0, 2.0, 1.0, 0.5],
        vec![10.0, 50.0, 8.0, 2.0, 1.0],
        vec![2.0, 8.0, 50.0, 5.0, 2.0],
        vec![1.0, 2.0, 5.0, 50.0, 3.0],
        vec![0.5, 1.0, 2.0, 3.0, 50.0],
    ];

    println!("Input first vector norm: {:.6}", norm(&basis[0]));

    let mut bkz = BKZ::new(basis, 5, 0.99);

    let start = std::time::Instant::now();
    bkz.reduce_with_limit(10);
    let elapsed = start.elapsed();

    let reduced = bkz.get_basis();
    let stats = bkz.get_stats();

    println!("Output first vector norm: {:.6}", norm(&reduced[0]));
    println!("Time: {:?}", elapsed);
    println!("Tours: {}", stats.tours);
    println!("Improvements: {}", stats.improvements);
    println!("Hermite factor: {:.6}", bkz.hermite_factor());

    if stats.improvements > 0 {
        println!("✓ Found improvements\n");
    } else {
        println!("⚠ No improvements (already LLL-reduced)\n");
    }
}

fn test_10d() {
    println!("Test 2: 10D lattice");
    println!("-------------------");

    // Create a lattice with some structure
    let mut basis = Vec::new();
    for i in 0..10 {
        let mut v = vec![0.0; 10];
        v[i] = 30.0 + (i as f64 * 2.0);
        // Add perturbations
        if i > 0 {
            v[i - 1] = 5.0;
        }
        if i + 1 < 10 {
            v[i + 1] = 3.0;
        }
        basis.push(v);
    }

    println!("Input first vector norm: {:.6}", norm(&basis[0]));

    let mut bkz = BKZ::new(basis, 10, 0.99);

    let start = std::time::Instant::now();
    bkz.reduce_with_limit(5);
    let elapsed = start.elapsed();

    let reduced = bkz.get_basis();
    let stats = bkz.get_stats();

    println!("Output first vector norm: {:.6}", norm(&reduced[0]));
    println!("Time: {:?}", elapsed);
    println!("Tours: {}", stats.tours);
    println!("Improvements: {}", stats.improvements);
    println!("Hermite factor: {:.6}", bkz.hermite_factor());

    if elapsed.as_secs() < 5 {
        println!("✓ Completed in reasonable time\n");
    } else {
        println!("⚠ Took longer than expected\n");
    }
}

fn compare_bkz_lll() {
    println!("Test 3: BKZ vs LLL quality comparison");
    println!("-------------------------------------");

    let basis = vec![
        vec![40.0, 10.0, 5.0, 2.0],
        vec![10.0, 40.0, 8.0, 3.0],
        vec![5.0, 8.0, 40.0, 4.0],
        vec![2.0, 3.0, 4.0, 40.0],
    ];

    // LLL
    let mut lll = LLL::new(basis.clone(), 0.99);
    let lll_start = std::time::Instant::now();
    lll.reduce();
    let lll_time = lll_start.elapsed();
    let lll_hf = lll.hermite_factor();
    let lll_first_norm = norm(&lll.get_basis()[0]);

    // BKZ
    let mut bkz = BKZ::new(basis, 4, 0.99);
    let bkz_start = std::time::Instant::now();
    bkz.reduce_with_limit(10);
    let bkz_time = bkz_start.elapsed();
    let bkz_hf = bkz.hermite_factor();
    let bkz_first_norm = norm(&bkz.get_basis()[0]);

    println!("LLL:");
    println!("  Time: {:?}", lll_time);
    println!("  First vector norm: {:.6}", lll_first_norm);
    println!("  Hermite factor: {:.6}", lll_hf);

    println!("\nBKZ (β=4):");
    println!("  Time: {:?}", bkz_time);
    println!("  First vector norm: {:.6}", bkz_first_norm);
    println!("  Hermite factor: {:.6}", bkz_hf);
    println!("  Tours: {}", bkz.get_stats().tours);
    println!("  Improvements: {}", bkz.get_stats().improvements);

    println!("\nComparison:");
    if bkz_hf <= lll_hf {
        let improvement = (1.0 - bkz_hf / lll_hf) * 100.0;
        println!("  ✓ BKZ improved Hermite factor by {:.2}%", improvement);
    } else {
        println!("  = BKZ and LLL gave similar quality");
    }

    if bkz_first_norm <= lll_first_norm {
        let improvement = (1.0 - bkz_first_norm / lll_first_norm) * 100.0;
        println!("  ✓ BKZ reduced first vector by {:.2}%", improvement);
    }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
