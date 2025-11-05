//! Test BKZ with normalization on synthetic lattices
//!
//! This tests the new `new_normalized()` method that prevents numerical
//! explosions from large basis entries.

use ga_engine::lattice_reduction::bkz_baseline::BKZ;
use ga_engine::lattice_reduction::lll_baseline::LLL;

fn main() {
    println!("=== BKZ Normalized Test ===\n");

    // Test 1: Small dimension with large entries
    test_large_entries_10d();

    // Test 2: Medium dimension with moderate entries
    test_moderate_entries_20d();

    // Test 3: Compare normalized vs standard on safe lattice
    test_normalized_correctness();
}

fn test_large_entries_10d() {
    println!("Test 1: 10D lattice with large entries (1000)");
    println!("-----------------------------------------------");

    // Diagonal lattice with perturbations and large scale
    let mut basis = Vec::new();
    for i in 0..10 {
        let mut v = vec![0.0; 10];
        v[i] = 1000.0; // Large diagonal
        // Add small perturbations
        if i + 1 < 10 {
            v[i + 1] = 10.0;
        }
        basis.push(v);
    }

    // Original LLL (for comparison)
    println!("Running LLL (baseline)...");
    let mut lll = LLL::new(basis.clone(), 0.99);
    let start_lll = std::time::Instant::now();
    lll.reduce();
    let lll_time = start_lll.elapsed();
    let lll_hf = lll.hermite_factor();
    println!("  LLL time: {:?}", lll_time);
    println!("  LLL Hermite factor: {:.6}", lll_hf);

    // BKZ with normalization
    println!("\nRunning BKZ (normalized, β=10)...");
    let (mut bkz, scale_factors) = BKZ::new_normalized(basis.clone(), 10, 0.99);

    let start_bkz = std::time::Instant::now();
    bkz.reduce_with_limit(5); // 5 tours max
    let bkz_time = start_bkz.elapsed();

    // Rescale back
    bkz.rescale(&scale_factors);

    let stats = bkz.get_stats();
    let bkz_hf = bkz.hermite_factor();

    println!("  BKZ time: {:?}", bkz_time);
    println!("  BKZ Hermite factor: {:.6}", bkz_hf);
    println!("  Tours: {}", stats.tours);
    println!("  Improvements: {}", stats.improvements);
    println!("  Enum calls: {}", stats.enum_calls);
    println!("  Nodes explored: {}", stats.enum_nodes);
    println!("  Timeouts: {}", stats.enum_timeouts);

    let reduced = bkz.get_basis();
    let first_norm: f64 = reduced[0].iter().map(|x| x * x).sum::<f64>().sqrt();
    println!("  First vector norm: {:.6}", first_norm);

    if bkz_hf <= lll_hf {
        println!("  ✓ BKZ improved over LLL");
    } else {
        println!("  ⚠ BKZ did not improve (may need more tours)");
    }

    println!();
}

fn test_moderate_entries_20d() {
    println!("Test 2: 20D lattice with moderate entries (100)");
    println!("------------------------------------------------");

    // Diagonal lattice with random-ish perturbations
    let mut basis = Vec::new();
    for i in 0..20 {
        let mut v = vec![0.0; 20];
        v[i] = 100.0;
        // Add perturbations
        if i + 1 < 20 {
            v[i + 1] = (i as f64 + 1.0) * 2.0;
        }
        if i + 2 < 20 {
            v[i + 2] = (i as f64 + 1.0);
        }
        basis.push(v);
    }

    println!("Running BKZ (normalized, β=10)...");
    let (mut bkz, scale_factors) = BKZ::new_normalized(basis.clone(), 10, 0.99);

    let start = std::time::Instant::now();
    bkz.reduce_with_limit(3); // 3 tours
    let elapsed = start.elapsed();

    bkz.rescale(&scale_factors);

    let stats = bkz.get_stats();
    let bkz_hf = bkz.hermite_factor();

    println!("  Time: {:?}", elapsed);
    println!("  Hermite factor: {:.6}", bkz_hf);
    println!("  Tours: {}", stats.tours);
    println!("  Improvements: {}", stats.improvements);

    let reduced = bkz.get_basis();
    let first_norm: f64 = reduced[0].iter().map(|x| x * x).sum::<f64>().sqrt();
    println!("  First vector norm: {:.6}", first_norm);

    if stats.improvements > 0 {
        println!("  ✓ Found improvements");
    }

    println!();
}

fn test_normalized_correctness() {
    println!("Test 3: Normalized vs Standard (small safe lattice)");
    println!("---------------------------------------------------");

    let basis = vec![
        vec![50.0, 10.0, 5.0],
        vec![10.0, 50.0, 8.0],
        vec![5.0, 8.0, 50.0],
    ];

    // Standard BKZ
    println!("Running standard BKZ...");
    let mut bkz_std = BKZ::new(basis.clone(), 3, 0.99);
    bkz_std.reduce_with_limit(5);
    let std_hf = bkz_std.hermite_factor();
    println!("  Standard Hermite factor: {:.6}", std_hf);

    // Normalized BKZ
    println!("Running normalized BKZ...");
    let (mut bkz_norm, scale_factors) = BKZ::new_normalized(basis.clone(), 3, 0.99);
    bkz_norm.reduce_with_limit(5);
    bkz_norm.rescale(&scale_factors);
    let norm_hf = bkz_norm.hermite_factor();
    println!("  Normalized Hermite factor: {:.6}", norm_hf);

    // Should give similar results (within numerical tolerance)
    let diff = (std_hf - norm_hf).abs();
    println!("  Difference: {:.6}", diff);

    if diff < 0.1 {
        println!("  ✓ Results match (normalized works correctly)");
    } else {
        println!("  ⚠ Results differ significantly");
    }

    println!();
}
