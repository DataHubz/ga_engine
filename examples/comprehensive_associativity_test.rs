//! Comprehensive associativity test

use ga_engine::ga::geometric_product;
use rand::Rng;

fn test_associativity(a: &[f64; 8], b: &[f64; 8], c: &[f64; 8]) -> (bool, f64) {
    let ab = geometric_product(a, b);
    let bc = geometric_product(b, c);

    let left = geometric_product(&ab, c);
    let right = geometric_product(a, &bc);

    let mut max_diff = 0.0;
    for i in 0..8 {
        let diff = (left[i] - right[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    (max_diff < 1e-10, max_diff)
}

fn main() {
    println!("=== Comprehensive Associativity Test ===\n");

    // Test 1: All basis elements
    println!("Test 1: All pairs of basis elements");
    let basis = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // 1
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // e1
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], // e2
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], // e3
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], // e23
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], // e31
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], // e12
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], // e123
    ];

    let names = ["1", "e1", "e2", "e3", "e23", "e31", "e12", "e123"];

    let mut failures = Vec::new();

    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                let (passed, diff) = test_associativity(&basis[i], &basis[j], &basis[k]);
                if !passed {
                    failures.push((names[i], names[j], names[k], diff));
                }
            }
        }
    }

    if failures.is_empty() {
        println!("✓ All {} basis triples passed!", 8*8*8);
    } else {
        println!("✗ {} failures:", failures.len());
        for (a, b, c, diff) in &failures {
            println!("  ({} * {}) * {} ≠ {} * ({} * {}), diff = {}", a, b, c, a, b, c, diff);
        }
    }

    // Test 2: Random multivectors
    println!("\nTest 2: Random multivectors");
    let mut rng = rand::thread_rng();
    let mut random_failures = 0;
    let num_tests = 100;

    for _ in 0..num_tests {
        let mut a = [0.0; 8];
        let mut b = [0.0; 8];
        let mut c = [0.0; 8];

        for i in 0..8 {
            a[i] = rng.gen::<f64>() * 2.0 - 1.0;
            b[i] = rng.gen::<f64>() * 2.0 - 1.0;
            c[i] = rng.gen::<f64>() * 2.0 - 1.0;
        }

        let (passed, _) = test_associativity(&a, &b, &c);
        if !passed {
            random_failures += 1;
        }
    }

    if random_failures == 0 {
        println!("✓ All {} random tests passed!", num_tests);
    } else {
        println!("✗ {} out of {} random tests failed", random_failures, num_tests);
    }

    // Test 3: The specific failing case
    println!("\nTest 3: Original failing case");
    let a = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let c = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];

    let (passed, diff) = test_associativity(&a, &b, &c);
    println!("a = [1, 2e1]");
    println!("b = [e1, e2]");
    println!("c = [e2, e3]");

    if passed {
        println!("✓ Passed!");
    } else {
        println!("✗ Failed with max diff = {}", diff);

        // Show the actual results
        let ab = geometric_product(&a, &b);
        let bc = geometric_product(&b, &c);
        let left = geometric_product(&ab, &c);
        let right = geometric_product(&a, &bc);

        println!("\n(a*b)*c = {:?}", left);
        println!("a*(b*c) = {:?}", right);

        println!("\nDifferences:");
        for i in 0..8 {
            let diff = left[i] - right[i];
            if diff.abs() > 1e-10 {
                println!("  [{}]: {} - {} = {}", names[i], left[i], right[i], diff);
            }
        }
    }
}
