//! Debug the specific associativity failure

use ga_engine::ga::geometric_product;

fn main() {
    println!("=== Debugging Associativity Failure ===\n");

    let a = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let c = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];

    println!("a = {:?} = 1 + 2e1", a);
    println!("b = {:?} = e1 + e2", b);
    println!("c = {:?} = e2 + e3\n", c);

    // Compute a⊗b
    let a_b = geometric_product(&a, &b);
    println!("a⊗b = {:?}", a_b);

    // Compute (a⊗b)⊗c
    let left = geometric_product(&a_b, &c);
    println!("(a⊗b)⊗c = {:?}\n", left);

    // Compute b⊗c
    let b_c = geometric_product(&b, &c);
    println!("b⊗c = {:?}", b_c);

    // Compute a⊗(b⊗c)
    let right = geometric_product(&a, &b_c);
    println!("a⊗(b⊗c) = {:?}\n", right);

    // Compare
    println!("=== Comparison ===");
    let mut is_equal = true;
    for i in 0..8 {
        let diff = (left[i] - right[i]).abs();
        if diff > 1e-10 {
            is_equal = false;
            println!("  Index {}: left={}, right={}, diff={}", i, left[i], right[i], diff);
        }
    }

    if is_equal {
        println!("✓ Results are equal - geometric product IS associative");
    } else {
        println!("✗ Results differ - potential bug in geometric product implementation!");
    }
}
