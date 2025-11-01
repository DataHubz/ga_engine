//! Test associativity and commutativity of geometric product

use ga_engine::clifford_ring::CliffordRingElement;

fn main() {
    println!("=== Testing Geometric Product Properties ===\n");

    // Test if geometric product is associative: (a⊗b)⊗c = a⊗(b⊗c)
    let a = CliffordRingElement::from_multivector([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let b = CliffordRingElement::from_multivector([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let c = CliffordRingElement::from_multivector([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    println!("Testing ASSOCIATIVITY: (a⊗b)⊗c = a⊗(b⊗c)");
    let left = a.multiply(&b).multiply(&c);
    let right = a.multiply(&b.multiply(&c));

    println!("(a⊗b)⊗c = {:?}", left.coeffs);
    println!("a⊗(b⊗c) = {:?}", right.coeffs);

    let mut is_associative = true;
    for i in 0..8 {
        if (left.coeffs[i] - right.coeffs[i]).abs() > 1e-10 {
            is_associative = false;
            println!("  Mismatch at index {}: {} vs {}", i, left.coeffs[i], right.coeffs[i]);
        }
    }

    if is_associative {
        println!("✓ Geometric product IS associative!\n");
    } else {
        println!("✗ Geometric product is NOT associative!\n");
    }

    // Test commutativity: a⊗b = b⊗a
    println!("Testing COMMUTATIVITY: a⊗b = b⊗a");
    let ab = a.multiply(&b);
    let ba = b.multiply(&a);

    println!("a⊗b = {:?}", ab.coeffs);
    println!("b⊗a = {:?}", ba.coeffs);

    let mut is_commutative = true;
    for i in 0..8 {
        if (ab.coeffs[i] - ba.coeffs[i]).abs() > 1e-10 {
            is_commutative = false;
            println!("  Mismatch at index {}: {} vs {}", i, ab.coeffs[i], ba.coeffs[i]);
        }
    }

    if is_commutative {
        println!("✓ Geometric product IS commutative!\n");
    } else {
        println!("✗ Geometric product is NOT commutative!\n");
    }

    // Test LWE decryption issue: (a⊗s)⊗r = s⊗(a⊗r)?
    println!("Testing LWE compatibility: (a⊗s)⊗r = s⊗(a⊗r)?");
    let s = CliffordRingElement::from_multivector([0.5, 0.3, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0]);
    let r = CliffordRingElement::from_multivector([0.7, -0.1, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0]);

    let left_lwe = a.multiply(&s).multiply(&r);
    let right_lwe = s.multiply(&a.multiply(&r));

    println!("(a⊗s)⊗r = {:?}", left_lwe.coeffs);
    println!("s⊗(a⊗r) = {:?}", right_lwe.coeffs);

    let mut lwe_works = true;
    let mut max_diff = 0.0;
    for i in 0..8 {
        let diff = (left_lwe.coeffs[i] - right_lwe.coeffs[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 1e-10 {
            lwe_works = false;
        }
    }

    if lwe_works {
        println!("✓ LWE decryption should work perfectly!\n");
    } else {
        println!("✗ LWE decryption has structural issue!");
        println!("  Maximum difference: {}", max_diff);
        println!("  This explains why Clifford-LWE decryption fails!\n");
    }
}
