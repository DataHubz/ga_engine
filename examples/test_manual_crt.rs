//! Manually test CRT reconstruction

fn main() {
    let q0 = 1099511627689i64;
    let q1 = 1099511627691i64;

    // Test 1: Small value
    println!("Test 1: CRT reconstruction of value 6");
    let c0 = 6;
    let c1 = 6;

    println!("  Residues: [{}, {}]", c0, c1);

    // Manual CRT
    let q_product = (q0 as i128) * (q1 as i128);
    let q_0 = q_product / (q0 as i128);  // = q1
    let q_1 = q_product / (q1 as i128);  // = q0

    // Need q_0^{-1} mod q0 and q_1^{-1} mod q1
    // For now, just check if residues are small
    if c0 == c1 && (c0 as i128) < (q0 as i128 / 2) {
        println!("  Both residues equal and small → result = {}", c0);
    }

    println!();

    // Test 2: The problematic values
    println!("Test 2: CRT reconstruction of problematic values");
    let c0 = 685153833332i64;
    let c1 = 741245090257i64;

    println!("  Residues: [{}, {}]", c0, c1);
    println!("  q0 = {}", q0);
    println!("  q1 = {}", q1);
    println!("  q0 / 2 = {}", q0 / 2);

    if c0 > q0 / 2 {
        println!("  c0 is in upper half → will be center-lifted to negative");
    }
    if c1 > q1 / 2 {
        println!("  c1 is in upper half → will be center-lifted to negative");
    }

    // These values are both > q/2, so they represent negative numbers!
    let c0_centered = c0 - q0;
    let c1_centered = c1 - q1;

    println!("\n  Center-lifted:");
    println!("  c0_centered = {}", c0_centered);
    println!("  c1_centered = {}", c1_centered);

    // If they're close, this represents a small negative number
    if (c0_centered - c1_centered).abs() < 1000 {
        println!("  → Residues represent approximately {}", c0_centered);
    }
}
