//! Test products involving e31 to find the pattern

use ga_engine::ga::geometric_product;

fn main() {
    let basis = [
        ("1",    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("e1",   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("e2",   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("e3",   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ("e23",  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ("e31",  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ("e12",  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ("e123", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    ];

    println!("=== Products involving e31 ===\n");

    // Expected values based on associativity:
    // e1*e31 = e1*(e3*e1) = (e1*e3)*e1 = -e31*e1 = -(e3*e1*e1) = -e3
    // e31*e1 = (e3*e1)*e1 = e3*(e1*e1) = e3*1 = e3
    // BUT WAIT - that assumes e31 = e3*e1, let me verify...

    println!("Manual computation:");
    println!("  If e31 = e3*e1, then:");
    println!("    e1*e31 = e1*(e3*e1) = (e1*e3)*e1");
    println!("    From table: e1*e3 = -e31");
    println!("    So: e1*e31 = -e31*e1 = -(e3*e1*e1) = -(e3*1) = -e3");
    println!();

    // Check e3*e1 to see if it gives +e31
    let e3_times_e1 = geometric_product(&basis[3].1, &basis[1].1);
    println!("Current table:");
    println!("  e3*e1 = {:?}", e3_times_e1);
    if e3_times_e1[5].abs() > 0.9 {
        println!("    → {} e31", if e3_times_e1[5] > 0.0 { "+" } else { "-" });
    }

    // Check e1*e31
    let e1_times_e31 = geometric_product(&basis[1].1, &basis[5].1);
    println!("  e1*e31 = {:?}", e1_times_e31);
    if e1_times_e31[3].abs() > 0.9 {
        println!("    → {} e3", if e1_times_e31[3] > 0.0 { "+" } else { "-" });
        println!("    Expected: -e3");
        if e1_times_e31[3] > 0.0 {
            println!("    ✗ WRONG SIGN!");
        } else {
            println!("    ✓ Correct");
        }
    }

    // Check e31*e1
    let e31_times_e1 = geometric_product(&basis[5].1, &basis[1].1);
    println!("  e31*e1 = {:?}", e31_times_e1);
    if e31_times_e1[3].abs() > 0.9 {
        println!("    → {} e3", if e31_times_e1[3] > 0.0 { "+" } else { "-" });
        println!("    Expected: +e3 (since e31*e1 = e3*e1*e1 = e3)");
        if e31_times_e1[3] < 0.0 {
            println!("    ✗ WRONG SIGN!");
        } else {
            println!("    ✓ Correct");
        }
    }
}
