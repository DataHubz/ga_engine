//! Print the geometric product table to verify correctness

fn main() {
    const BLADE_NAMES: [&str; 8] = ["1", "e1", "e2", "e3", "e23", "e31", "e12", "e123"];

    println!("=== Geometric Product Table for Cl(3,0) ===\n");
    println!("Order: [1, e1, e2, e3, e23, e31, e12, e123]\n");

    // Expected multiplication table (from geometric algebra theory):
    // e1*e1 = 1, e2*e2 = 1, e3*e3 = 1
    // e1*e2 = e12, e2*e1 = -e12
    // e2*e3 = e23, e3*e2 = -e23
    // e3*e1 = e31, e1*e3 = -e31
    // e12*e3 = e123, e3*e12 = -e123
    // etc.

    use ga_engine::ga::geometric_product;

    for i in 0..8 {
        for j in 0..8 {
            let mut a = [0.0; 8];
            let mut b = [0.0; 8];
            a[i] = 1.0;
            b[j] = 1.0;

            let result = geometric_product(&a, &b);

            // Find non-zero component
            let mut found = false;
            for (k, &val) in result.iter().enumerate() {
                if val.abs() > 1e-10 {
                    let sign_str = if val > 0.0 { "+" } else { "" };
                    println!("{:4} * {:4} = {}{:4.0} {}",
                        BLADE_NAMES[i], BLADE_NAMES[j], sign_str, val, BLADE_NAMES[k]);
                    found = true;
                }
            }

            if !found {
                println!("{:4} * {:4} = 0", BLADE_NAMES[i], BLADE_NAMES[j]);
            }
        }
    }

    // Test specific cases
    println!("\n=== Testing Associativity ===");

    // (e1*e2)*e3 should equal e1*(e2*e3)
    let e1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let e2 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let e3 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

    let e1_e2 = geometric_product(&e1, &e2);
    let left = geometric_product(&e1_e2, &e3);

    let e2_e3 = geometric_product(&e2, &e3);
    let right = geometric_product(&e1, &e2_e3);

    println!("(e1*e2)*e3 = {:?}", left);
    println!("e1*(e2*e3) = {:?}", right);

    let same = left.iter().zip(right.iter())
        .all(|(a, b)| (a - b).abs() < 1e-10);

    if same {
        println!("✓ Associative!");
    } else {
        println!("✗ NOT associative!");
    }
}
