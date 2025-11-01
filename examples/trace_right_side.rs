//! Trace the right side: a⊗(b⊗c)

use ga_engine::ga::geometric_product;

fn print_multivector(name: &str, v: &[f64]) {
    let labels = ["1", "e1", "e2", "e3", "e23", "e31", "e12", "e123"];
    print!("{} = ", name);
    let mut first = true;
    for (i, &coeff) in v.iter().enumerate() {
        if coeff.abs() > 1e-10 {
            if !first && coeff > 0.0 {
                print!(" + ");
            } else if !first {
                print!(" ");
            }
            if coeff == 1.0 {
                print!("{}", labels[i]);
            } else if coeff == -1.0 {
                print!("-{}", labels[i]);
            } else {
                print!("{}{}", coeff, labels[i]);
            }
            first = false;
        }
    }
    if first {
        print!("0");
    }
    println!();
}

fn main() {
    let a = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let c = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];

    print_multivector("a", &a);
    print_multivector("b", &b);
    print_multivector("c", &c);
    println!();

    let b_c = geometric_product(&b, &c);
    print_multivector("b⊗c", &b_c);
    println!();

    // Now compute term by term for a⊗(b⊗c)
    // a = 1 + 2e1
    // b⊗c = 1 + e23 - e31 + e12

    println!("Computing a⊗(b⊗c) term by term:");
    println!("  a has components: 1, 2·e1");
    println!("  b⊗c has components: 1, e23, -e31, e12");
    println!();

    // Test: 1 ⊗ (b⊗c) should just give b⊗c
    let scalar_times_bc = geometric_product(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &b_c);
    print_multivector("  1 ⊗ (b⊗c)", &scalar_times_bc);

    // Test: 2e1 ⊗ (b⊗c)
    let e1_times_bc = geometric_product(&[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &b_c);
    print_multivector("  2e1 ⊗ (b⊗c)", &e1_times_bc);

    // Let's also compute 2e1 ⊗ each component
    let e1_times_1 = geometric_product(&[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    print_multivector("    2e1 ⊗ 1", &e1_times_1);

    let e1_times_e23 = geometric_product(&[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    print_multivector("    2e1 ⊗ e23", &e1_times_e23);

    let e1_times_neg_e31 = geometric_product(&[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]);
    print_multivector("    2e1 ⊗ (-e31)", &e1_times_neg_e31);

    let e1_times_e12 = geometric_product(&[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    print_multivector("    2e1 ⊗ e12", &e1_times_e12);

    // Manual sum
    let mut manual = [0.0; 8];
    for i in 0..8 {
        manual[i] = e1_times_1[i] + e1_times_e23[i] + e1_times_neg_e31[i] + e1_times_e12[i];
    }
    print_multivector("    Sum of parts", &manual);

    // Compare with full product
    let full = geometric_product(&[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &b_c);
    print_multivector("    Full 2e1⊗(b⊗c)", &full);
    let matches = manual.iter().zip(full.iter()).all(|(a, b)| (a - b).abs() < 1e-10);
    println!("    Match: {}", matches);
}
