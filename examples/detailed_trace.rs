//! Detailed trace of the computation

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

    let a_b = geometric_product(&a, &b);
    print_multivector("a⊗b", &a_b);
    println!();

    // Now compute term by term for (a⊗b)⊗c
    // a⊗b = 2·1 + 1·e1 + 1·e2 + 2·e12
    // c = e2 + e3

    println!("Computing (a⊗b)⊗c term by term:");
    println!("  a⊗b has components: 2·1, 1·e1, 1·e2, 2·e12");
    println!("  c has components: e2, e3");
    println!();

    // Test each term individually
    let scalar_part = geometric_product(&[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &c);
    print_multivector("  2·1 ⊗ c", &scalar_part);

    let e1_part = geometric_product(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &c);
    print_multivector("  e1 ⊗ c", &e1_part);

    let e2_part = geometric_product(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], &c);
    print_multivector("  e2 ⊗ c", &e2_part);

    let e12_part = geometric_product(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0], &c);
    print_multivector("  2·e12 ⊗ c", &e12_part);

    // Sum them manually
    let mut manual_sum = [0.0; 8];
    for i in 0..8 {
        manual_sum[i] = scalar_part[i] + e1_part[i] + e2_part[i] + e12_part[i];
    }
    print_multivector("  Manual sum", &manual_sum);

    let full_product = geometric_product(&a_b, &c);
    print_multivector("  (a⊗b)⊗c", &full_product);

    println!("\n✓ Should match: {:?}", full_product == manual_sum);
}
