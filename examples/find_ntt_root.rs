//! Find primitive 2N-th root of unity for NTT
//!
//! For q = 3329, N = 32, we need ω such that:
//! - ω^64 ≡ 1 (mod 3329)
//! - ω^k ≢ 1 (mod 3329) for 0 < k < 64

fn mod_pow(base: i64, mut exp: i64, modulus: i64) -> i64 {
    let mut result = 1i64;
    let mut base = base % modulus;

    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }

    result
}

fn is_primitive_root(omega: i64, order: i64, q: i64) -> bool {
    // Check ω^order ≡ 1 (mod q)
    if mod_pow(omega, order, q) != 1 {
        return false;
    }

    // Check ω^k ≢ 1 for proper divisors of order
    let divisors = vec![2, 4, 8, 16, 32]; // Divisors of 64
    for &d in &divisors {
        if mod_pow(omega, d, q) == 1 {
            return false;
        }
    }

    true
}

fn main() {
    let q = 3329i64;
    let n = 32i64;
    let order = 2 * n; // 64

    println!("Finding primitive {}-th root of unity mod {}", order, q);
    println!();

    // Verify q is NTT-friendly
    if (q - 1) % order != 0 {
        println!("ERROR: q-1 = {} is not divisible by {}", q - 1, order);
        return;
    }

    println!("✓ q-1 = {} = {} × {}", q - 1, order, (q - 1) / order);
    println!();

    // Method 1: Try small values
    println!("Method 1: Searching small values...");
    for omega in 2..100 {
        if is_primitive_root(omega, order, q) {
            println!("  Found: ω = {}", omega);
            println!("  Verification: ω^{} mod {} = {}", order, q, mod_pow(omega, order, q));

            // Print powers
            println!("\n  Powers of ω:");
            for k in vec![1, 2, 4, 8, 16, 32, 64] {
                println!("    ω^{:2} mod {} = {}", k, q, mod_pow(omega, k, q));
            }
            println!();
        }
    }

    // Method 2: Use generator of multiplicative group
    println!("Method 2: Using group generator...");

    // For prime q, find generator g of (Z/qZ)*
    // Then ω = g^((q-1)/order) is a primitive order-th root

    // Find a generator (brute force for small q)
    let mut generator = 0;
    for g in 2..q {
        if mod_pow(g, q - 1, q) == 1 {
            // Check if g is a generator (order q-1)
            let mut is_gen = true;
            for divisor in vec![2, 4, 8, 16, 26, 52, 64, 128, 208, 416, 832, 1664] {
                if (q - 1) % divisor == 0 && mod_pow(g, (q - 1) / divisor, q) == 1 {
                    is_gen = false;
                    break;
                }
            }
            if is_gen {
                generator = g;
                break;
            }
        }
    }

    if generator > 0 {
        println!("  Generator g = {}", generator);

        let exponent = (q - 1) / order;
        let omega = mod_pow(generator, exponent, q);

        println!("  ω = g^{} mod {} = {}", exponent, q, omega);

        if is_primitive_root(omega, order, q) {
            println!("  ✓ Verified: ω is a primitive {}-th root of unity", order);

            // Compute ω^(-1) for inverse NTT
            let omega_inv = mod_pow(omega, order - 1, q); // ω^(-1) = ω^(order-1)
            println!("  ω^(-1) mod {} = {}", q, omega_inv);

            // Compute N^(-1) for inverse NTT normalization
            let n_inv = mod_pow(n, q - 2, q); // n^(-1) = n^(q-2) by Fermat's little theorem
            println!("  N^(-1) mod {} = {}", q, n_inv);

            println!("\n=== NTT Parameters for Clifford-LWE-256 ===");
            println!("q = {}", q);
            println!("N = {}", n);
            println!("ω (primitive {}-th root) = {}", order, omega);
            println!("ω^(-1) = {}", omega_inv);
            println!("N^(-1) = {}", n_inv);
        }
    } else {
        println!("  Could not find generator");
    }
}
