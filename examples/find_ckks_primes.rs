//! Find NTT-friendly primes for CKKS
//!
//! We need primes p ≡ 1 (mod 2N) to support NTT

fn is_prime(n: i64) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n % 2 == 0 { return false; }

    let mut i = 3;
    while i * i <= n {
        if n % i == 0 { return false; }
        i += 2;
    }
    true
}

fn find_prime_near(target: i64, modulus: i64) -> Option<i64> {
    // Search for prime ≡ 1 (mod modulus) near target
    let start = (target / modulus) * modulus + 1;

    // Search upward
    for offset in 0..1000 {
        let candidate = start + offset * modulus;
        if candidate > 0 && is_prime(candidate) {
            return Some(candidate);
        }
    }
    None
}

fn main() {
    let n = 64;
    let two_n = 2 * n;

    println!("Finding CKKS-friendly primes for N={}", n);
    println!("Requirement: p ≡ 1 (mod {})\n", two_n);

    // Find 60-bit prime
    let target_60 = 1i64 << 60;
    if let Some(p60) = find_prime_near(target_60, two_n as i64) {
        println!("60-bit prime: {}", p60);
        println!("  Bits: {}", (p60 as f64).log2());
        println!("  Check: {} mod {} = {}", p60, two_n, p60 % two_n as i64);
    }

    // Find 40-bit prime (special rescaling prime)
    let target_40 = 1i64 << 40;
    if let Some(p40) = find_prime_near(target_40, two_n as i64) {
        println!("\n40-bit prime (special): {}", p40);
        println!("  Bits: {}", (p40 as f64).log2());
        println!("  Check: {} mod {} = {}", p40, two_n, p40 % two_n as i64);
    }

    // Find another 40-bit prime for depth-2
    let target_40_2 = (1i64 << 40) + (1i64 << 38);
    if let Some(p40_2) = find_prime_near(target_40_2, two_n as i64) {
        println!("\n40-bit prime #2: {}", p40_2);
        println!("  Bits: {}", (p40_2 as f64).log2());
        println!("  Check: {} mod {} = {}", p40_2, two_n, p40_2 % two_n as i64);
    }

    println!("\nSuggested configuration for N=64:");
    println!("Δ = 2^40 = {}", 1i64 << 40);
    println!("Prime chain: [q60, q40_special]");
}
