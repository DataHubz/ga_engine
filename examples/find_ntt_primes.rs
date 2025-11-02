//! Find NTT-friendly primes for CKKS

fn is_prime(n: i64) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n % 2 == 0 { return false; }

    let limit = (n as f64).sqrt() as i64 + 1;
    for i in (3..=limit).step_by(2) {
        if n % i == 0 {
            return false;
        }
    }
    true
}

fn main() {
    println!("Finding 10 NTT-friendly 60-bit primes (p ≡ 1 mod 2048) close to powers of 2...\n");

    let target_bits = 60;
    let base = 1i64 << target_bits; // 2^60
    let ntt_modulus = 2048; // For N=1024, need p ≡ 1 (mod 2N)

    let mut primes = Vec::new();
    let mut candidate = base - base / 100; // Start at ~2^60 - 1%

    while primes.len() < 10 && candidate < base + base / 100 {
        // Ensure p ≡ 1 (mod 2048)
        let remainder = candidate % ntt_modulus;
        if remainder != 1 {
            candidate += (ntt_modulus + 1 - remainder) % ntt_modulus;
        }

        if is_prime(candidate) {
            primes.push(candidate);
            println!("Found prime #{}: {} (hex: 0x{:X})", primes.len(), candidate, candidate);
            println!("  Bits: {}", 64 - candidate.leading_zeros());
            println!("  p mod 2048 = {}", candidate % ntt_modulus);
            println!();
            candidate += ntt_modulus; // Jump to next candidate
        } else {
            candidate += ntt_modulus;
        }
    }

    if primes.len() == 10 {
        println!("\n✓ Found 10 NTT-friendly primes!\n");
        println!("Rust array:");
        println!("let moduli = vec![");
        for (i, p) in primes.iter().enumerate() {
            println!("    {},  // q_{} (60-bit, NTT-friendly)", p, i);
        }
        println!("];");
    } else {
        println!("\n✗ Only found {} primes", primes.len());
    }
}
