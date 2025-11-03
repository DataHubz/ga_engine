// Test NTT specifically with the 60-bit prime used in CKKS

fn main() {
    let q = 1141392289560813569i64; // The 60-bit prime
    let n = 1024usize;

    println!("Testing NTT with 60-bit prime:");
    println!("  q = {}", q);
    println!("  n = {}", n);

    // Check if (q-1) is divisible by 2n
    assert_eq!((q - 1) % (2 * n as i64), 0, "q-1 must be divisible by 2n");
    println!("✓ (q-1) divisible by 2n");

    // Find primitive root and omega
    let (psi, omega) = negacyclic_roots(q, n);
    println!("  psi = {}", psi);
    println!("  omega = {}", omega);

    // Verify psi^n = -1 (mod q) and psi^(2n) = 1 (mod q)
    let psi_n = mod_pow(psi, n as i64, q);
    let psi_2n = mod_pow(psi, 2 * n as i64, q);
    assert_eq!(psi_n, q - 1, "psi^n should equal -1 (mod q)");
    assert_eq!(psi_2n, 1, "psi^(2n) should equal 1 (mod q)");
    println!("✓ psi^n = -1 (mod q)");
    println!("✓ psi^(2n) = 1 (mod q)");

    // Verify omega = psi^2 has order n
    let omega_n = mod_pow(omega, n as i64, q);
    assert_eq!(omega_n, 1, "omega^n should equal 1 (mod q)");
    println!("✓ omega^n = 1 (mod q)");

    // Test NTT roundtrip with small values
    let test_cases = vec![
        vec![1i64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // [1, 0, ...]
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], // [1, 2, ..., 16]
        vec![100, 200, 300, 400, 500, 600, 700, 800, 0, 0, 0, 0, 0, 0, 0, 0], // Larger values
    ];

    for (idx, test) in test_cases.iter().enumerate() {
        let mut a = test.clone();
        // Pad to size n
        a.resize(n, 0);

        let original = a.clone();
        ntt_negacyclic(&mut a, q, psi, omega);
        intt_negacyclic(&mut a, q, psi, omega);

        // Check first 16 coefficients
        let matches = original[..16].iter().zip(&a[..16]).all(|(x, y)| x == y);

        if matches {
            println!("✓ Test case {} passed", idx + 1);
        } else {
            println!("✗ Test case {} FAILED", idx + 1);
            println!("  Expected: {:?}", &original[..16]);
            println!("  Got:      {:?}", &a[..16]);
            panic!("NTT roundtrip failed!");
        }
    }

    // Test polynomial multiplication
    println!("\nTesting polynomial multiplication:");
    let a = vec![1i64, 2, 0, 0]; // 1 + 2x
    let b = vec![3i64, 4, 0, 0]; // 3 + 4x
    // Expected: (1+2x)(3+4x) = 3 + 10x + 8x^2 (mod x^4 + 1) = 3 + 10x + 8x^2

    let mut a_full = a.clone();
    let mut b_full = b.clone();
    a_full.resize(n, 0);
    b_full.resize(n, 0);

    let result = polynomial_multiply_ntt(&a_full, &b_full, q, n);

    println!("  (1+2x) * (3+4x) mod (x^n+1):");
    println!("    result[0] = {} (expected 3)", result[0]);
    println!("    result[1] = {} (expected 10)", result[1]);
    println!("    result[2] = {} (expected 8)", result[2]);

    assert_eq!(result[0], 3);
    assert_eq!(result[1], 10);
    assert_eq!(result[2], 8);
    println!("✓ Polynomial multiplication correct");

    println!("\n✅ ALL TESTS PASSED for 60-bit prime!");
}

// === NTT implementation (copied from ckks_rns.rs) ===

#[inline(always)]
fn mod_add(a: i64, b: i64, q: i64) -> i64 {
    let s = a.wrapping_add(b);
    if s >= q { s - q } else if s < 0 { s + q } else { s }
}

#[inline(always)]
fn mod_mul(a: i64, b: i64, q: i64) -> i64 {
    let p = (a as i128) * (b as i128);
    (p % (q as i128)) as i64
}

fn mod_pow(mut base: i64, mut exp: i64, q: i64) -> i64 {
    let mut acc = 1i64;
    while exp > 0 {
        if (exp & 1) == 1 {
            acc = mod_mul(acc, base, q);
        }
        base = mod_mul(base, base, q);
        exp >>= 1;
    }
    acc
}

fn primitive_root(q: i64) -> i64 {
    let phi = q - 1;
    let mut odd = phi;
    while odd % 2 == 0 {
        odd /= 2;
    }
    for g in 2..q {
        if mod_pow(g, phi / 2, q) == 1 {
            continue;
        }
        if odd != 1 && mod_pow(g, phi / odd, q) == 1 {
            continue;
        }
        return g;
    }
    unreachable!("no primitive root found");
}

fn negacyclic_roots(q: i64, n: usize) -> (i64, i64) {
    let g = primitive_root(q);
    let two_n = 2i64 * (n as i64);
    assert_eq!((q - 1) % two_n, 0, "q-1 must be divisible by 2N for NTT");
    let exp = (q - 1) / two_n;
    let psi = mod_pow(g, exp, q);
    let omega = mod_mul(psi, psi, q);
    (psi, omega)
}

fn bit_reverse(a: &mut [i64]) {
    let n = a.len();
    let log_n = n.trailing_zeros() as usize;
    for i in 0..n {
        let rev = i.reverse_bits() >> (usize::BITS as usize - log_n);
        if i < rev {
            a.swap(i, rev);
        }
    }
}

fn ntt_in_place(a: &mut [i64], q: i64, omega: i64) {
    let n = a.len();
    bit_reverse(a);
    let mut len = 2;
    while len <= n {
        let log_len = len.trailing_zeros();
        let exp = (n >> log_len) as i64;
        let wlen = mod_pow(omega, exp, q);
        for i in (0..n).step_by(len) {
            let mut w = 1i64;
            for j in 0..(len / 2) {
                let u = a[i + j];
                let v = mod_mul(a[i + j + len / 2], w, q);
                a[i + j] = mod_add(u, v, q);
                a[i + j + len / 2] = mod_add(u, -v, q);
                w = mod_mul(w, wlen, q);
            }
        }
        len <<= 1;
    }
}

fn intt_in_place(a: &mut [i64], q: i64, omega: i64) {
    let n = a.len();
    let omega_inv = mod_pow(omega, q - 2, q);
    ntt_in_place(a, q, omega_inv);
    let n_inv = mod_pow(n as i64, q - 2, q);
    for v in a.iter_mut() {
        *v = mod_mul(*v, n_inv, q);
    }
}

fn ntt_negacyclic(a: &mut [i64], q: i64, psi: i64, _omega: i64) {
    let n = a.len();
    let mut psi_powers = vec![1i64; n];
    for i in 1..n {
        psi_powers[i] = mod_mul(psi_powers[i - 1], psi, q);
    }
    for i in 0..n {
        a[i] = mod_mul(a[i], psi_powers[i], q);
    }
    let omega = mod_mul(psi, psi, q);
    ntt_in_place(a, q, omega);
}

fn intt_negacyclic(a: &mut [i64], q: i64, psi: i64, _omega: i64) {
    let n = a.len();
    let omega = mod_mul(psi, psi, q);
    intt_in_place(a, q, omega);
    let psi_inv = mod_pow(psi, q - 2, q);
    let mut psi_inv_powers = vec![1i64; n];
    for i in 1..n {
        psi_inv_powers[i] = mod_mul(psi_inv_powers[i - 1], psi_inv, q);
    }
    for i in 0..n {
        a[i] = mod_mul(a[i], psi_inv_powers[i], q);
    }
}

fn polynomial_multiply_ntt(a: &[i64], b: &[i64], q: i64, n: usize) -> Vec<i64> {
    assert_eq!(a.len(), n);
    assert_eq!(b.len(), n);

    let (psi, omega) = negacyclic_roots(q, n);

    let mut a_ntt = a.to_vec();
    let mut b_ntt = b.to_vec();

    ntt_negacyclic(&mut a_ntt, q, psi, omega);
    ntt_negacyclic(&mut b_ntt, q, psi, omega);

    for i in 0..n {
        a_ntt[i] = mod_mul(a_ntt[i], b_ntt[i], q);
    }

    intt_negacyclic(&mut a_ntt, q, psi, omega);

    a_ntt
}
