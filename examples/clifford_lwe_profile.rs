//! Profile Clifford-LWE-256 to identify bottlenecks

use ga_engine::clifford_ring::{CliffordPolynomial, CliffordRingElement};
use rand::Rng;
use std::time::Instant;

struct CLWEParams {
    n: usize,
    q: f64,
    error_stddev: f64,
}

impl Default for CLWEParams {
    fn default() -> Self {
        Self {
            n: 32,
            q: 3329.0,
            error_stddev: 1.0,
        }
    }
}

struct PublicKey {
    a: CliffordPolynomial,
    b: CliffordPolynomial,
}

fn random_discrete_poly(n: usize) -> CliffordPolynomial {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut mv = [0.0; 8];
        for i in 0..8 {
            let val: f64 = rng.gen_range(0..3) as f64 - 1.0;
            mv[i] = val;
        }
        coeffs.push(CliffordRingElement::from_multivector(mv));
    }
    CliffordPolynomial::new(coeffs)
}

fn gaussian_error_poly(n: usize, stddev: f64) -> CliffordPolynomial {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut mv = [0.0; 8];
        for i in 0..8 {
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            mv[i] = z * stddev;
        }
        coeffs.push(CliffordRingElement::from_multivector(mv));
    }
    CliffordPolynomial::new(coeffs)
}

fn scale_poly(poly: &CliffordPolynomial, scalar: f64) -> CliffordPolynomial {
    let coeffs: Vec<_> = poly.coeffs.iter()
        .map(|c| c.scalar_mul(scalar))
        .collect();
    CliffordPolynomial::new(coeffs)
}

fn main() {
    println!("=== Clifford-LWE-256 Performance Profiling ===\n");

    let params = CLWEParams::default();
    const ITERS: usize = 1000;

    // Generate public key (one-time)
    let mut a = random_discrete_poly(params.n);
    a.reduce_modulo_xn_minus_1(params.n);
    let mut b = random_discrete_poly(params.n);
    b.reduce_modulo_xn_minus_1(params.n);
    let pk = PublicKey { a, b };

    // Generate message
    let mut msg_coeffs = Vec::with_capacity(params.n);
    for i in 0..params.n {
        let mut mv = [0.0; 8];
        mv[0] = if i % 3 == 0 { 1.0 } else { 0.0 };
        msg_coeffs.push(CliffordRingElement::from_multivector(mv));
    }
    let message = CliffordPolynomial::new(msg_coeffs);

    println!("Profiling encryption operation ({} iterations)...\n", ITERS);

    // Profile: Random discrete polynomial generation
    let start = Instant::now();
    for _ in 0..ITERS {
        let mut r = random_discrete_poly(params.n);
        r.reduce_modulo_xn_minus_1(params.n);
        std::hint::black_box(r);
    }
    let discrete_time = start.elapsed();
    let discrete_avg = discrete_time.as_micros() as f64 / ITERS as f64;

    // Profile: Gaussian error generation
    let start = Instant::now();
    for _ in 0..ITERS {
        let mut e = gaussian_error_poly(params.n, params.error_stddev);
        e.reduce_modulo_xn_minus_1(params.n);
        std::hint::black_box(e);
    }
    let gaussian_time = start.elapsed();
    let gaussian_avg = gaussian_time.as_micros() as f64 / ITERS as f64;

    // Profile: Message scaling
    let start = Instant::now();
    for _ in 0..ITERS {
        let scaled = scale_poly(&message, params.q / 2.0);
        std::hint::black_box(scaled);
    }
    let scaling_time = start.elapsed();
    let scaling_avg = scaling_time.as_micros() as f64 / ITERS as f64;

    // Profile: Polynomial multiplication (a * r)
    let r = random_discrete_poly(params.n);
    let start = Instant::now();
    for _ in 0..ITERS {
        let mut product = pk.a.multiply_karatsuba(&r);
        product.reduce_modulo_xn_minus_1(params.n);
        std::hint::black_box(product);
    }
    let multiply_time = start.elapsed();
    let multiply_avg = multiply_time.as_micros() as f64 / ITERS as f64;

    // Profile: Polynomial addition
    let e1 = gaussian_error_poly(params.n, params.error_stddev);
    let start = Instant::now();
    for _ in 0..ITERS {
        let sum = pk.a.add(&e1);
        std::hint::black_box(sum);
    }
    let add_time = start.elapsed();
    let add_avg = add_time.as_micros() as f64 / ITERS as f64;

    // Profile: Full encryption
    let start = Instant::now();
    for _ in 0..ITERS {
        // Random
        let mut r = random_discrete_poly(params.n);
        r.reduce_modulo_xn_minus_1(params.n);

        let mut e1 = gaussian_error_poly(params.n, params.error_stddev);
        e1.reduce_modulo_xn_minus_1(params.n);

        let mut e2 = gaussian_error_poly(params.n, params.error_stddev);
        e2.reduce_modulo_xn_minus_1(params.n);

        let scaled_msg = scale_poly(&message, params.q / 2.0);

        // u = a * r + e1
        let mut u = pk.a.multiply_karatsuba(&r);
        u.reduce_modulo_xn_minus_1(params.n);
        u = u.add(&e1);

        // v = b * r + e2 + scaled_msg
        let mut v = pk.b.multiply_karatsuba(&r);
        v.reduce_modulo_xn_minus_1(params.n);
        v = v.add(&e2);
        v = v.add(&scaled_msg);

        std::hint::black_box((u, v));
    }
    let full_time = start.elapsed();
    let full_avg = full_time.as_micros() as f64 / ITERS as f64;

    // Results
    println!("=== Breakdown ===\n");
    println!("Operation                      | Time (µs) | % of Total");
    println!("-------------------------------|-----------|------------");
    println!("Random discrete poly (r)       | {:7.2}   | {:5.1}%", discrete_avg, 100.0 * discrete_avg / full_avg);
    println!("Gaussian error (e1)            | {:7.2}   | {:5.1}%", gaussian_avg, 100.0 * gaussian_avg / full_avg);
    println!("Gaussian error (e2)            | {:7.2}   | {:5.1}%", gaussian_avg, 100.0 * gaussian_avg / full_avg);
    println!("Message scaling                | {:7.2}   | {:5.1}%", scaling_avg, 100.0 * scaling_avg / full_avg);
    println!("Karatsuba multiply (a*r)       | {:7.2}   | {:5.1}%", multiply_avg, 100.0 * multiply_avg / full_avg);
    println!("Karatsuba multiply (b*r)       | {:7.2}   | {:5.1}%", multiply_avg, 100.0 * multiply_avg / full_avg);
    println!("Polynomial additions (×3)      | {:7.2}   | {:5.1}%", 3.0 * add_avg, 100.0 * 3.0 * add_avg / full_avg);
    println!("-------------------------------|-----------|------------");
    println!("TOTAL (measured)               | {:7.2}   | 100.0%", full_avg);
    println!();

    let estimated = discrete_avg + 2.0 * gaussian_avg + scaling_avg + 2.0 * multiply_avg + 3.0 * add_avg;
    println!("Estimated from components: {:.2} µs", estimated);
    println!("Actual full encryption:    {:.2} µs", full_avg);
    println!("Overhead:                  {:.2} µs ({:.1}%)",
        full_avg - estimated,
        100.0 * (full_avg - estimated) / full_avg);

    println!("\n=== Optimization Opportunities ===\n");

    let rng_total = discrete_avg + 2.0 * gaussian_avg;
    println!("1. Random number generation: {:.2} µs ({:.1}%)", rng_total, 100.0 * rng_total / full_avg);
    println!("   → Use hardware RNG or faster PRNG");
    println!("   → Expected gain: 5-10 µs\n");

    let multiply_total = 2.0 * multiply_avg;
    println!("2. Polynomial multiplication: {:.2} µs ({:.1}%)", multiply_total, 100.0 * multiply_total / full_avg);
    println!("   → Precompute for fixed public key");
    println!("   → Expected gain: ~{:.2} µs (eliminate one multiply)\n", multiply_avg);

    println!("3. Message scaling: {:.2} µs ({:.1}%)", scaling_avg, 100.0 * scaling_avg / full_avg);
    println!("   → Precompute scaled message");
    println!("   → Expected gain: ~{:.2} µs\n", scaling_avg);

    println!("4. Memory allocations:");
    println!("   → Use stack buffers for small polynomials");
    println!("   → Reuse buffers across operations");
    println!("   → Expected gain: 2-5 µs\n");

    let potential_savings = multiply_avg + scaling_avg + 5.0;
    let optimized_time = full_avg - potential_savings;
    println!("=== Potential Performance ===\n");
    println!("Current:              {:.2} µs", full_avg);
    println!("With optimizations:   {:.2} µs (estimated)", optimized_time);
    println!("Speedup:              {:.2}×", full_avg / optimized_time);
    println!("vs Kyber-512 (15µs):  {:.1}× slower (vs current {:.1}×)",
        optimized_time / 15.0,
        full_avg / 15.0);
}
