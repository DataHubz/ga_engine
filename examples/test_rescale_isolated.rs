//! Test rescaling in isolation to verify it works correctly

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::rns::{RnsPolynomial, rns_rescale_exact, precompute_rescale_inv};

fn main() {
    let params = CliffordFHEParams::new_rns_mult();
    let primes = &params.moduli;
    let n = params.n;

    // Create a simple polynomial: [6×Δ², 0, 0, ...]
    // With Δ = 2^40, this is 6 × 2^80
    let delta_squared = params.scale * params.scale;
    let value_before_rescale = (6.0 * delta_squared).round() as i128;

    println!("Testing rescale with:");
    println!("  Δ = {:.3e}", params.scale);
    println!("  Δ² = {:.3e}", delta_squared);
    println!("  Value before rescale: 6×Δ² = {}", value_before_rescale);
    println!("  q_last = {}", primes[2]);
    println!("  Expected after rescale: 6×Δ²/q_last = {:.3e}\n", value_before_rescale as f64 / primes[2] as f64);

    // Create polynomial in RNS form with this value
    let mut coeffs = vec![0i64; n];
    coeffs[0] = value_before_rescale as i64;  // This will wrap, but that's OK for RNS

    let poly = RnsPolynomial::from_coeffs(&coeffs, primes, n, 0);

    println!("Polynomial before rescale (residues of coeffs[0]):");
    for j in 0..poly.num_primes() {
        println!("  mod q_{}: {}", j, poly.rns_coeffs[0][j]);
    }

    // Rescale
    let inv = precompute_rescale_inv(primes);
    let rescaled = rns_rescale_exact(&poly, primes, &inv);

    println!("\nPolynomial after rescale (residues of coeffs[0]):");
    for j in 0..rescaled.num_primes() {
        let qi = primes[j];
        let r = rescaled.rns_coeffs[0][j];
        let centered = if r > qi / 2 { r - qi } else { r };
        println!("  mod q_{}: {} (centered: {})", j, r, centered);
    }

    // Decode using single prime
    let decoded = rescaled.to_coeffs_single_prime(0, primes[0])[0];
    println!("\nDecoded value (from q_0): {}", decoded);
    println!("Expected: {:.0}", value_before_rescale as f64 / primes[2] as f64);
    println!("Ratio: {:.6}", decoded as f64 / (value_before_rescale as f64 / primes[2] as f64));
}
