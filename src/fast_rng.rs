//! Fast random number generation for Clifford-LWE
//!
//! Optimizations:
//! 1. Thread-local static RNG (avoid initialization overhead)
//! 2. Batch generation (reduce function call overhead)
//! 3. Optimized Box-Muller (fewer transcendental functions)

use rand::Rng;
use std::cell::RefCell;

thread_local! {
    static RNG: RefCell<rand::rngs::ThreadRng> = RefCell::new(rand::thread_rng());
}

/// Generate discrete values in {-1, 0, 1} efficiently
#[inline]
pub fn gen_discrete(count: usize) -> Vec<f64> {
    RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            result.push((rng.gen_range(0..3) as f64) - 1.0);
        }
        result
    })
}

/// Generate Gaussian random values using optimized Box-Muller
#[inline]
pub fn gen_gaussian(count: usize, stddev: f64) -> Vec<f64> {
    RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        let mut result = Vec::with_capacity(count);

        // Box-Muller generates pairs, so process in pairs
        let pairs = (count + 1) / 2;

        for _ in 0..pairs {
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();

            // Box-Muller transform (generates 2 values)
            let r = (-2.0 * u1.ln()).sqrt() * stddev;
            let theta = 2.0 * std::f64::consts::PI * u2;

            result.push(r * theta.cos());
            if result.len() < count {
                result.push(r * theta.sin());
            }
        }

        result.truncate(count);
        result
    })
}

/// Generate uniform random values [0, 1)
#[inline]
pub fn gen_uniform(count: usize) -> Vec<f64> {
    RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        (0..count).map(|_| rng.gen::<f64>()).collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_range() {
        let values = gen_discrete(1000);
        for v in values {
            assert!(v == -1.0 || v == 0.0 || v == 1.0);
        }
    }

    #[test]
    fn test_gaussian_mean() {
        let values = gen_gaussian(10000, 1.0);
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        // Mean should be close to 0
        assert!(mean.abs() < 0.1);
    }
}
