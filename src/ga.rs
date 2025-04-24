//! Geometric Algebra operations for 3D Euclidean space.
//!
//! This optimized implementation uses a compile-time-generated lookup table
//! to perform the full 8×8 multivector geometric product without bit-twiddling.

/// Bitmask for each GA basis blade: [1, e1, e2, e3, e23, e31, e12, e123]
const BLADE_MASKS: [u8; 8] = [0, 1, 2, 4, 6, 5, 3, 7];

/// Mapping from bitmask back to blade index in the multivector array.
const MASK2INDEX: [usize; 8] = [0, 1, 2, 6, 3, 5, 4, 7];

/// Compute the (sign, index) pair for blade i × blade j in a const context.
const fn sign_and_index(i: usize, j: usize) -> (f64, usize) {
    let mi = BLADE_MASKS[i];
    let mj = BLADE_MASKS[j];
    let k_mask = mi ^ mj;
    let k = MASK2INDEX[k_mask as usize];

    // Grade-0 (scalar) interaction
    if i == 0 || j == 0 {
        return (1.0, k);
    }

    // Grade-1 × grade-1: vector dot and wedge
    if i < 4 && j < 4 {
        if i == j {
            // e₁·e₁ = e₂·e₂ = e₃·e₃ = 1
            return (1.0, k);
        }
        // cyclic sign rule: e1→e2→e3→e1
        let i1 = (i - 1) as u32;
        let j1 = (j - 1) as u32;
        let diff = (j1 + 3 - i1) % 3;
        let sign = if diff == 1 { 1.0 } else { -1.0 };
        return (sign, k);
    }

    // Fallback: general blade×blade via bit-count ordering
    let mut sgn = 1i32;
    let mut bit = 0;
    while bit < 3 {
        if ((mi >> bit) & 1) == 1 {
            let mut lower = mj & ((1 << bit) - 1);
            let mut cnt = 0u8;
            while lower != 0 {
                cnt = cnt.wrapping_add(lower & 1);
                lower >>= 1;
            }
            if (cnt & 1) == 1 {
                sgn = -sgn;
            }
        }
        bit += 1;
    }
    (sgn as f64, k)
}

/// Build the full table of blade-pair products at compile time.
const fn make_gp_pairs() -> [(usize, usize, f64, usize); 64] {
    let mut table = [(0, 0, 0.0, 0); 64];
    let mut idx = 0;
    while idx < 64 {
        let i = idx / 8;
        let j = idx % 8;
        let (sign, k) = sign_and_index(i, j);
        table[idx] = (i, j, sign, k);
        idx += 1;
    }
    table
}

/// Lookup table of all 8×8 blade-pair products: (i, j, sign, k).
const GP_PAIRS: [(usize, usize, f64, usize); 64] = make_gp_pairs();

/// Compute the full 3D multivector geometric product in a tight loop.
///
/// # Arguments
/// - `a`, `b`: 8-component multivectors in the order
///   `[scalar, e1, e2, e3, e23, e31, e12, e123]`
/// - `out`: pre-allocated 8-element buffer for the result
#[inline(always)]
pub fn geometric_product_full(a: &[f64; 8], b: &[f64; 8], out: &mut [f64; 8]) {
    // Zero the output buffer
    *out = [0.0; 8];

    // Single pass over all precomputed blade products
    let mut idx = 0;
    while idx < 64 {
        let (i, j, sign, k) = GP_PAIRS[idx];
        out[k] += sign * a[i] * b[j];
        idx += 1;
    }
}

/// Compatibility wrapper for existing tests: takes slice inputs, calls
/// `geometric_product_full`, returns a `Vec<f64>`.
///
/// Panics if either slice is not length 8.
pub fn geometric_product(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert!(a.len() == 8 && b.len() == 8, "Expected 8 components for 3D multivectors");
    let mut a8 = [0.0; 8];
    let mut b8 = [0.0; 8];
    a8.copy_from_slice(a);
    b8.copy_from_slice(b);
    let mut out = [0.0; 8];
    geometric_product_full(&a8, &b8, &mut out);
    out.to_vec()
}
