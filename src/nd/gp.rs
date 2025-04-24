//! src/nd/gp.rs
//!
//! Geometric‐product table dispatcher for N dimensions.
//!
//! At runtime, builds a lookup table of size 2ⁿ×2ⁿ mapping
//! (i, j) → (sign, k) for blade multiplication: e_i * e_j = sign·e_k.

use crate::nd::types::Scalar;

/// Build a runtime GP‐table for a given dimension `n`.
///
/// Returns a Vec of length (2ⁿ)*(2ⁿ), indexed by `i * 2ⁿ + j`.
pub fn make_gp_table(n: usize) -> Vec<(Scalar, usize)> {
    let m = 1 << n;
    let mut table = Vec::with_capacity(m * m);
    for i in 0..m {
        for j in 0..m {
            table.push(sign_and_index(i, j, n));
        }
    }
    table
}

/// GP table for 2-D GA: 4 blades → 16 entries.
pub fn gp_table_2() -> Vec<(Scalar, usize)> {
    make_gp_table(2)
}

/// GP table for 3-D GA: 8 blades → 64 entries.
pub fn gp_table_3() -> Vec<(Scalar, usize)> {
    make_gp_table(3)
}

/// GP table for 4-D GA: 16 blades → 256 entries.
pub fn gp_table_4() -> Vec<(Scalar, usize)> {
    make_gp_table(4)
}

/// Count the sign (±1) and compute the output blade index for `i * j`.
///
/// Blades are represented by bitmasks in [0..2ⁿ).  The result mask is `i ^ j`,
/// and the sign is determined by the parity of swaps when reordering basis bits.
fn sign_and_index(i: usize, j: usize, n: usize) -> (Scalar, usize) {
    let mi = i;
    let mj = j;
    let k = mi ^ mj;

    // Compute sign by counting how many times basis vectors in `i`
    // must jump over those in `j` (parity of bit‐swaps).
    let mut sgn = 1i32;
    for bit in 0..n {
        if ((mi >> bit) & 1) != 0 {
            let mut lower = mj & ((1 << bit) - 1);
            let mut cnt = 0;
            while lower != 0 {
                cnt += lower & 1;
                lower >>= 1;
            }
            if (cnt & 1) != 0 {
                sgn = -sgn;
            }
        }
    }

    let sign: Scalar = if sgn > 0 { 1.0 } else { -1.0 };
    (sign, k)
}
