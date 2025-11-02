//! Verify that encrypt/decrypt preserves the scaled coefficient

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt};

fn main() {
    let params = CliffordFHEParams::new_rns_mult();

    let (pk, sk, _evk) = rns_keygen(&params);

    let msg = 5.0;
    let scaled = (msg * params.scale).round() as i64;

    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = scaled;

    let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);

    println!("Original:");
    println!("  Message: {}", msg);
    println!("  Scaled coeff: {}", scaled);
    println!("  Plaintext RNS[0]: {:?}\n", pt.coeffs.rns_coeffs[0]);

    let ct = rns_encrypt(&pk, &pt, &params);
    let pt_dec = rns_decrypt(&sk, &ct, &params);

    println!("After encrypt/decrypt:");
    println!("  Decrypted RNS[0]: {:?}\n", pt_dec.coeffs.rns_coeffs[0]);

    let coeffs_dec = pt_dec.to_coeffs(&params.moduli);
    println!("  Decrypted coeff: {}", coeffs_dec[0]);
    println!("  Original coeff:  {}", scaled);
    println!("  Difference:      {}\n", coeffs_dec[0] - scaled);

    let msg_dec = coeffs_dec[0] as f64 / params.scale;
    println!("  Decoded message: {:.6}", msg_dec);
    println!("  Original message: {:.6}", msg);
    println!("  Error: {:.6}", (msg_dec - msg).abs());
}
