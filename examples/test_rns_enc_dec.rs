use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::ckks_rns::{rns_encrypt, rns_decrypt, RnsPlaintext};
use ga_engine::clifford_fhe::keys_rns::rns_generate_keys;
use ga_engine::clifford_fhe::rns::mod_inverse;

fn main() {
    let params = CliffordFHEParams::new_rns_mult();
    let (sk, pk, _evk) = rns_generate_keys(&params);

    // Encrypt [2]
    let value = 2.0;
    let scaled = (value * params.scale).round() as i64;
    let coeffs = vec![scaled; params.n];  // Simple scalar encoding
    let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
    let ct = rns_encrypt(&pk, &pt, &params);
    
    println!("Encrypted value: {}", value);
    println!("Scale: {:.2e}\n", ct.scale);

    // Decrypt
    let pt_dec = rns_decrypt(&sk, &ct, &params);
    
    // CRT reconstruction
    let c0 = pt_dec.coeffs.rns_coeffs[0][0] as i128;
    let c1 = pt_dec.coeffs.rns_coeffs[0][1] as i128;
    let c2 = pt_dec.coeffs.rns_coeffs[0][2] as i128;
    
    let q0 = params.moduli[0] as i128;
    let q1 = params.moduli[1] as i128;
    let q2 = params.moduli[2] as i128;
    let Q = q0 * q1 * q2;
    
    println!("Decrypted RNS coefficients:");
    println!("  mod q_0: {}", c0);
    println!("  mod q_1: {}", c1);
    println!("  mod q_2: {}", c2);
    
    // Full CRT
    let Q0 = Q / q0;
    let Q1 = Q / q1;
    let Q2 = Q / q2;
    let Q0_inv = mod_inverse(Q0, q0);
    let Q1_inv = mod_inverse(Q1, q1);
    let Q2_inv = mod_inverse(Q2, q2);
    
    let mut c = ((c0 * Q0 % Q) * Q0_inv % Q + (c1 * Q1 % Q) * Q1_inv % Q + (c2 * Q2 % Q) * Q2_inv % Q) % Q;
    if c > Q / 2 {
        c -= Q;
    }
    
    println!("\nCRT reconstruction: {}", c);
    println!("Expected (value * scale): {:.0}", value * pt_dec.scale);
    println!("Recovered: {:.6}", (c as f64) / pt_dec.scale);
    println!("Expected: {:.6}", value);
}
