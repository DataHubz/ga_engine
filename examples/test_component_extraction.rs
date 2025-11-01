//! Test component extraction in isolation
//!
//! This debugs whether we can properly extract a single component
//! from an encrypted multivector.

use ga_engine::clifford_fhe::{
    ckks::{decrypt, encrypt, Plaintext},
    encoding::{decode_multivector, encode_multivector},
    keys::keygen,
    operations::extract_component,
    params::CliffordFHEParams,
};

fn main() {
    println!("=================================================================");
    println!("Component Extraction Test");
    println!("=================================================================\n");

    // Set up parameters
    let params = CliffordFHEParams::new_128bit();
    println!("Parameters:");
    println!("  Ring dimension (N): {}", params.n);
    println!("  Scaling factor: 2^{}\n", params.scale.log2() as u32);

    // Generate keys
    println!("Generating keys...");
    let (pk, sk, _evk) = keygen(&params);
    println!("✓ Keys generated\n");

    // Test: Encrypt [1, 2, 3, 4, 5, 6, 7, 8] and extract each component
    println!("Test: Extract each component from [1, 2, 3, 4, 5, 6, 7, 8]");
    println!("-------------------------------------------------------------\n");

    let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    println!("Original multivector: {:?}\n", mv);

    // Encrypt
    let pt_coeffs = encode_multivector(&mv, params.scale, params.n);
    let pt = Plaintext::new(pt_coeffs, params.scale);
    let ct = encrypt(&pk, &pt, &params);
    println!("✓ Encrypted\n");

    // Test extracting each component
    for i in 0..8 {
        println!("Extracting component {}...", i);

        let ct_component = extract_component(&ct, i, &params);
        let pt_component = decrypt(&sk, &ct_component, &params);
        let mv_component = decode_multivector(&pt_component.coeffs, params.scale);

        println!("  Decrypted: {:?}", &mv_component[..4]);
        println!("  Expected: component {} should be {}, rest should be ~0", i, mv[i]);

        // Check if extraction worked
        let value = mv_component[i];
        let error = (value - mv[i]).abs();

        if error < 0.1 {
            println!("  ✅ Component {}: {} (error: {:.2e})", i, value, error);
        } else {
            println!("  ❌ Component {}: {} (expected {}, error: {:.2e})", i, value, mv[i], error);
        }

        // Check that other components are near zero
        let mut max_other_error = 0.0;
        for j in 0..8 {
            if j != i {
                let other_error = mv_component[j].abs();
                if other_error > max_other_error {
                    max_other_error = other_error;
                }
            }
        }

        if max_other_error < 0.1 {
            println!("  ✅ Other components near zero (max: {:.2e})", max_other_error);
        } else {
            println!("  ⚠️  Other components not zero (max: {:.2e})", max_other_error);
        }
        println!();
    }

    println!("=================================================================");
    println!("Summary");
    println!("=================================================================");
    println!("If extraction works correctly:");
    println!("- Each extracted component should match the original value");
    println!("- All other components should be near zero");
    println!("\nThis test helps debug the component extraction logic.");
}
