//! Comprehensive CUDA GPU Homomorphic Division Test for RunPod
//!
//! This test validates the complete CUDA GPU homomorphic division pipeline:
//! - Ciphertext multiplication with relinearization
//! - Newton-Raphson inverse computation
//! - Full homomorphic division (a/b)
//!
//! **Run on RunPod.io with NVIDIA GPU:**
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example test_cuda_division_runpod
//! ```
//!
//! **Requirements:**
//! - NVIDIA GPU with CUDA support (Compute Capability 7.5+)
//! - CUDA Toolkit 12.0 or later
//! - Recommended: RTX 5090, RTX 4090, or A100

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    use ga_engine::clifford_fhe_v2::backends::gpu_cuda::ckks::CudaCkksContext;
    use ga_engine::clifford_fhe_v2::backends::gpu_cuda::inversion::{
        multiply_ciphertexts_gpu, newton_raphson_inverse_gpu, scalar_division_gpu,
    };
    use ga_engine::clifford_fhe_v2::backends::gpu_cuda::relin_keys::CudaRelinKeys;
    use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
    use std::time::Instant;

    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     CUDA GPU Homomorphic Division - RunPod Test Suite                 ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    // Step 1: Initialize parameters
    println!("Step 1: Initializing FHE parameters...");
    let params = CliffordFHEParams::new_128bit();
    println!("  ✅ Parameters: N={}, {} primes, {}-bit security",
             params.n, params.moduli.len(), 128);
    println!("  Modulus chain depth: {}", params.moduli.len() - 1);
    println!();

    // Step 2: Generate keys
    println!("Step 2: Generating cryptographic keys...");
    let key_ctx = KeyContext::new(params.clone());
    let start = Instant::now();
    let (pk, sk, _evk) = key_ctx.keygen();
    let keygen_time = start.elapsed();
    println!("  ✅ Keys generated in {:.2}s", keygen_time.as_secs_f64());
    println!();

    // Step 3: Initialize CUDA context
    println!("Step 3: Initializing CUDA CKKS context...");
    let start = Instant::now();
    let ctx = CudaCkksContext::new(params.clone())?;
    let init_time = start.elapsed();
    println!("  ✅ CUDA context ready in {:.2}ms", init_time.as_millis());
    println!();

    // Step 4: Generate relinearization keys
    println!("Step 4: Generating CUDA relinearization keys...");
    let start = Instant::now();

    // Extract secret key coefficients as flat vector
    let sk_flat: Vec<u64> = sk.coeffs.iter()
        .flat_map(|rns| rns.values.clone())
        .collect();

    let relin_keys = CudaRelinKeys::new(ctx.device().clone(), params.clone(), sk_flat, 20)?;
    let relin_keygen_time = start.elapsed();
    println!("  ✅ Relinearization keys generated in {:.2}s", relin_keygen_time.as_secs_f64());
    println!();

    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 1: Ciphertext Multiplication");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Test multiplication: 2.0 × 3.0 = 6.0
    let a = 2.0;
    let b = 3.0;
    let expected_product = a * b;

    println!("Test: {:.1} × {:.1} = {:.1}", a, b, expected_product);
    println!();

    // Encode and encrypt
    println!("  Encoding and encrypting...");
    let scale = params.scale;
    let level = (params.moduli.len() - 1) as usize;

    let pt_a = ctx.encode(&[a], scale, level)?;
    let pt_b = ctx.encode(&[b], scale, level)?;

    let ct_a = ctx.encrypt(&pt_a, &pk)?;
    let ct_b = ctx.encrypt(&pt_b, &pk)?;
    println!("  ✅ Encrypted at level {}", level);
    println!();

    // Multiply on GPU
    println!("  Performing GPU multiplication with relinearization...");
    let start = Instant::now();
    let ct_product = multiply_ciphertexts_gpu(&ct_a, &ct_b, &relin_keys, &ctx)?;
    let mult_time = start.elapsed();
    println!("  ✅ GPU multiplication completed in {:.2}ms", mult_time.as_millis());
    println!();

    // Rescale to manage scale
    println!("  Rescaling to manage scale...");
    let ct_product_rescaled = if ct_product.level > 0 {
        ct_product.rescale_to_next(&ctx)?
    } else {
        ct_product
    };
    println!("  Level after rescale: {}", ct_product_rescaled.level);
    println!();

    // Decrypt and decode
    println!("  Decrypting and decoding...");
    let pt_result = ctx.decrypt(&ct_product_rescaled, &sk)?;
    let result = ctx.decode(&pt_result)?;

    let error = (result[0] - expected_product).abs();
    let rel_error = error / expected_product;

    println!();
    println!("Results:");
    println!("  Expected:     {:.10}", expected_product);
    println!("  Computed:     {:.10}", result[0]);
    println!("  Error:        {:.2e}", error);
    println!("  Rel. Error:   {:.2e}", rel_error);

    if rel_error < 1e-6 {
        println!("  ✅ PASS - Multiplication working correctly!");
    } else {
        println!("  ❌ FAIL - Error too large");
        return Err(format!("Multiplication test failed: error = {:.2e}", error));
    }
    println!();

    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 2: Newton-Raphson Inverse");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Test inverse: 1/2.0 = 0.5
    let x = 2.0;
    let expected_inverse = 1.0 / x;
    let initial_guess = 0.5; // Good guess for 1/2

    println!("Test: 1/{:.1} = {:.1}", x, expected_inverse);
    println!("Initial guess: {:.2}", initial_guess);
    println!();

    // Encode and encrypt
    let pt_x = ctx.encode(&[x], scale, level)?;
    let ct_x = ctx.encrypt(&pt_x, &pk)?;
    println!("  ✅ Encrypted input");
    println!();

    // Run Newton-Raphson
    println!("  Running Newton-Raphson iteration (3 iterations)...");
    let iterations = 3;
    let start = Instant::now();
    let ct_inverse = newton_raphson_inverse_gpu(&ct_x, initial_guess, iterations, &relin_keys, &pk, &ctx)?;
    let inverse_time = start.elapsed();
    println!("  ✅ Newton-Raphson completed in {:.2}ms", inverse_time.as_millis());
    println!("  Performed {} iterations", iterations);
    println!();

    // Decrypt and decode
    let pt_inverse = ctx.decrypt(&ct_inverse, &sk)?;
    let result_inverse = ctx.decode(&pt_inverse)?;

    let error_inv = (result_inverse[0] - expected_inverse).abs();
    let rel_error_inv = error_inv / expected_inverse;

    println!("Results:");
    println!("  Expected:     {:.10}", expected_inverse);
    println!("  Computed:     {:.10}", result_inverse[0]);
    println!("  Error:        {:.2e}", error_inv);
    println!("  Rel. Error:   {:.2e}", rel_error_inv);

    if rel_error_inv < 1e-3 {
        println!("  ✅ PASS - Inverse computation working!");
    } else {
        println!("  ❌ FAIL - Error too large");
        return Err(format!("Inverse test failed: error = {:.2e}", error_inv));
    }
    println!();

    println!("════════════════════════════════════════════════════════════════════════");
    println!("TEST 3: Complete Homomorphic Division");
    println!("════════════════════════════════════════════════════════════════════════\n");

    // Test division: 6.0 / 2.0 = 3.0
    let numerator = 6.0;
    let denominator = 2.0;
    let expected_quotient = numerator / denominator;
    let initial_guess_div = 0.5; // Guess for 1/2

    println!("Test: {:.1} / {:.1} = {:.1}", numerator, denominator, expected_quotient);
    println!();

    // Encode and encrypt
    let pt_num = ctx.encode(&[numerator], scale, level)?;
    let pt_den = ctx.encode(&[denominator], scale, level)?;

    let ct_num = ctx.encrypt(&pt_num, &pk)?;
    let ct_den = ctx.encrypt(&pt_den, &pk)?;
    println!("  ✅ Encrypted numerator and denominator");
    println!();

    // Perform division on GPU
    println!("  Performing complete GPU division...");
    let start = Instant::now();
    let ct_quotient = scalar_division_gpu(&ct_num, &ct_den, initial_guess_div, iterations, &relin_keys, &pk, &ctx)?;
    let div_time = start.elapsed();
    println!("  ✅ GPU division completed in {:.2}ms", div_time.as_millis());
    println!();

    // Decrypt and decode
    let pt_quotient = ctx.decrypt(&ct_quotient, &sk)?;
    let result_quotient = ctx.decode(&pt_quotient)?;

    let error_div = (result_quotient[0] - expected_quotient).abs();
    let rel_error_div = error_div / expected_quotient;

    println!("Results:");
    println!("  Expected:     {:.10}", expected_quotient);
    println!("  Computed:     {:.10}", result_quotient[0]);
    println!("  Error:        {:.2e}", error_div);
    println!("  Rel. Error:   {:.2e}", rel_error_div);

    if rel_error_div < 1e-3 {
        println!("  ✅ PASS - Division working correctly!");
    } else {
        println!("  ❌ FAIL - Error too large");
        return Err(format!("Division test failed: error = {:.2e}", error_div));
    }
    println!();

    println!("════════════════════════════════════════════════════════════════════════");
    println!("PERFORMANCE SUMMARY");
    println!("════════════════════════════════════════════════════════════════════════\n");

    println!("Operation Timings:");
    println!("  Key Generation:      {:.2}s", keygen_time.as_secs_f64());
    println!("  Relin Key Gen:       {:.2}s", relin_keygen_time.as_secs_f64());
    println!("  Multiplication:      {:.2}ms", mult_time.as_millis());
    println!("  Newton-Raphson:      {:.2}ms", inverse_time.as_millis());
    println!("  Complete Division:   {:.2}ms", div_time.as_millis());
    println!();

    println!("════════════════════════════════════════════════════════════════════════");
    println!("✅ ALL TESTS PASSED - CUDA GPU DIVISION FULLY FUNCTIONAL");
    println!("════════════════════════════════════════════════════════════════════════\n");

    println!("Implementation Status: READY");
    println!("GPU Acceleration: VERIFIED");
    println!();

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                 CUDA GPU Division Test - RunPod                       ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");
    println!("❌ This test requires the 'v2' and 'v2-gpu-cuda' features.");
    println!();
    println!("Run with:");
    println!("  cargo run --release --features v2,v2-gpu-cuda --example test_cuda_division_runpod");
    println!();
    println!("Requirements:");
    println!("  • NVIDIA GPU with CUDA support (Compute Capability 7.5+)");
    println!("  • CUDA Toolkit 12.0 or later");
    println!();
    println!("For RunPod users:");
    println!("  1. Select RTX 5090, RTX 4090, or A100 GPU pod");
    println!("  2. Install CUDA Toolkit");
    println!("  3. Clone repository and run this test");
}
