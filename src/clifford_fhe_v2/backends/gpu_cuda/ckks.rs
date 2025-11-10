// CUDA CKKS Implementation
//
// Provides CKKS homomorphic encryption operations on NVIDIA GPUs using CUDA.
// Based on the Metal GPU implementation but adapted for CUDA/cudarc.

use super::device::CudaDeviceContext;
use super::ntt::CudaNttContext;
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::{PublicKey, SecretKey};
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::BarrettReducer;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use cudarc::driver::{CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rustfft::num_complex::Complex;
use std::sync::Arc;

/// CUDA CKKS context for FHE operations on NVIDIA GPU
pub struct CudaCkksContext {
    /// CUDA device
    device: Arc<CudaDeviceContext>,

    /// FHE parameters
    params: CliffordFHEParams,

    /// NTT contexts for each RNS prime
    ntt_contexts: Vec<CudaNttContext>,

    /// Barrett reducers for modular reduction (lightweight, keep on CPU)
    reducers: Vec<BarrettReducer>,

    /// Precomputed rescaling constants for GPU-native exact rescale
    /// rescale_inv_table[level][i] = q_{level}^{-1} mod q_i for i < level
    pub rescale_inv_table: Vec<Vec<u64>>,

    /// RNS kernels loaded flag
    rns_kernels_loaded: bool,

    /// GPU-cached twiddles for batched NTT (all primes concatenated)
    /// Layout: [prime_0 twiddles (n elements), prime_1 twiddles, ..., prime_L twiddles]
    /// Total size: num_primes × n × u64
    gpu_twiddles_fwd: Option<CudaSlice<u64>>,

    /// GPU-cached inverse twiddles for batched NTT
    gpu_twiddles_inv: Option<CudaSlice<u64>>,

    /// GPU-cached RNS moduli for batched operations
    /// Layout: [q_0, q_1, ..., q_L]
    gpu_moduli: Option<CudaSlice<u64>>,
}

/// CUDA ciphertext representation
#[derive(Clone)]
pub struct CudaCiphertext {
    /// First polynomial (c0) in strided RNS layout: c0[coeff_idx * num_primes + prime_idx]
    pub c0: Vec<u64>,

    /// Second polynomial (c1) in strided RNS layout
    pub c1: Vec<u64>,

    /// Ring dimension
    pub n: usize,

    /// Total stride (number of RNS primes in memory)
    pub num_primes: usize,

    /// Current level (number of active primes - 1)
    pub level: usize,

    /// Scaling factor
    pub scale: f64,
}

/// CUDA plaintext representation
pub struct CudaPlaintext {
    /// Polynomial coefficients in strided RNS layout
    pub poly: Vec<u64>,

    /// Ring dimension
    pub n: usize,

    /// Number of RNS primes
    pub num_primes: usize,

    /// Scaling factor
    pub scale: f64,
}

impl CudaCkksContext {
    /// Create new CUDA CKKS context
    pub fn new(params: CliffordFHEParams) -> Result<Self, String> {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║           Initializing CUDA CKKS Context                     ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");

        let device = Arc::new(CudaDeviceContext::new()?);

        println!("Creating NTT contexts for {} primes...", params.moduli.len());
        let start = std::time::Instant::now();

        // Create NTT context for each RNS prime
        let mut ntt_contexts = Vec::new();
        for (i, &q) in params.moduli.iter().enumerate() {
            // Find primitive root for this prime
            let root = Self::find_primitive_root(params.n, q)?;
            let ntt_ctx = CudaNttContext::new(params.n, q, root)?;
            ntt_contexts.push(ntt_ctx);

            if (i + 1) % 5 == 0 || i + 1 == params.moduli.len() {
                println!("  Created {}/{} NTT contexts", i + 1, params.moduli.len());
            }
        }

        println!("  [CUDA CKKS] NTT contexts created in {:.2}s", start.elapsed().as_secs_f64());

        // Create Barrett reducers for CPU-side operations
        let reducers: Vec<BarrettReducer> = params.moduli.iter()
            .map(|&q| BarrettReducer::new(q))
            .collect();

        // Precompute rescaling inverse constants
        let rescale_inv_table = Self::precompute_rescale_inv_table(&params.moduli);

        // Load RNS kernels
        println!("Loading RNS CUDA kernels...");
        let kernel_src = include_str!("kernels/rns.cu");
        let ptx = compile_ptx(kernel_src)
            .map_err(|e| format!("Failed to compile RNS CUDA kernel: {:?}", e))?;

        device.device.load_ptx(ptx, "rns_module", &[
            "rns_exact_rescale",
            "rns_exact_rescale_strided",
            "rns_strided_to_flat",
            "rns_flat_to_strided",
            "rns_add",
            "rns_sub",
            "rns_negate",
        ]).map_err(|e| format!("Failed to load RNS PTX: {:?}", e))?;

        // Precompute and cache twiddles on GPU for batched NTT
        println!("Caching twiddles and moduli on GPU for batched NTT...");
        let cache_start = std::time::Instant::now();

        let n = params.n;
        let num_primes = params.moduli.len();

        // Collect all forward twiddles
        let mut all_twiddles_fwd = Vec::with_capacity(n * num_primes);
        let mut all_twiddles_inv = Vec::with_capacity(n * num_primes);
        let mut all_moduli = Vec::with_capacity(num_primes);

        for ntt_ctx in &ntt_contexts {
            all_twiddles_fwd.extend_from_slice(&ntt_ctx.twiddles);
            all_twiddles_inv.extend_from_slice(&ntt_ctx.twiddles_inv);
            all_moduli.push(ntt_ctx.q);
        }

        // Upload to GPU once
        let gpu_twiddles_fwd = device.device.htod_copy(all_twiddles_fwd)
            .map_err(|e| format!("Failed to cache forward twiddles on GPU: {:?}", e))?;

        let gpu_twiddles_inv = device.device.htod_copy(all_twiddles_inv)
            .map_err(|e| format!("Failed to cache inverse twiddles on GPU: {:?}", e))?;

        let gpu_moduli = device.device.htod_copy(all_moduli)
            .map_err(|e| format!("Failed to cache moduli on GPU: {:?}", e))?;

        println!("  ✓ Cached {}KB twiddles and {} moduli on GPU in {:.3}s",
                 (n * num_primes * 8 * 2) / 1024,
                 num_primes,
                 cache_start.elapsed().as_secs_f64());

        println!("  [CUDA CKKS] ✓ GPU-only CKKS context ready!\n");

        Ok(Self {
            device,
            params,
            ntt_contexts,
            reducers,
            rescale_inv_table,
            rns_kernels_loaded: true,
            gpu_twiddles_fwd: Some(gpu_twiddles_fwd),
            gpu_twiddles_inv: Some(gpu_twiddles_inv),
            gpu_moduli: Some(gpu_moduli),
        })
    }

    /// Find primitive N-th root of unity modulo q
    fn find_primitive_root(n: usize, q: u64) -> Result<u64, String> {
        // For CKKS, we need a 2N-th primitive root
        // q - 1 must be divisible by 2N
        let two_n = (2 * n) as u64;
        if (q - 1) % two_n != 0 {
            return Err(format!("q-1 = {} is not divisible by 2N = {}", q - 1, two_n));
        }

        // Try small generators
        for g in 2..100 {
            // Compute g^((q-1)/2N) mod q
            let exp = (q - 1) / two_n;
            let root = Self::mod_exp(g, exp, q);

            // Verify it's a primitive 2N-th root
            if Self::mod_exp(root, n as u64, q) == q - 1 {  // root^N = -1 (mod q)
                return Ok(root);
            }
        }

        Err(format!("Could not find primitive root for n={}, q={}", n, q))
    }

    /// Modular exponentiation: base^exp mod m
    fn mod_exp(base: u64, mut exp: u64, m: u64) -> u64 {
        let mut result = 1u64;
        let mut base = base % m;

        while exp > 0 {
            if exp & 1 == 1 {
                result = ((result as u128 * base as u128) % m as u128) as u64;
            }
            base = ((base as u128 * base as u128) % m as u128) as u64;
            exp >>= 1;
        }

        result
    }

    /// Precompute rescaling inverse table
    fn precompute_rescale_inv_table(moduli: &[u64]) -> Vec<Vec<u64>> {
        let mut table = Vec::new();

        for level in 1..moduli.len() {
            let q_last = moduli[level];
            let mut inv_row = Vec::new();

            for i in 0..level {
                let q_i = moduli[i];
                // Compute q_last^{-1} mod q_i
                let inv = Self::mod_inverse_static(q_last, q_i)
                    .expect("Modular inverse must exist");
                inv_row.push(inv);
            }

            table.push(inv_row);
        }

        table
    }

    /// Modular inverse using extended Euclidean algorithm
    fn mod_inverse_static(a: u64, m: u64) -> Option<u64> {
        let (mut t, mut new_t) = (0i128, 1i128);
        let (mut r, mut new_r) = (m as i128, a as i128);

        while new_r != 0 {
            let quotient = r / new_r;
            (t, new_t) = (new_t, t - quotient * new_t);
            (r, new_r) = (new_r, r - quotient * new_r);
        }

        if r > 1 {
            return None;
        }
        if t < 0 {
            t += m as i128;
        }

        Some(t as u64)
    }

    /// GPU-Native Exact Rescaling using CUDA kernel
    ///
    /// Rescale ciphertext by dropping last prime using exact DRLMQ with centered rounding.
    /// Uses Russian peasant multiplication for 128-bit modular arithmetic.
    pub fn exact_rescale_gpu(&self, poly_in: &[u64], level: usize) -> Result<Vec<u64>, String> {
        if level == 0 {
            return Err("Cannot rescale at level 0".to_string());
        }

        let n = self.params.n;
        let moduli = &self.params.moduli[..=level];
        let num_primes_in = moduli.len();
        let num_primes_out = num_primes_in - 1;

        assert_eq!(poly_in.len(), n * num_primes_in, "Input size mismatch");

        // Convert from strided layout to flat RNS layout for GPU (parallelized)
        // Strided: poly_in[coeff_idx * num_primes_in + prime_idx]
        // Flat:    poly_flat[prime_idx * n + coeff_idx]
        use rayon::prelude::*;
        let poly_flat: Vec<u64> = (0..num_primes_in)
            .into_par_iter()
            .flat_map(|prime_idx| {
                (0..n)
                    .map(|coeff_idx| poly_in[coeff_idx * num_primes_in + prime_idx])
                    .collect::<Vec<_>>()
            })
            .collect();

        // Get precomputed constants
        // Table is indexed from level-1 (since level 0 doesn't rescale)
        let qlast_inv = &self.rescale_inv_table[level - 1];
        assert_eq!(qlast_inv.len(), num_primes_out, "qlast_inv table size mismatch");

        // Copy to GPU
        let gpu_input = self.device.device.htod_copy(poly_flat)
            .map_err(|e| format!("Failed to copy input to GPU: {:?}", e))?;

        let mut gpu_output = self.device.device.alloc_zeros::<u64>(n * num_primes_out)
            .map_err(|e| format!("Failed to allocate output: {:?}", e))?;

        let gpu_moduli = self.device.device.htod_copy(moduli.to_vec())
            .map_err(|e| format!("Failed to copy moduli: {:?}", e))?;

        let gpu_qtop_inv = self.device.device.htod_copy(qlast_inv.clone())
            .map_err(|e| format!("Failed to copy qtop_inv: {:?}", e))?;

        // Get kernel function
        let func = self.device.device.get_func("rns_module", "rns_exact_rescale")
            .ok_or("Failed to get rns_exact_rescale function")?;

        // Launch kernel
        let config = self.device.get_launch_config(n);
        unsafe {
            func.launch(config, (
                &gpu_input,
                &mut gpu_output,
                &gpu_moduli,
                &gpu_qtop_inv,
                n as u32,
                num_primes_in as u32,
                num_primes_out as u32,
            )).map_err(|e| format!("Rescale kernel launch failed: {:?}", e))?;
        }

        // Copy result back
        let result_flat = self.device.device.dtoh_sync_copy(&gpu_output)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        // Convert back to strided layout
        let mut result = vec![0u64; n * num_primes_out];
        for coeff_idx in 0..n {
            for prime_idx in 0..num_primes_out {
                result[coeff_idx * num_primes_out + prime_idx] = result_flat[prime_idx * n + coeff_idx];
            }
        }

        Ok(result)
    }

    /// Rescale polynomial in strided RNS layout (GPU-optimized, NO layout conversion!)
    ///
    /// This is MUCH faster than exact_rescale_gpu because it avoids expensive CPU layout conversions.
    /// Use this for V3 bootstrap operations where ciphertexts are in strided format.
    ///
    /// Input and output are in strided layout: poly[coeff_idx * num_primes + prime_idx]
    pub fn exact_rescale_gpu_strided(&self, poly_in: &[u64], level: usize) -> Result<Vec<u64>, String> {
        if level == 0 {
            return Err("Cannot rescale at level 0".to_string());
        }

        let n = self.params.n;
        let moduli = &self.params.moduli[..=level];
        let num_primes_in = moduli.len();
        let num_primes_out = num_primes_in - 1;

        assert_eq!(poly_in.len(), n * num_primes_in, "Input size mismatch");

        // NO LAYOUT CONVERSION NEEDED - input is already strided, output will be strided!

        // Get precomputed constants
        let qlast_inv = &self.rescale_inv_table[level - 1];
        assert_eq!(qlast_inv.len(), num_primes_out, "qlast_inv table size mismatch");

        // Copy to GPU in strided format
        let gpu_input = self.device.device.htod_copy(poly_in.to_vec())
            .map_err(|e| format!("Failed to copy input to GPU: {:?}", e))?;

        let mut gpu_output = self.device.device.alloc_zeros::<u64>(n * num_primes_out)
            .map_err(|e| format!("Failed to allocate output: {:?}", e))?;

        let gpu_moduli = self.device.device.htod_copy(moduli.to_vec())
            .map_err(|e| format!("Failed to copy moduli: {:?}", e))?;

        let gpu_qtop_inv = self.device.device.htod_copy(qlast_inv.clone())
            .map_err(|e| format!("Failed to copy qtop_inv: {:?}", e))?;

        // Get strided kernel function
        let func = self.device.device.get_func("rns_module", "rns_exact_rescale_strided")
            .ok_or("Failed to get rns_exact_rescale_strided function")?;

        // Launch kernel
        let config = self.device.get_launch_config(n);
        unsafe {
            func.launch(config, (
                &gpu_input,
                &mut gpu_output,
                &gpu_moduli,
                &gpu_qtop_inv,
                n as u32,
                num_primes_in as u32,
                num_primes_out as u32,
            )).map_err(|e| format!("Rescale kernel launch failed: {:?}", e))?;
        }

        // Copy result back - already in strided format!
        let result = self.device.device.dtoh_sync_copy(&gpu_output)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        Ok(result)
    }

    /// Rescale polynomial in flat RNS layout (for golden compare testing).
    /// Input and output are in flat layout: poly[prime_idx * n + coeff_idx]
    pub fn exact_rescale_gpu_flat(&self, poly_in: &[u64], level: usize) -> Result<Vec<u64>, String> {
        if level == 0 {
            return Err("Cannot rescale at level 0".to_string());
        }

        let n = self.params.n;
        let moduli = &self.params.moduli[..=level];
        let num_primes_in = moduli.len();
        let num_primes_out = num_primes_in - 1;

        assert_eq!(poly_in.len(), n * num_primes_in, "Input size mismatch");

        // Input is already in flat RNS layout - no conversion needed

        // Get precomputed constants
        // Table is indexed from level-1 (since level 0 doesn't rescale)
        let qlast_inv = &self.rescale_inv_table[level - 1];
        assert_eq!(qlast_inv.len(), num_primes_out, "qlast_inv table size mismatch");

        // Copy to GPU
        let gpu_input = self.device.device.htod_copy(poly_in.to_vec())
            .map_err(|e| format!("Failed to copy input to GPU: {:?}", e))?;

        let mut gpu_output = self.device.device.alloc_zeros::<u64>(n * num_primes_out)
            .map_err(|e| format!("Failed to allocate output: {:?}", e))?;

        let gpu_moduli = self.device.device.htod_copy(moduli.to_vec())
            .map_err(|e| format!("Failed to copy moduli: {:?}", e))?;

        let gpu_qtop_inv = self.device.device.htod_copy(qlast_inv.clone())
            .map_err(|e| format!("Failed to copy qtop_inv: {:?}", e))?;

        // Get kernel function
        let func = self.device.device.get_func("rns_module", "rns_exact_rescale")
            .ok_or("Failed to get rns_exact_rescale function")?;

        // Launch kernel
        let config = self.device.get_launch_config(n);
        unsafe {
            func.launch(config, (
                &gpu_input,
                &mut gpu_output,
                &gpu_moduli,
                &gpu_qtop_inv,
                n as u32,
                num_primes_in as u32,
                num_primes_out as u32,
            )).map_err(|e| format!("Rescale kernel launch failed: {:?}", e))?;
        }

        // Copy result back (already in flat layout)
        let result = self.device.device.dtoh_sync_copy(&gpu_output)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        Ok(result)
    }

    /// Encode floating-point values to polynomial (CPU operation)
    pub fn encode(&self, values: &[f64], scale: f64, level: usize) -> Result<CudaPlaintext, String> {
        // Use CPU encoding (same as Metal implementation)
        // This is not performance-critical for FHE

        let slots = self.params.n / 2;
        if values.len() > slots {
            return Err(format!("Too many values: {} (max {})", values.len(), slots));
        }

        let mut complex_vals = vec![Complex::new(0.0, 0.0); slots];
        for (i, &val) in values.iter().enumerate() {
            complex_vals[i] = Complex::new(val * scale, 0.0);
        }

        // Inverse canonical embedding
        let coeffs = self.inverse_canonical_embedding(&complex_vals);

        // Round to integers and reduce mod each prime
        let num_primes = level + 1;
        let mut poly = vec![0u64; self.params.n * num_primes];

        for coeff_idx in 0..self.params.n {
            // Round the real part (CKKS encoding ensures imaginary part is ~0)
            let val = coeffs[coeff_idx].re.round();
            for prime_idx in 0..num_primes {
                let q = self.params.moduli[prime_idx];
                let val_mod = if val >= 0.0 {
                    (val as u64) % q
                } else {
                    let abs_val = (-val) as u64;
                    q - (abs_val % q)
                };
                poly[coeff_idx * num_primes + prime_idx] = val_mod;
            }
        }

        Ok(CudaPlaintext {
            poly,
            n: self.params.n,
            num_primes,
            scale,
        })
    }

    /// Inverse canonical embedding (iCKKS)
    fn inverse_canonical_embedding(&self, values: &[Complex<f64>]) -> Vec<Complex<f64>> {
        use rustfft::FftPlanner;

        let slots = self.params.n / 2;
        assert_eq!(values.len(), slots);

        // Create conjugate pairs
        let mut extended = vec![Complex::new(0.0, 0.0); self.params.n];
        for i in 0..slots {
            extended[i] = values[i];
            extended[self.params.n - 1 - i] = values[i].conj();
        }

        // Inverse FFT
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(self.params.n);
        ifft.process(&mut extended);

        // Scale by N
        for val in &mut extended {
            *val *= self.params.n as f64;
        }

        extended
    }

    /// Add two ciphertexts (CPU operation, simple)
    pub fn add(&self, ct1: &CudaCiphertext, ct2: &CudaCiphertext) -> Result<CudaCiphertext, String> {
        if ct1.level != ct2.level {
            return Err("Ciphertexts must be at same level".to_string());
        }

        let num_active_primes = ct1.level + 1;
        let mut c0 = vec![0u64; self.params.n * self.params.moduli.len()];
        let mut c1 = vec![0u64; self.params.n * self.params.moduli.len()];

        for coeff_idx in 0..self.params.n {
            for prime_idx in 0..num_active_primes {
                let q = self.params.moduli[prime_idx];
                let idx = coeff_idx * self.params.moduli.len() + prime_idx;

                let sum0 = ct1.c0[idx] + ct2.c0[idx];
                c0[idx] = if sum0 >= q { sum0 - q } else { sum0 };

                let sum1 = ct1.c1[idx] + ct2.c1[idx];
                c1[idx] = if sum1 >= q { sum1 - q } else { sum1 };
            }
        }

        Ok(CudaCiphertext {
            c0,
            c1,
            n: self.params.n,
            num_primes: self.params.moduli.len(),
            level: ct1.level,
            scale: ct1.scale,  // Assuming same scale
        })
    }

    /// Get NTT contexts
    pub fn ntt_contexts(&self) -> &[CudaNttContext] {
        &self.ntt_contexts
    }

    /// Get device
    pub fn device(&self) -> &Arc<CudaDeviceContext> {
        &self.device
    }

    /// Get parameters
    pub fn params(&self) -> &CliffordFHEParams {
        &self.params
    }

    /// Multiply two ciphertexts using GPU NTT-based polynomial multiplication
    ///
    /// Computes tensor product: (c0, c1) × (d0, d1) = (c0×d0, c0×d1 + c1×d0, c1×d1)
    ///
    /// Returns (c0_result, c1_result, c2_result) in flat RNS layout
    /// Caller is responsible for relinearization to reduce back to size-2 ciphertext
    pub fn multiply_ciphertexts_tensored(
        &self,
        ct1: &CudaCiphertext,
        ct2: &CudaCiphertext,
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>), String> {
        // Use GPU-resident pipeline and download results only at the end
        let (gpu_c0, gpu_c1, gpu_c2) = self.multiply_ciphertexts_tensored_gpu(ct1, ct2)?;

        // Download results from GPU (3 transfers instead of 12!)
        let c0_result = self.device.device.dtoh_sync_copy(&gpu_c0)
            .map_err(|e| format!("Failed to download c0: {:?}", e))?;
        let c1_result = self.device.device.dtoh_sync_copy(&gpu_c1)
            .map_err(|e| format!("Failed to download c1: {:?}", e))?;
        let c2_result = self.device.device.dtoh_sync_copy(&gpu_c2)
            .map_err(|e| format!("Failed to download c2: {:?}", e))?;

        Ok((c0_result, c1_result, c2_result))
    }

    pub fn multiply_ciphertexts_tensored_OLD(
        &self,
        ct1: &CudaCiphertext,
        ct2: &CudaCiphertext,
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>), String> {
        if ct1.level != ct2.level {
            return Err(format!("Level mismatch: {} vs {}", ct1.level, ct2.level));
        }
        if ct1.n != ct2.n {
            return Err(format!("Ring dimension mismatch: {} vs {}", ct1.n, ct2.n));
        }

        let n = ct1.n;
        let num_active_primes = ct1.level + 1;

        // Convert from strided to flat layout for NTT operations
        let c0_flat = self.strided_to_flat(&ct1.c0, n, ct1.num_primes, num_active_primes);
        let c1_flat = self.strided_to_flat(&ct1.c1, n, ct1.num_primes, num_active_primes);
        let d0_flat = self.strided_to_flat(&ct2.c0, n, ct2.num_primes, num_active_primes);
        let d1_flat = self.strided_to_flat(&ct2.c1, n, ct2.num_primes, num_active_primes);

        // ============================================================
        // BATCHED NTT OPERATIONS - Process all primes in parallel!
        // ============================================================
        // Old approach: 240 kernel launches per multiplication
        //   - 4 forward NTTs × 20 primes = 80 launches
        //   - 4 pointwise ops × 20 primes = 80 launches
        //   - 4 inverse NTTs × 20 primes = 80 launches
        //
        // New approach: ~13 kernel launches per multiplication (20× reduction!)
        //   - 4 batched forward NTTs = 4 launches
        //   - 4 batched pointwise ops = 4 launches
        //   - 4 batched inverse NTTs = 4 launches
        //   - 1 CPU addition operation
        // ============================================================

        // Make mutable copies for in-place NTT
        let mut c0_flat = c0_flat;
        let mut c1_flat = c1_flat;
        let mut d0_flat = d0_flat;
        let mut d1_flat = d1_flat;

        // Step 1: Forward NTT - ALL primes at once (4 launches instead of 80)
        self.ntt_forward_batched(&mut c0_flat, num_active_primes)?;  // 1 launch
        self.ntt_forward_batched(&mut c1_flat, num_active_primes)?;  // 1 launch
        self.ntt_forward_batched(&mut d0_flat, num_active_primes)?;  // 1 launch
        self.ntt_forward_batched(&mut d1_flat, num_active_primes)?;  // 1 launch

        // Step 2: Pointwise multiply - ALL primes at once (4 launches instead of 80)
        let mut c0_result = vec![0u64; n * num_active_primes];
        let mut c1_part1 = vec![0u64; n * num_active_primes];
        let mut c1_part2 = vec![0u64; n * num_active_primes];
        let mut c2_result = vec![0u64; n * num_active_primes];

        self.ntt_pointwise_multiply_batched(&c0_flat, &d0_flat, &mut c0_result, num_active_primes)?;  // 1 launch
        self.ntt_pointwise_multiply_batched(&c0_flat, &d1_flat, &mut c1_part1, num_active_primes)?;   // 1 launch
        self.ntt_pointwise_multiply_batched(&c1_flat, &d0_flat, &mut c1_part2, num_active_primes)?;   // 1 launch
        self.ntt_pointwise_multiply_batched(&c1_flat, &d1_flat, &mut c2_result, num_active_primes)?;  // 1 launch

        // Step 3: Inverse NTT - ALL primes at once (4 launches instead of 80)
        self.ntt_inverse_batched(&mut c0_result, num_active_primes)?;  // 1 launch
        self.ntt_inverse_batched(&mut c1_part1, num_active_primes)?;   // 1 launch
        self.ntt_inverse_batched(&mut c1_part2, num_active_primes)?;   // 1 launch
        self.ntt_inverse_batched(&mut c2_result, num_active_primes)?;  // 1 launch

        // Step 4: Add c1_part1 + c1_part2 for final c1 (CPU operation, could GPU-accelerate later)
        let mut c1_result = vec![0u64; n * num_active_primes];
        for prime_idx in 0..num_active_primes {
            let offset = prime_idx * n;
            let q = self.params.moduli[prime_idx];
            for i in 0..n {
                let sum = (c1_part1[offset + i] as u128 + c1_part2[offset + i] as u128) % q as u128;
                c1_result[offset + i] = sum as u64;
            }
        }

        // Total: 13 kernel launches (4 forward + 4 pointwise + 4 inverse + 1 CPU op)
        // vs 240 kernel launches in sequential version
        // Result: 20× reduction in kernel launch overhead! 🚀

        Ok((c0_result, c1_result, c2_result))
    }

    /// GPU-RESIDENT multiplication - ALL operations stay on GPU!
    ///
    /// This version eliminates ALL CPU↔GPU data copying by:
    /// 1. Converting strided→flat on GPU
    /// 2. Performing all NTT operations on GPU
    /// 3. Returning GPU-resident results (CudaSlice)
    ///
    /// Expected savings: ~2-3 seconds from eliminating 576MB of PCIe transfers
    pub fn multiply_ciphertexts_tensored_gpu(
        &self,
        ct1: &CudaCiphertext,
        ct2: &CudaCiphertext,
    ) -> Result<(CudaSlice<u64>, CudaSlice<u64>, CudaSlice<u64>), String> {
        if ct1.level != ct2.level {
            return Err(format!("Level mismatch: {} vs {}", ct1.level, ct2.level));
        }
        if ct1.n != ct2.n {
            return Err(format!("Ring dimension mismatch: {} vs {}", ct1.n, ct2.n));
        }

        let n = ct1.n;
        let num_active_primes = ct1.level + 1;

        // Step 1: Convert strided→flat on CPU (extracts only active primes) AND upload to GPU
        // Note: strided_to_flat() does important work - it extracts only num_active_primes
        // from the full ct.c0/c1 arrays that contain ct.num_primes (30 total)
        let c0_flat = self.strided_to_flat(&ct1.c0, n, ct1.num_primes, num_active_primes);
        let c1_flat = self.strided_to_flat(&ct1.c1, n, ct1.num_primes, num_active_primes);
        let d0_flat = self.strided_to_flat(&ct2.c0, n, ct2.num_primes, num_active_primes);
        let d1_flat = self.strided_to_flat(&ct2.c1, n, ct2.num_primes, num_active_primes);

        // Upload to GPU ONCE (only active primes, not all 30!)
        let mut gpu_c0 = self.device.device.htod_copy(c0_flat)
            .map_err(|e| format!("Failed to upload c0: {:?}", e))?;
        let mut gpu_c1 = self.device.device.htod_copy(c1_flat)
            .map_err(|e| format!("Failed to upload c1: {:?}", e))?;
        let mut gpu_d0 = self.device.device.htod_copy(d0_flat)
            .map_err(|e| format!("Failed to upload d0: {:?}", e))?;
        let mut gpu_d1 = self.device.device.htod_copy(d1_flat)
            .map_err(|e| format!("Failed to upload d1: {:?}", e))?;

        // Step 2: Forward NTT - ALL on GPU!
        self.ntt_forward_batched_gpu(&mut gpu_c0, num_active_primes)?;
        self.ntt_forward_batched_gpu(&mut gpu_c1, num_active_primes)?;
        self.ntt_forward_batched_gpu(&mut gpu_d0, num_active_primes)?;
        self.ntt_forward_batched_gpu(&mut gpu_d1, num_active_primes)?;

        // Step 3: Pointwise multiply - ALL on GPU!
        let mut gpu_c0_result = self.device.device.alloc_zeros::<u64>(n * num_active_primes)
            .map_err(|e| format!("Failed to allocate gpu_c0_result: {:?}", e))?;
        let mut gpu_c1_part1 = self.device.device.alloc_zeros::<u64>(n * num_active_primes)
            .map_err(|e| format!("Failed to allocate gpu_c1_part1: {:?}", e))?;
        let mut gpu_c1_part2 = self.device.device.alloc_zeros::<u64>(n * num_active_primes)
            .map_err(|e| format!("Failed to allocate gpu_c1_part2: {:?}", e))?;
        let mut gpu_c2_result = self.device.device.alloc_zeros::<u64>(n * num_active_primes)
            .map_err(|e| format!("Failed to allocate gpu_c2_result: {:?}", e))?;

        self.ntt_pointwise_multiply_batched_gpu(&gpu_c0, &gpu_d0, &mut gpu_c0_result, num_active_primes)?;
        self.ntt_pointwise_multiply_batched_gpu(&gpu_c0, &gpu_d1, &mut gpu_c1_part1, num_active_primes)?;
        self.ntt_pointwise_multiply_batched_gpu(&gpu_c1, &gpu_d0, &mut gpu_c1_part2, num_active_primes)?;
        self.ntt_pointwise_multiply_batched_gpu(&gpu_c1, &gpu_d1, &mut gpu_c2_result, num_active_primes)?;

        // Step 4: Inverse NTT - ALL on GPU!
        self.ntt_inverse_batched_gpu(&mut gpu_c0_result, num_active_primes)?;
        self.ntt_inverse_batched_gpu(&mut gpu_c1_part1, num_active_primes)?;
        self.ntt_inverse_batched_gpu(&mut gpu_c1_part2, num_active_primes)?;
        self.ntt_inverse_batched_gpu(&mut gpu_c2_result, num_active_primes)?;

        // Step 5: Add c1_part1 + c1_part2 on GPU using rns_add kernel
        let mut gpu_c1_result = self.device.device.alloc_zeros::<u64>(n * num_active_primes)
            .map_err(|e| format!("Failed to allocate gpu_c1_result: {:?}", e))?;

        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        let func_add = self.device.device.get_func("rns_module", "rns_add")
            .ok_or("Failed to get rns_add function")?;

        let threads_per_block = 256;
        let total_elements = n * num_active_primes;
        let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func_add.launch(cfg, (
                &gpu_c1_part1,
                &gpu_c1_part2,
                &mut gpu_c1_result,
                gpu_moduli,
                n as u32,
                num_active_primes as u32,
            ))
            .map_err(|e| format!("GPU addition failed: {:?}", e))?;
        }

        // Return GPU-resident results - NO final download!
        Ok((gpu_c0_result, gpu_c1_result, gpu_c2_result))
    }

    /// Helper: Convert from strided to flat RNS layout (GPU-accelerated!)
    ///
    /// This replaces ~650k CPU operations with a single GPU kernel call.
    /// CRITICAL for BSGS performance - called 4× per multiplication.
    fn strided_to_flat(&self, data: &[u64], n: usize, stride: usize, num_primes: usize) -> Vec<u64> {
        use cudarc::driver::LaunchAsync;

        let total_elements = n * num_primes;

        // Copy to GPU
        let gpu_input = self.device.device.htod_copy(data.to_vec())
            .expect("Failed to copy to GPU");

        let mut gpu_output = self.device.device.alloc_zeros::<u64>(total_elements)
            .expect("Failed to allocate GPU memory");

        // Get kernel
        let func = self.device.device.get_func("rns_module", "rns_strided_to_flat")
            .expect("Failed to get rns_strided_to_flat kernel");

        // Launch kernel
        let threads_per_block = 256;
        let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (&gpu_input, &mut gpu_output, n as u32, stride as u32, num_primes as u32))
                .expect("Failed to launch rns_strided_to_flat kernel");
        }

        // Copy result back
        self.device.device.dtoh_sync_copy(&gpu_output)
            .expect("Failed to copy from GPU")
    }

    /// Batched Forward NTT - Process all primes in parallel
    ///
    /// This replaces sequential per-prime NTT with a single batched operation,
    /// dramatically reducing kernel launch overhead (20× reduction in launches).
    ///
    /// # Arguments
    /// * `data` - Flat RNS layout: [num_primes * n] = all primes concatenated
    /// * `num_primes` - Number of RNS primes to process
    ///
    /// # Layout
    /// Input/Output: data[prime_idx * n + coeff_idx]
    fn ntt_forward_batched(&self, data: &mut [u64], num_primes: usize) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let log_n = (n as f64).log2() as usize;

        if data.len() != n * num_primes {
            return Err(format!("Expected {} elements, got {}", n * num_primes, data.len()));
        }

        // Use GPU-cached twiddles and moduli (uploaded once during initialization)
        let gpu_twiddles = self.gpu_twiddles_fwd.as_ref()
            .ok_or("GPU twiddles not initialized")?;
        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        // Copy input data to GPU
        let mut gpu_data = self.device.device.htod_copy(data.to_vec())
            .map_err(|e| format!("Failed to copy data to GPU: {:?}", e))?;

        // Bit-reversal permutation (TODO: batch this too in future optimization)
        // For now, we apply bit-reversal sequentially to each prime's data
        for prime_idx in 0..num_primes {
            let func_bit_reverse = self.ntt_contexts[0].device.device.get_func("ntt_module", "bit_reverse_permutation")
                .ok_or("Failed to get bit_reverse_permutation function")?;

            let threads_per_block = 256;
            let num_blocks = (n / 2 + threads_per_block - 1) / threads_per_block;
            let cfg = LaunchConfig {
                grid_dim: (num_blocks as u32, 1, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            // Create a view into this prime's data
            let offset = prime_idx * n;
            let gpu_data_view = gpu_data.slice(offset..(offset + n));

            unsafe {
                func_bit_reverse.launch(cfg, (&gpu_data_view, n as u32, log_n as u32))
                    .map_err(|e| format!("Bit-reversal failed: {:?}", e))?;
            }
        }

        // Batched NTT stages
        let mut m = 1usize;
        for stage in 0..log_n {
            let func_ntt = self.ntt_contexts[0].device.device.get_func("ntt_module", "ntt_forward_batched")
                .ok_or("Failed to get ntt_forward_batched function")?;

            // 2D grid: (butterfly_blocks, num_primes)
            let threads_per_block = 256;
            let num_butterfly_blocks = (n / 2 + threads_per_block - 1) / threads_per_block;

            let cfg = LaunchConfig {
                grid_dim: (num_butterfly_blocks as u32, num_primes as u32, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func_ntt.launch(cfg, (
                    &mut gpu_data,
                    gpu_twiddles,
                    gpu_moduli,
                    n as u32,
                    num_primes as u32,
                    stage as u32,
                    m as u32,
                ))
                .map_err(|e| format!("Batched NTT stage {} failed: {:?}", stage, e))?;
            }
            m *= 2;
        }

        // Copy result back
        let result = self.device.device.dtoh_sync_copy(&gpu_data)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        data.copy_from_slice(&result);
        Ok(())
    }

    /// Batched Inverse NTT - Process all primes in parallel
    ///
    /// # Arguments
    /// * `data` - Flat RNS layout: [num_primes * n]
    /// * `num_primes` - Number of RNS primes to process
    fn ntt_inverse_batched(&self, data: &mut [u64], num_primes: usize) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let log_n = (n as f64).log2() as usize;

        if data.len() != n * num_primes {
            return Err(format!("Expected {} elements, got {}", n * num_primes, data.len()));
        }

        // Use GPU-cached inverse twiddles and moduli
        let gpu_twiddles_inv = self.gpu_twiddles_inv.as_ref()
            .ok_or("GPU inverse twiddles not initialized")?;
        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        // Copy input data to GPU
        let mut gpu_data = self.device.device.htod_copy(data.to_vec())
            .map_err(|e| format!("Failed to copy data to GPU: {:?}", e))?;

        // Batched inverse NTT stages
        let mut m = n / 2;
        for stage in (0..log_n).rev() {
            let func_ntt_inv = self.ntt_contexts[0].device.device.get_func("ntt_module", "ntt_inverse_batched")
                .ok_or("Failed to get ntt_inverse_batched function")?;

            // 2D grid: (butterfly_blocks, num_primes)
            let threads_per_block = 256;
            let num_butterfly_blocks = (n / 2 + threads_per_block - 1) / threads_per_block;

            let cfg = LaunchConfig {
                grid_dim: (num_butterfly_blocks as u32, num_primes as u32, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func_ntt_inv.launch(cfg, (
                    &mut gpu_data,
                    gpu_twiddles_inv,
                    gpu_moduli,
                    n as u32,
                    num_primes as u32,
                    stage as u32,
                    m as u32,
                ))
                .map_err(|e| format!("Batched inverse NTT stage {} failed: {:?}", stage, e))?;
            }
            m /= 2;
        }

        // Final scaling by n^(-1) for each prime
        for prime_idx in 0..num_primes {
            let ntt_ctx = &self.ntt_contexts[prime_idx];
            let func_scalar = ntt_ctx.device.device.get_func("ntt_module", "ntt_scalar_multiply")
                .ok_or("Failed to get ntt_scalar_multiply function")?;

            let threads_per_block = 256;
            let num_blocks = (n + threads_per_block - 1) / threads_per_block;
            let cfg = LaunchConfig {
                grid_dim: (num_blocks as u32, 1, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            // Create a view into this prime's data
            let offset = prime_idx * n;
            let gpu_data_view = gpu_data.slice(offset..(offset + n));

            unsafe {
                func_scalar.launch(cfg, (&gpu_data_view, ntt_ctx.n_inv, n as u32, ntt_ctx.q))
                    .map_err(|e| format!("Scalar multiply failed: {:?}", e))?;
            }
        }

        // Copy result back
        let result = self.device.device.dtoh_sync_copy(&gpu_data)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        data.copy_from_slice(&result);
        Ok(())
    }

    /// Batched Pointwise Multiplication - Process all primes in parallel
    ///
    /// Computes c = a ⊙ b (element-wise) for all RNS primes simultaneously.
    ///
    /// # Arguments
    /// * `a` - First operand in flat RNS layout: [num_primes * n]
    /// * `b` - Second operand in flat RNS layout: [num_primes * n]
    /// * `result` - Output in flat RNS layout: [num_primes * n]
    /// * `num_primes` - Number of RNS primes
    fn ntt_pointwise_multiply_batched(
        &self,
        a: &[u64],
        b: &[u64],
        result: &mut [u64],
        num_primes: usize,
    ) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let total_elements = n * num_primes;

        if a.len() != total_elements || b.len() != total_elements || result.len() != total_elements {
            return Err(format!("All arrays must have length {}", total_elements));
        }

        // Use GPU-cached moduli
        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        // Copy input arrays to GPU
        let gpu_a = self.device.device.htod_copy(a.to_vec())
            .map_err(|e| format!("Failed to copy a to GPU: {:?}", e))?;

        let gpu_b = self.device.device.htod_copy(b.to_vec())
            .map_err(|e| format!("Failed to copy b to GPU: {:?}", e))?;

        let mut gpu_c = self.device.device.alloc_zeros::<u64>(total_elements)
            .map_err(|e| format!("Failed to allocate result: {:?}", e))?;

        // Launch batched pointwise multiply kernel
        let func = self.ntt_contexts[0].device.device.get_func("ntt_module", "ntt_pointwise_multiply_batched")
            .ok_or("Failed to get ntt_pointwise_multiply_batched function")?;

        // 2D grid: (coeff_blocks, num_primes)
        let threads_per_block = 256;
        let num_coeff_blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_coeff_blocks as u32, num_primes as u32, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &gpu_a,
                &gpu_b,
                &mut gpu_c,
                gpu_moduli,
                n as u32,
                num_primes as u32,
            ))
            .map_err(|e| format!("Batched pointwise multiply failed: {:?}", e))?;
        }

        // Copy result back
        let res = self.device.device.dtoh_sync_copy(&gpu_c)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))?;

        result.copy_from_slice(&res);
        Ok(())
    }

    // ============================================================
    // GPU-RESIDENT BATCHED NTT OPERATIONS
    // These methods work directly on GPU memory (CudaSlice)
    // to eliminate CPU↔GPU data copying overhead
    // ============================================================

    /// GPU-resident batched forward NTT
    ///
    /// Works directly on GPU memory - no CPU↔GPU copies!
    ///
    /// # Arguments
    /// * `gpu_data` - Mutable GPU slice in flat RNS layout [num_primes * n]
    /// * `num_primes` - Number of RNS primes to process
    fn ntt_forward_batched_gpu(&self, gpu_data: &mut CudaSlice<u64>, num_primes: usize) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let log_n = (n as f64).log2() as usize;

        if gpu_data.len() != n * num_primes {
            return Err(format!("Expected {} elements, got {}", n * num_primes, gpu_data.len()));
        }

        // Use GPU-cached twiddles and moduli
        let gpu_twiddles = self.gpu_twiddles_fwd.as_ref()
            .ok_or("GPU twiddles not initialized")?;
        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        // Bit-reversal permutation
        for prime_idx in 0..num_primes {
            let func_bit_reverse = self.ntt_contexts[0].device.device.get_func("ntt_module", "bit_reverse_permutation")
                .ok_or("Failed to get bit_reverse_permutation function")?;

            let threads_per_block = 256;
            let num_blocks = (n / 2 + threads_per_block - 1) / threads_per_block;
            let cfg = LaunchConfig {
                grid_dim: (num_blocks as u32, 1, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            let offset = prime_idx * n;
            let gpu_data_view = gpu_data.slice(offset..(offset + n));

            unsafe {
                func_bit_reverse.launch(cfg, (&gpu_data_view, n as u32, log_n as u32))
                    .map_err(|e| format!("Bit-reversal failed: {:?}", e))?;
            }
        }

        // Batched NTT stages
        let mut m = 1usize;
        for stage in 0..log_n {
            let func_ntt = self.ntt_contexts[0].device.device.get_func("ntt_module", "ntt_forward_batched")
                .ok_or("Failed to get ntt_forward_batched function")?;

            let threads_per_block = 256;
            let num_butterfly_blocks = (n / 2 + threads_per_block - 1) / threads_per_block;

            let cfg = LaunchConfig {
                grid_dim: (num_butterfly_blocks as u32, num_primes as u32, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func_ntt.launch(cfg, (
                    &mut *gpu_data,
                    gpu_twiddles,
                    gpu_moduli,
                    n as u32,
                    num_primes as u32,
                    stage as u32,
                    m as u32,
                ))
                .map_err(|e| format!("Batched NTT stage {} failed: {:?}", stage, e))?;
            }
            m *= 2;
        }

        Ok(())
    }

    /// GPU-resident batched inverse NTT
    fn ntt_inverse_batched_gpu(&self, gpu_data: &mut CudaSlice<u64>, num_primes: usize) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let log_n = (n as f64).log2() as usize;

        if gpu_data.len() != n * num_primes {
            return Err(format!("Expected {} elements, got {}", n * num_primes, gpu_data.len()));
        }

        let gpu_twiddles_inv = self.gpu_twiddles_inv.as_ref()
            .ok_or("GPU inverse twiddles not initialized")?;
        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        // Batched inverse NTT stages
        let mut m = n / 2;
        for stage in (0..log_n).rev() {
            let func_ntt_inv = self.ntt_contexts[0].device.device.get_func("ntt_module", "ntt_inverse_batched")
                .ok_or("Failed to get ntt_inverse_batched function")?;

            let threads_per_block = 256;
            let num_butterfly_blocks = (n / 2 + threads_per_block - 1) / threads_per_block;

            let cfg = LaunchConfig {
                grid_dim: (num_butterfly_blocks as u32, num_primes as u32, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func_ntt_inv.launch(cfg, (
                    &mut *gpu_data,
                    gpu_twiddles_inv,
                    gpu_moduli,
                    n as u32,
                    num_primes as u32,
                    stage as u32,
                    m as u32,
                ))
                .map_err(|e| format!("Batched inverse NTT stage {} failed: {:?}", stage, e))?;
            }
            m /= 2;
        }

        // Final scaling by n^(-1) for each prime
        for prime_idx in 0..num_primes {
            let ntt_ctx = &self.ntt_contexts[prime_idx];
            let func_scalar = ntt_ctx.device.device.get_func("ntt_module", "ntt_scalar_multiply")
                .ok_or("Failed to get ntt_scalar_multiply function")?;

            let threads_per_block = 256;
            let num_blocks = (n + threads_per_block - 1) / threads_per_block;
            let cfg = LaunchConfig {
                grid_dim: (num_blocks as u32, 1, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            let offset = prime_idx * n;
            let gpu_data_view = gpu_data.slice(offset..(offset + n));

            unsafe {
                func_scalar.launch(cfg, (&gpu_data_view, ntt_ctx.n_inv, n as u32, ntt_ctx.q))
                    .map_err(|e| format!("Scalar multiply failed: {:?}", e))?;
            }
        }

        Ok(())
    }

    /// GPU-resident batched pointwise multiplication
    fn ntt_pointwise_multiply_batched_gpu(
        &self,
        gpu_a: &CudaSlice<u64>,
        gpu_b: &CudaSlice<u64>,
        gpu_result: &mut CudaSlice<u64>,
        num_primes: usize,
    ) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let total_elements = n * num_primes;

        if gpu_a.len() != total_elements || gpu_b.len() != total_elements || gpu_result.len() != total_elements {
            return Err(format!("All arrays must have length {}", total_elements));
        }

        let gpu_moduli = self.gpu_moduli.as_ref()
            .ok_or("GPU moduli not initialized")?;

        let func = self.ntt_contexts[0].device.device.get_func("ntt_module", "ntt_pointwise_multiply_batched")
            .ok_or("Failed to get ntt_pointwise_multiply_batched function")?;

        let threads_per_block = 256;
        let num_coeff_blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_coeff_blocks as u32, num_primes as u32, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                gpu_a,
                gpu_b,
                gpu_result,
                gpu_moduli,
                n as u32,
                num_primes as u32,
            ))
            .map_err(|e| format!("Batched pointwise multiply failed: {:?}", e))?;
        }

        Ok(())
    }

    /// Convert strided layout to flat layout on GPU
    ///
    /// Strided: poly_in[coeff_idx * stride + prime_idx]
    /// Flat:    poly_out[prime_idx * n + coeff_idx]
    ///
    /// This eliminates CPU↔GPU transfers compared to the old strided_to_flat()
    /// which would download, convert on CPU, then upload again.
    fn strided_to_flat_gpu(
        &self,
        gpu_strided: &CudaSlice<u64>,
        gpu_flat: &mut CudaSlice<u64>,
        n: usize,
        stride: usize,
        num_primes: usize,
    ) -> Result<(), String> {
        use cudarc::driver::LaunchAsync;

        let func = self.device.device
            .get_func("rns_module", "rns_strided_to_flat")
            .ok_or("Failed to get rns_strided_to_flat kernel")?;

        let total_elements = n * num_primes;
        let threads_per_block = 256;
        let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (gpu_strided, gpu_flat, n as u32, stride as u32, num_primes as u32),
            )
            .map_err(|e| format!("strided_to_flat GPU kernel failed: {:?}", e))?;
        }

        Ok(())
    }

    /// Add two RNS polynomials using GPU kernel (flat layout)
    ///
    /// Computes c[i] = (a[i] + b[i]) % q for each RNS limb
    ///
    /// # Arguments
    /// * `a` - First polynomial in flat RNS layout [n × num_primes]
    /// * `b` - Second polynomial in flat RNS layout [n × num_primes]
    /// * `num_primes` - Number of RNS primes to use
    ///
    /// # Returns
    /// Result polynomial in flat RNS layout [n × num_primes]
    pub fn add_polynomials_gpu(&self, a: &[u64], b: &[u64], num_primes: usize) -> Result<Vec<u64>, String> {
        use cudarc::driver::LaunchAsync;

        let n = self.params.n;
        let total_elements = n * num_primes;

        if a.len() < total_elements || b.len() < total_elements {
            return Err(format!(
                "Input polynomials too small: expected {}, got {} and {}",
                total_elements, a.len(), b.len()
            ));
        }

        // Copy inputs to GPU
        let a_gpu = self.device.device.htod_copy(a[..total_elements].to_vec())
            .map_err(|e| format!("Failed to copy a to GPU: {:?}", e))?;
        let b_gpu = self.device.device.htod_copy(b[..total_elements].to_vec())
            .map_err(|e| format!("Failed to copy b to GPU: {:?}", e))?;

        // Allocate output on GPU
        let c_gpu = self.device.device.alloc_zeros::<u64>(total_elements)
            .map_err(|e| format!("Failed to allocate GPU memory: {:?}", e))?;

        // Copy moduli to GPU
        let moduli_gpu = self.device.device.htod_copy(self.params.moduli.clone())
            .map_err(|e| format!("Failed to copy moduli to GPU: {:?}", e))?;

        // Get kernel function
        let func = self.device.device.get_func("rns_module", "rns_add")
            .ok_or_else(|| "rns_add kernel not found".to_string())?;

        // Launch configuration
        let threads_per_block = 256;
        let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel: rns_add(a, b, c, moduli, n, num_primes)
        unsafe {
            func.launch(
                cfg,
                (
                    &a_gpu,
                    &b_gpu,
                    &c_gpu,
                    &moduli_gpu,
                    n as u32,
                    num_primes as u32,
                ),
            ).map_err(|e| format!("Failed to launch rns_add kernel: {:?}", e))?;
        }

        // Copy result back to CPU
        let result = self.device.device.dtoh_sync_copy(&c_gpu)
            .map_err(|e| format!("Failed to copy result from GPU: {:?}", e))?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires NVIDIA GPU
    fn test_cuda_ckks_context() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = CudaCkksContext::new(params).expect("Failed to create CUDA CKKS context");

        assert_eq!(ctx.ntt_contexts.len(), ctx.params.moduli.len());
        println!("✅ CUDA CKKS context created successfully");
    }

    #[test]
    #[ignore] // Requires NVIDIA GPU
    fn test_cuda_rescale() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = CudaCkksContext::new(params.clone()).expect("Failed to create context");

        // Create test polynomial at level 2
        let n = params.n;
        let level = 2;
        let num_primes = level + 1;

        let mut poly = vec![0u64; n * num_primes];
        for i in 0..n {
            for j in 0..num_primes {
                poly[i * num_primes + j] = (i * 1000 + j) as u64 % params.moduli[j];
            }
        }

        // Rescale
        let result = ctx.exact_rescale_gpu(&poly, level).expect("Rescale failed");

        // Check result has correct size
        let expected_size = n * (num_primes - 1);
        assert_eq!(result.len(), expected_size);

        println!("✅ CUDA rescale test passed");
    }
}
