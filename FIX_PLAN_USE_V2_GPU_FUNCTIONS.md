# Fix Plan: Use V2 GPU Functions in V3 Bootstrap

**Date**: 2025-11-09
**Status**: 🎯 READY TO IMPLEMENT
**Goal**: Replace CPU loop duplicates in V3 with calls to V2's existing GPU functions

---

## 🎯 Objective

V3 bootstrap currently has duplicate implementations with CPU loops for operations that V2 already has GPU-accelerated versions of. We need to:

1. **DELETE the duplicate CPU implementations in V3**
2. **CALL V2's existing GPU functions instead**
3. **ADD any missing V2 functions that are needed**

---

## ✅ What V2 Already Has (Working GPU Functions)

| Function | File | Line | Status |
|----------|------|------|--------|
| `exact_rescale_gpu()` | ckks.rs | 273 | ✅ V3 already uses this |
| `encode()` | ckks.rs | 471 | ✅ V3 already uses this |
| `strided_to_flat()` | ckks.rs | 806 | ❌ V3 has duplicate CPU version |
| `add_polynomials_gpu()` | ckks.rs | 1352 | ❌ V3 has duplicate CPU version |

---

## ❌ What V2 Is Missing (Need to Add)

| Function | Kernel Exists? | What We Need |
|----------|----------------|--------------|
| `flat_to_strided()` | ✅ Yes (line 298 in rns.cu) | Add Rust function in ckks.rs |
| `subtract_polynomials_gpu()` | ✅ Yes (line 218 in rns.cu) | Add Rust function in ckks.rs |
| `multiply_plain_gpu()` | ❓ Need to check | May need new function |

---

## 🔧 Implementation Plan

### Step 1: Add Missing V2 GPU Functions (30 min)

#### 1.1: Add `flat_to_strided()` to V2

**File**: `src/clifford_fhe_v2/backends/gpu_cuda/ckks.rs`

**Add after line 839** (after `strided_to_flat()`):

```rust
/// Convert from flat to strided RNS layout using GPU
///
/// Flat:    poly_in[prime_idx * n + coeff_idx]
/// Strided: poly_out[coeff_idx * stride + prime_idx]
///
/// This is the inverse of strided_to_flat()
fn flat_to_strided(&self, data: &[u64], n: usize, stride: usize, num_primes: usize) -> Vec<u64> {
    use cudarc::driver::LaunchAsync;

    let total_elements = n * num_primes;

    // Copy to GPU
    let gpu_input = self.device.device.htod_copy(data.to_vec())
        .expect("Failed to copy to GPU");

    let mut gpu_output = self.device.device.alloc_zeros::<u64>(n * stride)
        .expect("Failed to allocate GPU memory");

    // Get kernel
    let func = self.device.device.get_func("rns_module", "rns_flat_to_strided")
        .expect("Failed to get rns_flat_to_strided kernel");

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
            .expect("Failed to launch rns_flat_to_strided kernel");
    }

    // Copy result back
    self.device.device.dtoh_sync_copy(&gpu_output)
        .expect("Failed to copy from GPU")
}
```

#### 1.2: Add `subtract_polynomials_gpu()` to V2

**File**: `src/clifford_fhe_v2/backends/gpu_cuda/ckks.rs`

**Add after `add_polynomials_gpu()`**:

```rust
/// Subtract two polynomials in RNS representation using GPU
///
/// Computes c[i] = (a[i] - b[i]) % q for each RNS limb
pub fn subtract_polynomials_gpu(&self, a: &[u64], b: &[u64], num_primes: usize) -> Result<Vec<u64>, String> {
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

    // Use cached moduli
    let moduli_gpu = self.gpu_moduli.as_ref()
        .ok_or("GPU moduli not cached")?;

    // Get kernel function
    let func = self.device.device.get_func("rns_module", "rns_sub")
        .ok_or_else(|| "rns_sub kernel not found".to_string())?;

    // Launch configuration
    let threads_per_block = 256;
    let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (threads_per_block as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch kernel
    unsafe {
        func.launch(
            cfg,
            (
                &a_gpu,
                &b_gpu,
                &c_gpu,
                moduli_gpu,
                n as u32,
                num_primes as u32,
            ),
        )
        .map_err(|e| format!("GPU subtraction failed: {:?}", e))?;
    }

    // Copy result back
    self.device.device.dtoh_sync_copy(&c_gpu)
        .map_err(|e| format!("Failed to copy result from GPU: {:?}", e))
}
```

---

### Step 2: Fix V3 `cuda_rotate_ciphertext()` (15 min)

**File**: `src/clifford_fhe_v3/bootstrapping/cuda_coeff_to_slot.rs`

**Lines 136-183**: Replace CPU loops with V2 GPU calls

**BEFORE** (CPU loops):
```rust
// Convert to flat RNS layout for GPU rotation
let mut c0_flat = vec![0u64; n * num_primes];
let mut c1_flat = vec![0u64; n * num_primes];

for coeff_idx in 0..n {
    for prime_idx in 0..num_primes {
        let strided_idx = coeff_idx * num_primes + prime_idx;
        let flat_idx = prime_idx * n + coeff_idx;
        c0_flat[flat_idx] = ct.c0[strided_idx];
        c1_flat[flat_idx] = ct.c1[strided_idx];
    }
}

// ... GPU rotation ...

// Convert back from flat to strided layout
let mut c0_strided = vec![0u64; n * num_primes];
let mut c1_strided = vec![0u64; n * num_primes];

for coeff_idx in 0..n {
    for prime_idx in 0..num_primes {
        let flat_idx = prime_idx * n + coeff_idx;
        let strided_idx = coeff_idx * num_primes + prime_idx;
        c0_strided[strided_idx] = c0_result[flat_idx];
        c1_strided[strided_idx] = c1_ks[flat_idx];
    }
}
```

**AFTER** (V2 GPU calls):
```rust
// Convert to flat RNS layout using V2 GPU function
let c0_flat = ckks_ctx.strided_to_flat(&ct.c0, n, ct.num_primes, num_primes);
let c1_flat = ckks_ctx.strided_to_flat(&ct.c1, n, ct.num_primes, num_primes);

// ... GPU rotation (unchanged) ...

// Convert back from flat to strided using V2 GPU function
let c0_strided = ckks_ctx.flat_to_strided(&c0_result, n, num_primes, num_primes);
let c1_strided = ckks_ctx.flat_to_strided(&c1_ks, n, num_primes, num_primes);
```

**Lines changed**: 139-146 and 173-183

---

### Step 3: Fix V3 `cuda_rotate_ciphertext()` Addition (10 min)

**File**: `src/clifford_fhe_v3/bootstrapping/cuda_coeff_to_slot.rs`

**Lines 164-170**: Replace CPU loop with V2 GPU call

**BEFORE** (CPU loop):
```rust
// Step 4: Add c0(X^g) + c0_ks
let mut c0_result = vec![0u64; n * num_primes];
for i in 0..(n * num_primes) {
    let prime_idx = i / n;
    let q = rotation_keys.modulus(prime_idx);
    c0_result[i] = (c0_galois[i] + c0_ks[i]) % q;
}
```

**AFTER** (V2 GPU call):
```rust
// Step 4: Add c0(X^g) + c0_ks using V2 GPU function
let c0_result = ckks_ctx.add_polynomials_gpu(&c0_galois, &c0_ks, num_primes)?;
```

**Lines changed**: 164-170 (replace with 1 line)

---

### Step 4: Fix V3 `cuda_add_ciphertexts()` (15 min)

**File**: `src/clifford_fhe_v3/bootstrapping/cuda_coeff_to_slot.rs`

**Lines 263-302**: Replace entire function with V2 calls

**BEFORE** (40 lines of CPU loops):
```rust
pub fn cuda_add_ciphertexts(
    ct1: &CudaCiphertext,
    ct2: &CudaCiphertext,
    ckks_ctx: &Arc<CudaCkksContext>,
) -> Result<CudaCiphertext, String> {
    // ... verification ...

    let mut c0 = vec![0u64; n * num_primes];
    let mut c1 = vec![0u64; n * num_primes];

    for coeff_idx in 0..n {
        for prime_idx in 0..num_active_primes {
            let q = ckks_ctx.params().moduli[prime_idx];
            let idx = coeff_idx * num_primes + prime_idx;

            let sum0 = ct1.c0[idx] + ct2.c0[idx];
            c0[idx] = if sum0 >= q { sum0 - q } else { sum0 };

            let sum1 = ct1.c1[idx] + ct2.c1[idx];
            c1[idx] = if sum1 >= q { sum1 - q } else { sum1 };
        }
    }

    Ok(CudaCiphertext { c0, c1, ... })
}
```

**AFTER** (use V2's GPU function):
```rust
pub fn cuda_add_ciphertexts(
    ct1: &CudaCiphertext,
    ct2: &CudaCiphertext,
    ckks_ctx: &Arc<CudaCkksContext>,
) -> Result<CudaCiphertext, String> {
    // Verify levels match
    if ct1.level != ct2.level {
        return Err(format!("Ciphertexts must be at same level: {} vs {}", ct1.level, ct2.level));
    }

    let n = ct1.n;
    let num_active_primes = ct1.level + 1;
    let num_primes = ct1.num_primes;

    // Convert to flat layout
    let c0_1_flat = ckks_ctx.strided_to_flat(&ct1.c0, n, num_primes, num_active_primes);
    let c1_1_flat = ckks_ctx.strided_to_flat(&ct1.c1, n, num_primes, num_active_primes);
    let c0_2_flat = ckks_ctx.strided_to_flat(&ct2.c0, n, num_primes, num_active_primes);
    let c1_2_flat = ckks_ctx.strided_to_flat(&ct2.c1, n, num_primes, num_active_primes);

    // Add on GPU
    let c0_flat_result = ckks_ctx.add_polynomials_gpu(&c0_1_flat, &c0_2_flat, num_active_primes)?;
    let c1_flat_result = ckks_ctx.add_polynomials_gpu(&c1_1_flat, &c1_2_flat, num_active_primes)?;

    // Convert back to strided
    let c0 = ckks_ctx.flat_to_strided(&c0_flat_result, n, num_primes, num_active_primes);
    let c1 = ckks_ctx.flat_to_strided(&c1_flat_result, n, num_primes, num_active_primes);

    Ok(CudaCiphertext {
        c0,
        c1,
        n,
        num_primes,
        level: ct1.level,
        scale: ct1.scale,
    })
}
```

---

### Step 5: Fix V3 `cuda_multiply_plain()` (20 min)

**File**: `src/clifford_fhe_v3/bootstrapping/cuda_coeff_to_slot.rs`

**Lines 210-257**: Check if we can optimize

This function does plaintext multiplication with 128-bit arithmetic. We need to check if there's a GPU kernel for this or if we should add one.

**Current**: CPU loops with `(c0_val as u128 * pt_val as u128) % q as u128`

**Options**:
1. If kernel exists - use it
2. If not - this may be acceptable to stay on CPU (called ~20 times, not the main bottleneck)

Let's **defer this** for now and focus on the bigger wins.

---

### Step 6: Fix V3 `modulus_raise()` in cuda_bootstrap.rs (10 min)

**File**: `src/clifford_fhe_v3/bootstrapping/cuda_bootstrap.rs`

**Lines 142-151**: This is just copying data to extend the array. Could add a GPU function but it's called only **once** per bootstrap, so it's low priority.

**Decision**: **Skip for now** - not on hot path.

---

## 📊 Expected Performance Impact

### Before Fix

| Operation | Current Implementation | Calls per Bootstrap | Cost |
|-----------|------------------------|---------------------|------|
| Layout conversion (strided↔flat) | CPU loops (30,720 iterations) | 36 (18 rotations × 2) | **180-360ms** |
| Addition after rotation | CPU loop (30,720 iterations) | 18 | **36-90ms** |
| Ciphertext addition | CPU loops (20,480 iterations) | ~50 (BSGS) | **100-250ms** |
| **Total** | - | - | **316-700ms** |

### After Fix

| Operation | New Implementation | Calls per Bootstrap | Cost |
|-----------|-------------------|---------------------|------|
| Layout conversion | V2 GPU (`strided_to_flat`) | 36 | **~40ms** (PCIe only) |
| Addition after rotation | V2 GPU (`add_polynomials_gpu`) | 18 | **~10ms** (PCIe only) |
| Ciphertext addition | V2 GPU (`add_polynomials_gpu`) | ~50 | **~20ms** (PCIe only) |
| **Total** | - | - | **~70ms** |

**Expected Speedup**: **0.25-0.6 seconds** (12.55s → 11.95-12.3s EvalMod)

---

## ✅ Testing Plan

### After Each Step

1. **Build**:
   ```bash
   cargo build --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap
   ```

2. **Run test**:
   ```bash
   cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap
   ```

3. **Verify**:
   - ✅ Compiles without errors
   - ✅ Performance improved
   - ✅ Results are mathematically correct

### Final Validation

Run 3 times to check consistency:
```bash
for i in {1..3}; do
  echo "=== Run $i ==="
  cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap 2>&1 | grep "EvalMod completed"
done
```

Expected: **11.9-12.3s EvalMod** (down from 12.55s)

---

## 📋 Implementation Checklist

### Step 1: Add V2 Functions
- [ ] Add `flat_to_strided()` to ckks.rs
- [ ] Add `subtract_polynomials_gpu()` to ckks.rs
- [ ] Build and verify no errors

### Step 2: Fix V3 cuda_rotate_ciphertext()
- [ ] Replace strided→flat CPU loops with v2 call
- [ ] Replace flat→strided CPU loops with v2 call
- [ ] Replace addition CPU loop with v2 call
- [ ] Build and test

### Step 3: Fix V3 cuda_add_ciphertexts()
- [ ] Replace entire function with v2 GPU calls
- [ ] Build and test

### Step 4: Verify Performance
- [ ] Run bootstrap 3 times
- [ ] Verify 11.9-12.3s EvalMod
- [ ] Check results are correct

---

## 🎯 Summary

**Changes Required**:
1. Add 2 functions to V2 (40 lines total)
2. Simplify V3 cuda_rotate_ciphertext() (delete 20 lines, add 4 lines)
3. Simplify V3 cuda_add_ciphertexts() (delete 30 lines, add 15 lines)

**Total**: Net reduction of ~30 lines of code, gain 0.25-0.6s performance

**Estimated Time**: 1.5 hours
**Expected Gain**: 0.25-0.6 seconds

Let's do it!
