# RunPod Quick Start Guide - CUDA CKKS Testing

## 🚀 Quick Setup (5 minutes)

### 1. Create RunPod Instance
1. Go to https://www.runpod.io/
2. Select **"Deploy"** → **"GPU Instances"**
3. Choose GPU: **RTX 4090** (48GB VRAM)
4. Template: **"RunPod Pytorch 2.0"** or **"CUDA 12.3"**
5. Storage: **50GB** minimum
6. Deploy pod

### 2. Connect to Pod
```bash
# Use SSH or Web Terminal (Jupyter)
# You'll get an SSH command like:
ssh root@<pod-id>.runpod.io -p <port>
```

### 3. Install Rust (30 seconds)
```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env
rustc --version  # Verify installation
```

### 4. Clone Repository
```bash
cd ~
git clone <your-repo-url> ga_engine
cd ga_engine
git checkout v2-cuda-v3-cuda-bootstrap  # Or your branch name
```

### 5. Build CUDA Backend
```bash
# Build with CUDA features (will take 3-5 minutes first time)
cargo build --release --features v2,v2-gpu-cuda

# Check for compilation success
echo $?  # Should print 0
```

### 6. Run CUDA CKKS Test
```bash
cargo run --release --features v2,v2-gpu-cuda --example test_cuda_ckks
```

## ✅ Expected Output

```
╔═══════════════════════════════════════════════════════════════╗
║              CUDA CKKS Basic Operations Test                 ║
╚═══════════════════════════════════════════════════════════════╝

Step 1: Creating FHE parameters...
  ✅ Parameters: N=1024, 20 primes

Step 2: Initializing CUDA CKKS context...

╔═══════════════════════════════════════════════════════════════╗
║           Initializing CUDA CKKS Context                     ║
╚═══════════════════════════════════════════════════════════════╝

CUDA Device: NVIDIA GeForce RTX 4090
CUDA Max Threads Per Threadgroup: 1024

Creating NTT contexts for 20 primes...
  Created 5/20 NTT contexts
  Created 10/20 NTT contexts
  Created 15/20 NTT contexts
  Created 20/20 NTT contexts
  [CUDA CKKS] NTT contexts created in X.XXs
Loading RNS CUDA kernels...
  [CUDA CKKS] ✓ GPU-only CKKS context ready!

  ✅ CUDA CKKS context ready!

Step 3: Testing encoding...
  ✅ Encoded 4 values at level 2
     Plaintext size: 1024 coefficients
     Number of RNS primes: 3

Step 4: Testing GPU rescaling...
  Input:  1024 coefficients × 3 primes = 3072 elements
  Output: 1024 coefficients × 2 primes = 2048 elements
  ✅ GPU rescaling successful!

═══════════════════════════════════════════════════════════════
✅ ALL TESTS PASSED
═══════════════════════════════════════════════════════════════

CUDA CKKS Operations Summary:
  • Context initialization: ✅
  • Encoding: ✅
  • GPU Rescaling: ✅

Next Steps:
  • Test on RunPod with NVIDIA GPU
  • Implement rotation operations
  • Implement full bootstrap
```

## 🔧 Troubleshooting

### Problem: "No CUDA device found"
**Cause**: CUDA drivers not loaded or GPU not available

**Fix**:
```bash
# Check GPU availability
nvidia-smi

# Should show RTX 4090
# If not, restart pod or contact RunPod support
```

### Problem: "Failed to compile CUDA kernel"
**Cause**: CUDA toolkit not installed or wrong version

**Fix**:
```bash
# Check CUDA version
nvcc --version

# Should show CUDA 12.0+
# If not, use a different pod template with CUDA 12.3
```

### Problem: Compilation takes too long
**Cause**: First build compiles all dependencies

**Fix**:
- First build takes 3-5 minutes (normal)
- Subsequent builds are fast (~10 seconds)
- Use `cargo build` first, then `cargo run`

### Problem: "error: linker cc not found"
**Cause**: Build essentials not installed

**Fix**:
```bash
apt update && apt install -y build-essential
```

## 📊 Performance Benchmarks

Run these to measure CUDA performance:

### 1. NTT Benchmark
```bash
# Run NTT tests (measures GPU NTT performance)
cargo test --release --features v2,v2-gpu-cuda ntt -- --ignored --nocapture
```

### 2. Rescaling Benchmark
```bash
# Create this after validating basic tests work
cargo run --release --features v2,v2-gpu-cuda --example bench_cuda_rescale
```

Expected performance on RTX 4090:
- **NTT (1024)**: ~0.2ms (vs ~0.5ms on Metal M3 Max)
- **GPU Rescaling**: ~0.4ms (vs ~1ms on Metal)
- **20 NTT contexts**: ~5-10s to create

## 🎯 Validation Checklist

- [ ] CUDA device detects RTX 4090
- [ ] 20 NTT contexts create successfully
- [ ] RNS kernels compile without errors
- [ ] GPU rescaling produces correct output size
- [ ] No runtime errors or crashes

Once all checks pass, you're ready for:
- [ ] Golden compare test (bit-exact validation)
- [ ] Rotation operations implementation
- [ ] Full V3 CUDA bootstrap

## 💰 RunPod Costs (Estimate)

- **RTX 4090**: ~$0.70-1.00/hour
- **Testing session**: ~1-2 hours
- **Total cost**: ~$1-2 for initial validation

**Tip**: Use **Spot Instances** for cheaper rates (~50% less)

## 📝 Next Steps After Validation

1. **Golden Compare Test**: Validate GPU rescaling is bit-exact
   ```bash
   cargo run --release --features v2,v2-gpu-cuda --example test_cuda_rescale_golden_compare
   ```
   (Need to create this example next)

2. **Rotation Operations**: Implement Galois automorphisms on GPU
   - Create `rotation.rs`
   - Create `kernels/galois.cu`

3. **Full Bootstrap**: Implement CoeffToSlot/SlotToCoeff
   - Create `rotation_keys.rs`
   - Create `bootstrap.rs`

## 🆘 Support

If you encounter issues:

1. **Check CUDA setup**:
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. **Check Rust/Cargo**:
   ```bash
   rustc --version
   cargo --version
   ```

3. **Check kernel logs**:
   Look for "Failed to compile CUDA kernel" errors
   These usually indicate syntax errors in `.cu` files

4. **Verify cudarc**:
   ```bash
   cargo tree | grep cudarc
   # Should show cudarc v0.11.x
   ```

## 📞 Quick Commands Reference

```bash
# Build
cargo build --release --features v2,v2-gpu-cuda

# Test CKKS
cargo run --release --features v2,v2-gpu-cuda --example test_cuda_ckks

# Run unit tests
cargo test --release --features v2,v2-gpu-cuda --lib

# Check GPU status
nvidia-smi
watch -n 1 nvidia-smi  # Live monitoring

# Check disk space
df -h

# Clean build artifacts (if running out of space)
cargo clean
```

---

**Ready to test on RunPod!** 🚀

The CUDA CKKS implementation compiles successfully and is ready for GPU validation.
