# V4 Geometric Product - Working Commands

## ✅ Clean Output (RECOMMENDED)

```bash
cargo run --release --features v4,v2-gpu-metal --example test_v4_geometric_product 2>&1 | \
  grep -v "ROTATION DEBUG\|GALOIS DEBUG\|NTT\|Metal Device\|Metal Max"
```

**Expected to see:**
```
✓ Geometric product completed in 5.060s
```

---

## 📊 Performance Benchmark

```bash
cargo run --release --features v4,v2-gpu-metal --example bench_v4_geometric_product 2>&1 | \
  grep -v "ROTATION DEBUG\|GALOIS DEBUG"
```

**Shows:**
- Pack A time
- Pack B time
- Geometric product time
- Statistics (mean, stddev, min, max)

---

## 🔍 Full Debug Output

If you want to see ALL the debug messages:

```bash
cargo run --release --features v4,v2-gpu-metal --example test_v4_geometric_product
```

⚠️ **Warning:** This produces ~50,000 lines of output showing every rotation and Galois operation.

---

## Why So Much Debug Output?

The rotation operations print debug info for:
- Every rotation step (16 rotations during packing)
- Every Galois automorphism
- Every key-switch operation
- RNS coordinate tracking

This is helpful for debugging rotation issues but overwhelming for normal use.

---

## Quick Verification (One-Liner)

```bash
cargo run --release --features v4,v2-gpu-metal --example test_v4_geometric_product 2>&1 | \
  grep "✓ Geometric product completed"
```

**Should output:**
```
✓ Geometric product completed in ~5s
```

✅ If you see this, the geometric product is working!

---

## What You Don't Like

The command without filtering:
```bash
cargo run --release --features v4,v2-gpu-metal --example test_v4_geometric_product
```

Produces thousands of lines like:
```
[ROTATION DEBUG] n=1024, num_primes_active=3, ct_stride=3
[GALOIS DEBUG] galois_map[0..5]: [0, 5, 10, 15, 20]
[ROTATION DEBUG] First 5 c0_active values: [...]
...
```

---

## Solution

**Always use the filtered command:**
```bash
cargo run --release --features v4,v2-gpu-metal --example test_v4_geometric_product 2>&1 | \
  grep -v "ROTATION DEBUG\|GALOIS DEBUG\|NTT\|Metal Device\|Metal Max"
```

Or add this alias to your `~/.zshrc` or `~/.bashrc`:
```bash
alias v4test='cargo run --release --features v4,v2-gpu-metal --example test_v4_geometric_product 2>&1 | grep -v "ROTATION DEBUG\|GALOIS DEBUG\|NTT\|Metal Device\|Metal Max"'
```

Then just run:
```bash
v4test
```

---

## Alternative: Disable Debug Output in Code

If you want to permanently remove the debug output, edit these files:

1. `src/clifford_fhe_v2/backends/gpu_metal/ckks.rs` (lines ~1565-1600)
2. `src/clifford_fhe_v2/backends/gpu_metal/rotation.rs` (various lines)

Comment out lines starting with:
- `eprintln!("[ROTATION DEBUG]`
- `eprintln!("[GALOIS DEBUG]`
- `eprintln!("[ROTATION WARNING]`

---

## Summary

| What You Want | Command |
|---------------|---------|
| **Clean output** | Use grep filter (see above) |
| **Quick check** | `... \| grep "✓ Geometric product completed"` |
| **Benchmark** | `cargo run ... bench_v4_geometric_product` with filter |
| **Debug everything** | Run without filter |
