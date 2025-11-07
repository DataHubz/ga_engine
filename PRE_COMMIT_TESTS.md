# Pre-Commit Test Checklist

Run these tests before committing the V3 bootstrap fixes to ensure all components are working correctly.

## Quick Test (Recommended - 1 minute)

Run this single command to test all core functionality:

```bash
cargo test --lib --features v2,v3 --quiet && \
cargo test --lib --features f64,nd,v1,v2 --no-default-features --quiet && \
echo "✓ All tests passed - ready to commit!"
```

**Expected Result**: Should show ~210 tests passing in under 1 minute.

## Component-by-Component Tests (Detailed - 5 minutes)

### 1. V1 Tests (31 tests)
```bash
cargo test --lib --features f64,nd,v1 --no-default-features
```
Expected: `test result: ok. 31 passed`

### 2. V2 CPU Tests (127 tests)
```bash
cargo test --lib --features f64,nd,v2 --no-default-features
```
Expected: `test result: ok. 127 passed`

### 3. V3 Tests (52 tests)
```bash
cargo test --lib --features v2,v3 clifford_fhe_v3
```
Expected: `test result: ok. 52 passed`

### 4. V3 Critical Module Tests

**Prime Generation** (verifies dynamic prime generation):
```bash
cargo test --lib clifford_fhe_v3::prime_gen --features v2,v3 -- --nocapture
```
Expected: All tests pass, including NTT-friendly verification

**CoeffToSlot** (verifies scale preservation fix):
```bash
cargo test --lib clifford_fhe_v3::bootstrapping::coeff_to_slot --features v2,v3 -- --nocapture
```
Expected: Scale stays constant (no exponential growth)

**SlotToCoeff** (verifies level budget fix):
```bash
cargo test --lib clifford_fhe_v3::bootstrapping::slot_to_coeff --features v2,v3 -- --nocapture
```
Expected: No "Cannot rescale at level 0" error

**Bootstrap Context**:
```bash
cargo test --lib clifford_fhe_v3::bootstrapping::bootstrap_context --features v2,v3 -- --nocapture
```
Expected: All context creation and rotation key tests pass

### 5. Build Verification
```bash
cargo build --release --features f64,nd,v1,v2,v3 --no-default-features
```
Expected: Clean build with only the existing warning about unused `mut`

## Metal GPU Tests (macOS only)

If on macOS with Apple Silicon:

```bash
cargo test --lib --features v2,v3,v2-gpu-metal --quiet
```
Expected: All tests pass with Metal GPU backend

## Critical Regression Tests

These specifically verify the fixes made:

### Test 1: Scale Preservation
```bash
cargo test --lib coeff_to_slot --features v2,v3 -- --nocapture | grep "scale"
```
Expected: Should show scale staying constant (e.g., `1.10e12`) throughout all levels

### Test 2: Level Budget
```bash
cargo test --lib slot_to_coeff --features v2,v3 -- --nocapture | grep "level"
```
Expected: Should complete without "Cannot rescale at level 0" panic

### Test 3: Dynamic Primes
```bash
cargo test --lib prime_gen --features v2,v3 -- --nocapture | grep "NTT-friendly"
```
Expected: All generated primes satisfy `q ≡ 1 (mod 2N)`

## Full Bootstrap Example (Optional - 10 minutes)

To verify the complete bootstrap pipeline works end-to-end:

```bash
cargo run --release --features v2,v3 --example test_v3_full_bootstrap 2>&1 | tee bootstrap_test_output.txt
```

Expected output:
- ✓ 41 primes generated
- ✓ Scale stays constant at `1.10e12` in CoeffToSlot
- ✓ Scale stays constant at `1.10e12` in SlotToCoeff
- ✓ Bootstrap completes successfully
- ✓ Final error: ~3-4 × 10⁻⁹
- ✓ Total time: ~600 seconds

## Test Summary

| Component | Test Count | Command | Expected Time |
|-----------|------------|---------|---------------|
| V1 | 31 | `cargo test --lib --features v1` | 1s |
| V2 | 127 | `cargo test --lib --features v2` | 2s |
| V3 | 52 | `cargo test --lib clifford_fhe_v3 --features v2,v3` | 3s |
| **Total** | **210** | All above | **6s** |

## Known Issues (Expected)

1. **Compiler Warning**: One unused `mut` warning in `test_v3_full_bootstrap.rs` (line 158) - this is expected and doesn't affect functionality.

2. **Lattice Reduction**: May fail if CMake issues exist - this is optional and not required for V3 bootstrap.

## Success Criteria

Before committing, verify:

- ✅ All 210 core tests pass (V1 + V2 + V3)
- ✅ Build succeeds with `--release --features v1,v2,v3`
- ✅ No new compiler errors (warnings are OK)
- ✅ V3 bootstrap example completes successfully (optional but recommended)

## If Tests Fail

If any test fails:

1. **Check error message** - look for specific test name
2. **Run isolated test** - `cargo test --lib <test_name> --features v2,v3 -- --nocapture`
3. **Check files modified** - ensure only intended changes were made
4. **Review recent commits** - verify no unintended modifications

## Quick Verification Script

I've created two test scripts:

1. **Quick Test** (~1 minute):
   ```bash
   chmod +x TEST_QUICK.sh
   ./TEST_QUICK.sh
   ```

2. **Comprehensive Test** (~5 minutes):
   ```bash
   chmod +x TEST_COMPREHENSIVE.sh
   ./TEST_COMPREHENSIVE.sh
   ```
