# Testing Guide

Complete guide for running all tests in GA Engine.

## Quick Start

### Single Command - Run Everything
```bash
cargo test --lib --features v2,v3
```

**Expected Result**: `ok. 249 passed; 0 failed; 2 ignored; 0 measured`

**Time**: ~70 seconds

## Test Breakdown

### 1. V3 Bootstrap Tests Only (52 tests)
```bash
cargo test --lib --features v2,v3 clifford_fhe_v3
```

**What's Tested**:
- Rotation infrastructure (Galois automorphisms)
- CoeffToSlot/SlotToCoeff transforms
- Diagonal matrix multiplication
- EvalMod (homomorphic modular reduction)
- ModRaise (modulus chain extension)
- SIMD batching (512× throughput)
- Component extraction (Pattern A)
- Bootstrap pipeline integration

**Result**: 52/52 passing (100%)

### 2. V2 Optimized Backend Tests (127 tests)
```bash
cargo test --lib --features v2 clifford_fhe_v2
```

**What's Tested**:
- NTT (Number Theoretic Transform)
- RNS (Residue Number System)
- CKKS encryption scheme
- Key generation and management
- Polynomial multiplication
- Geometric operations
- SIMD backends (NEON, scalar)

**Result**: 127/127 passing (100%)

### 3. V1 Baseline Tests (31 tests)
```bash
cargo test --lib --features v1 clifford_fhe_v1
```

**What's Tested**:
- Reference CKKS implementation
- Canonical embedding
- Slot encoding
- Automorphisms
- Geometric neural network
- Rotation keys

**Result**: 31/31 passing (100%)

### 4. Lattice Reduction Tests (~60 tests)
```bash
cargo test --lib lattice_reduction
```

**What's Tested**:
- Gram-Schmidt orthogonalization
- LLL reduction
- BKZ reduction
- Enumeration algorithms
- GA-accelerated lattice operations

**Note**: Requires CMake 4.0 fix (automatically applied via `.cargo/config.toml`)

**Result**: All passing

### 5. Medical Imaging Tests (~25 tests)
```bash
cargo test --lib medical_imaging --features v2,v3
```

**What's Tested**:
- Encrypted GNN inference
- SIMD batching for point clouds
- Clifford encoding
- Batched geometric products

**Result**: All passing (2 ignored V2 CPU tests)

## Specialized Test Commands

### Bootstrap-Specific Tests

#### Rotation Tests
```bash
cargo test --lib clifford_fhe_v3::bootstrapping::rotation --features v2,v3 -- --nocapture
```

#### CoeffToSlot/SlotToCoeff Tests
```bash
cargo test --lib clifford_fhe_v3::bootstrapping::coeff_to_slot --features v2,v3 -- --nocapture
cargo test --lib clifford_fhe_v3::bootstrapping::slot_to_coeff --features v2,v3 -- --nocapture
```

#### Extraction Tests (Pattern A)
```bash
cargo test --lib clifford_fhe_v3::batched::extraction --features v2,v3 -- --nocapture
```

#### EvalMod Tests
```bash
cargo test --lib clifford_fhe_v3::bootstrapping::eval_mod --features v2,v3 -- --nocapture
```

#### Bootstrap Context Tests
```bash
cargo test --lib clifford_fhe_v3::bootstrapping::bootstrap_context --features v2,v3 -- --nocapture
```

## Build Verification

### Compile All Examples
```bash
cargo build --release --features v2,v3 --examples
```

**Time**: ~60 seconds

**Expected**: `Finished release [optimized] target(s)`

## Performance Testing

### Run Integration Examples

#### V3 Bootstrap Simple
```bash
cargo run --release --features v2,v3 --example test_v3_bootstrap_simple
```

#### SIMD Batching Demo
```bash
cargo run --release --features v2,v3 --example test_batching
```

#### Medical Imaging Encrypted
```bash
cargo run --release --features v2,v3 --example medical_imaging_encrypted
```

## Test Output Interpretation

### Successful Test Run
```
running 249 tests
test clifford_fhe_v1::... ok
test clifford_fhe_v2::... ok
test clifford_fhe_v3::... ok
...

test result: ok. 249 passed; 0 failed; 2 ignored; 0 measured; 140 filtered out; finished in 70.22s
```

### What "2 ignored" Means
- These are 2 medical imaging V2 CPU tests that are temporarily ignored
- They don't affect V3 bootstrap functionality
- V3 has 0 ignored tests (52/52 passing)

## Clean Build Testing

### Complete Clean Rebuild
```bash
cargo clean
cargo test --lib --features v2,v3
```

**Purpose**: Verify everything builds from scratch (useful after pulling updates)

**Time**: ~2-3 minutes (first build) + ~70 seconds (tests)

## Troubleshooting

### CMake 4.0 Errors

**Symptom**: `Compatibility with CMake < 3.5 has been removed from CMake`

**Solution**: This is FIXED automatically via `.cargo/config.toml`. If you see this error:
1. Verify `.cargo/config.toml` exists and contains `CMAKE_POLICY_VERSION_MINIMUM = "3.5"`
2. Run `cargo clean` and rebuild
3. See [CMAKE_FIX.md](CMAKE_FIX.md) for details

### Test Timeouts

**Symptom**: Tests hang or timeout

**Solution**: Always use `--release` for performance-intensive tests:
```bash
cargo test --release --lib --features v2,v3
```

### Specific Test Failures

**Symptom**: One test fails, want to debug

**Solution**: Run in isolation with output:
```bash
cargo test --lib test_name --features v2,v3 -- --nocapture
```

## Continuous Integration

### Pre-Commit Checklist
```bash
# 1. Run full test suite
cargo test --lib --features v2,v3

# 2. Build all examples
cargo build --release --features v2,v3 --examples

# 3. Format code
cargo fmt

# 4. Run clippy
cargo clippy --features v2,v3 -- -D warnings
```

**All should pass before committing.**


## Test Statistics

| Category | Tests | Pass Rate | Time |
|----------|-------|-----------|------|
| V1 Baseline | 31 | 100% | ~5s |
| V2 Optimized | 127 | 100% | ~15s |
| V3 Bootstrap | 52 | 100% | ~40s |
| Lattice Reduction | ~60 | 100% | ~5s |
| Medical Imaging | ~25 | 100% (2 ignored) | ~5s |
| **Total** | **249** | **100%** | **~70s** |

## Related Documentation

- [COMMANDS.md](COMMANDS.md) - Complete command reference
- [CMAKE_FIX.md](CMAKE_FIX.md) - CMake 4.0 compatibility fix
- [V3_BOOTSTRAP.md](V3_BOOTSTRAP.md) - V3 implementation details
- [README.md](README.md) - Project overview

## Support

For test failures or questions:
- Check [COMMANDS.md](COMMANDS.md) troubleshooting section
- Review test output carefully (use `--nocapture` for details)
- Ensure CMake 4.0 fix is applied (see [CMAKE_FIX.md](CMAKE_FIX.md))
- File issue at: https://github.com/davidwilliamsilva/ga_engine/issues
