# CMake 4.0 Compatibility Fix

## Problem

When building with CMake 4.0+, the `netlib-src` dependency (used by the `lattice-reduction` feature) fails with:

```
CMake Error at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
```

## Root Cause

- CMake 4.0 removed support for projects specifying `cmake_minimum_required` < 3.5
- The `netlib-src` crate (v0.8.0) bundles BLAS/LAPACK source with old CMakeLists.txt files
- These files specify CMake versions older than 3.5

## Solution

Added `.cargo/config.toml` with the `CMAKE_POLICY_VERSION_MINIMUM` environment variable:

```toml
[env]
CMAKE_POLICY_VERSION_MINIMUM = "3.5"
```

This tells CMake 4.0+ to accept projects with older version requirements as if they specified 3.5.

## Verification

```bash
# Clean build with lattice-reduction feature
cargo clean
cargo test --lib --features v2,v3

# Expected result:
# test result: ok. 249 passed; 0 failed; 2 ignored
```

## Alternative Solutions

### Option 1: Temporary Environment Variable (Not Recommended)
```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 cargo test --lib --features v2,v3
```

### Option 2: Disable Lattice Reduction (Not Recommended)
```bash
cargo test --lib --features v2,v3 --no-default-features
```

### Option 3: Downgrade CMake (Not Recommended)
```bash
brew install cmake@3.31
```

## Notes

- The `.cargo/config.toml` solution is permanent and automatic
- No code changes required
- Works with all cargo commands (build, test, run, etc.)
- The lattice-reduction feature is used for security analysis, not core FHE operations
- This issue affects macOS/Linux builds; Windows may use different BLAS backends

## References

- [CMake 4.0 Release Notes](https://cmake.org/cmake/help/latest/release/4.0.html)
- [CMake Issue #25299](https://gitlab.kitware.com/cmake/cmake/-/issues/25299)
- [netlib-src crate](https://crates.io/crates/netlib-src)
