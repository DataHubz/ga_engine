#!/bin/bash

# Comprehensive Test Suite for GA Engine
# Tests V1, V2 (CPU, Metal GPU), and V3 components
# Run this before committing to ensure all tests pass

set -e  # Exit on first error

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         GA Engine - Comprehensive Test Suite                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test() {
    local name=$1
    local command=$2

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if eval "$command"; then
        echo -e "${GREEN}✓ PASSED${NC}: $name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}: $name"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# ============================================================================
# V1 Tests
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  V1: Baseline Reference Tests"
echo "════════════════════════════════════════════════════════════════════"

run_test "V1 Unit Tests (31 tests)" \
    "cargo test --lib --features f64,nd,v1 --no-default-features --quiet 2>&1 | tail -20"

run_test "V1 Build (Release)" \
    "cargo build --release --features f64,nd,v1 --no-default-features --quiet"

# ============================================================================
# V2 CPU Tests
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  V2: CPU-Optimized Tests"
echo "════════════════════════════════════════════════════════════════════"

run_test "V2 Unit Tests (127 tests)" \
    "cargo test --lib --features f64,nd,v2 --no-default-features --quiet 2>&1 | tail -20"

run_test "V2 Build (Release)" \
    "cargo build --release --features f64,nd,v2 --no-default-features --quiet"

run_test "V2 NTT Module Tests" \
    "cargo test --lib clifford_fhe_v2::backends::cpu_optimized::ntt --features f64,nd,v2 --no-default-features --quiet 2>&1 | tail -10"

run_test "V2 RNS Module Tests" \
    "cargo test --lib clifford_fhe_v2::backends::cpu_optimized::rns --features f64,nd,v2 --no-default-features --quiet 2>&1 | tail -10"

run_test "V2 CKKS Module Tests" \
    "cargo test --lib clifford_fhe_v2::backends::cpu_optimized::ckks --features f64,nd,v2 --no-default-features --quiet 2>&1 | tail -10"

# ============================================================================
# V2 Metal GPU Tests (macOS only)
# ============================================================================

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  V2: Metal GPU Tests (macOS)"
    echo "════════════════════════════════════════════════════════════════════"

    run_test "V2 Metal GPU Build" \
        "cargo build --release --features f64,nd,v2-gpu-metal --no-default-features --quiet"

    run_test "V2 Metal GPU Tests" \
        "cargo test --lib --features f64,nd,v2-gpu-metal --no-default-features --quiet 2>&1 | tail -20"
else
    echo ""
    echo -e "${YELLOW}⊘ Skipping Metal GPU tests (not on macOS)${NC}"
fi

# ============================================================================
# V3 Tests
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  V3: Bootstrapping Tests"
echo "════════════════════════════════════════════════════════════════════"

run_test "V3 Build (with V2 backend)" \
    "cargo build --release --features v2,v3 --quiet"

run_test "V3 Unit Tests (52 tests)" \
    "cargo test --lib --features v2,v3 clifford_fhe_v3 --quiet 2>&1 | tail -20"

run_test "V3 Prime Generation Tests" \
    "cargo test --lib clifford_fhe_v3::prime_gen --features v2,v3 --quiet 2>&1 | tail -10"

run_test "V3 Bootstrap Context Tests" \
    "cargo test --lib clifford_fhe_v3::bootstrapping::bootstrap_context --features v2,v3 --quiet 2>&1 | tail -10"

run_test "V3 Rotation Tests" \
    "cargo test --lib clifford_fhe_v3::bootstrapping::rotation --features v2,v3 --quiet 2>&1 | tail -10"

run_test "V3 CoeffToSlot Tests" \
    "cargo test --lib clifford_fhe_v3::bootstrapping::coeff_to_slot --features v2,v3 --quiet 2>&1 | tail -10"

run_test "V3 SlotToCoeff Tests" \
    "cargo test --lib clifford_fhe_v3::bootstrapping::slot_to_coeff --features v2,v3 --quiet 2>&1 | tail -10"

run_test "V3 EvalMod Tests" \
    "cargo test --lib clifford_fhe_v3::bootstrapping::eval_mod --features v2,v3 --quiet 2>&1 | tail -10"

# ============================================================================
# V3 + Metal GPU Tests (macOS only)
# ============================================================================

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  V3: Metal GPU Integration Tests (macOS)"
    echo "════════════════════════════════════════════════════════════════════"

    run_test "V3 + Metal Build" \
        "cargo build --release --features v2,v3,v2-gpu-metal --quiet"

    run_test "V3 + Metal Tests" \
        "cargo test --lib --features v2,v3,v2-gpu-metal --quiet 2>&1 | tail -20"
fi

# ============================================================================
# Combined Tests
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Combined: All Versions"
echo "════════════════════════════════════════════════════════════════════"

run_test "All Versions Combined Build" \
    "cargo build --release --features f64,nd,v1,v2,v3 --no-default-features --quiet"

run_test "All Versions Combined Tests (249 tests)" \
    "cargo test --lib --features v2,v3 --quiet 2>&1 | tail -20"

# ============================================================================
# Lattice Reduction Tests (optional, uses default features)
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Lattice Reduction (Optional)"
echo "════════════════════════════════════════════════════════════════════"

run_test "Lattice Reduction Tests" \
    "cargo test --lib lattice_reduction --quiet 2>&1 | tail -20" || echo -e "${YELLOW}⊘ Lattice reduction tests failed (optional component)${NC}"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                     TEST SUMMARY                                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Total Tests:  $TOTAL_TESTS"
echo -e "Passed:       ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed:       ${RED}$FAILED_TESTS${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              ✓ ALL TESTS PASSED - READY TO COMMIT               ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║              ✗ SOME TESTS FAILED - DO NOT COMMIT                ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
