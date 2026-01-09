#!/bin/bash
# scripts/run_cuda_benchmarks.sh
#
# Run comprehensive CUDA benchmarks for Clifford FHE
# Designed for RunPod RTX 4090 environment
#
# Usage:
#   chmod +x scripts/run_cuda_benchmarks.sh
#   ./scripts/run_cuda_benchmarks.sh [quick|full]
#
# Options:
#   quick - Run only essential benchmarks (~5-10 min)
#   full  - Run all benchmarks including stress tests (~30-60 min)

set -e

MODE=${1:-quick}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/cuda_benchmarks_$TIMESTAMP"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Clifford FHE CUDA Benchmark Suite                          ║"
echo "║   Mode: $MODE                                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Create results directory
mkdir -p $RESULTS_DIR
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Collect system information
echo "[1/7] Collecting system information..."
cat > $RESULTS_DIR/system_info.txt << EOF
================================================================================
SYSTEM INFORMATION
================================================================================
Date: $(date)
Hostname: $(hostname)

--- GPU Information ---
$(nvidia-smi)

--- CUDA Version ---
$(nvcc --version 2>/dev/null || echo "nvcc not in PATH")

--- Rust Version ---
$(rustc --version)
$(cargo --version)

--- CPU Information ---
$(lscpu | grep -E "Model name|CPU\(s\)|Thread|Core" || cat /proc/cpuinfo | grep -E "model name|cpu cores" | head -4)

--- Memory ---
$(free -h)

--- OS ---
$(cat /etc/os-release | head -5)
================================================================================
EOF
cat $RESULTS_DIR/system_info.txt
echo ""

# Build with CUDA features
echo "[2/7] Building with CUDA features..."
echo "Running: cargo build --release --features v2,v2-gpu-cuda,v4"
cargo build --release --features v2,v2-gpu-cuda,v4 2>&1 | tee $RESULTS_DIR/build.log
echo "Build complete!"
echo ""

# Verify CUDA integration
echo "[3/7] Verifying CUDA integration..."
cargo run --release --features v2,v2-gpu-cuda,v4 --example test_v4_cuda_basic 2>&1 | tee $RESULTS_DIR/cuda_verify.log
if [ $? -eq 0 ]; then
    echo "✓ CUDA integration verified!"
else
    echo "✗ CUDA verification failed. Check cuda_verify.log"
    exit 1
fi
echo ""

# Run V4 CUDA geometric product benchmark (quick version)
echo "[4/7] Running V4 CUDA Geometric Product Benchmark (Quick)..."
cargo run --release --features v2,v2-gpu-cuda,v4 --example bench_v4_cuda_geometric_quick 2>&1 | tee $RESULTS_DIR/v4_geometric_quick.log
echo ""

if [ "$MODE" = "full" ]; then
    # Run full V4 CUDA geometric product benchmark
    echo "[5/7] Running V4 CUDA Geometric Product Benchmark (Full)..."
    cargo run --release --features v2,v2-gpu-cuda,v4 --example bench_v4_cuda_geometric 2>&1 | tee $RESULTS_DIR/v4_geometric_full.log
    echo ""

    # Run V4 CUDA packing benchmark
    echo "[6/7] Running V4 CUDA Packing Benchmark..."
    cargo run --release --features v2,v2-gpu-cuda,v4 --example bench_v4_cuda_packing 2>&1 | tee $RESULTS_DIR/v4_packing.log
    echo ""

    # Run division benchmark
    echo "[7/7] Running CUDA Division Benchmark..."
    cargo run --release --features v2,v2-gpu-cuda --example bench_division_cuda_gpu 2>&1 | tee $RESULTS_DIR/division_cuda.log
    echo ""
else
    echo "[5/7] Skipping full geometric benchmark (quick mode)"
    echo "[6/7] Skipping packing benchmark (quick mode)"
    echo "[7/7] Skipping division benchmark (quick mode)"
    echo ""
fi

# Generate summary
echo "Generating summary..."
cat > $RESULTS_DIR/SUMMARY.md << EOF
# CUDA Benchmark Results: $TIMESTAMP

## Environment
- **GPU**: $(nvidia-smi --query-gpu=name --format=csv,noheader)
- **VRAM**: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
- **CUDA**: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo "N/A")
- **Mode**: $MODE

## Benchmarks Run
EOF

if [ "$MODE" = "full" ]; then
    cat >> $RESULTS_DIR/SUMMARY.md << EOF
1. ✓ CUDA Integration Verification
2. ✓ V4 Geometric Product (Quick)
3. ✓ V4 Geometric Product (Full)
4. ✓ V4 Packing
5. ✓ Division

## Key Results
Extract key timing from log files:
\`\`\`
grep -E "geometric product|Geometric Product|ms|µs|speedup" $RESULTS_DIR/*.log
\`\`\`
EOF
else
    cat >> $RESULTS_DIR/SUMMARY.md << EOF
1. ✓ CUDA Integration Verification
2. ✓ V4 Geometric Product (Quick)
3. ○ V4 Geometric Product (Full) - skipped
4. ○ V4 Packing - skipped
5. ○ Division - skipped

To run full benchmarks:
\`\`\`
./scripts/run_cuda_benchmarks.sh full
\`\`\`
EOF
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Benchmarks Complete!                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Files:"
ls -la $RESULTS_DIR/
echo ""
echo "To extract key metrics:"
echo "  grep -E 'ms|µs|speedup|Geometric Product' $RESULTS_DIR/*.log"
echo ""
echo "To download results:"
echo "  # From your local machine:"
echo "  # scp -r root@<pod-ip>:/workspace/ga_engine/$RESULTS_DIR ."
echo ""
