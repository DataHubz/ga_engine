#!/bin/bash
# Cleanup script: Remove everything except Geometric ML and Clifford-LWE

echo "=== Cleaning up ga_engine project ==="
echo "Keeping only: Core GA, Geometric ML, Clifford-LWE"
echo ""

# Remove NTRU/Kyber crypto experiments (failed)
echo "Removing NTRU/Kyber experiments..."
rm -rf src/ntru
rm -rf src/kyber

# Remove lattice-related experiments (not our focus)
echo "Removing lattice experiments..."
rm -f benches/lattice_*.rs
rm -f benches/lll_*.rs
rm -f benches/svp_*.rs

# Remove NTRU benchmarks
echo "Removing NTRU benchmarks..."
rm -f benches/ntru_*.rs
rm -f benches/kyber_*.rs

# Remove matrix-to-multivector experiments (superseded by clifford_ring)
echo "Removing old matrix experiments..."
rm -f benches/matrix_to_multivector_*.rs
rm -f benches/matrix_mult_block_ga.rs
rm -f benches/matrix_vector_as_sparse_matrix.rs
rm -f benches/matrix_subspace_comprehensive.rs
rm -f benches/block_matrix_*.rs
rm -f benches/multivector_2d_vs_matrix_2x2.rs

# Remove DFT experiments (GA was slower)
echo "Removing DFT experiments..."
rm -f benches/dft_*.rs

# Remove orthogonalization experiments
echo "Removing orthogonalization experiments..."
rm -f benches/ga_orthogonalization_*.rs

# Remove generic test benchmarks
echo "Removing generic benchmarks..."
rm -f benches/bench.rs
rm -f benches/classical.rs
rm -f benches/ga_variants.rs
rm -f benches/test_minimal.rs
rm -f benches/raw_operation_analysis.rs

# Remove misc benchmarks not related to our focus
echo "Removing misc benchmarks..."
rm -f benches/matrix_matrixmultiply.rs
rm -f benches/matrix_ndarray*.rs
rm -f benches/matrix_vector.rs
rm -f benches/motor_transform.rs
rm -f benches/reflection_chain.rs
rm -f benches/rotor_vs_matrix_rotation.rs

# Remove old paper drafts and analysis docs
echo "Removing old documentation..."
cd paper
rm -f ntru_*.md
rm -f why_block_decomposition_fails.md
rm -f matrix_mult_block_ga_SUCCESS.md
rm -f sparse_matrix_hypothesis_analysis.md
rm -f ga_performance_source_analysis.md
rm -f REAL_WORLD_APPLICATIONS_N32.md
rm -f RING_ISOMORPHISM_ANALYSIS.md
rm -f CLIFFORD_RING_BREAKTHROUGH.md
rm -f CONCRETE_NEXT_STEPS.md
rm -f PAPER_STATUS.md
rm -f PAPER_SUMMARY.md
rm -f candidate_*.tex

# Remove old LaTeX papers
rm -rf latex/

cd ..

# Remove old examples
echo "Removing old examples..."
rm -f examples/lattice_crypto_example.rs
rm -f examples/ml_feature_transform.rs
rm -f examples/block_matrix_concept.rs

# Remove presentation materials (will recreate focused ones)
echo "Removing old presentations..."
rm -rf presentation/

# Remove old root-level docs
echo "Removing old root docs..."
rm -f MATRIX_TO_MULTIVECTOR_RESULTS.md
rm -f NTRU_IMPLEMENTATION_SUMMARY.md

echo ""
echo "=== Cleanup complete! ==="
echo ""
echo "Remaining structure:"
echo "  src/"
echo "    ├── ga.rs (core GA operations)"
echo "    ├── clifford_ring.rs (our main contribution)"
echo "    ├── multivector.rs, rotor.rs, etc. (supporting)"
echo "    └── nd/ (N-dimensional GA)"
echo "  examples/"
echo "    ├── geometric_ml_3d_classification.rs"
echo "    └── clifford_lwe_mvp.rs"
echo "  benches/"
echo "    └── clifford_ring_crypto.rs"
echo "  paper/"
echo "    └── UNDENIABLE_PERFORMANCE_WINS.md"
