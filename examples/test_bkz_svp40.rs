use ga_engine::lattice_reduction::svp_challenge;
use ga_engine::lattice_reduction::bkz_baseline::BKZ;
use ga_engine::lattice_reduction::lll_baseline::LLL;
use std::time::Instant;

fn main() {
    println!("=== BKZ vs LLL on SVP Challenge Dimension 40 ===\n");

    let filename = "data/lattices/svpchallengedim40seed0.txt";

    match svp_challenge::parse_lattice_file(filename) {
        Ok(basis) => {
            println!("✓ Parsed {} x {} lattice", basis.len(), basis[0].len());

            // Check for overflow
            let max_entry = basis.iter()
                .flat_map(|row| row.iter())
                .map(|x| x.abs())
                .fold(0.0, f64::max);

            if max_entry.is_infinite() {
                println!("\n⚠️  Warning: Lattice contains inf values - f64 overflow!");
                return;
            }

            println!("Max entry magnitude: {:.3e}\n", max_entry);

            // Find shortest in original basis
            let (idx, orig_norm) = svp_challenge::find_shortest_vector(&basis);
            println!("Original shortest: b{} with norm {:.6e}\n", idx, orig_norm);

            // Test 1: LLL Baseline
            println!("--- LLL Baseline (δ=0.99) ---");
            let start = Instant::now();
            let mut lll = LLL::new(basis.clone(), 0.99);
            lll.reduce();
            let lll_time = start.elapsed();

            let lll_basis = lll.get_basis();
            let (_, lll_norm) = svp_challenge::find_shortest_vector(lll_basis);
            let lll_hf = lll.hermite_factor();
            let lll_stats = lll.get_stats();

            println!("Time: {:.2?}", lll_time);
            println!("Shortest norm: {:.6e}", lll_norm);
            println!("Hermite factor: {:.6}", lll_hf);
            println!("Stats: {} swaps, {} size_reductions\n",
                     lll_stats.swaps, lll_stats.size_reductions);

            // Test 2: BKZ with block size 10
            println!("--- BKZ (β=10, δ=0.99) ---");
            let start = Instant::now();
            let mut bkz10 = BKZ::new(basis.clone(), 10, 0.99);
            bkz10.reduce_with_limit(3); // 3 tours max
            let bkz10_time = start.elapsed();

            let bkz10_basis = bkz10.get_basis();
            let (_, bkz10_norm) = svp_challenge::find_shortest_vector(bkz10_basis);
            let bkz10_hf = bkz10.hermite_factor();
            let bkz10_stats = bkz10.get_stats();

            println!("Time: {:.2?}", bkz10_time);
            println!("Shortest norm: {:.6e}", bkz10_norm);
            println!("Hermite factor: {:.6}", bkz10_hf);
            println!("Stats: {:?}\n", bkz10_stats);

            // Test 3: BKZ with block size 20 (if reasonable time)
            if bkz10_time.as_secs() < 60 {
                println!("--- BKZ (β=20, δ=0.99) ---");
                let start = Instant::now();
                let mut bkz20 = BKZ::new(basis.clone(), 20, 0.99);
                bkz20.reduce_with_limit(2); // 2 tours max
                let bkz20_time = start.elapsed();

                let bkz20_basis = bkz20.get_basis();
                let (_, bkz20_norm) = svp_challenge::find_shortest_vector(bkz20_basis);
                let bkz20_hf = bkz20.hermite_factor();
                let bkz20_stats = bkz20.get_stats();

                println!("Time: {:.2?}", bkz20_time);
                println!("Shortest norm: {:.6e}", bkz20_norm);
                println!("Hermite factor: {:.6}", bkz20_hf);
                println!("Stats: {:?}\n", bkz20_stats);

                // Comparison
                println!("=== Comparison ===");
                println!("Original: {:.6e}", orig_norm);
                println!("LLL:      {:.6e} (HF: {:.6}, time: {:.2?})",
                         lll_norm, lll_hf, lll_time);
                println!("BKZ-10:   {:.6e} (HF: {:.6}, time: {:.2?})",
                         bkz10_norm, bkz10_hf, bkz10_time);
                println!("BKZ-20:   {:.6e} (HF: {:.6}, time: {:.2?})",
                         bkz20_norm, bkz20_hf, bkz20_time);

                println!("\nImprovements:");
                println!("LLL vs Original: {:.2}x shorter", orig_norm / lll_norm);
                println!("BKZ-10 vs LLL:   {:.2}x shorter", lll_norm / bkz10_norm);
                println!("BKZ-20 vs BKZ-10: {:.2}x shorter", bkz10_norm / bkz20_norm);
            } else {
                println!("⚠️  BKZ-10 took > 1 minute, skipping BKZ-20");
            }
        }
        Err(e) => {
            println!("✗ Failed to parse {}: {}", filename, e);
        }
    }
}
