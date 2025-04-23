# Makefile

.PHONY: coverage

coverage:
	@echo "⏳ Running coverage…"
	cargo llvm-cov \
	  --json \
	  --summary-only \
	  --output-path cov.json
	cargo run --bin coverage_summary