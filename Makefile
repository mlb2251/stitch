build:
	cargo build --release --bin=compress

test:
	cargo test --release --test integration_tests

claim-1:
	./bench_stitch_all_latest.sh compression_benchmark/benches
	./process_all.sh compression_benchmark/benches
	python3 analyze.py graph_all compression_benchmark/benches

.PHONY: build test claim-1