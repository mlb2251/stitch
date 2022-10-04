build:
	cargo build --release --bin=compress

test:
	cargo test --release --test integration_tests

.PHONY: build test