build:
	cargo build --release --bin=compress

test:
	cargo test

.PHONY: build test bindings-linux bindings-osx