build:
	cargo build --release --bin=compress

test:
	cargo test --release

.PHONY: build test bindings-linux bindings-osx