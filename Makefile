build:
	cargo build --release --bin=compress

test:
	cargo test --release

test-update:
	make test | grep "cp \"out/"

.PHONY: build test test-update