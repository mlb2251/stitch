build:
	cargo build --release --bin=compress

test:
	cargo test --release

update-tests:
	make test | grep "cp \"out/"

.PHONY: build test update-tests