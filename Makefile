build:
	cargo build --release --bin=compress

test:
	cargo test --release

test-update:
	make test | grep "cp \"out/"

claims: claim-1 claim-2 claim-3

claim-1:
	cd experiments && make claim-1

SEEDS := 3

claim-2:
	cd experiments && make claim-2 SEEDS=${SEEDS}

claim-3:
	cd experiments && make claim-3

.PHONY: build test test-update claim-1 claim-2 claim-3 claims