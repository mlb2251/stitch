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

check-dfa:
	@curl -s https://raw.githubusercontent.com/kavigupta/neurosym-lib/main/test_data/dfa.json > /tmp/remote_dfa.json; \
	if ! diff -q test_data/dfa.json /tmp/remote_dfa.json; then \
		echo "DFA files do not match!"; \
		rm -f /tmp/remote_dfa.json; \
		exit 1; \
	else \
		echo "DFA files match."; \
		rm -f /tmp/remote_dfa.json; \
	fi

flamegraph:
	CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --root --open --deterministic --output=out/flamegraph.svg --bin=compress -- data/cogsci/furniture.json -a3

.PHONY: build test test-update claim-1 claim-2 claim-3 claims check-dfa