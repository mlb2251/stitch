build:
	cargo build --release --bin=compress

test:
	cargo test

bindings-linux:
	./gen_bindings_linux.sh

bindings-osx:
	./gen_bindings_osx.sh

.PHONY: build test bindings-linux bindings-osx