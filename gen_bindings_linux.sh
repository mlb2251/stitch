#!/bin/bash
# This script just compiles the python bindings for stitch. While there are nicer ways to properly publish a library,
# for development purposes this is fine. If you add "path/to/stitch/target/release/" to your python path then 
# you can import stitch with `import stitch`. You can modify your python path with `sys.path.append("path/to/stitch/bindings/")`
# or in your ~/.bashrc with `export PYTHONPATH="$PYTHONPATH:path/to/stitch/bindings/"`
# alternatively, to test the bindings just do `cd bindings && python` and then `import stitch`
set -e
mkdir -p bindings
cargo rustc --release --example=stitch --features=python -- -C link-arg=-undefined
mv target/release/examples/libstitch.so bindings/stitch.so
echo "added bindings: bindings/stitch.so"
