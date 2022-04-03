#!/bin/bash
set -e

if [ -z $COMPRESSION_BIN ]
then
    echo "Please set environment variable \$COMPRESSION_BIN to the ocaml compression binary" 
    exit 1
fi

if [ -z $1 ]
then
    echo "Usage: ./bench_dreamcoder.sh benches/bench_name"
    exit 1
fi

BENCH_DIR=$1
OUT_DIR="${BENCH_DIR}/out/dc/$(TZ='America/New_York' date '+%Y-%m-%d_%H-%M-%S')"

mkdir -p $OUT_DIR/raw
mkdir -p $OUT_DIR/stderr

# run Dreamcoder on all the input files from the benchmark
for BENCH_PATH in $BENCH_DIR/bench*.json; do
    BENCH=$(basename -s .json $BENCH_PATH)
    echo "[bench_dreamcoder.sh] Running Dreamcoder with on: $BENCH"

    /usr/bin/time -v $COMPRESSION_BIN $BENCH_PATH > $OUT_DIR/raw/$BENCH.json 2> >(tee $OUT_DIR/stderr/$BENCH.stderr >&2)
done

echo "Done: $OUT_DIR"

python analyze.py process dreamcoder $OUT_DIR

echo "you can compare to stitch with: ./bench_stitch.sh $BENCH_DIR $OUT_DIR"

