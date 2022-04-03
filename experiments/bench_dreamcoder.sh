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

if [ -z $2 ]
then
    OUT_DIR="${BENCH_DIR}/out/dc/$(TZ='America/New_York' date '+%Y-%m-%d_%H-%M-%S')"
    mkdir -p $OUT_DIR/raw
    mkdir -p $OUT_DIR/stderr
    echo "Starting a new run: $OUT_DIR"
else
    OUT_DIR=$2
    echo "Resuming previous run: $OUT_DIR"
    if ! [[ $OUT_DIR == $BENCH_DIR* ]]; then
        echo "The output doesnt match with the bench dir"
        exit 1
    fi
fi



# run Dreamcoder on all the input files from the benchmark
for BENCH_PATH in $BENCH_DIR/bench*.json; do
    BENCH=$(basename -s .json $BENCH_PATH)
    RAW="$OUT_DIR/raw/$BENCH.json"
    STDERR="$OUT_DIR/stderr/$BENCH.stderr"

    if [ -s $RAW ] # check if file exists and nonempty
    then
        echo "Found existing output, skipping $BENCH"
        continue
    fi
    echo "[bench_dreamcoder.sh] Running Dreamcoder on: $BENCH"


    /usr/bin/time -v $COMPRESSION_BIN $BENCH_PATH > $RAW 2> >(tee $STDERR >&2)
done

echo "Done: $OUT_DIR"

python analyze.py process dreamcoder $OUT_DIR

echo "you can compare to stitch with: ./bench_stitch.sh $BENCH_DIR $OUT_DIR"

