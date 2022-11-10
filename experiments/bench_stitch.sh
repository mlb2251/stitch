#!/bin/bash
set -e

if [ -z $STITCH_DIR ]
then
    echo "[bench_stitch.sh] Please set environment variable \$STITCH_DIR to the stitch/ directory"
    exit 1
fi

MODE=$1

if [ $MODE == "compare" ]; then
    echo "[bench_stitch.sh] Running in strict mode"
    COMPARE_TO=$2
    BENCH_DIR=$(dirname $(dirname $(dirname $COMPARE_TO)))
elif [ $MODE == "nocompare" ]; then
    echo "[bench_stitch.sh] Running in relaxed mode"
    BENCH_DIR=$2
else
    echo "[bench_stitch.sh] Unknown mode: $MODE"
    exit 1
fi

if [[ $OSTYPE == 'darwin'* ]]
then
    GTIME="gtime"
else
    GTIME="/usr/bin/time"
fi


OUT_DIR="${BENCH_DIR}/out/stitch/$(TZ='America/New_York' date '+%Y-%m-%d_%H-%M-%S')"
# A dreamcoder run we should compare to (in particular used to get the iteration budget)
# or pass "none" for forcing 20 iterations

# compile Stitch
pushd $STITCH_DIR
cargo build --release --bin=compress
popd

mkdir -p $OUT_DIR/raw
mkdir -p $OUT_DIR/stderr


STITCH_FLAGS="--max-arity=3 --threads=1 --fmt=dreamcoder --dreamcoder-comparison"

# run Stitch on all the input files from the run
for BENCH_PATH in $BENCH_DIR/bench*.json; do
    BENCH=$(basename -s .json $BENCH_PATH)
    if [ $MODE == "compare" ]; then
        ITERATIONS=$(python3 analyze.py iteration_budget $COMPARE_TO $BENCH_PATH)
        if [ $ITERATIONS -eq 0 ]; then
            echo "skipping $BENCH_PATH since comparison is nonexistant or has zero iterations"
            continue
        fi
    else
        ITERATIONS=10 # whatever
    fi
    echo "[bench_stitch.sh] Running Stitch on: $BENCH"
    echo "$STITCH_DIR/target/release/compress $BENCH_PATH --iterations=$ITERATIONS $STITCH_FLAGS"
    $GTIME -v $STITCH_DIR/target/release/compress $BENCH_PATH --iterations=$ITERATIONS $STITCH_FLAGS --out=$OUT_DIR/raw/$BENCH.json 2>&1 &> $OUT_DIR/stderr/$BENCH.stderr
done

python3 analyze.py process stitch $OUT_DIR

echo "Done: $OUT_DIR"

echo "You can graph with: python3 analyze.py graphs bar $OUT_DIR $COMPARE_TO"

