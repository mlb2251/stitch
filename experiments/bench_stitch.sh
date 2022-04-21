#!/bin/bash
set -e

if [ -z $STITCH_DIR ]
then
    echo "[bench_stitch.sh] Please set environment variable \$STITCH_DIR to the stitch/ directory"
    exit 1
fi

# if [ -z $2 ]
# then
#     echo "[bench_stitch.sh] Usage: ./launch_stitch_run.sh mod ebenches/bench_name benches/bench_name/out/dc/run_name"
#     exit 1
# fi

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



OUT_DIR="${BENCH_DIR}/out/stitch/$(TZ='America/New_York' date '+%Y-%m-%d_%H-%M-%S')"
# A dreamcoder run we should compare to (in particular used to get the iteration budget)
# or pass "none" for forcing 20 iterations

# LOOSE=$3 # set to "loose" if you want to ignore failures to match up iteration budgets

# compile Stitch
pushd $STITCH_DIR
cargo build --release --bin=compress
popd

mkdir -p $OUT_DIR/raw
mkdir -p $OUT_DIR/stderr

# run Stitch on all the input files from the run
for BENCH_PATH in $BENCH_DIR/bench*.json; do
    BENCH=$(basename -s .json $BENCH_PATH)
    if [ $MODE == "compare" ]; then
        ITERATIONS=$(python3 analyze.py iteration_budget $COMPARE_TO $BENCH_PATH)
    else
        ITERATIONS=10 # whatever
    fi
    echo "[bench_stitch.sh] Running Stitch with -a3 on: $BENCH"
    /usr/bin/time -v $STITCH_DIR/target/release/compress $BENCH_PATH --max-arity=3 --threads=8 --iterations=$ITERATIONS --fmt=dreamcoder --dreamcoder-comparison --out=$OUT_DIR/raw/$BENCH.json 2>&1 | tee $OUT_DIR/stderr/$BENCH.stderr
done

python analyze.py process stitch $OUT_DIR

echo "Done: $OUT_DIR"

echo "You can graph with: python3 analyze.py graphs bar $OUT_DIR $COMPARE_TO"


# python3 analyze.py run_invention_info_stitch out/$DOMAIN/$RUN
# echo "Comparing..."
# python3 analyze.py compare out/$DOMAIN/$RUN
