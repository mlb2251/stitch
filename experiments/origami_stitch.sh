#!/bin/bash
set -e

if [ -z $STITCH_DIR ]
then
    echo "[bench_stitch.sh] Please set environment variable \$STITCH_DIR to the stitch/ directory"
    exit 1
fi

# compile Stitch
pushd $STITCH_DIR
cargo build --release --bin=compress
popd


OUT_DIR="$STITCH_DIR/data/dc/origami/out/stitch/$(TZ='America/New_York' date '+%Y-%m-%d_%H-%M-%S')"
mkdir -p $OUT_DIR/raw
mkdir -p $OUT_DIR/stderr

BENCH0=$STITCH_DIR/data/dc/origami/iteration_0_3.json
BENCH1=$STITCH_DIR/data/dc/origami/iteration_1_6.json
BENCH2=$STITCH_DIR/data/dc/origami/iteration_2_1.json
BENCH3=$STITCH_DIR/data/dc/origami/iteration_3_1.json


$BENCH_PATH=
BENCH=$(basename -s .json $BENCH_PATH)

/usr/bin/time -v $STITCH_DIR/target/release/compress -a4 -t8 -i3 --fmt=dreamcoder --dreamcoder-drop-last --out=$OUT_DIR/raw/$BENCH.json 2>&1 | tee $OUT_DIR/stderr/$BENCH.stderr

python analyze.py process stitch $OUT_DIR
